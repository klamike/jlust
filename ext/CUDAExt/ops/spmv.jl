# ─── cuSPARSE sparse matrix-vector product ───────────────────────────────────
#
# y ← alpha * op(A) * x + beta * y

# ─── Handle ──────────────────────────────────────────────────────────────────

mutable struct CUSPARSESpMVHandle{T} <: JLUST.AbstractKernelHandle
    spmat_desc::CuSparseMatrixDescriptor
    dnvec_x::CuDenseVectorDescriptor
    dnvec_y::CuDenseVectorDescriptor
    workspace::CuVector{UInt8}
    transa::Char
    algo::cusparseSpMVAlg_t
end

export CUSPARSESpMVHandle

function JLUST.prepare(::CUSPARSEBackend, ::Type{<:Op{:SpMV}}, u_A::USTensor{T,Ti};
                        transa::Char='N',
                        algo::cusparseSpMVAlg_t=CUSPARSE_SPMV_ALG_DEFAULT) where {T<:_CUSPARSE_ELTYPES, Ti}
    idx   = _cusparse_index(u_A)
    m, n  = Int64.(extents(u_A))
    spmat = CuSparseMatrixDescriptor(_to_cuspmat(u_A), idx)
    x_len = transa == 'N' ? n : m
    y_len = transa == 'N' ? m : n
    descX = CuDenseVectorDescriptor(T, x_len)
    descY = CuDenseVectorDescriptor(T, y_len)
    α_ref = Ref{T}(one(T));  β_ref = Ref{T}(zero(T))

    ws = _cusparse_workspace() do buf_sz, buf
        if buf === CUDA.CU_NULL
            cusparseSpMV_bufferSize(handle(), transa, α_ref, spmat, descX, β_ref, descY, T, algo, buf_sz)
        else
            cusparseSpMV_preprocess(handle(), transa, α_ref, spmat, descX, β_ref, descY, T, algo, buf)
        end
    end
    CUSPARSESpMVHandle{T}(spmat, descX, descY, ws, transa, algo)
end

function JLUST.update_values!(h::CUSPARSESpMVHandle, u_A::USTensor)
    cusparseSpMatSetValues(h.spmat_desc, nonzeros(u_A))
    return h
end

# ─── Execution ────────────────────────────────────────────────────────────────
#
# Direct path uses CUDA.jl's high-level `CUSPARSE.mv!` — graph-capture safe and
# avoids per-call descriptor allocation.  The handle path (below) is for code
# that calls SpMV repeatedly with the same matrix; it pre-builds descriptors
# and workspace at `prepare()` time and reuses them across invocations.

function JLUST.execute(::CUSPARSEBackend, ::Op{:SpMV, F},
                       u_A::USTensor{T,Ti}, u_x::USTensor, u_y::USTensor;
                       transa::Char='N',
                       alpha=one(T), beta=zero(T)) where {F, T<:_CUSPARSE_ELTYPES, Ti}
    cusA = _to_cuspmat(u_A)
    idx  = _cusparse_index(u_A)
    CUSPARSE.mv!(transa, T(alpha), cusA, nonzeros(u_x), T(beta), nonzeros(u_y), idx)
    return u_y
end

# Handle path: A's descriptor is baked in at prepare time, so execute only
# takes the dense operands.  Refresh A's values via `update_values!(h, u_A)`.
#
# Two argument forms exist: `(h, u_x::USTensor, u_y::USTensor)` for the public
# API, and `(h, x::CuVector, y::CuVector)` for hot internal paths that already
# hold raw device arrays — the USTensor wrappers allocate ~10 small host objects
# per call which would dominate at the ~10μs scale.
function JLUST.execute(h::CUSPARSESpMVHandle{T},
                       x::CuVector, y::CuVector;
                       alpha=one(T), beta=zero(T)) where T
    _cusparse_set_dense!(h.dnvec_x, x)
    _cusparse_set_dense!(h.dnvec_y, y)
    cusparseSpMV(handle(), h.transa,
                 Ref{T}(alpha), h.spmat_desc, h.dnvec_x,
                 Ref{T}(beta),  h.dnvec_y,
                 T, h.algo, h.workspace)
    return y
end

@inline JLUST.execute(h::CUSPARSESpMVHandle{T}, u_x::USTensor, u_y::USTensor;
                      alpha=one(T), beta=zero(T)) where T =
    (JLUST.execute(h, nonzeros(u_x), nonzeros(u_y); alpha=alpha, beta=beta); u_y)


# ─── Walker-driven SpMV specializations ─────────────────────────────────────
#
# Two CUDA-specific SpMV kernels used to live here:
#
#   1. `_csr_spmv_specialized!` — VS-threads-per-row warp-vector with shuffle
#      reduce + LDG-wrapped x.  Now emitted by the generic walker for any
#      (Dense/Batch outer + Compressed-unique inner) format, gated by the
#      `_supports_warp_vector(::CUDABackend) = true` trait declared in
#      CUDAExt.jl.  See KernelAbstractionsExt/ops/_walker.jl + spmv.jl.
#
#   2. `_coo_spmv_specialized!` / `_coo_spmv_warp_kernel!` — segmented-warp-
#      reduce COO SpMV (1 thread/NNZ + log2(32) shfl_down + per-segment-head
#      atomic).  Now emitted by the same walker for any non-unique
#      Compressed outer (sorted-COO row list), again gated by the warp-vector
#      trait.  `_warp_seg_reduce_sum_down` (defined in CUDAExt.jl) provides
#      the device-side primitive the walker calls.
#
# Both retirements mean DCSR, custom user formats, sorted-row-list shapes, and
# any future variant get the warp-shuffle treatment without per-format work —
# the CSR/COO specializations are no longer load-bearing.

# ─── Merge-based (NNZ-partitioned) CSR SpMV ──────────────────────────────────
#
# For graph-like / low-mean-degree CSR matrices, the walker's row-parallel
# warp-vector kernel underperforms because most threads land on rows with 0–1
# NNZ.  The merge-based kernel partitions the NNZ axis into fixed-size chunks
# (NPC NNZ per warp); each warp:
#
#   1. Has each thread binary-search rowptr to find the row of its NNZ.
#   2. Computes nzval[k] * x[col[k]].
#   3. Segmented warp-reduce: same-row contributions sum within the warp.
#   4. Each segment head atomic-adds to y[row].
#
# Matches cuSPARSE's CSR_ALG1 strategy at a high level (load-balance over
# NNZ, not rows).  Same CSR storage; no format change visible to users.
#
# NPC=64 is empirically the best on L40S across the SuiteSparse curated set:
# small enough to keep many warps in flight, large enough to amortize the
# binary-search overhead.

const _CSR_SPMV_MERGE_NPC = 64
const _CSR_SPMV_MERGE_THREADS = 256

function _csr_spmv_merge_kernel!(_pos1, _crd1, _nzval, _x_raw, _y, _origin_off,
                                  _n_rows, _n_nnz, _alpha, ::Val{NPC}) where NPC
    bid  = blockIdx().x
    tid  = threadIdx().x
    bs   = blockDim().x
    warp_in_block = (tid - Int32(1)) ÷ Int32(32)
    lane = (tid - Int32(1)) % Int32(32)
    n_warps_per_block = bs ÷ Int32(32)
    global_warp = (bid - Int32(1)) * n_warps_per_block + warp_in_block

    nnz_lo = Int32(global_warp) * Int32(NPC)
    nnz_hi = min(nnz_lo + Int32(NPC), _n_nnz)
    if nnz_lo >= _n_nnz
        return nothing
    end

    _x = Base.Experimental.Const(_x_raw)
    T = eltype(_nzval)
    MASK = UInt32(0xffffffff)

    chunk_pos = nnz_lo
    while chunk_pos < nnz_hi
        my_nnz = chunk_pos + lane    # 0-based NNZ index
        my_row = Int32(-1)
        v = zero(T)
        if my_nnz < nnz_hi
            # Per-thread binary search: smallest k in [2..n_rows+1] s.t.
            # rowptr[k] - origin_off > my_nnz.  Then row = k - 2 (0-based).
            lo = Int32(2); hi = _n_rows + Int32(2)
            while lo < hi
                mid = (lo + hi) ÷ Int32(2)
                vp = Int32(_pos1[mid]) - _origin_off
                if vp <= my_nnz
                    lo = mid + Int32(1)
                else
                    hi = mid
                end
            end
            my_row = lo - Int32(2)
            col = Int(_crd1[my_nnz + Int32(1)]) - Int(_origin_off) + Int(1)
            v = _alpha * _nzval[my_nnz + Int32(1)] * _x[col]
        end

        # Segmented warp reduce by my_row.  shfl_down returns own value when
        # source lane >= warpsize, so we explicitly bound (lane + δ < 32) to
        # avoid summing into ourselves at the warp's right edge.
        orig_row = my_row
        for δ in (Int32(1), Int32(2), Int32(4), Int32(8), Int32(16))
            peer_v   = CUDA.shfl_down_sync(MASK, v, δ)
            peer_row = CUDA.shfl_down_sync(MASK, orig_row, δ)
            if (lane + δ < Int32(32)) && peer_row == orig_row && orig_row >= Int32(0)
                v += peer_v
            end
        end
        prev_row = CUDA.shfl_up_sync(MASK, orig_row, UInt32(1))
        is_head = (lane == Int32(0)) | (prev_row != orig_row)
        if is_head & (orig_row >= Int32(0))
            CUDA.@atomic _y[orig_row + Int32(1)] += v
        end
        chunk_pos += Int32(32)
    end
    return nothing
end

# Override of the JLUST hook — only fires for CuArray-backed buffers.
# Returns true to signal that execution happened (skip the walker fallback).
function JLUST._csr_spmv_merge!(rowptr::CuVector, colind::CuVector, nzval::CuVector{T},
                                 x::CuVector{T}, y::CuVector{T},
                                 origin_off::Int32, n_rows::Int32, n_nnz::Int32,
                                 alpha::T) where T
    n_nnz == Int32(0) && return true
    npc = _CSR_SPMV_MERGE_NPC
    threads = _CSR_SPMV_MERGE_THREADS
    n_warps = cld(Int(n_nnz), npc)
    n_warps_per_block = threads ÷ 32
    n_blocks = cld(n_warps, n_warps_per_block)
    CUDA.@cuda threads=threads blocks=n_blocks _csr_spmv_merge_kernel!(
        rowptr, colind, nzval, x, y, origin_off, n_rows, n_nnz, alpha, Val(npc))
    return true
end
