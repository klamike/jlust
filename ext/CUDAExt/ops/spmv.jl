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
                        transa::Char='N') where {T<:_CUSPARSE_ELTYPES, Ti}
    idx   = _cusparse_index(u_A)
    m, n  = Int64.(extents(u_A))
    spmat = CuSparseMatrixDescriptor(_to_cuspmat(u_A), idx)
    x_len = transa == 'N' ? n : m
    y_len = transa == 'N' ? m : n
    descX = CuDenseVectorDescriptor(T, x_len)
    descY = CuDenseVectorDescriptor(T, y_len)
    algo  = CUSPARSE_SPMV_ALG_DEFAULT
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
function JLUST.execute(h::CUSPARSESpMVHandle{T},
                       u_x::USTensor, u_y::USTensor;
                       alpha=one(T), beta=zero(T)) where T
    _cusparse_set_dense!(h.dnvec_x, nonzeros(u_x))
    _cusparse_set_dense!(h.dnvec_y, nonzeros(u_y))
    cusparseSpMV(handle(), h.transa,
                 Ref{T}(alpha), h.spmat_desc, h.dnvec_x,
                 Ref{T}(beta),  h.dnvec_y,
                 T, h.algo, h.workspace)
    return u_y
end


# ─── CUDA warp-level COO SpMV (hook override) ────────────────────────────────
#
# Warp segmented reduce: each thread handles one NNZ.  Within each warp of 32,
# shfl_down accumulates same-row contributions rightward; the leftmost thread
# of each row segment holds the segment sum and performs one @atomic write.
# Reduces global atomics by ~min(32, avg_nnz_per_row)× vs one-per-NNZ, and
# avoids branch-heavy sequential accumulation of the chunked 8-NNZ kernel.

function _coo_spmv_warp_kernel!(row_crd, col_crd, nzval, x, y, origin_off, n_nnz)
    T    = eltype(nzval)
    tid  = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    lane0 = (threadIdx().x - Int32(1)) % Int32(32)   # 0-based lane (0–31)
    MASK  = UInt32(0xffffffff)

    my_row = Int32(-1)
    my_val = zero(T)

    if tid <= n_nnz
        col    = Int(col_crd[tid]) - Int(origin_off) + 1
        my_row = Int32(row_crd[tid]) - origin_off        # 0-based
        my_val = nzval[tid] * x[col]
    end

    orig_row = my_row   # immutable copy for segment boundary detection

    # Segmented warp reduce (shfl_down, offsets 1/2/4/8/16):
    # After this, the leftmost thread of each same-row segment holds the sum.
    for δ in (Int32(1), Int32(2), Int32(4), Int32(8), Int32(16))
        peer_val = CUDA.shfl_down_sync(MASK, my_val, δ)
        peer_row = CUDA.shfl_down_sync(MASK, orig_row, δ)
        if peer_row == orig_row && orig_row >= Int32(0)
            my_val += peer_val
        end
    end

    # Segment head: lane 0 is always a head; otherwise check if left neighbor
    # has a different row (using the immutable orig_row for correctness).
    prev_row = CUDA.shfl_up_sync(MASK, orig_row, UInt32(1))
    is_head  = (lane0 == Int32(0)) | (prev_row != orig_row)

    if is_head & (orig_row >= Int32(0))
        CUDA.@atomic y[orig_row + 1] += my_val
    end

    return nothing
end

# Dispatches on CuVector to intercept the KernelAbstractionsExt COO path.
function JLUST._coo_spmv_specialized!(
        row_crd::CuVector, col_crd::CuVector,
        nzval::CuVector{T}, x::AbstractVector{T}, y::AbstractVector{T},
        origin_off::Int32, n_nnz::Int32) where T
    n_nnz == Int32(0) && return true
    threads = 256
    blocks  = cld(Int(n_nnz), threads)
    CUDA.@cuda threads=threads blocks=blocks _coo_spmv_warp_kernel!(
        row_crd, col_crd, nzval, x, y, origin_off, n_nnz)
    return true
end

# ─── CUDA vector-per-row CSR SpMV (hook override) ────────────────────────────
#
# Val{VS} threads collaborate on each row: each handles NNZ at stride VS.
# A log2(VS)-step warp-shuffle tree reduces within the VS-thread group;
# lane 0 writes y[row]. VS is selected adaptively by average NNZ/row so that
# the number of inner-loop iterations per thread stays near the sweet spot
# (enough to hide latency, not so many that parallelism is wasted).
#
# Group mask: VS consecutive bits starting at the group's base lane.
# All threads in the group participate unconditionally so that shfl_sync
# sees a consistent mask; threads beyond n_outer contribute 0.
#
# log2(VS)-step warp shuffle reduction emitted at compile time.  VS must be a
# power of two ≤ 32; verified at the @generated boundary.
@generated function _csr_group_reduce(val, mask::UInt32, ::Val{VS}) where VS
    @assert VS isa Integer && VS > 0 && (VS & (VS - 1)) == 0 && VS <= 32 "VS must be a power of 2 in {1,2,4,8,16,32}"
    body = Expr(:block)
    δ = 1
    while δ < VS
        push!(body.args, :(val += CUDA.shfl_down_sync(mask, val, Int32($δ))))
        δ <<= 1
    end
    push!(body.args, :(return val))
    body
end

# ZERO_BETA: compile-time flag; when true, writes my_acc directly (no y read).
# When false, performs y[row] = my_acc + beta * y[row] accumulation.
function _csr_spmv_vector_kernel!(pos, crd, nzval, x, y, origin_off, n_outer,
                                   ::Val{VS}, beta, ::Val{ZERO_BETA}) where {VS, ZERO_BETA}
    T        = eltype(nzval)
    vs       = Int32(VS)
    tid      = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    row_id   = (tid - Int32(1)) ÷ vs + Int32(1)
    vec_lane = (tid - Int32(1)) % vs

    lane_in_warp  = (threadIdx().x - Int32(1)) % Int32(32)
    group_in_warp = lane_in_warp ÷ vs
    group_bits    = (UInt32(1) << UInt32(VS)) - UInt32(1)   # VS consecutive 1-bits
    group_mask    = group_bits << (UInt32(group_in_warp) * UInt32(VS))

    my_acc = zero(T)
    if row_id <= n_outer
        lo = Int(pos[row_id])     - Int(origin_off)
        hi = Int(pos[row_id + 1]) - Int(origin_off)
        k  = lo + Int(vec_lane) + 1
        while k <= hi
            col     = Int(crd[k]) - Int(origin_off) + 1
            my_acc += nzval[k] * x[col]
            k      += Int(vs)
        end
    end

    my_acc = _csr_group_reduce(my_acc, group_mask, Val(VS))

    if vec_lane == Int32(0) && row_id <= n_outer
        y[row_id] = ZERO_BETA ? my_acc : my_acc + beta * y[row_id]
    end
    return nothing
end

# Cached SM count: queried once from CUDA device, reused across all SpMV calls.
# Reset to 0 to force a re-query (e.g. after device switch).
const _cuda_n_SMs = Ref{Int}(0)
function _cuda_sm_count()
    if _cuda_n_SMs[] == 0
        _cuda_n_SMs[] = Int(CUDA.attribute(CUDA.device(),
                                           CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
    end
    _cuda_n_SMs[]
end

# Select VS (threads per CSR row) using two criteria combined with max():
#
#   NNZ-based floor (vs_nnz): enough inner-loop iterations per thread to
#   amortise the warp-shuffle overhead and hide latency within the thread.
#     avg_nnz < 4  → VS=2,  < 8  → VS=4,  < 16 → VS=8,
#     < 32         → VS=16, ≥ 32 → VS=32.
#
#   Occupancy-based floor (vs_occ): enough thread blocks so that the SM
#   stays bandwidth-bound rather than latency-bound.  Target ≥ 6 blocks/SM
#   (= 48 warps/SM at 8 warps/256-thread block, matching the hardware limit).
#   blocks = ⌈n_rows × VS / 256⌉ ≥ 6 × n_SMs  →  VS ≥ ⌈6 × n_SMs × 256 / n_rows⌉.
#   Result rounded up to the next power of two in {2,4,8,16,32}.
#
#   Example — L40S (142 SMs), negI (35 393 rows, avg_nnz=1):
#     vs_nnz=2 (avg_nnz<4), vs_occ=⌈6×142×256/35393⌉=⌈6.2⌉=8  →  VS=8.
#     With VS=2: 277 blocks/142 SMs = 15 warps/SM (latency-bound).
#     With VS=8: 1106 blocks/142 SMs = 62 warps/SM → capped at 48 (bandwidth-bound). ✓
#
# For VS=32 the warp mask formula (UInt32(1)<<32)-1 wraps to 0xffffffff by
# UInt32 overflow semantics on GPU, giving the correct full-warp mask.
function JLUST._csr_spmv_specialized!(
        pos::CuVector, crd::CuVector,
        nzval::CuVector{T}, x::AbstractVector{T}, y::AbstractVector{T},
        origin_off::Int32, n_outer::Int32, beta::T) where T
    n_outer == Int32(0) && return true
    avg_nnz = length(nzval) / Int(n_outer)
    n_SMs   = _cuda_sm_count()
    threads = 256

    # NNZ-based floor: enough inner iterations per thread to hide latency.
    # Split at 2 so avg_nnz ∈ [2,4) maps to VS=4 rather than VS=2.
    # Values must be powers of 2 in {2,4,8,16,32} to match the kernel dispatch below.
    vs_nnz = avg_nnz < 2.0  ? 2  :
             avg_nnz < 4.0  ? 4  :
             avg_nnz < 8.0  ? 8  :
             avg_nnz < 16.0 ? 16 : 32

    # Occupancy floor: VS such that blocks/SM ≥ 6.
    # Capped at vs_nnz × 2: beyond one doubling, idle threads outweigh the
    # occupancy gain (e.g. VS=32 for avg_nnz=4 wastes 87% of threads).
    vs_occ_raw = cld(6 * n_SMs * threads, Int(n_outer))
    vs_occ = vs_occ_raw <= 2  ? 2  :
             vs_occ_raw <= 4  ? 4  :
             vs_occ_raw <= 8  ? 8  :
             vs_occ_raw <= 16 ? 16 : 32

    vs        = min(max(vs_nnz, min(vs_occ, vs_nnz * 2)), 32)
    zero_beta = Val(iszero(beta))
    blocks    = cld(Int(n_outer) * vs, threads)
    _vs_dispatch(vs) do vsv
        CUDA.@cuda threads=threads blocks=blocks _csr_spmv_vector_kernel!(
            pos, crd, nzval, x, y, origin_off, n_outer, vsv, beta, zero_beta)
    end
    return true
end

# Lift a runtime `vs ∈ {2,4,8,16,32}` to a compile-time `Val(vs)` and call `f`.
# Anything else is rejected — the kernel only specializes for these unroll widths.
@inline function _vs_dispatch(f::F, vs::Int) where F
    vs == 2  && return f(Val(2))
    vs == 4  && return f(Val(4))
    vs == 8  && return f(Val(8))
    vs == 16 && return f(Val(16))
    vs == 32 && return f(Val(32))
    error("_vs_dispatch: vs=$vs not in {2,4,8,16,32}")
end
