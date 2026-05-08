# ─── cuSPARSE sparse matrix-vector product ───────────────────────────────────
#
# y ← alpha * op(A) * x + beta * y

# ─── Handle ──────────────────────────────────────────────────────────────────

mutable struct CUSPARSESpMVHandle{T}
    spmat_desc::CuSparseMatrixDescriptor
    dnvec_x::CuDenseVectorDescriptor
    dnvec_y::CuDenseVectorDescriptor
    workspace::CuVector{UInt8}
    transa::Char
    algo::cusparseSpMVAlg_t
end

export CUSPARSESpMVHandle

function JLUST.prepare(::CUSPARSEBackend, ::Type{SpMVOp}, u_A::USTensor{T,Ti};
                        transa::Char='N') where {T<:_CUSPARSE_ELTYPES, Ti}
    idx   = _cusparse_index(u_A)
    m, n  = Int64.(extents(u_A))
    spmat_desc = CuSparseMatrixDescriptor(_to_cuspmat(u_A), idx)

    x_len = transa == 'N' ? n : m
    y_len = transa == 'N' ? m : n
    descX = CuDenseVectorDescriptor(T, x_len)
    descY = CuDenseVectorDescriptor(T, y_len)
    algo  = CUSPARSE_SPMV_ALG_DEFAULT

    alpha_ref = Ref{T}(one(T));  beta_ref = Ref{T}(zero(T))
    buf_sz = Ref{Csize_t}(0)
    cusparseSpMV_bufferSize(
        handle(), transa, alpha_ref, spmat_desc, descX, beta_ref, descY,
        T, algo, buf_sz)
    ws = CUDA.zeros(UInt8, max(1, Int(buf_sz[])))
    cusparseSpMV_preprocess(
        handle(), transa, alpha_ref, spmat_desc, descX, beta_ref, descY,
        T, algo, ws)

    CUSPARSESpMVHandle{T}(spmat_desc, descX, descY, ws, transa, algo)
end

function JLUST.update_values!(h::CUSPARSESpMVHandle, u_A::USTensor)
    cusparseSpMatSetValues(h.spmat_desc, nonzeros(u_A))
    return h
end

# ─── Execution ────────────────────────────────────────────────────────────────

# Direct path — builds a fresh descriptor each call (no handle caching).
function JLUST.sparse_mv!(::CUSPARSEBackend,
                           u_A::USTensor{T,Ti}, u_x::USTensor, u_y::USTensor;
                           transa::Char='N',
                           alpha=one(T), beta=zero(T)) where {T<:_CUSPARSE_ELTYPES,Ti}
    cusA = _to_cuspmat(u_A)
    idx  = _cusparse_index(u_A)
    CUSPARSE.mv!(transa, T(alpha), cusA, nonzeros(u_x), T(beta), nonzeros(u_y), idx)
    return u_y
end

# Handle path — only updates dense data pointers; no allocation, no descriptor rebuild.
function JLUST.sparse_mv!(h::CUSPARSESpMVHandle{T},
                           u_x::USTensor, u_y::USTensor;
                           alpha=one(T), beta=zero(T)) where T
    cusparseDnVecSetValues(h.dnvec_x, nonzeros(u_x))
    cusparseDnVecSetValues(h.dnvec_y, nonzeros(u_y))
    cusparseSpMV(
        handle(), h.transa,
        Ref{T}(alpha), h.spmat_desc, h.dnvec_x,
        Ref{T}(beta),  h.dnvec_y,
        T, h.algo, h.workspace)
    return u_y
end

# Convenience wrapper — overrides KernelAbstractionsExt's default when CUDA is loaded.
# Defaults to CUSPARSEBackend for CUDA arrays, EmitterBackend for everything else.
function JLUST.sparse_mv!(u_A::USTensor, u_x::USTensor, u_y::USTensor;
                           backend=nothing, kw...)
    be = something(backend,
                   nonzeros(u_A) isa CuArray ? CUSPARSEBackend() : EmitterBackend())
    JLUST.sparse_mv!(be, u_A, u_x, u_y; kw...)
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
        nzval::CuVector{T}, x::CuVector{T}, y::CuVector{T},
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

@inline function _csr_group_reduce(val::T, mask::UInt32, ::Val{2}) where T
    val += CUDA.shfl_down_sync(mask, val, Int32(1))
    val
end
@inline function _csr_group_reduce(val::T, mask::UInt32, ::Val{4}) where T
    val += CUDA.shfl_down_sync(mask, val, Int32(1))
    val += CUDA.shfl_down_sync(mask, val, Int32(2))
    val
end
@inline function _csr_group_reduce(val::T, mask::UInt32, ::Val{8}) where T
    val += CUDA.shfl_down_sync(mask, val, Int32(1))
    val += CUDA.shfl_down_sync(mask, val, Int32(2))
    val += CUDA.shfl_down_sync(mask, val, Int32(4))
    val
end
@inline function _csr_group_reduce(val::T, mask::UInt32, ::Val{16}) where T
    val += CUDA.shfl_down_sync(mask, val, Int32(1))
    val += CUDA.shfl_down_sync(mask, val, Int32(2))
    val += CUDA.shfl_down_sync(mask, val, Int32(4))
    val += CUDA.shfl_down_sync(mask, val, Int32(8))
    val
end
@inline function _csr_group_reduce(val::T, mask::UInt32, ::Val{32}) where T
    val += CUDA.shfl_down_sync(mask, val, Int32(1))
    val += CUDA.shfl_down_sync(mask, val, Int32(2))
    val += CUDA.shfl_down_sync(mask, val, Int32(4))
    val += CUDA.shfl_down_sync(mask, val, Int32(8))
    val += CUDA.shfl_down_sync(mask, val, Int32(16))
    val
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

# Select VS based on average NNZ/row (computed over all rows, including empty).
# Thresholds derived empirically from the L40S benchmark suite:
#   < 4  → VS=2   (ultra-sparse; 2 threads/row keeps occupancy)
#   < 8  → VS=4   (sparse with moderate or heavy empty-row fraction)
#   < 16 → VS=8   (moderately dense; more parallelism per row)
#   < 32 → VS=16  (dense rows; parallelize within-row loads aggressively)
#   ≥ 32 → VS=32  (warp-per-row; row NNZ >> warp width)
#
# For VS=32 the warp mask formula (UInt32(1)<<32)-1 wraps to 0xffffffff by
# UInt32 overflow semantics on GPU, giving the correct full-warp mask.
function JLUST._csr_spmv_specialized!(
        pos::CuVector, crd::CuVector,
        nzval::CuVector{T}, x::CuVector{T}, y::CuVector{T},
        origin_off::Int32, n_outer::Int32, beta::T) where T
    n_outer == Int32(0) && return true
    avg_nnz = length(nzval) / Int(n_outer)
    vs = avg_nnz < 4.0  ? 2  :
         avg_nnz < 8.0  ? 4  :
         avg_nnz < 16.0 ? 8  :
         avg_nnz < 32.0 ? 16 : 32
    zero_beta = Val(iszero(beta))
    threads = 256
    blocks  = cld(Int(n_outer) * vs, threads)
    if vs == 2
        CUDA.@cuda threads=threads blocks=blocks _csr_spmv_vector_kernel!(
            pos, crd, nzval, x, y, origin_off, n_outer, Val(2), beta, zero_beta)
    elseif vs == 4
        CUDA.@cuda threads=threads blocks=blocks _csr_spmv_vector_kernel!(
            pos, crd, nzval, x, y, origin_off, n_outer, Val(4), beta, zero_beta)
    elseif vs == 8
        CUDA.@cuda threads=threads blocks=blocks _csr_spmv_vector_kernel!(
            pos, crd, nzval, x, y, origin_off, n_outer, Val(8), beta, zero_beta)
    elseif vs == 16
        CUDA.@cuda threads=threads blocks=blocks _csr_spmv_vector_kernel!(
            pos, crd, nzval, x, y, origin_off, n_outer, Val(16), beta, zero_beta)
    else
        CUDA.@cuda threads=threads blocks=blocks _csr_spmv_vector_kernel!(
            pos, crd, nzval, x, y, origin_off, n_outer, Val(32), beta, zero_beta)
    end
    return true
end
