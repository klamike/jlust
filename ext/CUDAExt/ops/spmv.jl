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
function JLUST.sparse_mv!(u_A::USTensor, u_x::USTensor, u_y::USTensor;
                           backend=CUSPARSEBackend(), kw...)
    JLUST.sparse_mv!(backend, u_A, u_x, u_y; kw...)
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
# _CSR_VECTOR_SIZE threads collaborate on each row: each handles NNZ at
# stride _CSR_VECTOR_SIZE from its lane offset. A log2(_CSR_VECTOR_SIZE)-step
# warp-shuffle tree reduces within each group; lane 0 writes y[row].
# Compared to one-thread-per-row: better parallelism for rows with many NNZ
# and removes the cuSPARSE descriptor-rebuild overhead from the direct path.

const _CSR_VECTOR_SIZE = Int32(4)   # threads per row; must divide 32

function _csr_spmv_vector_kernel!(pos, crd, nzval, x, y, origin_off, n_outer)
    T        = eltype(nzval)
    tid      = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    row_id   = (tid - Int32(1)) ÷ _CSR_VECTOR_SIZE + Int32(1)   # 1-based row
    vec_lane = (tid - Int32(1)) % _CSR_VECTOR_SIZE               # 0-based within group

    lane_in_warp  = (threadIdx().x - Int32(1)) % Int32(32)
    group_in_warp = lane_in_warp ÷ _CSR_VECTOR_SIZE
    group_mask    = UInt32(0x0f) << (UInt32(group_in_warp) * UInt32(4))

    my_acc = zero(T)
    if row_id <= n_outer
        lo = Int(pos[row_id])     - Int(origin_off)
        hi = Int(pos[row_id + 1]) - Int(origin_off)
        k  = lo + Int(vec_lane) + 1
        while k <= hi
            col     = Int(crd[k]) - Int(origin_off) + 1
            my_acc += nzval[k] * x[col]
            k      += Int(_CSR_VECTOR_SIZE)
        end
    end

    # All threads in the group participate in the reduce for shfl correctness.
    my_acc += CUDA.shfl_down_sync(group_mask, my_acc, Int32(1))
    my_acc += CUDA.shfl_down_sync(group_mask, my_acc, Int32(2))

    if vec_lane == Int32(0) && row_id <= n_outer
        y[row_id] = my_acc
    end
    return nothing
end

function JLUST._csr_spmv_specialized!(
        pos::CuVector, crd::CuVector,
        nzval::CuVector{T}, x::CuVector{T}, y::CuVector{T},
        origin_off::Int32, n_outer::Int32) where T
    n_outer == Int32(0) && return true
    threads = 256
    blocks  = cld(Int(n_outer) * Int(_CSR_VECTOR_SIZE), threads)
    CUDA.@cuda threads=threads blocks=blocks _csr_spmv_vector_kernel!(
        pos, crd, nzval, x, y, origin_off, n_outer)
    return true
end
