# ─── cuSPARSE sparse matrix-matrix product ───────────────────────────────────
#
# C ← alpha * op(A) * op(B) + beta * C
# B and C must be column-major dense CuMatrices.

# ─── Handle ──────────────────────────────────────────────────────────────────

mutable struct CUSPARSESpMMHandle{T}
    spmat_desc::CuSparseMatrixDescriptor
    dnmat_B::CuDenseMatrixDescriptor
    dnmat_C::CuDenseMatrixDescriptor
    workspace::CuVector{UInt8}
    transa::Char
    transb::Char
    algo::cusparseSpMMAlg_t
end

export CUSPARSESpMMHandle

function JLUST.prepare(::CUSPARSEBackend, ::Type{SpMMOp}, u_A::USTensor{T,Ti};
                        transa::Char='N', transb::Char='N',
                        n_cols::Int) where {T<:_CUSPARSE_ELTYPES, Ti}
    idx   = _cusparse_index(u_A)
    m, n  = Int64.(extents(u_A))
    spmat_desc = CuSparseMatrixDescriptor(_to_cuspmat(u_A), idx)

    k_dim = transa == 'N' ? n : m
    m_out = transa == 'N' ? m : n
    descB = CuDenseMatrixDescriptor(T, k_dim, n_cols)
    descC = CuDenseMatrixDescriptor(T, m_out, n_cols)
    algo  = CUSPARSE_SPMM_ALG_DEFAULT

    alpha_ref = Ref{T}(one(T));  beta_ref = Ref{T}(zero(T))
    buf_sz = Ref{Csize_t}(1000)
    cusparseSpMM_bufferSize(
        handle(), transa, transb, alpha_ref, spmat_desc, descB, beta_ref, descC,
        T, algo, buf_sz)
    ws = CUDA.zeros(UInt8, max(1, Int(buf_sz[])))
    cusparseSpMM_preprocess(
        handle(), transa, transb, alpha_ref, spmat_desc, descB, beta_ref, descC,
        T, algo, ws)

    CUSPARSESpMMHandle{T}(spmat_desc, descB, descC, ws, transa, transb, algo)
end

function JLUST.update_values!(h::CUSPARSESpMMHandle, u_A::USTensor)
    cusparseSpMatSetValues(h.spmat_desc, nonzeros(u_A))
    return h
end

# ─── Execution ────────────────────────────────────────────────────────────────

# Direct path — builds a fresh descriptor each call (no handle caching).
function JLUST.sparse_mm!(::CUSPARSEBackend,
                           u_A::USTensor{T,Ti}, u_B::USTensor, u_C::USTensor;
                           transa::Char='N', transb::Char='N',
                           alpha=one(T), beta=zero(T)) where {T<:_CUSPARSE_ELTYPES,Ti}
    cusA = _to_cuspmat(u_A)
    idx  = _cusparse_index(u_A)
    CUSPARSE.mm!(transa, transb, T(alpha), cusA, nonzeros(u_B), T(beta), nonzeros(u_C), idx)
    return u_C
end

# Handle path — only updates dense data pointers; no allocation, no descriptor rebuild.
function JLUST.sparse_mm!(h::CUSPARSESpMMHandle{T},
                           u_B::USTensor, u_C::USTensor;
                           alpha=one(T), beta=zero(T)) where T
    cusparseDnMatSetValues(h.dnmat_B, nonzeros(u_B))
    cusparseDnMatSetValues(h.dnmat_C, nonzeros(u_C))
    cusparseSpMM(
        handle(), h.transa, h.transb,
        Ref{T}(alpha), h.spmat_desc, h.dnmat_B,
        Ref{T}(beta),  h.dnmat_C,
        T, h.algo, h.workspace)
    return u_C
end

# Convenience wrapper — overrides KernelAbstractionsExt's default when CUDA is loaded.
function JLUST.sparse_mm!(u_A::USTensor, u_B::USTensor, u_C::USTensor;
                           backend=CUSPARSEBackend(), kw...)
    JLUST.sparse_mm!(backend, u_A, u_B, u_C; kw...)
end

# cuSPARSE rejects SubArray outputs; route to EmitterBackend (our KA kernel) instead.
# This fires when row_bufs are views into the stacked diag_out buffer.
function JLUST.sparse_mm!(u_A::USTensor, u_B::USTensor,
                           u_C::USTensor{T,Ti,N,<:SubArray}; kw...) where {T,Ti,N}
    JLUST.sparse_mm!(EmitterBackend(), u_A, u_B, u_C; kw...)
end
