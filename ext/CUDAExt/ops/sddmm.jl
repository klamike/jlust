# ─── cuSPARSE SDDMM ──────────────────────────────────────────────────────────
#
# Sampled dense-dense matrix multiply:
#   C ← alpha * (op(A) * op(B)) ∘ sparsity(C) + beta * C
#
# A and B are dense; C is sparse (CSR or COO) and acts as both mask and output.

# ─── Handle ──────────────────────────────────────────────────────────────────

mutable struct CUSPARSESDDMMHandle{T}
    spmat_C::CuSparseMatrixDescriptor
    dnmat_A::CuDenseMatrixDescriptor
    dnmat_B::CuDenseMatrixDescriptor
    workspace::CuVector{UInt8}
    transa::Char
    transb::Char
    algo::cusparseSDDMMAlg_t
end

export CUSPARSESDDMMHandle

function JLUST.prepare(::CUSPARSEBackend, ::Type{<:Op{:SDDMM}},
                        u_A::USTensor{T}, u_B::USTensor, u_C::USTensor;
                        transa::Char='N', transb::Char='N') where {T<:_CUSPARSE_ELTYPES}
    idx   = _cusparse_index(u_C)
    descC = CuSparseMatrixDescriptor(_to_cuspmat(u_C), idx)
    algo  = CUSPARSE_SDDMM_ALG_DEFAULT

    # Physical stored dimensions — transposition is handled by transa/transb args
    rows_A, cols_A = Int64.(extents(u_A))
    rows_B, cols_B = Int64.(extents(u_B))
    descA = CuDenseMatrixDescriptor(T, rows_A, cols_A)
    descB = CuDenseMatrixDescriptor(T, rows_B, cols_B)

    alpha_ref = Ref{T}(one(T));  beta_ref = Ref{T}(zero(T))
    buf_sz = Ref{Csize_t}(0)
    cusparseSDDMM_bufferSize(
        handle(), transa, transb, alpha_ref, descA, descB, beta_ref, descC,
        T, algo, buf_sz)
    ws = CUDA.zeros(UInt8, max(1, Int(buf_sz[])))
    cusparseSDDMM_preprocess(
        handle(), transa, transb, alpha_ref, descA, descB, beta_ref, descC,
        T, algo, ws)

    CUSPARSESDDMMHandle{T}(descC, descA, descB, ws, transa, transb, algo)
end

function JLUST.update_values!(h::CUSPARSESDDMMHandle, u_C::USTensor)
    cusparseSpMatSetValues(h.spmat_C, nonzeros(u_C))
    return h
end

# Handle path — preprocess cached; only values updated per call.
function JLUST.sparse_sddmm!(h::CUSPARSESDDMMHandle{T},
                               u_A::USTensor, u_B::USTensor, u_C::USTensor;
                               alpha=one(T), beta=zero(T)) where T
    cusparseDnMatSetValues(h.dnmat_A, nonzeros(u_A))
    cusparseDnMatSetValues(h.dnmat_B, nonzeros(u_B))
    cusparseSpMatSetValues(h.spmat_C, nonzeros(u_C))
    cusparseSDDMM(
        handle(), h.transa, h.transb,
        Ref{T}(alpha), h.dnmat_A, h.dnmat_B,
        Ref{T}(beta),  h.spmat_C,
        T, h.algo, h.workspace)
    return u_C
end

# ─── Execution ────────────────────────────────────────────────────────────────

# Direct path — preprocesses and computes each call (no handle caching).
function JLUST.sparse_sddmm!(::CUSPARSEBackend,
                               u_A::USTensor{T}, u_B::USTensor, u_C::USTensor;
                               transa::Char='N', transb::Char='N',
                               alpha=one(T), beta=zero(T)) where {T<:_CUSPARSE_ELTYPES}
    cusC = _to_cuspmat(u_C)
    idx  = _cusparse_index(u_C)
    CUSPARSE.sddmm!(transa, transb, T(alpha), nonzeros(u_A), nonzeros(u_B),
                    T(beta), cusC, idx)
    return u_C
end

