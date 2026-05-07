# ─── cuSPARSE SDDMM ──────────────────────────────────────────────────────────
#
# Sampled dense-dense matrix multiply:
#   C ← alpha * (op(A) * op(B)) ∘ sparsity(C) + beta * C
#
# A and B are dense; C is sparse (CSR or BSR) and acts as both mask and output.
# Uses CUSPARSE.sddmm! which calls cusparseSDDMM.

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

function JLUST.sparse_sddmm!(u_A::USTensor, u_B::USTensor, u_C::USTensor;
                               backend=CUSPARSEBackend(), kw...)
    JLUST.sparse_sddmm!(backend, u_A, u_B, u_C; kw...)
end
