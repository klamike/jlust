# ─── cuSPARSE sparse-sparse matrix product (SpGEMM) ──────────────────────────
#
# C ← alpha * op(A) * op(B) + beta * C
#
# All three operands must be CSR (cuSPARSE SpGEMM restriction).
# SpGEMM can change the sparsity structure of C, so the returned USTensor
# may have different index buffers than the input u_C.  When beta = 0 the
# input u_C is used only to determine the output dimensions; when beta ≠ 0
# u_C must already carry the exact non-zero pattern of op(A)*op(B).
#
# Uses the high-level CUSPARSE.gemm! wrapper from CUDA.jl, which handles the
# three-phase cuSPARSE SpGEMM protocol internally.

function JLUST.sparse_gemm!(::CUSPARSEBackend,
                              u_A::USTensor{T,Ti}, u_B::USTensor, u_C::USTensor;
                              transa::Char='N', transb::Char='N',
                              alpha=one(T), beta=zero(T)) where {T<:_CUSPARSE_ELTYPES, Ti}
    cusA = _to_cuspmat(u_A)::CuSparseMatrixCSR{T,Ti}
    cusB = _to_cuspmat(u_B)::CuSparseMatrixCSR{T,Ti}
    cusC = _to_cuspmat(u_C)::CuSparseMatrixCSR{T,Ti}
    idx  = _cusparse_index(u_A)
    out  = CUSPARSE.gemm!(transa, transb, T(alpha), cusA, cusB, T(beta), cusC, idx)
    ust(out)
end

function JLUST.sparse_gemm!(u_A::USTensor, u_B::USTensor, u_C::USTensor;
                              backend=CUSPARSEBackend(), kw...)
    JLUST.sparse_gemm!(backend, u_A, u_B, u_C; kw...)
end
