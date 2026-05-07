# ─── cuSPARSE sparse triangular solve (single RHS) ───────────────────────────
#
# Solves  op(A) * x = alpha * b  for x.
# Uses cusparseSpSV.  A must be CSR or CSC.
#
# Handle path: pre-runs bufferSize + analysis once so each call only pays the
# O(nnz) SpSV_solve cost.  Dense vector data pointers are updated via
# cusparseDnVecSetValues before each solve.

# ─── Handle ──────────────────────────────────────────────────────────────────

mutable struct CUSPARSESpSVHandle{T}
    spmat_desc :: CuSparseMatrixDescriptor
    dnvec_b    :: CuDenseVectorDescriptor   # RHS (input)
    dnvec_x    :: CuDenseVectorDescriptor   # solution (output)
    spsv_desc  :: CuSparseSpSVDescriptor
    workspace  :: CuVector{UInt8}
    transa     :: Char
    algo       :: cusparseSpSVAlg_t
end

export CUSPARSESpSVHandle

function JLUST.prepare(::CUSPARSEBackend, ::Type{SpSVOp},
                        u_A::USTensor{T,Ti};
                        transa::Char='N', uplo::Char='L', diag::Char='N',
                        alpha=one(T)) where {T<:_CUSPARSE_ELTYPES, Ti}
    idx  = _cusparse_index(u_A)
    m, _ = Int64.(extents(u_A))
    cusA = _to_cuspmat(u_A)

    descA = CuSparseMatrixDescriptor(cusA, idx)
    cusparse_uplo = Ref{cusparseFillMode_t}(uplo)
    cusparse_diag = Ref{cusparseDiagType_t}(diag)
    cusparseSpMatSetAttribute(descA, CUSPARSE_SPMAT_FILL_MODE,
                               cusparse_uplo, Csize_t(sizeof(cusparse_uplo)))
    cusparseSpMatSetAttribute(descA, CUSPARSE_SPMAT_DIAG_TYPE,
                               cusparse_diag, Csize_t(sizeof(cusparse_diag)))

    # Placeholder vectors for analysis (cuSPARSE only reads their dimensions).
    b_tmp = CUDA.zeros(T, m)
    x_tmp = CUDA.zeros(T, m)
    descB = CuDenseVectorDescriptor(b_tmp)
    descX = CuDenseVectorDescriptor(x_tmp)

    algo      = CUSPARSE_SPSV_ALG_DEFAULT
    spsv_desc = CuSparseSpSVDescriptor()
    alpha_ref = Ref{T}(T(alpha))

    buf_sz = Ref{Csize_t}(0)
    cusparseSpSV_bufferSize(handle(), transa, alpha_ref, descA, descB, descX,
                             T, algo, spsv_desc, buf_sz)
    ws = CUDA.zeros(UInt8, max(1, Int(buf_sz[])))

    cusparseSpSV_analysis(handle(), transa, alpha_ref, descA, descB, descX,
                           T, algo, spsv_desc, ws)

    CUSPARSESpSVHandle{T}(descA, descB, descX, spsv_desc, ws, transa, algo)
end

function JLUST.update_values!(h::CUSPARSESpSVHandle, u_A::USTensor)
    cusparseSpMatSetValues(h.spmat_desc, nonzeros(u_A))
    return h
end

# ─── Execution ────────────────────────────────────────────────────────────────

# Direct path — rebuilds descriptors each call.
function JLUST.sparse_sv!(::CUSPARSEBackend,
                           u_A::USTensor{T,Ti}, u_b::USTensor, u_x::USTensor;
                           transa::Char='N', uplo::Char='L', diag::Char='N',
                           alpha=one(T)) where {T<:_CUSPARSE_ELTYPES, Ti}
    cusA = _to_cuspmat(u_A)
    idx  = _cusparse_index(u_A)
    CUSPARSE.sv!(transa, uplo, diag, T(alpha), cusA,
                 nonzeros(u_b), nonzeros(u_x), idx)
    return u_x
end

# Handle path — only updates data pointers; no allocation, no re-analysis.
function JLUST.sparse_sv!(h::CUSPARSESpSVHandle{T},
                           u_b::USTensor, u_x::USTensor;
                           alpha=one(T)) where T
    cusparseDnVecSetValues(h.dnvec_b, nonzeros(u_b))
    cusparseDnVecSetValues(h.dnvec_x, nonzeros(u_x))
    cusparseSpSV_solve(
        handle(), h.transa, Ref{T}(T(alpha)),
        h.spmat_desc, h.dnvec_b, h.dnvec_x,
        T, h.algo, h.spsv_desc)
    return u_x
end

function JLUST.sparse_sv!(u_A::USTensor, u_b::USTensor, u_x::USTensor;
                           backend=CUSPARSEBackend(), kw...)
    JLUST.sparse_sv!(backend, u_A, u_b, u_x; kw...)
end
