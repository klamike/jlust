# ─── cuSPARSE sparse triangular solve (single RHS) ───────────────────────────
#
# Solves  op(A) * x = alpha * b  for x.
# Uses cusparseSpSV.  A must be CSR or CSC.
#
# Handle path: pre-runs bufferSize + analysis once so each call only pays the
# O(nnz) SpSV_solve cost.  Dense vector data pointers are updated via
# cusparseDnVecSetValues before each solve.

# ─── Handle ──────────────────────────────────────────────────────────────────

mutable struct CUSPARSESpSVHandle{T} <: JLUST.AbstractKernelHandle
    spmat_desc :: CuSparseMatrixDescriptor
    dnvec_b    :: CuDenseVectorDescriptor   # RHS (input)
    dnvec_x    :: CuDenseVectorDescriptor   # solution (output)
    spsv_desc  :: CuSparseSpSVDescriptor
    workspace  :: CuVector{UInt8}
    transa     :: Char
    algo       :: cusparseSpSVAlg_t
end

export CUSPARSESpSVHandle

function JLUST.prepare(::CUSPARSEBackend, ::Type{<:Op{:SpSV}},
                        u_A::USTensor{T,Ti};
                        transa::Char='N', uplo::Char='L', diag::Char='N',
                        alpha=one(T)) where {T<:_CUSPARSE_ELTYPES, Ti}
    idx  = _cusparse_index(u_A)
    m, _ = Int64.(extents(u_A))
    descA = CuSparseMatrixDescriptor(_to_cuspmat(u_A), idx)
    cusparseSpMatSetAttribute(descA, CUSPARSE_SPMAT_FILL_MODE,
                               Ref{cusparseFillMode_t}(uplo), Csize_t(sizeof(cusparseFillMode_t)))
    cusparseSpMatSetAttribute(descA, CUSPARSE_SPMAT_DIAG_TYPE,
                               Ref{cusparseDiagType_t}(diag), Csize_t(sizeof(cusparseDiagType_t)))

    # Placeholder vectors for analysis (cuSPARSE only reads their dimensions).
    descB = CuDenseVectorDescriptor(CUDA.zeros(T, m))
    descX = CuDenseVectorDescriptor(CUDA.zeros(T, m))
    algo      = CUSPARSE_SPSV_ALG_DEFAULT
    spsv_desc = CuSparseSpSVDescriptor()
    α_ref     = Ref{T}(T(alpha))

    ws = _cusparse_workspace() do buf_sz, buf
        if buf === CUDA.CU_NULL
            cusparseSpSV_bufferSize(handle(), transa, α_ref, descA, descB, descX, T, algo, spsv_desc, buf_sz)
        else
            cusparseSpSV_analysis(handle(), transa, α_ref, descA, descB, descX, T, algo, spsv_desc, buf)
        end
    end
    CUSPARSESpSVHandle{T}(descA, descB, descX, spsv_desc, ws, transa, algo)
end

function JLUST.update_values!(h::CUSPARSESpSVHandle, u_A::USTensor)
    cusparseSpMatSetValues(h.spmat_desc, nonzeros(u_A))
    return h
end

# ─── Execution ────────────────────────────────────────────────────────────────

function JLUST.execute(::CUSPARSEBackend, ::Op{:SpSV, F_op},
                       u_A::USTensor{T,Ti}, u_b::USTensor, u_x::USTensor;
                       transa::Char='N', uplo::Char='L', diag::Char='N',
                       alpha=one(T)) where {F_op, T<:_CUSPARSE_ELTYPES, Ti}
    cusA = _to_cuspmat(u_A)
    idx  = _cusparse_index(u_A)
    CUSPARSE.sv!(transa, uplo, diag, T(alpha), cusA,
                 nonzeros(u_b), nonzeros(u_x), idx)
    return u_x
end

function JLUST.execute(h::CUSPARSESpSVHandle{T},
                       u_b::USTensor, u_x::USTensor;
                       alpha=one(T)) where T
    _cusparse_set_dense!(h.dnvec_b, nonzeros(u_b))
    _cusparse_set_dense!(h.dnvec_x, nonzeros(u_x))
    cusparseSpSV_solve(handle(), h.transa, Ref{T}(T(alpha)),
                       h.spmat_desc, h.dnvec_b, h.dnvec_x,
                       T, h.algo, h.spsv_desc)
    return u_x
end

