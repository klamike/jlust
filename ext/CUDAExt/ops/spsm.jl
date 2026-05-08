# ─── cuSPARSE sparse triangular solve (multiple RHS) ─────────────────────────
#
# Solves  op(A) * C = alpha * B  for C.
# Uses cusparseSpSM.  A must be CSR or CSC; B and C are dense matrices.
#
# Handle path: pre-runs bufferSize + analysis once; each call only pays solve.

# ─── Handle ──────────────────────────────────────────────────────────────────

mutable struct CUSPARSESpSMHandle{T} <: JLUST.AbstractKernelHandle
    spmat_desc :: CuSparseMatrixDescriptor
    dnmat_B    :: CuDenseMatrixDescriptor   # RHS (input)
    dnmat_C    :: CuDenseMatrixDescriptor   # solution (output)
    spsm_desc  :: CuSparseSpSMDescriptor
    workspace  :: CuVector{UInt8}
    transa     :: Char
    transb     :: Char
    algo       :: cusparseSpSMAlg_t
end

export CUSPARSESpSMHandle

function JLUST.prepare(::CUSPARSEBackend, ::Type{<:Op{:SpSM}},
                        u_A::USTensor{T,Ti};
                        transa::Char='N', transb::Char='N',
                        uplo::Char='L', diag::Char='N',
                        n_cols::Int=1, alpha=one(T)) where {T<:_CUSPARSE_ELTYPES, Ti}
    idx  = _cusparse_index(u_A)
    m, _ = Int64.(extents(u_A))
    descA = CuSparseMatrixDescriptor(_to_cuspmat(u_A), idx)
    cusparseSpMatSetAttribute(descA, CUSPARSE_SPMAT_FILL_MODE,
                               Ref{cusparseFillMode_t}(uplo), Csize_t(sizeof(cusparseFillMode_t)))
    cusparseSpMatSetAttribute(descA, CUSPARSE_SPMAT_DIAG_TYPE,
                               Ref{cusparseDiagType_t}(diag), Csize_t(sizeof(cusparseDiagType_t)))

    # Placeholder dense matrices (cuSPARSE only reads dimensions during analysis).
    descB = CuDenseMatrixDescriptor(CUDA.zeros(T, m, n_cols))
    descC = CuDenseMatrixDescriptor(CUDA.zeros(T, m, n_cols))
    algo      = CUSPARSE_SPSM_ALG_DEFAULT
    spsm_desc = CuSparseSpSMDescriptor()
    α_ref     = Ref{T}(T(alpha))

    ws = _cusparse_workspace() do buf_sz, buf
        if buf === CUDA.CU_NULL
            cusparseSpSM_bufferSize(handle(), transa, transb, α_ref, descA, descB, descC, T, algo, spsm_desc, buf_sz)
        else
            cusparseSpSM_analysis(handle(), transa, transb, α_ref, descA, descB, descC, T, algo, spsm_desc, buf)
        end
    end
    CUSPARSESpSMHandle{T}(descA, descB, descC, spsm_desc, ws, transa, transb, algo)
end

function JLUST.update_values!(h::CUSPARSESpSMHandle, u_A::USTensor)
    cusparseSpMatSetValues(h.spmat_desc, nonzeros(u_A))
    return h
end

# ─── Execution ────────────────────────────────────────────────────────────────

function JLUST.execute(::CUSPARSEBackend, ::Op{:SpSM, F_op},
                       u_A::USTensor{T,Ti}, u_B::USTensor, u_C::USTensor;
                       transa::Char='N', transb::Char='N',
                       uplo::Char='L', diag::Char='N',
                       alpha=one(T)) where {F_op, T<:_CUSPARSE_ELTYPES, Ti}
    cusA = _to_cuspmat(u_A)
    idx  = _cusparse_index(u_A)
    CUSPARSE.sm!(transa, transb, uplo, diag, T(alpha), cusA,
                 nonzeros(u_B), nonzeros(u_C), idx)
    return u_C
end

function JLUST.execute(h::CUSPARSESpSMHandle{T},
                       u_B::USTensor, u_C::USTensor;
                       alpha=one(T)) where T
    _cusparse_set_dense!(h.dnmat_B, nonzeros(u_B))
    _cusparse_set_dense!(h.dnmat_C, nonzeros(u_C))
    cusparseSpSM_solve(handle(), h.transa, h.transb, Ref{T}(T(alpha)),
                       h.spmat_desc, h.dnmat_B, h.dnmat_C,
                       T, h.algo, h.spsm_desc)
    return u_C
end

