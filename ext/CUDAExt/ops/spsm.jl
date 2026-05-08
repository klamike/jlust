# ─── cuSPARSE sparse triangular solve (multiple RHS) ─────────────────────────
#
# Solves  op(A) * C = alpha * B  for C.
# Uses cusparseSpSM.  A must be CSR or CSC; B and C are dense matrices.
#
# Handle path: pre-runs bufferSize + analysis once; each call only pays solve.

# ─── Handle ──────────────────────────────────────────────────────────────────

mutable struct CUSPARSESpSMHandle{T}
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
    cusA = _to_cuspmat(u_A)

    descA = CuSparseMatrixDescriptor(cusA, idx)
    cusparse_uplo = Ref{cusparseFillMode_t}(uplo)
    cusparse_diag = Ref{cusparseDiagType_t}(diag)
    cusparseSpMatSetAttribute(descA, CUSPARSE_SPMAT_FILL_MODE,
                               cusparse_uplo, Csize_t(sizeof(cusparse_uplo)))
    cusparseSpMatSetAttribute(descA, CUSPARSE_SPMAT_DIAG_TYPE,
                               cusparse_diag, Csize_t(sizeof(cusparse_diag)))

    # Placeholder dense matrices (cuSPARSE only reads dimensions during analysis).
    B_tmp = CUDA.zeros(T, m, n_cols)
    C_tmp = CUDA.zeros(T, m, n_cols)
    descB = CuDenseMatrixDescriptor(B_tmp)
    descC = CuDenseMatrixDescriptor(C_tmp)

    algo      = CUSPARSE_SPSM_ALG_DEFAULT
    spsm_desc = CuSparseSpSMDescriptor()
    alpha_ref = Ref{T}(T(alpha))

    buf_sz = Ref{Csize_t}(0)
    cusparseSpSM_bufferSize(handle(), transa, transb, alpha_ref, descA, descB, descC,
                             T, algo, spsm_desc, buf_sz)
    ws = CUDA.zeros(UInt8, max(1, Int(buf_sz[])))

    cusparseSpSM_analysis(handle(), transa, transb, alpha_ref, descA, descB, descC,
                           T, algo, spsm_desc, ws)

    CUSPARSESpSMHandle{T}(descA, descB, descC, spsm_desc, ws, transa, transb, algo)
end

function JLUST.update_values!(h::CUSPARSESpSMHandle, u_A::USTensor)
    cusparseSpMatSetValues(h.spmat_desc, nonzeros(u_A))
    return h
end

# ─── Execution ────────────────────────────────────────────────────────────────

# Direct path
function JLUST.sparse_sm!(::CUSPARSEBackend,
                           u_A::USTensor{T,Ti}, u_B::USTensor, u_C::USTensor;
                           transa::Char='N', transb::Char='N',
                           uplo::Char='L', diag::Char='N',
                           alpha=one(T)) where {T<:_CUSPARSE_ELTYPES, Ti}
    cusA = _to_cuspmat(u_A)
    idx  = _cusparse_index(u_A)
    CUSPARSE.sm!(transa, transb, uplo, diag, T(alpha), cusA,
                 nonzeros(u_B), nonzeros(u_C), idx)
    return u_C
end

# Handle path — only updates dense data pointers; no descriptor rebuild.
function JLUST.sparse_sm!(h::CUSPARSESpSMHandle{T},
                           u_B::USTensor, u_C::USTensor;
                           alpha=one(T)) where T
    cusparseDnMatSetValues(h.dnmat_B, nonzeros(u_B))
    cusparseDnMatSetValues(h.dnmat_C, nonzeros(u_C))
    cusparseSpSM_solve(
        handle(), h.transa, h.transb, Ref{T}(T(alpha)),
        h.spmat_desc, h.dnmat_B, h.dnmat_C,
        T, h.algo, h.spsm_desc)
    return u_C
end

