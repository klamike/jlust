# ─── cuSPARSE sparse-sparse matrix product (SpGEMM) ──────────────────────────
#
# C ← alpha * op(A) * op(B) + beta * C
#
# All three operands must be CSR (cuSPARSE SpGEMM restriction).

# ─── Handle (SpGEMMreuse API) ─────────────────────────────────────────────────
#
# SpGEMMreuse separates symbolic analysis (phases 1–3, done once at prepare()
# time) from numeric computation (phase 4, called every sparse_gemm! call).
# Phases 1–3 determine the output sparsity structure and cache it in the
# CuSpGEMMDescriptor.  Phase 4 (reuse_compute) only fills in numeric values —
# no additional workspace, no reanalysis.

mutable struct CUSPARSESpGEMMHandle{T,Ti}
    spgemm_desc::CuSpGEMMDescriptor
    descA::CuSparseMatrixDescriptor
    descB::CuSparseMatrixDescriptor
    descC::CuSparseMatrixDescriptor
    C_rowPtr::CuVector{Ti}
    C_colInd::CuVector{Ti}
    C_nzVal::CuVector{T}
    algo::cusparseSpGEMMAlg_t
    transa::Char
    transb::Char
    m::Int64
    n::Int64
end

export CUSPARSESpGEMMHandle

function JLUST.prepare(::CUSPARSEBackend, ::Type{SpGEMMOp},
                        u_A::USTensor{T,Ti}, u_B::USTensor;
                        transa::Char='N', transb::Char='N') where {T<:_CUSPARSE_ELTYPES, Ti}
    idx  = _cusparse_index(u_A)
    algo = CUSPARSE_SPGEMM_DEFAULT

    descA = CuSparseMatrixDescriptor(_to_cuspmat(u_A), idx)
    descB = CuSparseMatrixDescriptor(_to_cuspmat(u_B), idx)

    m = transa == 'N' ? Int64(extents(u_A)[1]) : Int64(extents(u_A)[2])
    n = transb == 'N' ? Int64(extents(u_B)[2]) : Int64(extents(u_B)[1])

    # Placeholder rowPtr for C (all zeros); colInd and values are null until
    # SpMatGetSize tells us nnzC after the symbolic phases.
    C_rowPtr = CUDA.zeros(Ti, m + 1)
    descC = CuSparseMatrixDescriptor(CuSparseMatrixCSR, C_rowPtr, T, Ti, m, n, idx)

    spgemm_desc = CuSpGEMMDescriptor()

    # ── Phase 1: workEstimation (call twice — first for buffer size, then compute) ──
    buf1_sz = Ref{Csize_t}(0)
    cusparseSpGEMMreuse_workEstimation(
        handle(), transa, transb, descA, descB, descC,
        algo, spgemm_desc, buf1_sz, CUDA.CU_NULL)
    buf1 = CUDA.zeros(UInt8, max(1, Int(buf1_sz[])))
    cusparseSpGEMMreuse_workEstimation(
        handle(), transa, transb, descA, descB, descC,
        algo, spgemm_desc, buf1_sz, buf1)

    # ── Phase 2: nnz (three buffer pairs; call twice each) ──
    buf2_sz = Ref{Csize_t}(0); buf3_sz = Ref{Csize_t}(0); buf4_sz = Ref{Csize_t}(0)
    cusparseSpGEMMreuse_nnz(
        handle(), transa, transb, descA, descB, descC, algo, spgemm_desc,
        buf2_sz, CUDA.CU_NULL, buf3_sz, CUDA.CU_NULL, buf4_sz, CUDA.CU_NULL)
    buf2 = CUDA.zeros(UInt8, max(1, Int(buf2_sz[])))
    buf3 = CUDA.zeros(UInt8, max(1, Int(buf3_sz[])))
    buf4 = CUDA.zeros(UInt8, max(1, Int(buf4_sz[])))
    cusparseSpGEMMreuse_nnz(
        handle(), transa, transb, descA, descB, descC, algo, spgemm_desc,
        buf2_sz, buf2, buf3_sz, buf3, buf4_sz, buf4)

    # ── Query nnzC and allocate C arrays ──
    nnz_ref = Ref{Int64}(0)
    cusparseSpMatGetSize(descC, Ref{Int64}(0), Ref{Int64}(0), nnz_ref)
    nnzC = Int(nnz_ref[])

    C_colInd = CUDA.zeros(Ti, nnzC)
    C_nzVal  = CUDA.zeros(T,  nnzC)
    cusparseCsrSetPointers(descC, C_rowPtr, C_colInd, C_nzVal)

    # ── Phase 3: copy (call twice) ──
    buf5_sz = Ref{Csize_t}(0)
    cusparseSpGEMMreuse_copy(
        handle(), transa, transb, descA, descB, descC,
        algo, spgemm_desc, buf5_sz, CUDA.CU_NULL)
    buf5 = CUDA.zeros(UInt8, max(1, Int(buf5_sz[])))
    cusparseSpGEMMreuse_copy(
        handle(), transa, transb, descA, descB, descC,
        algo, spgemm_desc, buf5_sz, buf5)

    # Workspace buffers 1–5 are no longer needed after symbolic phases complete.
    # GC will reclaim them; spgemm_desc retains the symbolic result internally.

    CUSPARSESpGEMMHandle{T,Ti}(
        spgemm_desc, descA, descB, descC,
        C_rowPtr, C_colInd, C_nzVal,
        algo, transa, transb, m, n)
end

function JLUST.update_values!(h::CUSPARSESpGEMMHandle, u_A::USTensor, u_B::USTensor)
    cusparseSpMatSetValues(h.descA, nonzeros(u_A))
    cusparseSpMatSetValues(h.descB, nonzeros(u_B))
    return h
end

# Handle path — numeric-only reuse; no workspace, no reanalysis.
function JLUST.sparse_gemm!(h::CUSPARSESpGEMMHandle{T,Ti};
                              alpha=one(T), beta=zero(T)) where {T, Ti}
    cusparseSpGEMMreuse_compute(
        handle(), h.transa, h.transb,
        Ref{T}(alpha), h.descA, h.descB,
        Ref{T}(beta),  h.descC,
        T, h.algo, h.spgemm_desc)
    m, n = h.m, h.n
    cusA = CuSparseMatrixCSR{T,Ti}(h.C_rowPtr, h.C_colInd, h.C_nzVal, (m, n))
    return ust(cusA)
end

# ─── Execution ────────────────────────────────────────────────────────────────

# Direct path — builds fresh descriptors and runs the full SpGEMM each call.
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
