# ─── cuSPARSE sparse matrix-matrix product ───────────────────────────────────
#
# C ← alpha * op(A) * op(B) + beta * C
# B and C must be column-major dense CuMatrices.

# ─── Handle ──────────────────────────────────────────────────────────────────

mutable struct CUSPARSESpMMHandle{T} <: JLUST.AbstractKernelHandle
    spmat_desc::CuSparseMatrixDescriptor
    dnmat_B::CuDenseMatrixDescriptor
    dnmat_C::CuDenseMatrixDescriptor
    workspace::CuVector{UInt8}
    transa::Char
    transb::Char
    algo::cusparseSpMMAlg_t
end

export CUSPARSESpMMHandle

function JLUST.prepare(::CUSPARSEBackend, ::Type{<:Op{:SpMM}}, u_A::USTensor{T,Ti};
                        transa::Char='N', transb::Char='N',
                        n_cols::Int,
                        algo::cusparseSpMMAlg_t=CUSPARSE_SPMM_ALG_DEFAULT) where {T<:_CUSPARSE_ELTYPES, Ti}
    idx   = _cusparse_index(u_A)
    m, n  = Int64.(extents(u_A))
    spmat = CuSparseMatrixDescriptor(_to_cuspmat(u_A), idx)
    k_dim = transa == 'N' ? n : m
    m_out = transa == 'N' ? m : n
    descB = CuDenseMatrixDescriptor(T, k_dim, n_cols)
    descC = CuDenseMatrixDescriptor(T, m_out, n_cols)
    α_ref = Ref{T}(one(T));  β_ref = Ref{T}(zero(T))

    ws = _cusparse_workspace() do buf_sz, buf
        if buf === CUDA.CU_NULL
            cusparseSpMM_bufferSize(handle(), transa, transb, α_ref, spmat, descB, β_ref, descC, T, algo, buf_sz)
        else
            cusparseSpMM_preprocess(handle(), transa, transb, α_ref, spmat, descB, β_ref, descC, T, algo, buf)
        end
    end
    CUSPARSESpMMHandle{T}(spmat, descB, descC, ws, transa, transb, algo)
end

function JLUST.update_values!(h::CUSPARSESpMMHandle, u_A::USTensor)
    cusparseSpMatSetValues(h.spmat_desc, nonzeros(u_A))
    return h
end

# ─── Execution ────────────────────────────────────────────────────────────────

function JLUST.execute(::CUSPARSEBackend, ::Op{:SpMM, F},
                       u_A::USTensor{T,Ti}, u_B::USTensor, u_C::USTensor;
                       transa::Char='N', transb::Char='N',
                       alpha=one(T), beta=zero(T)) where {F, T<:_CUSPARSE_ELTYPES, Ti}
    cusA = _to_cuspmat(u_A)
    idx  = _cusparse_index(u_A)
    CUSPARSE.mm!(transa, transb, T(alpha), cusA, nonzeros(u_B), T(beta), nonzeros(u_C), idx)
    return u_C
end

function JLUST.execute(h::CUSPARSESpMMHandle{T},
                       u_B::USTensor, u_C::USTensor;
                       alpha=one(T), beta=zero(T)) where T
    _cusparse_set_dense!(h.dnmat_B, nonzeros(u_B))
    _cusparse_set_dense!(h.dnmat_C, nonzeros(u_C))
    cusparseSpMM(handle(), h.transa, h.transb,
                 Ref{T}(alpha), h.spmat_desc, h.dnmat_B,
                 Ref{T}(beta),  h.dnmat_C,
                 T, h.algo, h.workspace)
    return u_C
end

# cuSPARSE rejects SubArray outputs; the BBM scatter path routes row_bufs that
# are views into a stacked diag_out buffer.  Force EmitterBackend for SpMM
# with a SubArray-backed C, regardless of the other operands' storage.
function JLUST.execute(::Type{OT},
                       A::USTensor, B::USTensor, C::USTensor{T,Ti,N,<:SubArray};
                       backend=nothing, kw...) where {OT<:Op{:SpMM}, T, Ti, N}
    be = something(backend, EmitterBackend())
    JLUST.execute(be, OT(format(A), format(B), format(C)), A, B, C; kw...)
end
