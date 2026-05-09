# ─── cuSPARSE sparse matrix-vector product ───────────────────────────────────
#
# y ← alpha * op(A) * x + beta * y

# ─── Handle ──────────────────────────────────────────────────────────────────

mutable struct CUSPARSESpMVHandle{T} <: JLUST.AbstractKernelHandle
    spmat_desc::CuSparseMatrixDescriptor
    dnvec_x::CuDenseVectorDescriptor
    dnvec_y::CuDenseVectorDescriptor
    workspace::CuVector{UInt8}
    transa::Char
    algo::cusparseSpMVAlg_t
end

export CUSPARSESpMVHandle

function JLUST.prepare(::CUSPARSEBackend, ::Type{<:Op{:SpMV}}, u_A::USTensor{T,Ti};
                        transa::Char='N',
                        algo::cusparseSpMVAlg_t=CUSPARSE_SPMV_ALG_DEFAULT) where {T<:_CUSPARSE_ELTYPES, Ti}
    idx   = _cusparse_index(u_A)
    m, n  = Int64.(extents(u_A))
    spmat = CuSparseMatrixDescriptor(_to_cuspmat(u_A), idx)
    x_len = transa == 'N' ? n : m
    y_len = transa == 'N' ? m : n
    descX = CuDenseVectorDescriptor(T, x_len)
    descY = CuDenseVectorDescriptor(T, y_len)
    α_ref = Ref{T}(one(T));  β_ref = Ref{T}(zero(T))

    ws = _cusparse_workspace() do buf_sz, buf
        if buf === CUDA.CU_NULL
            cusparseSpMV_bufferSize(handle(), transa, α_ref, spmat, descX, β_ref, descY, T, algo, buf_sz)
        else
            cusparseSpMV_preprocess(handle(), transa, α_ref, spmat, descX, β_ref, descY, T, algo, buf)
        end
    end
    CUSPARSESpMVHandle{T}(spmat, descX, descY, ws, transa, algo)
end

function JLUST.update_values!(h::CUSPARSESpMVHandle, u_A::USTensor)
    cusparseSpMatSetValues(h.spmat_desc, nonzeros(u_A))
    return h
end

# ─── Execution ────────────────────────────────────────────────────────────────
#
# Direct path uses CUDA.jl's high-level `CUSPARSE.mv!` — graph-capture safe and
# avoids per-call descriptor allocation.  The handle path (below) is for code
# that calls SpMV repeatedly with the same matrix; it pre-builds descriptors
# and workspace at `prepare()` time and reuses them across invocations.

function JLUST.execute(::CUSPARSEBackend, ::Op{:SpMV, F},
                       u_A::USTensor{T,Ti}, u_x::USTensor, u_y::USTensor;
                       transa::Char='N',
                       alpha=one(T), beta=zero(T)) where {F, T<:_CUSPARSE_ELTYPES, Ti}
    cusA = _to_cuspmat(u_A)
    idx  = _cusparse_index(u_A)
    CUSPARSE.mv!(transa, T(alpha), cusA, nonzeros(u_x), T(beta), nonzeros(u_y), idx)
    return u_y
end

# Handle path: A's descriptor is baked in at prepare time, so execute only
# takes the dense operands.  Refresh A's values via `update_values!(h, u_A)`.
#
# Two argument forms exist: `(h, u_x::USTensor, u_y::USTensor)` for the public
# API, and `(h, x::CuVector, y::CuVector)` for hot internal paths that already
# hold raw device arrays — the USTensor wrappers allocate ~10 small host objects
# per call which would dominate at the ~10μs scale.
function JLUST.execute(h::CUSPARSESpMVHandle{T},
                       x::CuVector, y::CuVector;
                       alpha=one(T), beta=zero(T)) where T
    _cusparse_set_dense!(h.dnvec_x, x)
    _cusparse_set_dense!(h.dnvec_y, y)
    cusparseSpMV(handle(), h.transa,
                 Ref{T}(alpha), h.spmat_desc, h.dnvec_x,
                 Ref{T}(beta),  h.dnvec_y,
                 T, h.algo, h.workspace)
    return y
end

@inline JLUST.execute(h::CUSPARSESpMVHandle{T}, u_x::USTensor, u_y::USTensor;
                      alpha=one(T), beta=zero(T)) where T =
    (JLUST.execute(h, nonzeros(u_x), nonzeros(u_y); alpha=alpha, beta=beta); u_y)


# ─── Walker-driven SpMV specializations ─────────────────────────────────────
#
# Two CUDA-specific SpMV kernels used to live here:
#
#   1. `_csr_spmv_specialized!` — VS-threads-per-row warp-vector with shuffle
#      reduce + LDG-wrapped x.  Now emitted by the generic walker for any
#      (Dense/Batch outer + Compressed-unique inner) format, gated by the
#      `_supports_warp_vector(::CUDABackend) = true` trait declared in
#      CUDAExt.jl.  See KernelAbstractionsExt/ops/_walker.jl + spmv.jl.
#
#   2. `_coo_spmv_specialized!` / `_coo_spmv_warp_kernel!` — segmented-warp-
#      reduce COO SpMV (1 thread/NNZ + log2(32) shfl_down + per-segment-head
#      atomic).  Now emitted by the same walker for any non-unique
#      Compressed outer (sorted-COO row list), again gated by the warp-vector
#      trait.  `_warp_seg_reduce_sum_down` (defined in CUDAExt.jl) provides
#      the device-side primitive the walker calls.
#
# Both retirements mean DCSR, custom user formats, sorted-row-list shapes, and
# any future variant get the warp-shuffle treatment without per-format work —
# the CSR/COO specializations are no longer load-bearing.
