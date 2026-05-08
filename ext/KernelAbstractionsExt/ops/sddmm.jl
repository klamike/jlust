# ─── EmitterBackend: SDDMM ───────────────────────────────────────────────────
#
# Sampled dense-dense matrix multiply:
#   C ← alpha * (A * B) ∘ sparsity(C) + beta * C
#
# C is sparse (the mask and result); A and B are dense.
# Parallelism: one thread per outer fiber of C (row for CSR, fiber for DCSR,
# NNZ for COO). The inner k-loop is sequential within each thread.
#
# Shares _sparse_arg_names / _sparse_args / _spmv_ndrange from spmv.jl.

function JLUST.supports_backend(::EmitterBackend, op::SDDMMOp)
    _is_dense_fmt(op.A) && _is_dense_fmt(op.B) && _is_emittable(op.C)
end

# ─── Body emitter ─────────────────────────────────────────────────────────────
#
# Built via the unified level walker.  The walker binds `_y_idx` (output row)
# and `_x_idx` (output column from the inner level); the leaf computes the
# k-loop dot product against dense A and B and writes back to nzval.
# No row-level init/write is needed — the leaf handles its own scratch.

function _emit_sddmm_body(fmt::TensorFormat, ::Type{T}) where T
    leaf = quote
        _dot = $(zero(T))
        for _k in 1:_n_inner
            _dot += _A[_y_idx, _k] * _B[_k, _x_idx]
        end
        _nzval[_nnz_pos] = _alpha * _dot + _beta * _nzval[_nnz_pos]
    end
    # No row contention: each NNZ has a unique nzval position, so atomic == non-atomic.
    row_body = inner -> inner
    emit_kernel_body(fmt;
                     row_body_unique = row_body, row_body_atomic = row_body,
                     leaf_unique = leaf, leaf_atomic = leaf)
end

# ─── Kernel cache and launch ──────────────────────────────────────────────────

function _get_sddmm_kernel(fmt::TensorFormat, ::Type{T}) where T
    key = (fmt.name, T, :sddmm)
    haskey(_emitter_cache, key) && return _emitter_cache[key]

    body      = _emit_sddmm_body(fmt, T)
    buf_names = _sparse_arg_names(fmt)
    arg_names = vcat(buf_names, [:_A, :_B, :_alpha, :_beta, :_origin_off, :_n_outer, :_n_inner])
    fname     = gensym(:ust_sddmm)

    kern = @eval begin
        @kernel inbounds=true function $fname($(arg_names...))
            $body
        end
        $fname
    end

    _emitter_cache[key] = kern
    return kern
end

# ─── sparse_sddmm! ────────────────────────────────────────────────────────────

function JLUST.sparse_sddmm!(::EmitterBackend,
                               u_A::USTensor{T}, u_B::USTensor, u_C::USTensor;
                               alpha=one(T), beta=zero(T)) where T
    fmt     = format(u_C)
    ka      = KernelAbstractions.get_backend(nonzeros(u_C))
    off     = Int32(index_origin(u_C) isa OneBased ? 1 : 0)
    n_outer = Int32(_spmv_ndrange(u_C))
    n_inner = Int32(size(nonzeros(u_A), 2))   # cols of A = shared inner dimension

    kern = _get_sddmm_kernel(fmt, T)

    sparse_bufs = _sparse_args(u_C)
    all_args    = (sparse_bufs..., nonzeros(u_A), nonzeros(u_B),
                   T(alpha), T(beta), off, n_outer, n_inner)

    kernel_obj = Base.invokelatest(kern, ka, 64)
    Base.invokelatest(kernel_obj, all_args...; ndrange=Int(n_outer))

    return u_C
end

function JLUST.sparse_sddmm!(u_A::USTensor, u_B::USTensor, u_C::USTensor;
                               backend=EmitterBackend(), kw...)
    JLUST.sparse_sddmm!(backend, u_A, u_B, u_C; kw...)
end
