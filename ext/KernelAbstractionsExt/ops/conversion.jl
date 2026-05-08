# ─── EmitterBackend: format conversions ──────────────────────────────────────
#
# sparse_to_dense: scatter stored NNZ into a pre-zeroed dense matrix.
#   Parallelism: one thread per outer fiber of the sparse tensor (same as SpMV).

function JLUST.supports_backend(::EmitterBackend, op::SparseToDenseOp)
    _is_emittable(op.src)
end

# ─── sparse_to_dense body emitter ─────────────────────────────────────────────
#
# Built via the unified level walker.  The walker binds `_y_idx` (row) and
# `_x_idx` (col); the leaf scatters `_nzval[_nnz_pos]` into the dense matrix.
# Each NNZ writes to a distinct (row, col), so atomic == non-atomic.

function _emit_s2d_body(fmt::TensorFormat)
    leaf = :(_dense[_y_idx, _x_idx] = _nzval[_nnz_pos])
    row_body = inner -> inner
    emit_kernel_body(fmt;
                     row_body_unique = row_body, row_body_atomic = row_body,
                     leaf_unique = leaf, leaf_atomic = leaf)
end

# ─── Kernel cache and launch ──────────────────────────────────────────────────

function _get_s2d_kernel(fmt::TensorFormat)
    key = (fmt.name, Any, :s2d)
    haskey(_emitter_cache, key) && return _emitter_cache[key]

    body      = _emit_s2d_body(fmt)
    buf_names = _sparse_arg_names(fmt)
    arg_names = vcat(buf_names, [:_dense, :_origin_off, :_n_outer])
    fname     = gensym(:ust_s2d)

    kern = @eval begin
        @kernel inbounds=true function $fname($(arg_names...))
            $body
        end
        $fname
    end

    _emitter_cache[key] = kern
    return kern
end

# ─── sparse_to_dense ──────────────────────────────────────────────────────────

function JLUST.sparse_to_dense(::EmitterBackend, u::USTensor{T,I,N,VA,VI,O}) where {T,I,N,VA,VI,O}
    fmt     = format(u)
    ka      = KernelAbstractions.get_backend(nonzeros(u))
    off     = Int32(O === OneBased ? 1 : 0)
    n_outer = Int32(_spmv_ndrange(u))

    dense_val = KernelAbstractions.zeros(ka, T, extents(u)...)
    VA2 = typeof(dense_val)
    kern = _get_s2d_kernel(fmt)

    sparse_bufs = _sparse_args(u)
    all_args    = (sparse_bufs..., dense_val, off, n_outer)

    kernel_obj = Base.invokelatest(kern, ka, 64)
    Base.invokelatest(kernel_obj, all_args...; ndrange=Int(n_outer))

    fmt_dense = Formats.DensedRight(N)
    USTensor{T,I,N,VA2,VI,O}(extents(u), fmt_dense,
                               Dict{Int,VI}(), Dict{Int,VI}(), dense_val, nothing)
end

function JLUST.sparse_to_dense(u::USTensor; backend=EmitterBackend(), kw...)
    JLUST.sparse_to_dense(backend, u; kw...)
end

# ─── dense_to_sparse ──────────────────────────────────────────────────────────

function JLUST.dense_to_sparse(::EmitterBackend, u::USTensor, fmt::TensorFormat; kw...)
    error("EmitterBackend does not support dense_to_sparse — requires parallel prefix " *
          "sums. Use CUSPARSEBackend() or convert_format() via a COO intermediate.")
end
