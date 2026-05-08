# ─── EmitterBackend: format conversions ──────────────────────────────────────
#
# sparse_to_dense: scatter stored NNZ into a pre-zeroed dense matrix.
#   Parallelism: one thread per outer fiber of the sparse tensor (same as SpMV).

function JLUST.supports_backend(::EmitterBackend, ::Op{:SparseToDense, Tuple{S}}) where {S}
    _is_emittable(S)
end

# ─── sparse_to_dense body emitter ─────────────────────────────────────────────
#
# Built via the unified level walker.  The walker binds `_y_idx` (row) and
# `_x_idx` (col); the leaf scatters `_nzval[_nnz_pos]` into the dense matrix.
# Each NNZ writes to a distinct (row, col), so atomic == non-atomic.

function _emit_s2d_body(levels::Tuple)
    leaf = :(_dense[_y_idx, _x_idx] = _nzval[_nnz_pos])
    row_body = inner -> inner
    emit_kernel_body(levels;
                     row_body_unique = row_body, row_body_atomic = row_body,
                     leaf_unique = leaf, leaf_atomic = leaf)
end

_emit_s2d_body(fmt::TensorFormat) = _emit_s2d_body(fmt.levels)

# ─── @generated sparse_to_dense kernel ───────────────────────────────────────

@generated function _ust_s2d_kern(::Type{FMT}, args::Vararg{Any, M}) where {FMT<:TensorFormat, M}
    LT          = FMT.parameters[1]
    levels      = ntuple(i -> LT.parameters[i](), Val(length(LT.parameters)))
    sparse_nms  = _sparse_arg_names_for_levels(levels)
    standard_nm = (:_dense, :_origin_off, :_n_outer)
    all_nms     = (sparse_nms..., standard_nm...)
    bindings    = [Expr(:(=), nm, :(args[$i])) for (i, nm) in enumerate(all_nms)]
    body        = _emit_s2d_body(levels)
    quote
        @inbounds begin
            $(bindings...)
            $body
        end
        return nothing
    end
end

# ─── sparse_to_dense ──────────────────────────────────────────────────────────

function JLUST.execute(::EmitterBackend, ::Op{:SparseToDense, F},
                        u::USTensor{T,I,N,VA,VI,O}) where {F, T,I,N,VA,VI,O}
    fmt     = format(u)
    ka      = KernelAbstractions.get_backend(nonzeros(u))
    off     = Int32(O === OneBased ? 1 : 0)
    n_outer = Int32(_spmv_ndrange(u))

    dense_val = KernelAbstractions.zeros(ka, T, extents(u)...)
    VA2 = typeof(dense_val)

    sparse_bufs = _sparse_args(u)
    args = (typeof(fmt), sparse_bufs..., dense_val, off, n_outer)
    _launch_kern(ka, _ust_s2d_kern, args, Int(n_outer))

    fmt_dense = Formats.DensedRight(N)
    USTensor{T,I,N,VA2,VI,O}(extents(u), fmt_dense,
                               JLUST._no_bufs(Val(N), VI), JLUST._no_bufs(Val(N), VI),
                               dense_val, nothing)
end


# ─── dense_to_sparse ──────────────────────────────────────────────────────────

function JLUST.execute(::EmitterBackend, ::Op{:DenseToSparse}, u::USTensor, fmt::TensorFormat; kw...)
    error("EmitterBackend does not support dense_to_sparse — requires parallel prefix " *
          "sums. Use CUSPARSEBackend() or convert_format() via a COO intermediate.")
end
