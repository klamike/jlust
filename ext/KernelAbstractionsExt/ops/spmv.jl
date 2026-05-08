# ─── Emitter: arg name/value collection ───────────────────────────────────────
#
# These hooks are the extension points for custom AbstractLevelFormat subtypes.
# Add methods in user code (or JLUST extensions) to plug a new level type into
# the emitter without touching _sparse_arg_names / _sparse_args.
#
#   _level_arg_names(lv, pc, cc; outermost::Bool) → Vector{Symbol}
#       Returns the @kernel arg names contributed by level `lv` at the given
#       depth.  `outermost=true` means this is the outermost level — non-unique
#       Compressed at the outer level has no pos buffer (no parent fiber), so
#       the pos slot is omitted.  pc/cc track pos/crd naming counters.
#
#   _level_has_nzval(lv) → Bool
#       Return false if the level encodes values implicitly (e.g. IncidenceLevel
#       where values are always ±1).  If ANY level returns false, :_nzval is
#       omitted from the @kernel signature.
#
#   _level_args(lv, u, lvl_idx; outermost::Bool) → Vector{AbstractArray}
#       Runtime arrays in the same order as _level_arg_names.

# AbstractLevelFormat fallbacks delegate to the public JLUST extension hooks.
# Custom levels are only ever inner (paired with a Dense/Compressed outer), so
# the outermost flag is forwarded to the public hook only for completeness.
_level_arg_names(lv::AbstractLevelFormat, pc::Ref, cc::Ref; outermost::Bool=false) =
    JLUST.level_arg_names(lv, pc, cc)

# Compressed/Delta: pos+crd, except at the outermost level — no parent fiber to
# index into, so only crd is bound (the format omits the pos buffer entirely).
function _level_arg_names(::CompressedLevel, pc::Ref, cc::Ref; outermost::Bool=false)
    outermost && return [Symbol(:_crd, cc[] += 1)]
    return [Symbol(:_pos, pc[] += 1), Symbol(:_crd, cc[] += 1)]
end
function _level_arg_names(::DeltaLevel, pc::Ref, cc::Ref; outermost::Bool=false)
    outermost && return [Symbol(:_crd, cc[] += 1)]
    return [Symbol(:_pos, pc[] += 1), Symbol(:_crd, cc[] += 1)]
end
_level_arg_names(::SingletonLevel, pc::Ref, cc::Ref; outermost::Bool=false) =
    [Symbol(:_crd, cc[] += 1)]

_level_has_nzval(lv::AbstractLevelFormat) = JLUST.level_has_nzval(lv)
_level_has_nzval(::Union{DenseLevel,BatchLevel,CompressedLevel,SingletonLevel,DeltaLevel,RangeLevel}) = true

_level_args(lv::AbstractLevelFormat, u::USTensor, lvl::Int; outermost::Bool=false) =
    JLUST.level_args(lv, u, lvl)

function _level_args(::CompressedLevel, u::USTensor, lvl::Int; outermost::Bool=false)
    outermost && return AbstractArray[coordinates(u, lvl)]
    return AbstractArray[positions(u, lvl), coordinates(u, lvl)]
end
function _level_args(::DeltaLevel, u::USTensor, lvl::Int; outermost::Bool=false)
    outermost && return AbstractArray[coordinates(u, lvl)]
    return AbstractArray[positions(u, lvl), coordinates(u, lvl)]
end
_level_args(::SingletonLevel, u::USTensor, lvl::Int; outermost::Bool=false) =
    AbstractArray[coordinates(u, lvl)]

# Symbolic arg names matching the order the emitter expects for the sparse A.
function _sparse_arg_names_for_levels(levels::Tuple)
    names = Symbol[]
    pc = Ref(0); cc = Ref(0)
    for (i, lv) in enumerate(levels)
        append!(names, _level_arg_names(lv, pc, cc; outermost = (i == 1)))
    end
    all(_level_has_nzval(lv) for lv in levels) && push!(names, :_nzval)
    names
end

_sparse_arg_names(fmt::TensorFormat) = _sparse_arg_names_for_levels(fmt.levels)

# Actual arrays in the same order as _sparse_arg_names.
function _sparse_args(u::USTensor)
    args = AbstractArray[]
    for (lvl, lv) in enumerate(u.format.levels)
        append!(args, _level_args(lv, u, lvl; outermost = (lvl == 1)))
    end
    all(_level_has_nzval(lv) for lv in u.format.levels) && push!(args, nonzeros(u))
    args
end

# Compute thread count (ndrange) for SpMV based on outermost format level.
# DenseLevel: one thread per dense row. CompressedLevel: one thread per outer fiber.
# For non-unique Compressed (COO-like): one thread per NNZ (length of crd, not pos).
function _spmv_ndrange(u_A::USTensor)
    lv1 = format(u_A).levels[1]
    if lv1 isa Union{DenseLevel,BatchLevel}
        extents(u_A)[1]
    else
        length(coordinates(u_A, 1))   # fibers for unique, NNZ for non-unique
    end
end

# ─── SpMV body emitter ────────────────────────────────────────────────────────
#
# Recursively builds a Julia Expr for the inner body of a @kernel SpMV function.
# Two passes of counters (pc, cc) track pos/crd buffer names to match
# _sparse_arg_names ordering exactly.
#
# p_var:         Symbol name of the current 1-based fiber position variable,
#                or nothing for the outermost level (becomes thread index).
# needs_atomic:  true when y writes must use @atomic (e.g. COO-like patterns).
# input_fn_sym:  Symbol for the per-element input transform (default :_input_fn).
#                Applied as _input_fn(_x[_x_idx]) at each leaf.
# output_fn_sym: Symbol for the post-accumulation output transform (default :_output_fn).
#                Applied as _output_fn(_acc) at each row write.
#
# Julia specializes kernels on the concrete types of _input_fn/_output_fn, so
# passing `identity` compiles to zero overhead.

function _emit_spmv_body(levels::Tuple, ::Type{T};
                          input_fn_sym::Symbol=:_input_fn,
                          output_fn_sym::Symbol=:_output_fn) where T
    leaf_unique = :(_acc += _nzval[_nnz_pos] * $input_fn_sym(_x[_x_idx]))
    leaf_atomic = :(KernelAbstractions.@atomic _y[_y_idx] +=
                        _alpha * _nzval[_nnz_pos] * $input_fn_sym(_x[_x_idx]))
    row_body_unique = inner -> quote
        _acc = $(zero(T))
        $inner
        _y[_y_idx] = _alpha * $output_fn_sym(_acc) + _beta * _y[_y_idx]
    end
    # Non-unique outer (COO-like): pre-scaled-by-beta y is accumulated into via the atomic leaf.
    row_body_atomic = inner -> inner
    emit_kernel_body(levels;
                     row_body_unique, row_body_atomic, leaf_unique, leaf_atomic)
end

_emit_spmv_body(fmt::TensorFormat, T; kw...) = _emit_spmv_body(fmt.levels, T; kw...)

# ─── COO chunked kernel (reduced atomic contention) ──────────────────────────
#
# For sorted COO, consecutive NNZ often share the same row.  Each thread
# processes CHUNK consecutive NNZ and accumulates locally before doing a single
# atomicAdd per row encountered.  With CHUNK=8 this reduces atomic operations
# by ~min(CHUNK, avg_nnz_per_row)× compared to one-atomic-per-NNZ.
#
# ndrange = ceil(nnz / CHUNK); each thread handles [i_lo, i_hi] inclusive.

const _COO_CHUNK = 8   # NNZ per thread; Int for type stability

@kernel inbounds=true function _coo_spmv_chunked!(
        row_crd, col_crd, _nzval, _x, _y, _origin_off, _n_nnz)
    tid  = @index(Global, Linear)
    i_lo = (tid - 1) * _COO_CHUNK + 1
    i_hi = min(i_lo + _COO_CHUNK - 1, Int(_n_nnz))

    cur_row = -1   # sentinel: no current row
    acc     = zero(eltype(_nzval))

    for i in i_lo:i_hi
        row = Int(row_crd[i]) - Int(_origin_off)   # 0-based row index
        col = Int(col_crd[i]) - Int(_origin_off) + 1
        if row != cur_row
            if cur_row >= 0
                KernelAbstractions.@atomic _y[cur_row + 1] += acc
            end
            cur_row = row
            acc = zero(eltype(_nzval))
        end
        acc += _nzval[i] * _x[col]
    end
    if cur_row >= 0
        KernelAbstractions.@atomic _y[cur_row + 1] += acc
    end
end

# ─── Kernel functions and launch ─────────────────────────────────────────────
#
# Each op's kernel is a regular `@generated` Julia function that emits its body
# from the format type's level structure, using `KernelIntrinsics` (KI)
# primitives directly (`KI.get_global_id().x`, `Atomix.@atomic`).  At the call
# site, `_launch_kern` dispatches via `KI.kernel_function` — Julia specializes
# per (FMT, T, ...) tuple type, the generated body compiles in the call site's
# world age, and there is no global Dict cache and no `Base.invokelatest`.

# Launch helper: call `KI.kernel_function` and dispatch the kernel.  Bypasses
# the `KI.@kernel` macro so we can splat varargs cleanly.
@inline function _launch_kern(ka, kfunc, args::Tuple, ndrange::Int, ws::Int=64)
    ng        = cld(ndrange, ws)
    kf        = KI.argconvert(ka, kfunc)
    kargs     = map(x -> KI.argconvert(ka, x), args)
    ktt       = Tuple{map(Core.Typeof, kargs)...}
    kobj      = KI.kernel_function(ka, kf, ktt)
    kobj(kargs...; numworkgroups=ng, workgroupsize=ws)
end

# SpMV kernel — body emitted from FMT's level types.  `args` is the variadic
# tail: sparse buffers, x, y, origin_off, n_outer, input_fn, output_fn, alpha, beta.
@generated function _ust_spmv_kern(::Type{FMT}, ::Type{T}, args::Vararg{Any, M}) where {FMT<:TensorFormat, T, M}
    LT          = FMT.parameters[1]
    levels      = ntuple(i -> LT.parameters[i](), Val(length(LT.parameters)))
    sparse_nms  = _sparse_arg_names_for_levels(levels)
    standard_nm = (:_x, :_y, :_origin_off, :_n_outer, :_input_fn, :_output_fn, :_alpha, :_beta)
    all_nms     = (sparse_nms..., standard_nm...)
    bindings    = [Expr(:(=), nm, :(args[$i])) for (i, nm) in enumerate(all_nms)]
    body        = _emit_spmv_body(levels, T)
    quote
        @inbounds begin
            $(bindings...)
            $body
        end
        return nothing
    end
end

# ─── sparse_mv! ───────────────────────────────────────────────────────────────

function JLUST.sparse_mv!(::EmitterBackend, u_A::USTensor, u_x::USTensor, u_y::USTensor;
                          alpha=one(eltype(u_A)), beta=zero(eltype(u_A)),
                          input_fn::IF=identity, output_fn::OF=identity) where {IF, OF}
    fmt          = format(u_A)
    T            = eltype(u_A)
    T_alpha      = T(alpha)
    T_beta       = T(beta)
    ka           = KernelAbstractions.get_backend(nonzeros(u_A))
    off          = Int32(index_origin(u_A) isa OneBased ? 1 : 0)

    lv1          = fmt.levels[1]
    # COO-like: outermost non-unique Compressed (row coords) + Singleton inner (col coords).
    # Checking the inner level prevents routing generic non-unique-Compressed-outer formats
    # through code that assumes coordinates(A,1)=rows, coordinates(A,2)=cols.
    is_coo_like  = lv1 isa CompressedLevel && !is_unique(lv1) &&
                   length(fmt.levels) >= 2 && fmt.levels[2] isa SingletonLevel
    use_identity = IF === typeof(identity) && OF === typeof(identity)

    if is_coo_like
        # COO-like: pre-scale y by beta; atomic += accumulates into this baseline.
        iszero(T_beta) ? fill!(nonzeros(u_y), zero(T)) : (nonzeros(u_y) .*= T_beta)
        if use_identity && isone(T_alpha)
            # Specialized chunked kernel: lower atomic contention than generic emitter.
            row_crd = coordinates(u_A, 1)
            col_crd = coordinates(u_A, 2)
            nzv     = nonzeros(u_A)
            nx      = nonzeros(u_x)
            ny      = nonzeros(u_y)
            n_nnz   = Int32(length(row_crd))
            if !JLUST._coo_spmv_specialized!(row_crd, col_crd, nzv, nx, ny, off, n_nnz)
                n_chunks = Int32(cld(Int(n_nnz), _COO_CHUNK))
                _coo_spmv_chunked!(ka, 64)(
                    row_crd, col_crd, nzv, nx, ny, off, n_nnz;
                    ndrange=Int(n_chunks))
            end
        else
            # General emitter: _alpha applied in atomic leaf; beta already pre-scaled.
            n_outer     = Int32(_spmv_ndrange(u_A))
            sparse_bufs = _sparse_args(u_A)
            all_args    = (typeof(fmt), T,
                           sparse_bufs..., nonzeros(u_x), nonzeros(u_y), off, n_outer,
                           input_fn, output_fn, T_alpha, zero(T))
            _launch_kern(ka, _ust_spmv_kern, all_args, Int(n_outer))
        end
    else
        n_outer     = Int32(_spmv_ndrange(u_A))
        sparse_bufs = _sparse_args(u_A)

        # CSR-vector hook: DenseLevel outer + unique CompressedLevel inner, no fusion.
        # CUDAExt overrides with a warp-shuffle vector kernel (supports all beta values).
        # Val{ZERO_BETA} specialization avoids the y-read when beta=0.
        dispatched = false
        if use_identity &&
                lv1 isa Union{DenseLevel,BatchLevel} && length(fmt.levels) >= 2
            lv2 = fmt.levels[2]
            if lv2 isa CompressedLevel && is_unique(lv2)
                dispatched = JLUST._csr_spmv_specialized!(
                    sparse_bufs[1], sparse_bufs[2],
                    nonzeros(u_A), nonzeros(u_x), nonzeros(u_y), off, n_outer, T_beta)
            end
        end

        if !dispatched
            all_args = (typeof(fmt), T,
                        sparse_bufs..., nonzeros(u_x), nonzeros(u_y), off, n_outer,
                        input_fn, output_fn, T_alpha, T_beta)
            _launch_kern(ka, _ust_spmv_kern, all_args, Int(n_outer))
        end
    end

    return u_y
end

# ─── EmitterSpMVHandle ────────────────────────────────────────────────────────
#
# With the @generated kernel architecture, dispatch is free — Julia's method
# specialization caches the kernel automatically per (FMT, T, IF, OF) tuple.
# The handle just preserves metadata (n_outer, off, input/output transforms)
# so repeated calls don't re-extract them; there is no separate "kernel object".
#
# input_fn:  applied element-wise to x before each multiply: y = A * input_fn.(x)
# output_fn: applied to the accumulated row sum before writing:  y[i] = output_fn(acc)
#
# Pass identity (the default) for either to generate zero-overhead kernel paths.

struct EmitterSpMVHandle{T, IF, OF, FMT}
    n_outer   :: Int32
    off       :: Int32
    input_fn  :: IF
    output_fn :: OF
end

export EmitterSpMVHandle

function JLUST.prepare(::EmitterBackend, ::Type{<:Op{:SpMV}}, u_A::USTensor{T};
                        input_fn::IF=identity, output_fn::OF=identity) where {T, IF, OF}
    off     = Int32(index_origin(u_A) isa OneBased ? 1 : 0)
    n_outer = Int32(_spmv_ndrange(u_A))
    EmitterSpMVHandle{T, IF, OF, typeof(format(u_A))}(n_outer, off, input_fn, output_fn)
end

function JLUST.sparse_mv!(h::EmitterSpMVHandle, u_A::USTensor, x::AbstractVector, y::AbstractVector; kw...)
    JLUST.sparse_mv!(h, u_A, ust(x), ust(y); kw...)
end

function JLUST.sparse_mv!(h::EmitterSpMVHandle{T, IF, OF, FMT},
                           u_A::USTensor, u_x::USTensor, u_y::USTensor;
                           alpha=one(T), beta=zero(T)) where {T, IF, OF, FMT}
    ka          = KernelAbstractions.get_backend(nonzeros(u_A))
    sparse_bufs = _sparse_args(u_A)
    args = (FMT, T,
            sparse_bufs..., nonzeros(u_x), nonzeros(u_y),
            h.off, h.n_outer, h.input_fn, h.output_fn, T(alpha), T(beta))
    _launch_kern(ka, _ust_spmv_kern, args, Int(h.n_outer))
    return u_y
end
