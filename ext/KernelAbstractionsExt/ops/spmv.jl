# ─── Emitter: arg name/value collection ───────────────────────────────────────
#
# These three functions are the extension points for custom AbstractLevelFormat
# subtypes.  Add methods in user code (or JLUST extensions) to plug a new level
# type into the emitter without touching _sparse_arg_names / _sparse_args.
#
#   _level_arg_names(lv, pc, cc) → Vector{Symbol}
#       Returns the @kernel arg names contributed by level `lv`.
#       pc/cc are Ref counters for pos/crd naming (increment as needed).
#
#   _level_has_nzval(lv) → Bool
#       Return false if the level encodes values implicitly (e.g. IncidenceLevel
#       where values are always ±1).  If ANY level returns false, :_nzval is
#       omitted from the @kernel signature.
#
#   _level_args(lv, u, lvl_idx, dummy_pos) → Vector{AbstractArray}
#       Returns the actual array arguments for level `lv` in USTensor `u`.
#       Must be consistent with the names returned by _level_arg_names.

# AbstractLevelFormat fallbacks delegate to the public JLUST extension hooks so
# that user-defined level types only need to extend JLUST.level_* functions.
_level_arg_names(lv::AbstractLevelFormat,             pc::Ref, cc::Ref) = JLUST.level_arg_names(lv, pc, cc)
_level_arg_names(::Union{CompressedLevel,DeltaLevel}, pc::Ref, cc::Ref) =
    [Symbol(:_pos, pc[] += 1), Symbol(:_crd, cc[] += 1)]
_level_arg_names(::SingletonLevel,                    pc::Ref, cc::Ref) =
    [Symbol(:_crd, cc[] += 1)]

_level_has_nzval(lv::AbstractLevelFormat) = JLUST.level_has_nzval(lv)
_level_has_nzval(::Union{DenseLevel,BatchLevel,CompressedLevel,SingletonLevel,DeltaLevel,RangeLevel}) = true

_level_args(lv::AbstractLevelFormat, u::USTensor, lvl::Int, dummy_pos) = JLUST.level_args(lv, u, lvl, dummy_pos)
function _level_args(lv::Union{CompressedLevel,DeltaLevel}, u::USTensor, lvl::Int, dummy_pos)
    pos = has_positions(u, lvl) ? positions(u, lvl) : dummy_pos
    AbstractArray[pos, coordinates(u, lvl)]
end
_level_args(::SingletonLevel, u::USTensor, lvl::Int, dummy_pos) =
    AbstractArray[coordinates(u, lvl)]

# Symbolic arg names matching the order the emitter expects for the sparse A.
function _sparse_arg_names(fmt::TensorFormat)
    names = Symbol[]
    pc = Ref(0); cc = Ref(0)
    for (_, lv) in fmt.levels
        append!(names, _level_arg_names(lv, pc, cc))
    end
    all(_level_has_nzval(p.second) for p in fmt.levels) && push!(names, :_nzval)
    names
end

# Actual arrays in the same order as _sparse_arg_names.
# Outermost CompressedLevel (DCSR, COO) has no pos buffer; pass a typed empty
# array as a placeholder — the emitter never generates an access to _pos1.
function _sparse_args(u::USTensor{T,I,N,VA,VI,O}) where {T,I,N,VA,VI,O}
    dummy_pos = VI(undef, 0)
    args = AbstractArray[]
    for (lvl, (_, lv)) in enumerate(u.format.levels)
        append!(args, _level_args(lv, u, lvl, dummy_pos))
    end
    all(_level_has_nzval(p.second) for p in u.format.levels) && push!(args, nonzeros(u))
    args
end

# Compute thread count (ndrange) for SpMV based on outermost format level.
# DenseLevel: one thread per dense row. CompressedLevel: one thread per outer fiber.
# For non-unique Compressed (COO-like): one thread per NNZ (length of crd, not pos).
function _spmv_ndrange(u_A::USTensor)
    _, lv1 = format(u_A).levels[1]
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

function _emit_spmv_body(fmt::TensorFormat, ::Type{T};
                          input_fn_sym::Symbol=:_input_fn,
                          output_fn_sym::Symbol=:_output_fn) where T
    pc = Ref(0); cc = Ref(0)
    _, lv1 = fmt.levels[1]
    na = lv1 isa CompressedLevel && !is_unique(lv1)   # COO-like → atomic
    _emit_spmv_level(fmt.levels, 1, nothing, pc, cc, T, na, input_fn_sym, output_fn_sym)
end

function _emit_spmv_level(levels, lvl, p_var, pc, cc, T, needs_atomic,
                           input_fn_sym, output_fn_sym)
    if lvl > length(levels)
        # Leaf: accumulate or atomic-write, applying input_fn to each x element.
        return needs_atomic ?
            :(KernelAbstractions.@atomic _y[_y_idx] += _alpha * _nzval[_nnz_pos] * $input_fn_sym(_x[_x_idx])) :
            :(_acc += _nzval[_nnz_pos] * $input_fn_sym(_x[_x_idx]))
    end
    _, lv = levels[lvl]
    _emit_spmv_lv(lv, levels, lvl, p_var, pc, cc, T, needs_atomic, input_fn_sym, output_fn_sym)
end

# AbstractLevelFormat (custom user types) → delegate to public JLUST.emit_spmv_lv.
# Inner position: forward (lv, p_var, input_fn_sym) — the minimal signature users implement.
# Outermost (::Nothing): custom levels must be inner levels paired with a DenseLevel outer.
function _emit_spmv_lv(lv::AbstractLevelFormat, _levels, _lvl, p_var::Symbol,
                        _pc, _cc, _T, _na, input_fn_sym, _output_fn_sym)
    JLUST.emit_spmv_lv(lv, p_var, input_fn_sym)
end

function _emit_spmv_lv(lv::AbstractLevelFormat, _levels, _lvl, ::Nothing,
                        _pc, _cc, _T, _na, _input_fn_sym, _output_fn_sym)
    error("EmitterBackend SpMV: $(typeof(lv)) cannot be the outermost level; pair with DenseLevel.")
end

# DenseLevel / BatchLevel (outermost) → thread = dense row index
function _emit_spmv_lv(::Union{DenseLevel,BatchLevel}, levels, lvl, ::Nothing, pc, cc, T, _,
                        input_fn_sym, output_fn_sym)
    inner = _emit_spmv_level(levels, lvl + 1, :_tid, pc, cc, T, false, input_fn_sym, output_fn_sym)
    quote
        _tid = @index(Global, Linear)
        if _tid <= _n_outer
            _acc  = $(zero(T))
            $inner
            _y[_tid] = _alpha * $output_fn_sym(_acc) + _beta * _y[_tid]
        end
    end
end

# DenseLevel / BatchLevel (non-outermost) → dense loop (uncommon for SpMV)
function _emit_spmv_lv(::Union{DenseLevel,BatchLevel}, levels, lvl, p_var::Symbol, pc, cc, T, na,
                        input_fn_sym, output_fn_sym)
    sz  = Symbol(:_sz, lvl)
    lv2 = Symbol(:_i, lvl)
    inner = _emit_spmv_level(levels, lvl + 1, lv2, pc, cc, T, na, input_fn_sym, output_fn_sym)
    quote
        for $lv2 in 1:$sz
            $inner
        end
    end
end

# CompressedLevel (outermost)
#   unique   → fiber-parallel (DCSR-like): one thread per non-empty row, no atomics
#   non-unique → NNZ-parallel (COO-like): one thread per NNZ, atomic y update
function _emit_spmv_lv(lv::CompressedLevel, levels, lvl, ::Nothing, pc, cc, T, _,
                        input_fn_sym, output_fn_sym)
    pc[] += 1; ci = cc[] += 1
    cs = Symbol(:_crd, ci)
    if is_unique(lv)
        inner = _emit_spmv_level(levels, lvl + 1, :_tid, pc, cc, T, false, input_fn_sym, output_fn_sym)
        quote
            _tid = @index(Global, Linear)
            if _tid <= _n_outer
                _y_idx = Int($cs[_tid]) - Int(_origin_off) + 1   # 1-based row
                _acc   = $(zero(T))
                $inner
                _y[_y_idx] = _alpha * $output_fn_sym(_acc) + _beta * _y[_y_idx]
            end
        end
    else
        # Non-unique: thread = NNZ index within the single outer fiber.
        # Row index comes directly from crd (not from a fiber pos lookup).
        inner = _emit_spmv_level(levels, lvl + 1, :_tid, pc, cc, T, true, input_fn_sym, output_fn_sym)
        quote
            _tid = @index(Global, Linear)
            if _tid <= _n_outer
                _y_idx = Int($cs[_tid]) - Int(_origin_off) + 1
                $inner
            end
        end
    end
end

# CompressedLevel (non-outermost) → inner fiber loop; crd stores x dimension index
function _emit_spmv_lv(::CompressedLevel, levels, lvl, p_var::Symbol, pc, cc, T, na,
                        input_fn_sym, output_fn_sym)
    pi = pc[] += 1; ci = cc[] += 1
    ps = Symbol(:_pos, pi); cs = Symbol(:_crd, ci)
    lvar = Symbol(:_i, lvl)
    inner = _emit_spmv_level(levels, lvl + 1, lvar, pc, cc, T, na, input_fn_sym, output_fn_sym)
    quote
        _lo = Int($ps[$p_var])     - Int(_origin_off)
        _hi = Int($ps[$p_var + 1]) - Int(_origin_off)
        for $lvar in (_lo + 1):_hi
            _x_idx   = Int($cs[$lvar]) - Int(_origin_off) + 1
            _nnz_pos = $lvar
            $inner
        end
    end
end

# SingletonLevel → one coordinate per position (COO column index)
function _emit_spmv_lv(::SingletonLevel, levels, lvl, p_var::Symbol, pc, cc, T, na,
                        input_fn_sym, output_fn_sym)
    ci = cc[] += 1
    cs = Symbol(:_crd, ci)
    inner = _emit_spmv_level(levels, lvl + 1, p_var, pc, cc, T, na, input_fn_sym, output_fn_sym)
    quote
        _x_idx   = Int($cs[$p_var]) - Int(_origin_off) + 1
        _nnz_pos = $p_var
        $inner
    end
end

# DeltaLevel (non-outermost) → accumulated delta decode + inner loop
function _emit_spmv_lv(::DeltaLevel, levels, lvl, p_var::Symbol, pc, cc, T, na,
                        input_fn_sym, output_fn_sym)
    pi = pc[] += 1; ci = cc[] += 1
    ps = Symbol(:_pos, pi); cs = Symbol(:_crd, ci)
    lvar   = Symbol(:_i, lvl)
    corig  = Symbol(:_corig, lvl)
    inner  = _emit_spmv_level(levels, lvl + 1, lvar, pc, cc, T, na, input_fn_sym, output_fn_sym)
    quote
        _lo = Int($ps[$p_var])     - Int(_origin_off)
        _hi = Int($ps[$p_var + 1]) - Int(_origin_off)
        $corig = 0
        for $lvar in (_lo + 1):_hi
            $corig  += Int($cs[$lvar])
            _x_idx   = $corig - Int(_origin_off) + 1
            _nnz_pos = $lvar
            $inner
            $corig += 1
        end
    end
end

function _emit_spmv_lv(::RangeLevel, levels, lvl, p_var, pc, cc, T, na,
                        input_fn_sym, output_fn_sym)
    error("EmitterBackend SpMV: RangeLevel (DIA-style) kernels not supported. " *
          "Use convert_format to CSR or DCSR first.")
end

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

# ─── Kernel cache and launch ─────────────────────────────────────────────────

# Cache: (TensorFormat, element_type, :spmv/:spmm/:spmm_nf) [or with extra n_col] → @kernel fn.
const _emitter_cache = Dict{Any, Any}()

function _get_spmv_kernel(fmt::TensorFormat, ::Type{T}) where T
    # Julia JIT-specializes on the concrete (input_fn, output_fn) types at call
    # time — no need for separate @kernel functions per transform pair.
    key = (fmt, T, :spmv)
    haskey(_emitter_cache, key) && return _emitter_cache[key]

    body      = _emit_spmv_body(fmt, T)
    buf_names = _sparse_arg_names(fmt)
    arg_names = vcat(buf_names, [:_x, :_y, :_origin_off, :_n_outer, :_input_fn, :_output_fn, :_alpha, :_beta])
    fname     = gensym(:ust_spmv)

    kern = @eval begin
        @kernel inbounds=true function $fname($(arg_names...))
            $body
        end
        $fname
    end

    _emitter_cache[key] = kern
    return kern
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

    _, lv1       = fmt.levels[1]
    # COO-like: outermost non-unique Compressed (row coords) + Singleton inner (col coords).
    # Checking the inner level prevents routing generic non-unique-Compressed-outer formats
    # through code that assumes coordinates(A,1)=rows, coordinates(A,2)=cols.
    is_coo_like  = lv1 isa CompressedLevel && !is_unique(lv1) &&
                   length(fmt.levels) >= 2 && fmt.levels[2].second isa SingletonLevel
    use_identity = IF === typeof(identity) && OF === typeof(identity)

    if is_coo_like
        # Pre-scale y by beta; atomic += accumulates into this baseline.
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
                Base.invokelatest(_coo_spmv_chunked!, ka, 64)(
                    row_crd, col_crd, nzv, nx, ny, off, n_nnz;
                    ndrange=Int(n_chunks))
            end
        else
            # General emitter: _alpha applied in atomic leaf; beta already pre-scaled.
            n_outer     = Int32(_spmv_ndrange(u_A))
            sparse_bufs = _sparse_args(u_A)
            kern        = _get_spmv_kernel(fmt, T)
            all_args    = (sparse_bufs..., nonzeros(u_x), nonzeros(u_y), off, n_outer,
                           input_fn, output_fn, T_alpha, zero(T))
            kernel_obj  = Base.invokelatest(kern, ka, 64)
            Base.invokelatest(kernel_obj, all_args...; ndrange=Int(n_outer))
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
            lv2 = fmt.levels[2][2]
            if lv2 isa CompressedLevel && is_unique(lv2)
                dispatched = JLUST._csr_spmv_specialized!(
                    sparse_bufs[1], sparse_bufs[2],
                    nonzeros(u_A), nonzeros(u_x), nonzeros(u_y), off, n_outer, T_beta)
            end
        end

        if !dispatched
            kern       = _get_spmv_kernel(fmt, T)
            all_args   = (sparse_bufs..., nonzeros(u_x), nonzeros(u_y), off, n_outer,
                          input_fn, output_fn, T_alpha, T_beta)
            kernel_obj = Base.invokelatest(kern, ka, 64)
            Base.invokelatest(kernel_obj, all_args...; ndrange=Int(n_outer))
        end
    end

    return u_y
end

# ─── EmitterSpMVHandle ────────────────────────────────────────────────────────
#
# prepare() compiles and caches the specialized kernel once, including the
# concrete types of input_fn/output_fn.  Subsequent sparse_mv! calls skip
# format inspection and invokelatest overhead on the kernel lookup.
#
# input_fn:  applied element-wise to x before each multiply: y = A * input_fn.(x)
# output_fn: applied to the accumulated row sum before writing:  y[i] = output_fn(acc)
#
# Pass identity (the default) for either to generate zero-overhead kernel paths.
# Example — fused relu-input SpMV followed by relu-output SpMV:
#
#   h1 = prepare(EmitterBackend(), SpMVOp, u_A1; output_fn=relu)
#   h2 = prepare(EmitterBackend(), SpMVOp, u_A2; input_fn=relu)
#   sparse_mv!(h1, u_A1, u_x, u_tmp)   # y_tmp = relu.(A1 * x)  (1 kernel)
#   sparse_mv!(h2, u_A2, u_tmp, u_y)   # y     = A2 * relu.(y_tmp) (1 kernel)
#
# Compared to the unfused form, this saves one broadcast kernel launch per chain link.

mutable struct EmitterSpMVHandle{T, IF, OF}
    kernel_obj  # compiled KA kernel (backend + block size already bound)
    n_outer::Int32
    off::Int32
    input_fn::IF
    output_fn::OF
end

export EmitterSpMVHandle

function JLUST.prepare(::EmitterBackend, ::Type{SpMVOp}, u_A::USTensor{T};
                        input_fn::IF=identity, output_fn::OF=identity) where {T, IF, OF}
    fmt     = format(u_A)
    ka      = KernelAbstractions.get_backend(nonzeros(u_A))
    off     = Int32(index_origin(u_A) isa OneBased ? 1 : 0)
    n_outer = Int32(_spmv_ndrange(u_A))

    kern = _get_spmv_kernel(fmt, T)
    # Bind backend + block size; GPU JIT fires on first actual launch.
    kobj = Base.invokelatest(kern, ka, 64)

    EmitterSpMVHandle{T, IF, OF}(kobj, n_outer, off, input_fn, output_fn)
end

function JLUST.sparse_mv!(h::EmitterSpMVHandle, u_A::USTensor, x::AbstractVector, y::AbstractVector; kw...)
    JLUST.sparse_mv!(h, u_A, ust(x), ust(y); kw...)
end

function JLUST.sparse_mv!(h::EmitterSpMVHandle, u_A::USTensor, u_x::USTensor, u_y::USTensor;
                           alpha=one(eltype(u_A)), beta=zero(eltype(u_A)))
    T_A         = eltype(u_A)
    sparse_bufs = _sparse_args(u_A)
    all_args    = (sparse_bufs..., nonzeros(u_x), nonzeros(u_y),
                   h.off, h.n_outer, h.input_fn, h.output_fn, T_A(alpha), T_A(beta))
    Base.invokelatest(h.kernel_obj, all_args...; ndrange=Int(h.n_outer))
    return u_y
end
