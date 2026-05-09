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
_level_has_nzval(::ShiftedDiagLevel) = false

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
                          vs::Int=1,
                          zero_beta::Bool=false,
                          const_x::Bool=false,
                          seg::Bool=false,
                          input_fn_sym::Symbol=:_input_fn,
                          output_fn_sym::Symbol=:_output_fn) where T
    leaf_unique = :(_acc += _nzval[_nnz_pos] * $input_fn_sym(_x[_x_idx]))
    # Atomic-mode leaf: when `seg=true` the outer wraps with an all-warp
    # segmented reduce + single per-segment-head atomic, so the leaf just
    # accumulates into the per-thread `_my_val`.  Otherwise the leaf does the
    # @atomic write directly (one atomic per NNZ — simple but contended).
    leaf_atomic = seg ?
        :(_my_val += _alpha * _nzval[_nnz_pos] * $input_fn_sym(_x[_x_idx])) :
        :(KernelAbstractions.@atomic _y[_y_idx] +=
              _alpha * _nzval[_nnz_pos] * $input_fn_sym(_x[_x_idx]))
    # When beta=0 we know y is write-only, so skip the read; otherwise the
    # accumulator is fused with `beta * y[idx]` for the SpMV identity
    # y ← α·A·x + β·y.  The compile-time `zero_beta` flag eliminates the y-read
    # and the runtime `_beta` multiply when applicable.
    write_unique(acc_expr) = zero_beta ?
        :(_y[_y_idx] = _alpha * $output_fn_sym($acc_expr)) :
        :(_y[_y_idx] = _alpha * $output_fn_sym($acc_expr) + _beta * _y[_y_idx])
    # Row body: scalar mode writes y unconditionally; vector mode warp-shuffle-
    # reduces _acc across the VS-thread group, then only `_vec_lane == 0`
    # writes the final y[row].
    row_body_unique = if vs == 1
        inner -> quote
            _acc = $(zero(T))
            $inner
            $(write_unique(:_acc))
        end
    else
        inner -> quote
            _acc = $(zero(T))
            $inner
            _acc = JLUST._warp_reduce_sum_down(_acc, _group_mask, Val($vs))
            if _vec_lane == Int32(0)
                $(write_unique(:_acc))
            end
        end
    end
    # Non-unique outer (COO-like): in seg mode the outer body owns init+reduce,
    # so the row body just emits the inner walk.  In atomic mode (no seg), the
    # leaf does the atomic write itself; row body is a no-op wrap.
    row_body_atomic = inner -> inner
    inner_body = emit_kernel_body(levels;
                                   row_body_unique, row_body_atomic,
                                   leaf_unique, leaf_atomic, vs=vs, seg=seg)
    # Optional read-only wrap of x.  When `const_x` is true the kernel body
    # rebinds `_x = Base.Experimental.Const(_x)` so that on backends with a
    # read-only data cache (CUDA's LDG), x[col] reads route through it.  Set
    # only by backends whose `_supports_ldg(ka) === true`.
    const_x ? quote
        _x = Base.Experimental.Const(_x)
        $inner_body
    end : inner_body
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

# ─── Kernel launch ───────────────────────────────────────────────────────────
#
# All emitter-backed kernels share the unified `_ust_emit_kern` (defined in
# _walker.jl); each op defines a singleton `KT` plus methods for
# `_kern_standard_nms(::KT)` and `_kern_emit_body(::KT, levels, T)`.
# `_launch_kern` calls `KI.kernel_function` to dispatch via Julia's method
# table — specialization caches per (KT, FMT, T, …); no global Dict, no
# `Base.invokelatest`.

@inline _launch_kern(ka, kfunc, args::Tuple, ndrange::Int) =
    _launch_kern(ka, kfunc, args, ndrange, JLUST._default_workgroup_size(ka))

@inline function _launch_kern(ka, kfunc, args::Tuple, ndrange::Int, ws::Int)
    ng    = cld(ndrange, ws)
    kf    = KI.argconvert(ka, kfunc)
    kargs = map(x -> KI.argconvert(ka, x), args)
    ktt   = Tuple{map(Core.Typeof, kargs)...}
    kobj  = KI.kernel_function(ka, kf, ktt)
    kobj(kargs...; numworkgroups=ng, workgroupsize=ws)
end

# SpMV kernel singleton.  Body emits via the unified `_ust_emit_kern`.  Type
# parameters select per-backend strategies that the walker bakes into the
# emitted body:
#
#   `LDG :: Bool` — wrap x in `Base.Experimental.Const` for read-only-cache
#                   loads.  Backends with `_supports_ldg(ka) === true` opt in.
#   `VS  :: Int`  — warp-vector size.  When VS > 1 the walker emits VS-strided
#                   inner loops + warp-shuffle reduce + lane-0-writes-y.
#                   Backends with `_supports_warp_vector(ka) === true` opt in
#                   for formats with variable-length inner iteration.

struct _SpMVKern{LDG, VS, ZB, SEG} end
_SpMVKern() = _SpMVKern{false, 1, false, false}()
_kern_standard_nms(::_SpMVKern) =
    (:_x, :_y, :_origin_off, :_n_outer, :_input_fn, :_output_fn, :_alpha, :_beta)
_kern_emit_body(::_SpMVKern{LDG, VS, ZB, SEG}, levels, ::Type{T}) where {LDG, VS, ZB, SEG, T} =
    _emit_spmv_body(levels, T; const_x=LDG, vs=VS, zero_beta=ZB, seg=SEG)

# `_supports_ldg(ka)` and `_supports_warp_vector(ka)` are declared in JLUST
# core (src/backends.jl) so each backend extension overrides for its concrete
# backend type without coupling extensions to each other.
const _supports_ldg          = JLUST._supports_ldg
const _supports_warp_vector  = JLUST._supports_warp_vector

# Pick warp-vector width VS in {1, 2, 4, 8, 16, 32}.  VS=1 disables vector
# mode (scalar walker).  Heuristic balances:
#   • per-thread inner-loop length: aim for ≥ avg_nnz/vs ≥ 1 to avoid wasted threads
#   • SM occupancy: enough thread blocks to fill the device
# The chosen VS becomes a kernel type parameter (`_SpMVKern{LDG, VS}`) so the
# walker emits stride-VS inner loops + warp-shuffle reduce at compile time.
function _spmv_vs(ka, fmt::TensorFormat, n_outer::Int, total_nnz::Int)
    _supports_warp_vector(ka) || return 1
    n_outer == 0 && return 1
    levels = fmt.levels
    length(levels) >= 2 || return 1
    lv1, lv2 = levels[1], levels[2]
    # Vector mode applies when the *inner* level has variable-length iteration
    # (CompressedLevel) and the outer is row-per-thread (Dense / Batch).  Other
    # combinations (Singleton inner, COO-like outer) get scalar mode.
    (lv1 isa Union{DenseLevel,BatchLevel} && lv2 isa CompressedLevel && is_unique(lv2)) || return 1

    avg_nnz = total_nnz / n_outer
    vs_nnz = avg_nnz < 2.0  ? 2  :
             avg_nnz < 4.0  ? 4  :
             avg_nnz < 8.0  ? 8  :
             avg_nnz < 16.0 ? 16 : 32
    # Occupancy floor: enough warp-rows to saturate the device.  We don't know
    # the SM count from KernelAbstractions, so use a conservative estimate of
    # 6 × ws threads per SM × ~150 SMs (roughly L40S / H100 ballpark).  When
    # n_outer is huge the floor reduces to vs_nnz.
    target_threads = 6 * 256 * 150
    vs_occ_raw = cld(target_threads, n_outer)
    vs_occ = vs_occ_raw <= 2  ? 2  :
             vs_occ_raw <= 4  ? 4  :
             vs_occ_raw <= 8  ? 8  :
             vs_occ_raw <= 16 ? 16 : 32
    min(max(vs_nnz, min(vs_occ, vs_nnz * 2)), 32)
end

# Pick segmented-warp-reduce mode for sorted-COO-style outers (non-unique
# Compressed) when the backend has warp shuffles.  In this mode the walker
# emits 1 thread per NNZ + warp-segmented sum + atomic-add only at segment
# heads — replaces the format-specific COO warp kernel that used to live in
# CUDAExt.  Format-agnostic: applies to any user format whose outer level is
# a non-unique CompressedLevel (custom row-list shapes get it for free).
@inline function _spmv_seg(ka, fmt::TensorFormat)
    _supports_warp_vector(ka) || return false
    levels = fmt.levels
    isempty(levels) && return false
    lv1 = levels[1]
    lv1 isa CompressedLevel && !is_unique(lv1)
end

# Lift the runtime VS / ZB / SEG to compile-time `Val`s so the walker @generated
# body specializes per (LDG, VS, ZB, SEG) combination — different stride literal,
# warp-reduce depth, y-write pattern, and atomic vs segmented-reduce leaf.
@inline function _launch_spmv_kern(ka, fmt::TensorFormat, all_args, ndrange::Int,
                                     total_nnz::Int, beta)
    ldg = _supports_ldg(ka)
    vs  = _spmv_vs(ka, fmt, ndrange, total_nnz)
    seg = _spmv_seg(ka, fmt)
    _launch_spmv_kern_dispatch(ka, all_args, ndrange, Val(ldg), vs, Val(iszero(beta)), Val(seg))
end

@inline function _launch_spmv_kern_dispatch(ka, all_args, ndrange::Int,
                                              ::Val{LDG}, vs::Int, ::Val{ZB},
                                              ::Val{SEG}) where {LDG, ZB, SEG}
    # Vector mode requires VS threads per row, so the launch ndrange grows.
    n_threads = vs == 1 ? ndrange : ndrange * vs
    if     vs == 1  ; _launch_kern(ka, _ust_emit_kern, (_SpMVKern{LDG, 1 , ZB, SEG}(), all_args...), n_threads)
    elseif vs == 2  ; _launch_kern(ka, _ust_emit_kern, (_SpMVKern{LDG, 2 , ZB, SEG}(), all_args...), n_threads)
    elseif vs == 4  ; _launch_kern(ka, _ust_emit_kern, (_SpMVKern{LDG, 4 , ZB, SEG}(), all_args...), n_threads)
    elseif vs == 8  ; _launch_kern(ka, _ust_emit_kern, (_SpMVKern{LDG, 8 , ZB, SEG}(), all_args...), n_threads)
    elseif vs == 16 ; _launch_kern(ka, _ust_emit_kern, (_SpMVKern{LDG, 16, ZB, SEG}(), all_args...), n_threads)
    elseif vs == 32 ; _launch_kern(ka, _ust_emit_kern, (_SpMVKern{LDG, 32, ZB, SEG}(), all_args...), n_threads)
    else error("_launch_spmv_kern: vs=$vs not in {1,2,4,8,16,32}")
    end
end

# ─── sparse_mv! ───────────────────────────────────────────────────────────────

function JLUST.execute(::EmitterBackend, ::Op{:SpMV, F},
                       u_A::USTensor, u_x::USTensor, u_y::USTensor;
                       alpha=one(eltype(u_A)), beta=zero(eltype(u_A)),
                       input_fn::IF=identity, output_fn::OF=identity) where {F, IF, OF}
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
        # COO-like: pre-scale y by beta; the atomic / segmented-reduce paths
        # accumulate additively into this baseline.
        iszero(T_beta) ? fill!(nonzeros(u_y), zero(T)) : (nonzeros(u_y) .*= T_beta)
        if _supports_warp_vector(ka)
            # Walker emits the segmented warp-reduce kernel (1 thread/NNZ +
            # log2(32) shfl_down + segment-head atomic).  Replaces the old
            # format-specific `_coo_spmv_specialized!` CUDA hook; format-
            # agnostic — applies to any non-unique-Compressed outer.
            n_outer     = Int32(_spmv_ndrange(u_A))
            sparse_bufs = _sparse_args(u_A)
            all_args    = (typeof(fmt), T,
                           sparse_bufs..., nonzeros(u_x), nonzeros(u_y), off, n_outer,
                           input_fn, output_fn, T_alpha, zero(T))
            _launch_spmv_kern(ka, fmt, all_args, Int(n_outer), length(nonzeros(u_A)), T_beta)
        elseif use_identity && isone(T_alpha)
            # Non-warp backend (CPU, POCL): the chunked kernel batches CHUNK
            # consecutive NNZ per thread + accumulates locally before each
            # row-boundary atomic — fewer atomics than the generic per-NNZ
            # emit, important when warp shuffles aren't available.  Limited to
            # identity transforms and α=1 because the chunked kernel hardcodes
            # those (the walker fallback below covers the general case).
            row_crd = coordinates(u_A, 1)
            col_crd = coordinates(u_A, 2)
            nzv     = nonzeros(u_A)
            nx      = nonzeros(u_x)
            ny      = nonzeros(u_y)
            n_nnz   = Int32(length(row_crd))
            n_chunks = Int32(cld(Int(n_nnz), _COO_CHUNK))
            _coo_spmv_chunked!(ka, 64)(
                row_crd, col_crd, nzv, nx, ny, off, n_nnz;
                ndrange=Int(n_chunks))
        else
            # Generic atomic-per-NNZ via walker (handles α ≠ 1 / non-identity
            # transforms on non-warp backends).
            n_outer     = Int32(_spmv_ndrange(u_A))
            sparse_bufs = _sparse_args(u_A)
            all_args    = (typeof(fmt), T,
                           sparse_bufs..., nonzeros(u_x), nonzeros(u_y), off, n_outer,
                           input_fn, output_fn, T_alpha, zero(T))
            _launch_spmv_kern(ka, fmt, all_args, Int(n_outer), length(nonzeros(u_A)), T_beta)
        end
    else
        n_outer     = Int32(_spmv_ndrange(u_A))
        sparse_bufs = _sparse_args(u_A)
        all_args    = (typeof(fmt), T,
                       sparse_bufs..., nonzeros(u_x), nonzeros(u_y), off, n_outer,
                       input_fn, output_fn, T_alpha, T_beta)
        # Walker dispatch.  `_launch_spmv_kern` consults the backend traits
        # (`_supports_ldg`, `_supports_warp_vector`) and the format's level
        # structure to pick LDG-wrap, warp-vector size, ZERO_BETA spec, and
        # segmented-reduce mode.  For CUDA + (Dense outer, Compressed-unique
        # inner): VS=2..32 stride + warp-shuffle reduce.  For CUDA + non-unique
        # Compressed outer: 1 thread/NNZ + warp-segmented-sum + per-segment
        # atomic.  Both replace the format-specific CUDA hooks that used to
        # live in CUDAExt; both benefit any future format with the same shape.
        _launch_spmv_kern(ka, fmt, all_args, Int(n_outer), length(nonzeros(u_A)), T_beta)
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

struct EmitterSpMVHandle{T, IF, OF, FMT} <: JLUST.AbstractKernelHandle
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

function JLUST.execute(h::EmitterSpMVHandle{T, IF, OF, FMT},
                        u_A::USTensor, u_x::USTensor, u_y::USTensor;
                        alpha=one(T), beta=zero(T)) where {T, IF, OF, FMT}
    ka          = KernelAbstractions.get_backend(nonzeros(u_A))
    sparse_bufs = _sparse_args(u_A)
    T_beta      = T(beta)
    all_args    = (FMT, T, sparse_bufs..., nonzeros(u_x), nonzeros(u_y),
                   h.off, h.n_outer, h.input_fn, h.output_fn, T(alpha), T_beta)
    _launch_spmv_kern(ka, format(u_A), all_args, Int(h.n_outer), length(nonzeros(u_A)), T_beta)
    return u_y
end
