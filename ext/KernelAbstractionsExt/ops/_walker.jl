# ─── Unified level walker for kernel codegen ──────────────────────────────────
#
# All EmitterBackend ops (SpMV, SpMM, SDDMM, sparse_to_dense) share the same
# level-traversal structure: an outermost level mapped to one thread per output
# row (or NNZ for COO-like layouts), then a recursive walk over inner levels
# that binds `_x_idx` (the inner coordinate) and `_nnz_pos` (the position into
# `_nzval`) at each leaf.  Differences between ops are confined to:
#
#   • leaf body            — what to compute at the deepest level
#   • row body             — how to wrap the inner-loop recursion (e.g. allocate
#                            an accumulator, loop over n_col, etc.)
#
# The walker emits code using `KernelIntrinsics` (KI) primitives directly —
# `KI.get_global_id().x` for the thread index, `Atomix.@atomic` for COO-style
# atomic writes.  The kernel functions are regular `@generated` Julia functions
# (specialized per format type), called via `KI.@kernel` at the call site.  No
# lazy `@eval`, no global cache, no `Base.invokelatest`.
#
# The walker takes four Exprs:
#   row_body_unique : invoked at outer Dense / Compressed-unique levels;
#                     receives the inner-walk Expr and returns the row-level
#                     wrapper (typically: init accumulator → recurse → write).
#   row_body_atomic : same shape, used at outer non-unique CompressedLevel;
#                     usually just `il -> il` since the leaf does the atomic
#                     write itself.
#   leaf_unique     : leaf body for the unique outer (uses `_acc`-style accumulator).
#   leaf_atomic     : leaf body for the non-unique outer (atomic write at leaf).
#
# Names bound by the walker into the kernel scope:
#   _tid, _y_idx                          (outer level)
#   _x_idx, _nnz_pos                      (each inner level, depth-dependent)
#   _origin_off, _n_outer, plus format buffers (`_pos<i>`, `_crd<i>`, `_nzval`)
#                                          (kernel arguments — see _sparse_arg_names)
#
# Custom AbstractLevelFormat: the walker delegates to JLUST.emit_spmv_lv at
# inner positions when the level lacks a dedicated method.  Outermost custom
# levels are an error and must be paired with a DenseLevel outer.
#
# ── Direction (gather vs scatter) ─────────────────────────────────────────────
#
# `direction === :gather` (default): outer level walks the y-output dim, inner
# levels walk the x-input dim.  Walker binds `_y_idx = _row_id` (outer) and
# `_x_idx = crd[k]` (inner crd).  Used by CSR / DCSR / COO — the SpMV gather
# pattern `y[i] = Σ A[i,j] x[j]`.
#
# `direction === :scatter`: outer level walks the x-input dim, inner levels
# walk the y-output dim (the role flip CSC requires).  Walker swaps the
# bindings: `_x_idx = _row_id` (outer = input col), `_y_idx = crd[k]` (inner
# crd = output row).  Op-side leaf templates can still write `_y[_y_idx] += A
# * _x[_x_idx]` and have it mean the right thing — semantics live in the
# walker, not in every per-strategy emit body.
#
# Direction is a compile-time `Symbol` argument; the conditional bind is
# selected at codegen time, so the emitted Expr for direction=:gather is
# identical to pre-direction code (zero runtime cost in the gather path).
#
# Custom levels (AbstractLevelFormat) work in either direction: `level_step`
# is direction-agnostic ("the coord this level walks") so the walker binds
# the returned coord to `_y_idx` or `_x_idx` based on the format's level→dim
# mapping.  A user can build a CSC-like custom format `(j:dense, i:MyLevel)`
# and the walker routes it through scatter automatically.

const KI = KernelAbstractions.KernelIntrinsics

# ── Inner-level walker ────────────────────────────────────────────────────────

# Pick which symbol the inner crd binds to based on direction.
@inline _inner_idx_sym(direction::Symbol) =
    direction === :scatter ? :_y_idx : :_x_idx

function _walk_inner(levels, lvl, p_var, pc, cc, leaf, vs::Int=1; direction::Symbol=:gather)
    lvl > length(levels) && return leaf
    _walk_inner_lv(levels[lvl], levels, lvl, p_var, pc, cc, leaf, vs; direction)
end

# Inner Dense / Batch — dense loop.  When the level type carries a static size
# (e.g. blocked inner dims in BSR: DenseLevel{2}), the loop bound is a literal
# integer the compiler can fully unroll.  When the size is dynamic
# (DenseLevel{nothing}, e.g. an outer-but-not-outermost Dense), a runtime
# `_sz<lvl>` kernel argument is referenced — the kernel signature must bind it.
function _walk_inner_lv(lv::Union{DenseLevel{Sz},BatchLevel{Sz}},
                        levels, lvl, p_var::Symbol, pc, cc, leaf,
                        vs::Int=1; direction::Symbol=:gather) where {Sz}
    lvar  = Symbol(:_i, lvl)
    inner = _walk_inner(levels, lvl + 1, lvar, pc, cc, leaf, vs; direction)
    if Sz === nothing
        sz = Symbol(:_sz, lvl)
        quote
            for $lvar in 1:$sz
                $inner
            end
        end
    else
        quote
            for $lvar in 1:$Sz
                $inner
            end
        end
    end
end

# Inner CompressedLevel — variable-length inner loop, optionally strided by VS
# threads working on the same parent (warp-vector mode).  In scalar mode (vs=1)
# `_vec_lane` is bound to 0 by the outer walker, so the strided form reduces
# to the sequential form `for k in lo+1:hi` — same code as before.
#
# Arithmetic stays in the native types of pos/crd/origin_off (typically Int32
# on GPU buffers): `pos[r] - origin_off` does *not* upcast to Int.  Forcing
# Int upcasts compiled to 64-bit GPU arithmetic and was a 20%+ perf hit in the
# inner loop versus equivalent hand-rolled Int32 kernels.
function _walk_inner_lv(::CompressedLevel, levels, lvl, p_var::Symbol, pc, cc, leaf,
                        vs::Int=1; direction::Symbol=:gather)
    pi = pc[] += 1; ci = cc[] += 1
    ps = Symbol(:_pos, pi); cs = Symbol(:_crd, ci)
    lvar = Symbol(:_i, lvl)
    inner = _walk_inner(levels, lvl + 1, lvar, pc, cc, leaf, vs; direction)
    idx_sym = _inner_idx_sym(direction)
    if vs == 1
        quote
            _lo = $ps[$p_var]      - _origin_off
            _hi = $ps[$p_var + 1]  - _origin_off
            for $lvar in (_lo + Int32(1)):_hi
                $idx_sym = $cs[$lvar] - _origin_off + Int32(1)
                _nnz_pos = $lvar
                $inner
            end
        end
    else
        quote
            _lo = $ps[$p_var]      - _origin_off
            _hi = $ps[$p_var + 1]  - _origin_off
            $lvar = _lo + _vec_lane + Int32(1)
            while $lvar <= _hi
                $idx_sym = $cs[$lvar] - _origin_off + Int32(1)
                _nnz_pos = $lvar
                $inner
                $lvar += Int32($vs)
            end
        end
    end
end

function _walk_inner_lv(::SingletonLevel, levels, lvl, p_var::Symbol, pc, cc, leaf,
                        vs::Int=1; direction::Symbol=:gather)
    ci = cc[] += 1
    cs = Symbol(:_crd, ci)
    inner = _walk_inner(levels, lvl + 1, p_var, pc, cc, leaf, vs; direction)
    idx_sym = _inner_idx_sym(direction)
    quote
        $idx_sym = $cs[$p_var] - _origin_off + Int32(1)
        _nnz_pos = $p_var
        $inner
    end
end

function _walk_inner_lv(::DeltaLevel, levels, lvl, p_var::Symbol, pc, cc, leaf,
                        vs::Int=1; direction::Symbol=:gather)
    pi = pc[] += 1; ci = cc[] += 1
    ps = Symbol(:_pos, pi); cs = Symbol(:_crd, ci)
    lvar  = Symbol(:_i, lvl)
    corig = Symbol(:_corig, lvl)
    inner = _walk_inner(levels, lvl + 1, lvar, pc, cc, leaf, vs; direction)
    idx_sym = _inner_idx_sym(direction)
    quote
        _lo = Int($ps[$p_var])     - Int(_origin_off)
        _hi = Int($ps[$p_var + 1]) - Int(_origin_off)
        $corig = 0
        for $lvar in (_lo + 1):_hi
            $corig  += Int($cs[$lvar])
            $idx_sym = $corig - Int(_origin_off) + 1
            _nnz_pos = $lvar
            $inner
            $corig += 1
        end
    end
end

function _walk_inner_lv(::RangeLevel, levels, lvl, p_var, pc, cc, leaf,
                        vs::Int=1; direction::Symbol=:gather)
    error("EmitterBackend: RangeLevel (DIA-style) inner kernels not supported. " *
          "Use convert_format to CSR or DCSR first.")
end

# Custom AbstractLevelFormat: bind the coord returned by `level_step` to the
# direction-appropriate symbol (`_x_idx` for gather, `_y_idx` for scatter):
#   - if the level carries nzval (DiagonalLevel: val = nzval[i]) → standard
#     leaf path, leaf reads `_nzval[_nnz_pos]` as usual
#   - if the level has no nzval (ShiftedDiagLevel, RampLevel, etc.) → bind
#     `_val` from level_step and emit the SpMV leaf inline; the standard leaf
#     can't be used because `_nzval` isn't a kernel argument.  Restricted to
#     leaf-position custom levels (no further inner recursion possible since
#     we're emitting a complete leaf).
#
# `level_step` is direction-agnostic (returns "the coord this level walks"),
# so a user CSC-like custom format `(j:dense, i:MyLevel)` works under scatter
# without any change in the user's level code — the walker binds `_y_idx` from
# the returned coord and the inlined leaf becomes an atomic-add.
function _walk_inner_lv(lv::AbstractLevelFormat, levels, lvl, p_var::Symbol, pc, cc, leaf,
                        vs::Int=1; direction::Symbol=:gather)
    idx_sym = _inner_idx_sym(direction)
    if JLUST.level_has_nzval(lv)
        inner = _walk_inner(levels, lvl + 1, p_var, pc, cc, leaf, vs; direction)
        quote
            _p1 = Int($p_var) - Int(_origin_off) + 1
            ($idx_sym, _) = JLUST.level_step($lv, _p1, :_nzval)
            _nnz_pos = $p_var
            $inner
        end
    else
        lvl == length(levels) || error(
            "EmitterBackend: nzval-less custom level $(typeof(lv)) at level $lvl ",
            "must be the innermost level (got $(length(levels)) total levels)")
        # SpMV leaf inlined: gather accumulates into `_acc`; scatter atomic-adds
        # into y at the inner-supplied coord (bound to `_y_idx` by direction).
        # Caller pre-scales y by beta in scatter mode.
        if direction === :scatter
            quote
                _p1 = Int($p_var) - Int(_origin_off) + 1
                ($idx_sym, _val) = JLUST.level_step($lv, _p1, nothing)
                _nnz_pos = $p_var
                KernelAbstractions.@atomic _y[_y_idx] += _alpha * _val * _input_fn(_x[_x_idx])
            end
        else
            quote
                _p1 = Int($p_var) - Int(_origin_off) + 1
                ($idx_sym, _val) = JLUST.level_step($lv, _p1, nothing)
                _nnz_pos = $p_var
                _acc += _val * _input_fn(_x[_x_idx])
            end
        end
    end
end

# ── Outer-level walker ────────────────────────────────────────────────────────
#
# Walks the outermost level, binds `_y_idx`, and stitches together the row body
# (which embeds the inner walk via the row_body callback).
#
# Returns: a single Expr that is the complete @kernel body.

function emit_kernel_body(levels::Tuple;
                          row_body_unique, row_body_atomic,
                          leaf_unique,     leaf_atomic,
                          vs::Int=1, seg::Bool=false,
                          direction::Symbol=:gather)
    pc = Ref(0); cc = Ref(0)
    isempty(levels) && error("emit_kernel_body: no levels")
    _emit_outer(levels[1], levels, pc, cc,
                row_body_unique, row_body_atomic, leaf_unique, leaf_atomic,
                vs, seg; direction)
end

emit_kernel_body(fmt::TensorFormat; kw...) = emit_kernel_body(fmt.levels; kw...)

# Outer Dense / Batch.  Thread → row directly when `vs == 1`; otherwise VS
# threads cooperate per row.  In vector mode (`vs > 1`) the outer also binds
# `_vec_lane` and `_group_mask`, which the inner CompressedLevel walker reads
# to stride its loop and which the row body uses for the warp-reduce.
#
# When `vs == 1` the bindings still exist (`_vec_lane = 0`, `_group_mask = 0`)
# so the inner walker code is identical regardless of mode — Julia constant-
# folds them away in the scalar path.
function _emit_outer(::Union{DenseLevel,BatchLevel}, levels, pc, cc,
                     row_body_unique, _row_body_atomic, leaf_unique, _leaf_atomic,
                     vs::Int=1, _seg::Bool=false; direction::Symbol=:gather)
    inner = _walk_inner(levels, 2, :_row_id, pc, cc, leaf_unique, vs; direction)
    body  = row_body_unique(inner)
    # Outer Dense walks the y-dim (row) under gather, the x-dim (col) under
    # scatter — bind to `_y_idx` or `_x_idx` accordingly.  Op-side leaf can
    # always read `_y[_y_idx]` and `_x[_x_idx]` and have it mean the right
    # thing because the walker resolves direction at codegen time.
    out_idx_sym = direction === :scatter ? :_x_idx : :_y_idx
    if vs == 1
        quote
            _tid       = KI.get_global_id().x
            _row_id    = _tid
            _vec_lane  = Int32(0)
            _group_mask = UInt32(0)
            if _row_id <= _n_outer
                $out_idx_sym = _row_id
                $body
            end
        end
    else
        quote
            _tid       = KI.get_global_id().x
            _row_id    = (_tid - Int32(1)) ÷ Int32($vs) + Int32(1)
            _vec_lane  = (_tid - Int32(1)) % Int32($vs)
            # Warp-shuffle group mask: VS consecutive 1-bits, shifted by group.
            _lane_in_warp  = (KI.get_local_id().x - Int32(1)) % Int32(32)
            _group_in_warp = _lane_in_warp ÷ Int32($vs)
            _group_bits    = (UInt32(1) << UInt32($vs)) - UInt32(1)
            _group_mask    = _group_bits << (UInt32(_group_in_warp) * UInt32($vs))
            if _row_id <= _n_outer
                $out_idx_sym = _row_id
                $body
            end
        end
    end
end

# Outer CompressedLevel — fiber-parallel (unique) or NNZ-parallel (non-unique).
# Outer Compressed has no parent-fiber pos buffer (no pos slot), so pc stays
# unincremented — matches _level_arg_names which emits crd only at the outer level.
# The vector mode (vs>1) doesn't apply to outer-Compressed structures (DCSR
# etc.) — VS makes sense only when the *inner* level has variable-length
# iteration that VS threads can stride across.  We accept `vs` for signature
# uniformity but ignore it here.
function _emit_outer(lv::CompressedLevel, levels, pc, cc,
                     row_body_unique, row_body_atomic, leaf_unique, leaf_atomic,
                     vs::Int=1, seg::Bool=false; direction::Symbol=:gather)
    ci = cc[] += 1
    cs = Symbol(:_crd, ci)
    out_idx_sym = direction === :scatter ? :_x_idx : :_y_idx
    if is_unique(lv)
        inner = _walk_inner(levels, 2, :_row_id, pc, cc, leaf_unique, 1; direction)
        body  = row_body_unique(inner)
        quote
            _tid       = KI.get_global_id().x
            _row_id    = _tid
            _vec_lane  = Int32(0)
            _group_mask = UInt32(0)
            if _row_id <= _n_outer
                $out_idx_sym = Int($cs[_row_id]) - Int(_origin_off) + 1
                $body
            end
        end
    elseif seg
        # Segmented warp-reduce mode (sorted COO + warp-shuffle backend).
        # Every lane in the warp must reach the warp-reduce primitive — out-of-
        # range threads contribute (val=0, row=-1) so they're skipped at the
        # head check.  The walker writes the partial inner result into _my_val
        # via the `leaf_atomic` body (rewritten by `_emit_spmv_body` to
        # `_my_val += α·nzval·input_fn(x)` when seg=true).
        # Seg path is gather-only (folds row-sums); scatter via non-unique
        # Compressed outer would need a different reduction.
        direction === :gather ||
            error("EmitterBackend: segmented-reduce path is gather-only.  " *
                  "Scatter via non-unique Compressed outer is not yet supported.")
        inner = _walk_inner(levels, 2, :_row_id, pc, cc, leaf_atomic, 1; direction)
        body  = row_body_atomic(inner)   # `inner -> inner` for seg
        quote
            _tid        = KI.get_global_id().x
            _row_id     = _tid
            _vec_lane   = Int32(0)
            _group_mask = UInt32(0)
            _MASK       = UInt32(0xffffffff)
            _my_row     = Int32(-1)
            _my_val     = zero(eltype(_y))
            if _row_id <= _n_outer
                _y_idx  = Int($cs[_row_id]) - Int(_origin_off) + 1
                _my_row = Int32(_y_idx) - Int32(1)
                $body
            end
            (_my_val, _is_head) = JLUST._warp_seg_reduce_sum_down(_my_val, _my_row, _MASK)
            if _is_head & (_my_row >= Int32(0))
                KernelAbstractions.@atomic _y[_my_row + Int32(1)] += _my_val
            end
        end
    else
        # Non-unique fallback: thread = NNZ; leaf is the atomic-add y-write.
        inner = _walk_inner(levels, 2, :_row_id, pc, cc, leaf_atomic, 1; direction)
        body  = row_body_atomic(inner)
        quote
            _tid       = KI.get_global_id().x
            _row_id    = _tid
            _vec_lane  = Int32(0)
            _group_mask = UInt32(0)
            if _row_id <= _n_outer
                $out_idx_sym = Int($cs[_row_id]) - Int(_origin_off) + 1
                $body
            end
        end
    end
end

function _emit_outer(lv::AbstractLevelFormat, _levels, _pc, _cc, args...)
    error("EmitterBackend: $(typeof(lv)) cannot be the outermost level; pair with a DenseLevel.")
end

function _emit_outer(::Union{SingletonLevel,RangeLevel,DeltaLevel}, _levels, _pc, _cc, args...)
    error("EmitterBackend: outermost SingletonLevel / RangeLevel / DeltaLevel is invalid; " *
          "pair with a DenseLevel or CompressedLevel.")
end

# ─── Unified @generated kernel ────────────────────────────────────────────────
#
# Every walker-driven kernel (SpMV, SpMM column-first / NNZ-first / tiled, SDDMM)
# shares the same shape:
#   1. extract sparse-arg names from the format's level types
#   2. append op-specific standard arg names (`_x`, `_y`, `_alpha`, …)
#   3. bind args[i] to those names inside an @inbounds block
#   4. splice in the op-specific body Expr
#
# The two bits that vary per op are encoded as method dispatch on a kernel
# singleton type `KT`:
#
#   `_kern_standard_nms(::KT)            → Tuple of Symbols`
#   `_kern_emit_body(::KT, levels, ::Type{T}) → Expr`
#
# Adding a new emitter-driven op = define a singleton + the two methods + call
# `_launch_kern(ka, _ust_emit_kern, (KT(), FMT, T, args...), ndrange)`.

function _kern_standard_nms end
function _kern_emit_body end

@generated function _ust_emit_kern(::KT, ::Type{FMT}, ::Type{T},
                                    args::Vararg{Any, M}) where {KT, FMT<:TensorFormat, T, M}
    kt          = KT()
    LT          = FMT.parameters[1]
    levels      = ntuple(i -> LT.parameters[i](), Val(length(LT.parameters)))
    sparse_nms  = _sparse_arg_names_for_levels(levels)
    standard_nm = _kern_standard_nms(kt)
    all_nms     = (sparse_nms..., standard_nm...)
    bindings    = [Expr(:(=), nm, :(args[$i])) for (i, nm) in enumerate(all_nms)]
    body        = _kern_emit_body(kt, levels, T)
    quote
        @inbounds begin
            $(bindings...)
            $body
        end
        return nothing
    end
end
