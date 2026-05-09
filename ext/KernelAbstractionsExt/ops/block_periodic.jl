# ─── Block-Toeplitz periodic + BSM-with-patches SpMV ────────────────────────
#
# A block-banded matrix with one diagonal block (USTensor or BSM) shared
# across T periods + one off-diagonal block pair shared across the T-1
# transitions can be SpMV'd without ever materialising the T-fold replica.
# Each thread owns one output row tid; from tid we recover (period_t,
# kind, in-period-row) and walk the appropriate block's level structure
# with a per-period column offset.
#
# Architecture: the kernel is fully walker-emitted from each block's
# `TensorFormat`.  A new singleton `_BBMPeriodicSpMVKern` + `@generated`
# kernel `_bbm_periodic_emit_kern` dispatch on the tuple
# `(DiagFmt, NegFmt, PosFmt)` of block formats; the body splices three
# `_emit_block_inner_walk` calls — one per block — into one fused row body.
# Each sub-walk is wrapped in a `let`-binding that aliases the walker's
# default `_pos1` / `_crd1` / `_nzval` names to per-block prefixed buffers
# (e.g. `_diag_pos1`) so the three block formats coexist without
# colliding, and sets `_period_col_off` to the right per-period column
# shift (`col_off_t` for diag/neg, `col_off_next` for pos).
#
# Generality:
#   - any block format the standard inner walker handles (CSR, DCSR-shape,
#     ShiftedDiag, custom user `level_step` formats) drops in unchanged
#   - selector off-diag = pass `ShiftedDiag` blocks for (neg, pos) → the
#     walker's nzval-less custom-level path emits `_acc += val * x[col]`
#     with no indirect loads; replaces the dedicated selector kernel
#   - selector blocks inside the BSM diag continue to ride the existing
#     `SelectorPatch` mechanism — `_compile_bsm` extracts them from the
#     diag's CSR; the kernel walks the leaner CSR plus the patches tuple
#
# Backend-agnostic: every backend KA targets (CUDA, ROCm, CPU, POCL,
# oneAPI) runs the same code.  CUDA-specific tricks (LDG read-only
# cache via `Base.Experimental.Const`) gate on `_supports_ldg(ka)`.

# Helper: apply the tuple of selector patches to row r, returning the
# accumulated patch contribution.  Tuple length is type-known so the loop
# unrolls; each iteration specialises on the patch's static type so
# `patch.val::T` reads as a literal.
@inline function _apply_patches(patches::Tuple, r::Int32,
                                  col_offset_extra::Int32, x, ::Type{T}) where T
    acc = zero(T)
    for patch in patches
        if r >= patch.row_start && r <= patch.row_end
            col  = (r - patch.row_start + Int32(1)) + patch.col_offset + col_offset_extra
            acc += patch.val * x[col]
        end
    end
    acc
end

# ── BSM SpMV: assembled CSR + selector patches ────────────────────────────────
#
# The classic BlockSparseMatrix SpMV routes through the assembled CSR.  When
# some blocks are constant-scaled identities (ShiftedDiag-formatted), they're
# extracted as patches at compile time — the kernel walks the leaner CSR plus
# the patches in one fused pass.

@kernel inbounds=true function _bsm_with_patches_spmv_kernel!(
        asm_pos, asm_crd, asm_nzval,
        _y, x_raw,
        n_total_rows::Int32, patches::Tuple,
        ::Val{ZERO_BETA}, beta,
        ::Val{LDG}) where {ZERO_BETA, LDG}
    T   = eltype(_y)
    tid = @index(Global, Linear)
    if tid <= n_total_rows
        x = LDG ? Base.Experimental.Const(x_raw) : x_raw
        r = Int32(tid)
        acc = zero(T)
        lo = asm_pos[r]
        hi = asm_pos[r + Int32(1)] - Int32(1)
        for k in lo:hi
            acc += asm_nzval[k] * x[asm_crd[k]]
        end
        acc += _apply_patches(patches, r, Int32(0), x, T)
        _y[tid] = ZERO_BETA ? acc : acc + beta * _y[tid]
    end
end

function JLUST._bsm_with_patches_spmv_launch!(ka,
        asm_pos, asm_crd, asm_nzval,
        y, x,
        n_total_rows::Int32, patches::Tuple,
        zero_beta::Bool, beta)
    ws = JLUST._default_workgroup_size(ka)
    _bsm_with_patches_spmv_kernel!(ka, ws)(
        asm_pos, asm_crd, asm_nzval,
        y, x,
        n_total_rows, patches,
        Val(zero_beta), beta,
        Val(JLUST._supports_ldg(ka));
        ndrange = Int(n_total_rows))
    return y
end

# ── BBM-periodic walker-emit kernel ───────────────────────────────────────────
#
# Singleton type carries the launch-time strategy bits as type params:
#   LDG :: Bool — wrap x in `Base.Experimental.Const` for read-only cache
#   ZB  :: Bool — caller knows beta == 0; skip the y read on the write
#
# `_kern_emit_body` consumes the *tuple* of (DiagFmt, NegFmt, PosFmt) types
# and emits one fused kernel body via `_emit_block_inner_walk` per block.

struct _BBMPeriodicSpMVKern{LDG, ZB} end
_BBMPeriodicSpMVKern() = _BBMPeriodicSpMVKern{false, false}()

# Emit the fused kernel body.  The kernel sees:
#   - per-block buffer args, prefixed `_diag_…` / `_neg_…` / `_pos_…`,
#     bound by the @generated dispatch wrapper
#   - standard args (`_x`, `_y`, `_n_diag`, `_n_off`, `_n_cols`,
#     `_n_total_rows`, `_input_fn`, `_alpha`, `_beta`, `_origin_off`,
#     `_patches`)
function _emit_bbm_periodic_body(diag_levels, neg_levels, pos_levels, ::Type{T},
                                   ::Val{LDG}, ::Val{ZB}) where {T, LDG, ZB}
    diag_overrides = _block_buf_overrides(diag_levels, "_diag_")
    neg_overrides  = _block_buf_overrides(neg_levels,  "_neg_")
    pos_overrides  = _block_buf_overrides(pos_levels,  "_pos_")
    diag_walk = _emit_block_inner_walk(diag_levels, T;
                                         row_id_var = :_row_id,
                                         col_off_expr = :_col_off_t,
                                         buf_overrides = diag_overrides)
    neg_walk  = _emit_block_inner_walk(neg_levels, T;
                                         row_id_var = :_row_id,
                                         col_off_expr = :_col_off_t,
                                         buf_overrides = neg_overrides)
    pos_walk  = _emit_block_inner_walk(pos_levels, T;
                                         row_id_var = :_row_id,
                                         col_off_expr = :_col_off_next,
                                         buf_overrides = pos_overrides)
    write_y = ZB ? :(_y[_tid] = _alpha * _acc) :
                   :(_y[_tid] = _alpha * _acc + _beta * _y[_tid])
    quote
        # Non-Periodic outers don't bind these — but the inner walks expect
        # them in scope (warp-vector and segmented-reduce paths read them).
        # For the BBM-periodic kernel we always run scalar (vs=1, no seg).
        _vec_lane   = Int32(0)
        _group_mask = UInt32(0)

        $(LDG ? :(_x = Base.Experimental.Const(_x)) : nothing)

        _tid = KI.get_global_id().x
        if _tid <= _n_total_rows
            _period_size  = _n_diag + _n_off
            _period_m1    = (Int32(_tid) - Int32(1)) ÷ _period_size
            _rem          = (Int32(_tid) - Int32(1)) - _period_m1 * _period_size
            _col_off_t    = _period_m1 * _n_cols
            _col_off_next = (_period_m1 + Int32(1)) * _n_cols
            _y_idx        = _tid
            _acc          = zero($T)
            if _rem < _n_diag
                _row_id = _rem + Int32(1)
                $diag_walk
                # Selector patches (BSM diag blocks extracted by _compile_bsm)
                # contribute one inline `val * x[col + col_off_t]` per matching
                # row.  Patches tuple is type-known → loop unrolls.
                _acc += _apply_patches(_patches, _row_id, _col_off_t, _x, $T)
            else
                _row_id = _rem - _n_diag + Int32(1)
                $neg_walk
                $pos_walk
            end
            $write_y
        end
    end
end

# @generated dispatch: extracts the three block formats from the tuple type
# parameter, reconstructs each format's level instances, computes per-block
# arg names with prefixes + standard names, binds args by name, and splices
# in the fused body.  Mirrors `_ust_emit_kern` but for the 3-block periodic
# pattern.
@generated function _bbm_periodic_emit_kern(::KT, ::Type{Fmts}, ::Type{T},
                                              args::Vararg{Any, M}) where {KT, Fmts<:Tuple, T, M}
    diag_fmt_t, neg_fmt_t, pos_fmt_t = Fmts.parameters
    diag_levels = ntuple(i -> diag_fmt_t.parameters[1].parameters[i](),
                         Val(length(diag_fmt_t.parameters[1].parameters)))
    neg_levels  = ntuple(i -> neg_fmt_t.parameters[1].parameters[i](),
                         Val(length(neg_fmt_t.parameters[1].parameters)))
    pos_levels  = ntuple(i -> pos_fmt_t.parameters[1].parameters[i](),
                         Val(length(pos_fmt_t.parameters[1].parameters)))
    diag_nms = [Symbol("_diag_", String(n)) for n in _sparse_arg_names_for_levels(diag_levels)]
    neg_nms  = [Symbol("_neg_",  String(n)) for n in _sparse_arg_names_for_levels(neg_levels)]
    pos_nms  = [Symbol("_pos_",  String(n)) for n in _sparse_arg_names_for_levels(pos_levels)]
    standard_nms = (:_x, :_y, :_n_diag, :_n_off, :_n_cols, :_n_total_rows,
                    :_input_fn, :_alpha, :_beta, :_origin_off, :_patches)
    all_nms  = (diag_nms..., neg_nms..., pos_nms..., standard_nms...)
    bindings = [Expr(:(=), nm, :(args[$i])) for (i, nm) in enumerate(all_nms)]
    kt       = KT()
    body     = _emit_bbm_periodic_body(diag_levels, neg_levels, pos_levels, T,
                                         Val(_bbm_periodic_ldg(kt)),
                                         Val(_bbm_periodic_zb(kt)))
    quote
        @inbounds begin
            $(bindings...)
            $body
        end
        return nothing
    end
end

@inline _bbm_periodic_ldg(::_BBMPeriodicSpMVKern{LDG, ZB}) where {LDG, ZB} = LDG
@inline _bbm_periodic_zb( ::_BBMPeriodicSpMVKern{LDG, ZB}) where {LDG, ZB} = ZB

# Public launcher.  Dispatches on (LDG, ZB) and routes args in the order the
# @generated kernel expects.
function JLUST._bbm_periodic_spmv_launch!(ka,
        diag_bufs::Tuple, neg_bufs::Tuple, pos_bufs::Tuple,
        diag_fmt::TensorFormat, neg_fmt::TensorFormat, pos_fmt::TensorFormat,
        y, x,
        n_diag::Int32, n_off::Int32, n_cols::Int32, n_total_rows::Int32,
        patches::Tuple,
        zero_beta::Bool, beta;
        alpha = oneunit(eltype(y)), input_fn = identity, origin_off::Int32 = Int32(1))
    ws  = JLUST._default_workgroup_size(ka)
    ldg = JLUST._supports_ldg(ka)
    Fmts = Tuple{typeof(diag_fmt), typeof(neg_fmt), typeof(pos_fmt)}
    T    = eltype(y)
    _launch_bbm_periodic(ka, ws, Val(ldg), Val(zero_beta), Fmts, T,
        diag_bufs, neg_bufs, pos_bufs,
        x, y, n_diag, n_off, n_cols, n_total_rows,
        input_fn, T(alpha), T(beta), origin_off, patches)
end

@inline function _launch_bbm_periodic(ka, ws,
        ::Val{LDG}, ::Val{ZB}, ::Type{Fmts}, ::Type{T},
        diag_bufs::Tuple, neg_bufs::Tuple, pos_bufs::Tuple,
        x, y, n_diag::Int32, n_off::Int32, n_cols::Int32, n_total_rows::Int32,
        input_fn, alpha, beta, origin_off::Int32, patches::Tuple) where {LDG, ZB, Fmts, T}
    kt = _BBMPeriodicSpMVKern{LDG, ZB}()
    args = (kt, Fmts, T,
            diag_bufs..., neg_bufs..., pos_bufs...,
            x, y, n_diag, n_off, n_cols, n_total_rows,
            input_fn, alpha, beta, origin_off, patches)
    _launch_kern(ka, _bbm_periodic_emit_kern, args, Int(n_total_rows), ws)
end
