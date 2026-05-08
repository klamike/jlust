# ─── SpMM body emitter ────────────────────────────────────────────────────────
#
# Column-first 1-D kernel: each thread handles one outer fiber (row for CSR,
# fiber for DCSR, NNZ for COO) and sequentially loops over n_col columns of B.
# Builds via the unified level walker in _walker.jl.

function _emit_spmm_body(levels::Tuple, ::Type{T}) where T
    leaf_unique = :(_acc += _nzval[_nnz_pos] * _B[_x_idx, _col])
    leaf_atomic = :(KernelAbstractions.@atomic _C[_y_idx, _col] +=
                        _alpha * _nzval[_nnz_pos] * _B[_x_idx, _col])
    row_body_unique = inner -> quote
        for _col in 1:_n_col
            _acc = $(zero(T))
            $inner
            _C[_y_idx, _col] = _alpha * _acc + _beta * _C[_y_idx, _col]
        end
    end
    # Non-unique outer (COO-like): pre-scaled-by-beta C accumulated via atomic leaf.
    row_body_atomic = inner -> quote
        for _col in 1:_n_col
            $inner
        end
    end
    emit_kernel_body(levels;
                     row_body_unique, row_body_atomic, leaf_unique, leaf_atomic)
end

_emit_spmm_body(fmt::TensorFormat, T) = _emit_spmm_body(fmt.levels, T)

# ─── SpMM kernel singletons (column-first / NNZ-first / tiled) ──────────────
# All three emit through the unified `_ust_emit_kern`; only the standard arg
# names and the body builder differ.  The 2D kernel (further down) uses a
# different signature and stays separate.

struct _SpMMColKern end
struct _SpMMNNZFirstKern{NCOL,ZB} end
struct _SpMMTiledKern end

_kern_standard_nms(::_SpMMColKern)        = (:_B, :_C, :_origin_off, :_n_outer, :_n_col, :_alpha, :_beta)
_kern_standard_nms(::_SpMMNNZFirstKern)   = (:_B, :_C, :_origin_off, :_n_outer, :_alpha, :_beta)
_kern_standard_nms(::_SpMMTiledKern)      = (:_B, :_C, :_origin_off, :_n_outer, :_n_col, :_alpha, :_beta)

_kern_emit_body(::_SpMMColKern,                levels, ::Type{T}) where T          = _emit_spmm_body(levels, T)
_kern_emit_body(::_SpMMNNZFirstKern{NCOL,ZB},  levels, ::Type{T}) where {NCOL,ZB,T} = _emit_spmm_body_nnzfirst(levels, T, NCOL, ZB)
_kern_emit_body(::_SpMMTiledKern,              levels, ::Type{T}) where T          = _emit_spmm_body_tiled(levels, T, _SPMM_TILE_K)

# ─── NNZ-first (tiled) SpMM emitter ─────────────────────────────────────────
#
# For each row, iterates NNZ in the outer loop and all k output columns in the
# inner loop — loading each B[x_idx, :] row once and broadcasting the scalar
# nzval across all k accumulators.  Inlining n_col as a compile-time constant
# allows the Julia compiler to fully unroll the column loop and keep all k
# accumulators in registers.
#
# Effective for small/medium k (≤ SPMM_NCOL_INLINE_THRESHOLD).  For larger k,
# we fall back to the column-first emitter (avoids register spill).

const _SPMM_NCOL_INLINE_THRESHOLD = 32   # inline up to 32 output columns
const _SPMM_2D_AVG_NNZ            = 3    # avg NNZ/row above this → 2D for large matrices

# Per-backend cache: SM count queried once, then reused across all SpMM calls.
const _sm_count_cache = Dict{Type, Int}()

# Conservative default for backends whose SM count we cannot determine.
_probe_sm_count(::Any) = 80   # ~V100 level; underestimates for L40S, safe for T4+

function _probe_sm_count(ka::KernelAbstractions.GPU)
    # Try to query the CUDA SM count via Base.loaded_modules.
    # Falls back to the default if CUDA is not loaded or the query fails.
    cuda_id = Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")
    if haskey(Base.loaded_modules, cuda_id)
        cuda = Base.loaded_modules[cuda_id]
        try
            dev = cuda.device()
            return Int(cuda.attribute(dev, cuda.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
        catch
        end
    end
    return 80
end

# Threshold n_rows below which 2D is always chosen (1D has < ~6 warps/SM).
# Scales with SM count: threshold ≈ 3 blocks/SM × group_size(64) × n_SMs = 192 × n_SMs.
# Cached after first call; device properties are constant within a session.
function _spmm_2d_max_rows(ka)
    get!(_sm_count_cache, typeof(ka)) do
        192 * _probe_sm_count(ka)
    end
end

function _emit_spmm_body_nnzfirst(levels::Tuple, ::Type{T}, n_col::Int, zero_beta::Bool=false) where T
    # k accumulators (one per output column), accumulated in registers, written together.
    leaf_unique = Expr(:block, [
        :($(Symbol(:_acc_, c)) += _nzval[_nnz_pos] * _B[_x_idx, $(c)])
        for c in 1:n_col]...)
    leaf_atomic = Expr(:block, [
        :(KernelAbstractions.@atomic _C[_y_idx, $(c)] += _alpha * _nzval[_nnz_pos] * _B[_x_idx, $(c)])
        for c in 1:n_col]...)
    acc_inits  = Expr(:block, [:($(Symbol(:_acc_, c)) = $(zero(T))) for c in 1:n_col]...)
    acc_writes = zero_beta ?
        Expr(:block, [:(_C[_y_idx, $(c)] = _alpha * $(Symbol(:_acc_, c)))
                      for c in 1:n_col]...) :
        Expr(:block, [:(_C[_y_idx, $(c)] = _alpha * $(Symbol(:_acc_, c)) + _beta * _C[_y_idx, $(c)])
                      for c in 1:n_col]...)
    row_body_unique = inner -> quote
        $acc_inits
        $inner
        $acc_writes
    end
    row_body_atomic = inner -> inner
    emit_kernel_body(levels;
                     row_body_unique, row_body_atomic, leaf_unique, leaf_atomic)
end

_emit_spmm_body_nnzfirst(fmt::TensorFormat, T, n_col, zero_beta=false) =
    _emit_spmm_body_nnzfirst(fmt.levels, T, n_col, zero_beta)

_is_csr_like_for_guard(fmt::TensorFormat) =
    length(fmt.levels) == 2 &&
    fmt.levels[1] isa Union{DenseLevel,BatchLevel} &&
    fmt.levels[2] isa CompressedLevel &&
    is_unique(fmt.levels[2])

# ─── 2D NNZ-first SpMM (one thread per (row, col) pair) ─────────────────────
#
# Problem: 1D NNZ-first launches n_rows threads.  For medium networks (9K–14K
# buses) this gives only 2–3 warps/SM on the L40S (142 SMs), leaving most SMs
# bandwidth-starved.  For 30K buses it's still only ~6 warps/SM.
#
# Fix: launch n_rows × n_col threads.  Each thread handles exactly one output
# element — 1 accumulator instead of n_col.  For T=24 this is 24× more blocks:
#   9241 Bbus  (1D):  9241 rows  →  145 blocks →  2 warps/SM
#   9241 Bbus  (2D): 221784      → 3466 blocks → 48 warps/SM  (hardware limit)
#   30000 Bbus (2D): 720000      → fully saturated at 48 warps/SM
# Lower register pressure (1 vs 24 Float64 accumulators) lets more blocks coexist
# per SM, reinforcing the occupancy gain.
#
# The B reads B[x_idx, col] remain non-coalesced across a warp (consecutive
# threads have different col values at stride n_rows_B).  Latency hiding via the
# abundant concurrent warps compensates; no layout change is required.
#
# Applied for all CSR-like formats when n_col ≤ _SPMM_NCOL_INLINE_THRESHOLD.
# Non-CSR formats (DCSR, COO) continue to use the 1D NNZ-first path.

function _emit_spmm_2d_body_csr(::Type{T}, n_col::Int, zero_beta::Bool, guard::Bool) where T
    acc_write = if zero_beta
        :(_C[_row, _col] = _alpha * _acc)
    else
        :(_C[_row, _col] = _alpha * _acc + _beta * _C[_row, _col])
    end
    inner = quote
        _acc = $(zero(T))
        for _i2 in (_lo + 1):_hi
            _x_idx = Int(_crd1[_i2]) - Int(_origin_off) + 1
            _acc  += _nzval[_i2] * _B[_x_idx, _col]
        end
        $acc_write
    end
    work = guard ? :(if _lo < _hi; $inner; end) : inner
    quote
        _lin = KI.get_global_id().x - 1
        _row = _lin ÷ $(n_col) + 1
        _col = _lin % $(n_col) + 1
        if _row <= _n_outer
            _lo = Int(_pos1[_row])     - Int(_origin_off)
            _hi = Int(_pos1[_row + 1]) - Int(_origin_off)
            $work
        end
    end
end

# 2D SpMM kernel — n_col, zero_beta, guard are compile-time type parameters.
# Hardcoded CSR signature (level structure is uniform; the guard is a separate axis).
@generated function _ust_spmm_2d_kern(::Type{T}, ::Val{NCOL}, ::Val{ZB}, ::Val{GUARD},
                                       _pos1, _crd1, _nzval,
                                       _B, _C, _origin_off, _n_outer, _alpha, _beta) where {T, NCOL, ZB, GUARD}
    body = _emit_spmm_2d_body_csr(T, NCOL, ZB, GUARD)
    quote
        @inbounds begin
            $body
        end
        return nothing
    end
end

# ─── Tiled NNZ-first SpMM emitter ────────────────────────────────────────────
#
# For large k (> _SPMM_NCOL_INLINE_THRESHOLD), the full NNZ-first emitter would
# need k accumulators in registers — likely spilling for k=64.  The tiled
# variant processes k in TILE_K-wide strips, keeping only TILE_K accumulators
# in registers at once.
#
# Each strip iterates the NNZ once (loading pos/crd/nzval), writing to TILE_K
# output columns via runtime _tile_start + compile-time offset.  For k=64 with
# TILE_K=8: 8 NNZ passes vs column-first's 64 passes — 8× less NNZ bandwidth.
#
# Applied when n_col > _SPMM_NCOL_INLINE_THRESHOLD and n_col % _SPMM_TILE_K == 0.

const _SPMM_TILE_K = 8   # columns per tile; governs register pressure

# Tiled body — reuses the unified walker (`_walk_inner` / `_emit_outer`).  The
# row/leaf shape differs from full-k NNZ-first only in: (a) tile_k accumulators
# tracked via `_tile_start` runtime offset, (b) outer while-loop over tiles.
# No second walker is needed — the leaf and row_body callbacks parameterize it.
function _emit_spmm_body_tiled(levels::Tuple, ::Type{T}, tile_k::Int) where T
    acc_inits   = Expr(:block, [:($(Symbol(:_acc_, c)) = $(zero(T))) for c in 1:tile_k]...)
    acc_writes  = Expr(:block, [
        :(_C[_y_idx, _tile_start + $(c-1)] =
              _alpha * $(Symbol(:_acc_, c)) + _beta * _C[_y_idx, _tile_start + $(c-1)])
        for c in 1:tile_k]...)
    leaf_unique = Expr(:block, [
        :($(Symbol(:_acc_, c)) += _nzval[_nnz_pos] * _B[_x_idx, _tile_start + $(c - 1)])
        for c in 1:tile_k]...)
    leaf_atomic = Expr(:block, [
        :(KernelAbstractions.@atomic _C[_y_idx, _tile_start + $(c - 1)] +=
              _alpha * _nzval[_nnz_pos] * _B[_x_idx, _tile_start + $(c - 1)])
        for c in 1:tile_k]...)
    row_body_unique = inner -> quote
        _tile_start = Int32(1)
        while _tile_start <= _n_col
            $acc_inits
            $inner
            $acc_writes
            _tile_start += Int32($tile_k)
        end
    end
    row_body_atomic = inner -> quote
        _tile_start = Int32(1)
        while _tile_start <= _n_col
            $inner
            _tile_start += Int32($tile_k)
        end
    end
    emit_kernel_body(levels;
                     row_body_unique, row_body_atomic, leaf_unique, leaf_atomic)
end

_emit_spmm_body_tiled(fmt::TensorFormat, T, tile_k) =
    _emit_spmm_body_tiled(fmt.levels, T, tile_k)

# ─── SpMM kernel strategies ───────────────────────────────────────────────────
#
# Four parallelization shapes; selection is pure function of (format, n_outer,
# n_col, nnz, ka, skip_empty_rows).  Adding a new strategy = define a singleton
# type, a `_run_spmm!` method, and add a branch to `_spmm_strategy`.
#
#   ColFirst  : 1 thread/row, runtime n_col loop, 1 accumulator     (universal fallback)
#   NNZFirst  : 1 thread/row, compile-time NCOL accumulators        (k ≤ NCOL_INLINE)
#   Tiled     : 1 thread/row, TILE_K accs/strip, n_col/TILE_K passes (k > NCOL_INLINE, k % TILE_K = 0)
#   2D        : 1 thread/cell, 1 accumulator, CSR-only              (saturate SMs at small n_outer)

struct _SpMMColFirst end
struct _SpMMNNZFirst end
struct _SpMMTiled    end
struct _SpMM2D       end

function _spmm_strategy(fmt::TensorFormat, n_outer::Int, n_col::Int,
                         nnz::Int, ka, skip_empty_rows::Bool)
    lv1 = fmt.levels[1]
    is_coo_like = lv1 isa CompressedLevel && !is_unique(lv1) &&
                  length(fmt.levels) >= 2 && fmt.levels[2] isa SingletonLevel
    is_coo_like && return _SpMMColFirst()

    if n_col <= _SPMM_NCOL_INLINE_THRESHOLD
        # 2D wins when CSR-shaped AND (skip-empty mask requested ∨ 1D underfills SMs ∨ NNZ-dense).
        if _is_csr_like_for_guard(fmt) && (skip_empty_rows ||
                                           n_outer <= _spmm_2d_max_rows(ka) ||
                                           nnz > _SPMM_2D_AVG_NNZ * n_outer)
            return _SpMM2D()
        end
        return _SpMMNNZFirst()
    end
    n_col % _SPMM_TILE_K == 0 && return _SpMMTiled()
    _SpMMColFirst()
end

# Common runtime args bundle: (fmt_type, T, sparse_bufs..., B, C, off, n_outer, ...).
@inline _spmm_common(u_A, u_B, u_C, off, n_outer) =
    (_sparse_args(u_A)..., nonzeros(u_B), nonzeros(u_C), off, n_outer)

function _run_spmm!(::_SpMM2D, ka, u_A, u_B, u_C, off, n_outer::Int32, n_col::Int,
                    T_alpha, T_beta, ::Val{Z}, ::Val{SE}) where {Z, SE}
    # 2D kernel uses a hardcoded CSR signature, so it bypasses _ust_emit_kern.
    T = eltype(u_A)
    args = (T, Val(n_col), Val(Z), Val(SE),
            _spmm_common(u_A, u_B, u_C, off, n_outer)..., T_alpha, T_beta)
    _launch_kern(ka, _ust_spmm_2d_kern, args, Int(n_outer) * n_col)
end

function _run_spmm!(::_SpMMNNZFirst, ka, u_A, u_B, u_C, off, n_outer::Int32, n_col::Int,
                    T_alpha, T_beta, ::Val{Z}, ::Val{SE}) where {Z, SE}
    T = eltype(u_A)
    args = (_SpMMNNZFirstKern{n_col, Z}(), typeof(format(u_A)), T,
            _spmm_common(u_A, u_B, u_C, off, n_outer)..., T_alpha, T_beta)
    _launch_kern(ka, _ust_emit_kern, args, Int(n_outer))
end

function _run_spmm!(::_SpMMTiled, ka, u_A, u_B, u_C, off, n_outer::Int32, n_col::Int,
                    T_alpha, T_beta, ::Val{Z}, ::Val{SE}) where {Z, SE}
    T = eltype(u_A)
    args = (_SpMMTiledKern(), typeof(format(u_A)), T,
            _spmm_common(u_A, u_B, u_C, off, n_outer)..., Int32(n_col), T_alpha, T_beta)
    _launch_kern(ka, _ust_emit_kern, args, Int(n_outer))
end

function _run_spmm!(::_SpMMColFirst, ka, u_A, u_B, u_C, off, n_outer::Int32, n_col::Int,
                    T_alpha, T_beta, ::Val{Z}, ::Val{SE}) where {Z, SE}
    T = eltype(u_A)
    args = (_SpMMColKern(), typeof(format(u_A)), T,
            _spmm_common(u_A, u_B, u_C, off, n_outer)..., Int32(n_col), T_alpha, T_beta)
    _launch_kern(ka, _ust_emit_kern, args, Int(n_outer))
end

# ─── sparse_mm! ───────────────────────────────────────────────────────────────

function JLUST.execute(::EmitterBackend, ::Op{:SpMM, F},
                       u_A::USTensor, u_B::USTensor, u_C::USTensor;
                       alpha=one(eltype(u_A)), beta=zero(eltype(u_A)),
                       skip_empty_rows::Bool=false) where {F}
    fmt     = format(u_A)
    T       = eltype(u_A)
    T_alpha = T(alpha)
    T_beta  = T(beta)
    ka      = KernelAbstractions.get_backend(nonzeros(u_A))
    off     = Int32(index_origin(u_A) isa OneBased ? 1 : 0)
    n_outer = Int32(_spmv_ndrange(u_A))
    n_col   = Int(extents(u_B)[2])

    # COO-like atomic-leaf path needs C pre-scaled by beta.
    lv1 = fmt.levels[1]
    if lv1 isa CompressedLevel && !is_unique(lv1) &&
       length(fmt.levels) >= 2 && fmt.levels[2] isa SingletonLevel
        iszero(T_beta) ? fill!(nonzeros(u_C), zero(T)) : (nonzeros(u_C) .*= T_beta)
    end

    strat = _spmm_strategy(fmt, Int(n_outer), n_col, length(nonzeros(u_A)), ka, skip_empty_rows)
    _run_spmm!(strat, ka, u_A, u_B, u_C, off, n_outer, n_col,
               T_alpha, T_beta, Val(iszero(T_beta)), Val(skip_empty_rows))
    return u_C
end

# ─── Fused BBM scatter kernels ────────────────────────────────────────────────
#
# Replace T*nb_r tiny copyto! calls (48 for T=24, nb_r=2) with a single kernel.
# diag scatter: diag_out[r, t] → y[(t-1)*d_period + r]  (0-indexed t, r)
# off  scatter: off_buf[r, t]  → y[(t-1)*d_period + off_start + r]

@kernel inbounds=true function _bbm_scatter_diag_kernel!(_y, @Const(_diag), _d_period, _n_diag)
    gid = @index(Global, Linear) - 1
    t   = gid ÷ _n_diag
    r   = gid % _n_diag
    _y[t * _d_period + r + 1] = _diag[r + 1, t + 1]
end

@kernel inbounds=true function _bbm_scatter_off_kernel!(_y, @Const(_off), _d_period, _off_start, _n_off)
    gid = @index(Global, Linear) - 1
    t   = gid ÷ _n_off
    r   = gid % _n_off
    _y[t * _d_period + _off_start + r + 1] = _off[r + 1, t + 1]
end

function JLUST._bbm_scatter_diag!(y, diag_out, d_period, n_diag, T)
    ka = KernelAbstractions.get_backend(y)
    _bbm_scatter_diag_kernel!(ka, 256)(y, diag_out, Int(d_period), Int(n_diag);
                                       ndrange=Int(n_diag) * Int(T))
end

function JLUST._bbm_scatter_off!(y, off_buf, d_period, off_start, n_off, T_off)
    ka = KernelAbstractions.get_backend(y)
    _bbm_scatter_off_kernel!(ka, 256)(y, off_buf, Int(d_period), Int(off_start), Int(n_off);
                                      ndrange=Int(n_off) * Int(T_off))
end
