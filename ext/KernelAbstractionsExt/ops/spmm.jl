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

# ─── @generated SpMM kernel functions (column-first 1-D) ────────────────────

@generated function _ust_spmm_kern(::Type{FMT}, ::Type{T}, args::Vararg{Any, M}) where {FMT<:TensorFormat, T, M}
    LT          = FMT.parameters[1]
    levels      = ntuple(i -> LT.parameters[i](), Val(length(LT.parameters)))
    sparse_nms  = _sparse_arg_names_for_levels(levels)
    standard_nm = (:_B, :_C, :_origin_off, :_n_outer, :_n_col, :_alpha, :_beta)
    all_nms     = (sparse_nms..., standard_nm...)
    bindings    = [Expr(:(=), nm, :(args[$i])) for (i, nm) in enumerate(all_nms)]
    body        = _emit_spmm_body(levels, T)
    quote
        @inbounds begin
            $(bindings...)
            $body
        end
        return nothing
    end
end

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

# ─── Kernel cache and launch ─────────────────────────────────────────────────

# NNZ-first kernel — n_col and zero_beta are compile-time type parameters
# (Julia specializes the @generated body per (FMT, T, NCOL, ZB) combination).
@generated function _ust_spmm_nf_kern(::Type{FMT}, ::Type{T}, ::Val{NCOL}, ::Val{ZB},
                                       args::Vararg{Any, M}) where {FMT<:TensorFormat, T, NCOL, ZB, M}
    LT          = FMT.parameters[1]
    levels      = ntuple(i -> LT.parameters[i](), Val(length(LT.parameters)))
    sparse_nms  = _sparse_arg_names_for_levels(levels)
    standard_nm = (:_B, :_C, :_origin_off, :_n_outer, :_alpha, :_beta)
    all_nms     = (sparse_nms..., standard_nm...)
    bindings    = [Expr(:(=), nm, :(args[$i])) for (i, nm) in enumerate(all_nms)]
    body        = _emit_spmm_body_nnzfirst(levels, T, NCOL, ZB)
    quote
        @inbounds begin
            $(bindings...)
            $body
        end
        return nothing
    end
end

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

# Inner handlers: identical structure to NNZ-first but recurse into the tiled
# leaf.  Only non-outermost levels are defined here; the outermost level and
# the leaf differ from the full-k NNZ-first path.

function _emit_spmm_tiled_inner(levels, lvl, p_var, pc, cc, T, tile_k)
    if lvl > length(levels)
        return Expr(:block, [
            :($(Symbol(:_acc_, c)) += _nzval[_nnz_pos] * _B[_x_idx, _tile_start + $(c - 1)])
            for c in 1:tile_k]...)
    end
    _emit_spmm_tiled_lv(levels[lvl], levels, lvl, p_var, pc, cc, T, tile_k)
end

function _emit_spmm_tiled_lv(::CompressedLevel, levels, lvl, p_var::Symbol, pc, cc, T, tile_k)
    pi = pc[] += 1; ci = cc[] += 1
    ps = Symbol(:_pos, pi); cs = Symbol(:_crd, ci)
    lvar = Symbol(:_i, lvl)
    inner = _emit_spmm_tiled_inner(levels, lvl + 1, lvar, pc, cc, T, tile_k)
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

function _emit_spmm_tiled_lv(::SingletonLevel, levels, lvl, p_var::Symbol, pc, cc, T, tile_k)
    ci = cc[] += 1
    cs = Symbol(:_crd, ci)
    inner = _emit_spmm_tiled_inner(levels, lvl + 1, p_var, pc, cc, T, tile_k)
    quote
        _x_idx   = Int($cs[$p_var]) - Int(_origin_off) + 1
        _nnz_pos = $p_var
        $inner
    end
end

function _emit_spmm_tiled_lv(::DeltaLevel, levels, lvl, p_var::Symbol, pc, cc, T, tile_k)
    pi = pc[] += 1; ci = cc[] += 1
    ps = Symbol(:_pos, pi); cs = Symbol(:_crd, ci)
    lvar  = Symbol(:_i, lvl)
    corig = Symbol(:_corig, lvl)
    inner = _emit_spmm_tiled_inner(levels, lvl + 1, lvar, pc, cc, T, tile_k)
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

# AbstractLevelFormat inner level: delegate to level_step hook (mirrors non-tiled path).
function _emit_spmm_tiled_lv(lv::AbstractLevelFormat, levels, lvl, p_var::Symbol, pc, cc, T, tile_k)
    nz_sym = JLUST.level_has_nzval(lv) ? :_nzval : :nothing
    inner  = _emit_spmm_tiled_inner(levels, lvl + 1, p_var, pc, cc, T, tile_k)
    quote
        _p1 = Int($p_var) - Int(_origin_off) + 1
        (_x_idx, _) = JLUST.level_step($lv, _p1, $nz_sym)
        _nnz_pos = $p_var
        $inner
    end
end

function _emit_spmm_tiled_lv(lv, args...)
    error("EmitterBackend tiled SpMM: unsupported level $(typeof(lv)); convert to CSR/DCSR first")
end

function _emit_spmm_body_tiled(levels::Tuple, ::Type{T}, tile_k::Int) where T
    lv1 = levels[1]
    acc_inits = Expr(:block, [:($(Symbol(:_acc_, c)) = $(zero(T))) for c in 1:tile_k]...)

    if lv1 isa Union{DenseLevel,BatchLevel}
        # CSR-like: thread → row; output row index is _tid
        pc = Ref(0); cc = Ref(0)
        inner = _emit_spmm_tiled_inner(levels, 2, :_tid, pc, cc, T, tile_k)
        acc_writes = Expr(:block, [:(_C[_tid, _tile_start + $(c-1)] = _alpha * $(Symbol(:_acc_, c)) + _beta * _C[_tid, _tile_start + $(c-1)]) for c in 1:tile_k]...)
        quote
            _tid = KI.get_global_id().x
            if _tid <= _n_outer
                _tile_start = Int32(1)
                while _tile_start <= _n_col
                    $acc_inits
                    $inner
                    $acc_writes
                    _tile_start += Int32($tile_k)
                end
            end
        end
    elseif lv1 isa CompressedLevel && is_unique(lv1)
        # DCSR-like: thread → fiber; crd1 → output row index _y_idx
        # Outer CompressedLevel consumed pc=1, cc=1 → _pos1, _crd1
        pc = Ref(1); cc = Ref(1)
        inner = _emit_spmm_tiled_inner(levels, 2, :_tid, pc, cc, T, tile_k)
        acc_writes = Expr(:block, [:(_C[_y_idx, _tile_start + $(c-1)] = _alpha * $(Symbol(:_acc_, c)) + _beta * _C[_y_idx, _tile_start + $(c-1)]) for c in 1:tile_k]...)
        quote
            _tid = KI.get_global_id().x
            if _tid <= _n_outer
                _y_idx = Int(_crd1[_tid]) - Int(_origin_off) + 1
                _tile_start = Int32(1)
                while _tile_start <= _n_col
                    $acc_inits
                    $inner
                    $acc_writes
                    _tile_start += Int32($tile_k)
                end
            end
        end
    else
        error("EmitterBackend tiled SpMM: $(typeof(lv1)) cannot be the outermost level; pair with DenseLevel or use unique CompressedLevel.")
    end
end

_emit_spmm_body_tiled(fmt::TensorFormat, T, tile_k) =
    _emit_spmm_body_tiled(fmt.levels, T, tile_k)

# Tiled kernel — TILE_K is a compile-time constant; runtime _n_col drives the tile loop.
@generated function _ust_spmm_tiled_kern(::Type{FMT}, ::Type{T}, args::Vararg{Any, M}) where {FMT<:TensorFormat, T, M}
    LT          = FMT.parameters[1]
    levels      = ntuple(i -> LT.parameters[i](), Val(length(LT.parameters)))
    sparse_nms  = _sparse_arg_names_for_levels(levels)
    standard_nm = (:_B, :_C, :_origin_off, :_n_outer, :_n_col, :_alpha, :_beta)
    all_nms     = (sparse_nms..., standard_nm...)
    bindings    = [Expr(:(=), nm, :(args[$i])) for (i, nm) in enumerate(all_nms)]
    body        = _emit_spmm_body_tiled(levels, T, _SPMM_TILE_K)
    quote
        @inbounds begin
            $(bindings...)
            $body
        end
        return nothing
    end
end

# ─── sparse_mm! ───────────────────────────────────────────────────────────────

function JLUST.sparse_mm!(::EmitterBackend, u_A::USTensor, u_B::USTensor, u_C::USTensor;
                           alpha=one(eltype(u_A)), beta=zero(eltype(u_A)),
                           skip_empty_rows::Bool=false)
    fmt     = format(u_A)
    T       = eltype(u_A)
    T_alpha = T(alpha)
    T_beta  = T(beta)
    ka      = KernelAbstractions.get_backend(nonzeros(u_A))
    off     = Int32(index_origin(u_A) isa OneBased ? 1 : 0)
    n_outer = Int32(_spmv_ndrange(u_A))
    n_col   = Int(extents(u_B)[2])

    lv1 = fmt.levels[1]
    is_coo_like = lv1 isa CompressedLevel && !is_unique(lv1) &&
                  length(fmt.levels) >= 2 && fmt.levels[2] isa SingletonLevel

    # COO-like: pre-scale C by beta; atomic += accumulates into this baseline.
    if is_coo_like
        iszero(T_beta) ? fill!(nonzeros(u_C), zero(T)) : (nonzeros(u_C) .*= T_beta)
    end

    sparse_bufs = _sparse_args(u_A)
    z           = iszero(T_beta)

    if !is_coo_like && n_col <= _SPMM_NCOL_INLINE_THRESHOLD
        # 2D kernel (one thread per output cell) for CSR-like matrices when: skip_empty_rows
        # is set, the matrix is too small for 1D to saturate SMs, or high NNZ density means
        # bandwidth hiding from concurrent warps outweighs 2D launch overhead.
        # All other cases (DCSR, COO, large sparse CSR) use the 1D NNZ-first path.
        use_2d = _is_csr_like_for_guard(fmt) && (skip_empty_rows ||
                     Int(n_outer) <= _spmm_2d_max_rows(ka) ||
                     length(nonzeros(u_A)) > _SPMM_2D_AVG_NNZ * Int(n_outer))
        if use_2d
            # 2D kernel uses fixed CSR signature (pos1, crd1, nzval, B, C, ...).
            # The guard axis becomes a Val parameter so dispatch picks the right specialization.
            args = (T, Val(n_col), Val(z), Val(skip_empty_rows),
                    sparse_bufs..., nonzeros(u_B), nonzeros(u_C),
                    off, n_outer, T_alpha, T_beta)
            _launch_kern(ka, _ust_spmm_2d_kern, args, Int(n_outer) * n_col)
        else
            args = (typeof(fmt), T, Val(n_col), Val(z),
                    sparse_bufs..., nonzeros(u_B), nonzeros(u_C),
                    off, n_outer, T_alpha, T_beta)
            _launch_kern(ka, _ust_spmm_nf_kern, args, Int(n_outer))
        end
    elseif !is_coo_like && n_col % _SPMM_TILE_K == 0
        # Tiled NNZ-first: TILE_K accumulators per strip, n_col/TILE_K NNZ passes.
        args = (typeof(fmt), T,
                sparse_bufs..., nonzeros(u_B), nonzeros(u_C),
                off, n_outer, Int32(n_col), T_alpha, T_beta)
        _launch_kern(ka, _ust_spmm_tiled_kern, args, Int(n_outer))
    else
        # Column-first: runtime n_col loop (fallback for COO and non-divisible k).
        args = (typeof(fmt), T,
                sparse_bufs..., nonzeros(u_B), nonzeros(u_C),
                off, n_outer, Int32(n_col), T_alpha, T_beta)
        _launch_kern(ka, _ust_spmm_kern, args, Int(n_outer))
    end

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
