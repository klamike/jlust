# ─── SpMM body emitter ────────────────────────────────────────────────────────
#
# Emits a 1-D @kernel for sparse_mm!: each thread handles one outer fiber (row
# for CSR, fiber for DCSR, NNZ for COO) and sequentially loops over the n_col
# columns of B.  Accumulation pattern mirrors SpMV; leaf writes B[x_idx, col]
# instead of x[x_idx], and final write goes to C[y_idx, col].

function _emit_spmm_body(fmt::TensorFormat, ::Type{T}) where T
    pc = Ref(0); cc = Ref(0)
    _, lv1 = fmt.levels[1]
    na = lv1 isa CompressedLevel && !is_unique(lv1)
    _emit_spmm_level(fmt.levels, 1, nothing, pc, cc, T, na)
end

function _emit_spmm_level(levels, lvl, p_var, pc, cc, T, needs_atomic)
    if lvl > length(levels)
        return needs_atomic ?
            :(KernelAbstractions.@atomic _C[_y_idx, _col] += _nzval[_nnz_pos] * _B[_x_idx, _col]) :
            :(_acc += _nzval[_nnz_pos] * _B[_x_idx, _col])
    end
    _, lv = levels[lvl]
    _emit_spmm_lv(lv, levels, lvl, p_var, pc, cc, T, needs_atomic)
end

# DenseLevel / BatchLevel (outermost) → thread = row; sequential col loop
function _emit_spmm_lv(::Union{DenseLevel,BatchLevel}, levels, lvl, ::Nothing, pc, cc, T, _)
    inner = _emit_spmm_level(levels, lvl + 1, :_tid, pc, cc, T, false)
    quote
        _tid = @index(Global, Linear)
        if _tid <= _n_outer
            for _col in 1:_n_col
                _acc = $(zero(T))
                $inner
                _C[_tid, _col] = _alpha * _acc + _beta * _C[_tid, _col]
            end
        end
    end
end

# DenseLevel / BatchLevel (non-outermost) → dense loop
function _emit_spmm_lv(::Union{DenseLevel,BatchLevel}, levels, lvl, p_var::Symbol, pc, cc, T, na)
    sz  = Symbol(:_sz, lvl)
    lv2 = Symbol(:_i, lvl)
    inner = _emit_spmm_level(levels, lvl + 1, lv2, pc, cc, T, na)
    quote
        for $lv2 in 1:$sz
            $inner
        end
    end
end

# CompressedLevel (outermost)
#   unique   → fiber-parallel (DCSR-like): one thread per non-empty row
#   non-unique → NNZ-parallel (COO-like): atomic C update per NNZ × col
function _emit_spmm_lv(lv::CompressedLevel, levels, lvl, ::Nothing, pc, cc, T, _)
    pc[] += 1; ci = cc[] += 1
    cs = Symbol(:_crd, ci)
    if is_unique(lv)
        inner = _emit_spmm_level(levels, lvl + 1, :_tid, pc, cc, T, false)
        quote
            _tid = @index(Global, Linear)
            if _tid <= _n_outer
                _y_idx = Int($cs[_tid]) - Int(_origin_off) + 1
                for _col in 1:_n_col
                    _acc = $(zero(T))
                    $inner
                    _C[_y_idx, _col] = _alpha * _acc + _beta * _C[_y_idx, _col]
                end
            end
        end
    else
        inner = _emit_spmm_level(levels, lvl + 1, :_tid, pc, cc, T, true)
        quote
            _tid = @index(Global, Linear)
            if _tid <= _n_outer
                _y_idx = Int($cs[_tid]) - Int(_origin_off) + 1
                for _col in 1:_n_col
                    $inner
                end
            end
        end
    end
end

# CompressedLevel (non-outermost) → inner fiber loop; same as SpMV
function _emit_spmm_lv(::CompressedLevel, levels, lvl, p_var::Symbol, pc, cc, T, na)
    pi = pc[] += 1; ci = cc[] += 1
    ps = Symbol(:_pos, pi); cs = Symbol(:_crd, ci)
    lvar = Symbol(:_i, lvl)
    inner = _emit_spmm_level(levels, lvl + 1, lvar, pc, cc, T, na)
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
function _emit_spmm_lv(::SingletonLevel, levels, lvl, p_var::Symbol, pc, cc, T, na)
    ci = cc[] += 1
    cs = Symbol(:_crd, ci)
    inner = _emit_spmm_level(levels, lvl + 1, p_var, pc, cc, T, na)
    quote
        _x_idx   = Int($cs[$p_var]) - Int(_origin_off) + 1
        _nnz_pos = $p_var
        $inner
    end
end

# DeltaLevel (non-outermost) → accumulated delta decode + inner loop
function _emit_spmm_lv(::DeltaLevel, levels, lvl, p_var::Symbol, pc, cc, T, na)
    pi = pc[] += 1; ci = cc[] += 1
    ps = Symbol(:_pos, pi); cs = Symbol(:_crd, ci)
    lvar  = Symbol(:_i, lvl)
    corig = Symbol(:_corig, lvl)
    inner = _emit_spmm_level(levels, lvl + 1, lvar, pc, cc, T, na)
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

function _emit_spmm_lv(::RangeLevel, levels, lvl, p_var, pc, cc, T, na)
    error("EmitterBackend SpMM: RangeLevel (DIA-style) kernels not yet emitted. " *
          "Use convert_format to CSR or DCSR first.")
end

# ─── Kernel cache and launch ─────────────────────────────────────────────────

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

function _emit_spmm_body_nnzfirst(fmt::TensorFormat, ::Type{T}, n_col::Int, zero_beta::Bool=false) where T
    pc = Ref(0); cc = Ref(0)
    _, lv1 = fmt.levels[1]
    na = lv1 isa CompressedLevel && !is_unique(lv1)
    _emit_spmm_level_nf(fmt.levels, 1, nothing, pc, cc, T, na, n_col, zero_beta)
end

function _emit_spmm_level_nf(levels, lvl, p_var, pc, cc, T, needs_atomic, n_col, zero_beta=false)
    if lvl > length(levels)
        # Leaf: accumulate nzval * B[x_idx, col] for all output columns.
        col_updates = Expr(:block, [
            needs_atomic ?
                :(KernelAbstractions.@atomic _C[_y_idx, $(c)] += _nzval[_nnz_pos] * _B[_x_idx, $(c)]) :
                :($(Symbol(:_acc_, c)) += _nzval[_nnz_pos] * _B[_x_idx, $(c)])
            for c in 1:n_col]...)
        return col_updates
    end
    _, lv = levels[lvl]
    _emit_spmm_lv_nf(lv, levels, lvl, p_var, pc, cc, T, needs_atomic, n_col, zero_beta)
end

function _emit_spmm_lv_nf(::Union{DenseLevel,BatchLevel}, levels, lvl, ::Nothing, pc, cc, T, _na, n_col, zero_beta=false)
    inner      = _emit_spmm_level_nf(levels, lvl + 1, :_tid, pc, cc, T, false, n_col)
    acc_inits  = Expr(:block, [:($(Symbol(:_acc_, c)) = $(zero(T))) for c in 1:n_col]...)
    acc_writes = if zero_beta
        Expr(:block, [:(_C[_tid, $(c)] = _alpha * $(Symbol(:_acc_, c))) for c in 1:n_col]...)
    else
        Expr(:block, [:(_C[_tid, $(c)] = _alpha * $(Symbol(:_acc_, c)) + _beta * _C[_tid, $(c)]) for c in 1:n_col]...)
    end
    quote
        _tid = @index(Global, Linear)
        if _tid <= _n_outer
            $acc_inits
            $inner
            $acc_writes
        end
    end
end

function _emit_spmm_lv_nf(lv::CompressedLevel, levels, lvl, ::Nothing, pc, cc, T, _na, n_col, zero_beta=false)
    pc[] += 1; ci = cc[] += 1
    cs = Symbol(:_crd, ci)
    if is_unique(lv)
        inner      = _emit_spmm_level_nf(levels, lvl + 1, :_tid, pc, cc, T, false, n_col)
        acc_inits  = Expr(:block, [:($(Symbol(:_acc_, c)) = $(zero(T))) for c in 1:n_col]...)
        acc_writes = if zero_beta
            Expr(:block, [:(_C[_y_idx, $(c)] = _alpha * $(Symbol(:_acc_, c))) for c in 1:n_col]...)
        else
            Expr(:block, [:(_C[_y_idx, $(c)] = _alpha * $(Symbol(:_acc_, c)) + _beta * _C[_y_idx, $(c)]) for c in 1:n_col]...)
        end
        quote
            _tid = @index(Global, Linear)
            if _tid <= _n_outer
                _y_idx = Int($cs[_tid]) - Int(_origin_off) + 1
                $acc_inits
                $inner
                $acc_writes
            end
        end
    else
        inner = _emit_spmm_level_nf(levels, lvl + 1, :_tid, pc, cc, T, true, n_col)
        quote
            _tid = @index(Global, Linear)
            if _tid <= _n_outer
                _y_idx = Int($cs[_tid]) - Int(_origin_off) + 1
                $inner
            end
        end
    end
end

function _emit_spmm_lv_nf(::CompressedLevel, levels, lvl, p_var::Symbol, pc, cc, T, na, n_col, zero_beta=false)
    pi = pc[] += 1; ci = cc[] += 1
    ps = Symbol(:_pos, pi); cs = Symbol(:_crd, ci)
    lvar = Symbol(:_i, lvl)
    inner = _emit_spmm_level_nf(levels, lvl + 1, lvar, pc, cc, T, na, n_col)
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

function _emit_spmm_lv_nf(::SingletonLevel, levels, lvl, p_var::Symbol, pc, cc, T, na, n_col, zero_beta=false)
    ci = cc[] += 1
    cs = Symbol(:_crd, ci)
    inner = _emit_spmm_level_nf(levels, lvl + 1, p_var, pc, cc, T, na, n_col)
    quote
        _x_idx   = Int($cs[$p_var]) - Int(_origin_off) + 1
        _nnz_pos = $p_var
        $inner
    end
end

function _emit_spmm_lv_nf(::DeltaLevel, levels, lvl, p_var::Symbol, pc, cc, T, na, n_col, zero_beta=false)
    pi = pc[] += 1; ci = cc[] += 1
    ps = Symbol(:_pos, pi); cs = Symbol(:_crd, ci)
    lvar  = Symbol(:_i, lvl)
    corig = Symbol(:_corig, lvl)
    inner = _emit_spmm_level_nf(levels, lvl + 1, lvar, pc, cc, T, na, n_col)
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

# Fallback for unsupported levels in NNZ-first path: use column-first emitter.
function _emit_spmm_lv_nf(lv, levels, lvl, p_var, pc, cc, T, na, n_col, zero_beta=false)
    _emit_spmm_lv(lv, levels, lvl, p_var, pc, cc, T, na)
end

# ─── Kernel cache and launch ─────────────────────────────────────────────────

function _get_spmm_kernel(fmt::TensorFormat, ::Type{T}) where T
    key = (fmt, T, :spmm)
    haskey(_emitter_cache, key) && return _emitter_cache[key]

    body      = _emit_spmm_body(fmt, T)
    buf_names = _sparse_arg_names(fmt)
    arg_names = vcat(buf_names, [:_B, :_C, :_origin_off, :_n_outer, :_n_col, :_alpha, :_beta])
    fname     = gensym(:ust_spmm)

    kern = @eval begin
        @kernel inbounds=true function $fname($(arg_names...))
            $body
        end
        $fname
    end

    _emitter_cache[key] = kern
    return kern
end

# NNZ-first kernel with n_col baked in: cached per (fmt, T, :spmm_nf, n_col).
function _get_spmm_nf_kernel(fmt::TensorFormat, ::Type{T}, n_col::Int) where T
    key = (fmt, T, :spmm_nf, n_col)
    haskey(_emitter_cache, key) && return _emitter_cache[key]

    body      = _emit_spmm_body_nnzfirst(fmt, T, n_col)
    buf_names = _sparse_arg_names(fmt)
    arg_names = vcat(buf_names, [:_B, :_C, :_origin_off, :_n_outer, :_alpha, :_beta])
    fname     = gensym(:ust_spmm_nf)

    kern = @eval begin
        @kernel inbounds=true function $fname($(arg_names...))
            $body
        end
        $fname
    end

    _emitter_cache[key] = kern
    return kern
end

# Beta=0 NNZ-first kernel: no C read in write path (C = alpha * acc only).
# Cached per (fmt, T, :spmm_nf0, n_col). No _beta argument.
function _get_spmm_nf_beta0_kernel(fmt::TensorFormat, ::Type{T}, n_col::Int) where T
    key = (fmt, T, :spmm_nf0, n_col)
    haskey(_emitter_cache, key) && return _emitter_cache[key]

    body      = _emit_spmm_body_nnzfirst(fmt, T, n_col, true)  # zero_beta=true
    buf_names = _sparse_arg_names(fmt)
    arg_names = vcat(buf_names, [:_B, :_C, :_origin_off, :_n_outer, :_alpha])  # no _beta

    fname = gensym(:ust_spmm_nf0)
    kern = @eval begin
        @kernel inbounds=true function $fname($(arg_names...))
            $body
        end
        $fname
    end

    _emitter_cache[key] = kern
    return kern
end

# ─── Guarded NNZ-first kernels (CSR only) ────────────────────────────────────
#
# For CSR matrices where NNZ << n_rows (e.g. Cg in DCOPF: 375 NNZ in 30k rows),
# normal kernels waste bandwidth on empty rows (writing zeros or reading C for no-ops).
# Guarded variants skip the work entirely for empty rows.
#
# Two variants:
#   guarded beta=0 (spmm_nf0g): C[tid,:] = alpha*acc  — for sparse blocks with no prior write
#   guarded beta=1 (spmm_nfg):  C[tid,:] += alpha*acc  — for sparse accumulation onto
#                                                          a C already initialized by a dense block
#
# In _bbm_apply_diags! two-pass ordering:
#   Pass 1: dense blocks → beta=0/1, writes all rows (including zeros for their empty rows)
#   Pass 2: sparse blocks → guarded beta=1, adds contribution only for non-empty sparse rows;
#           rows empty in the sparse block retain the dense block's correctly-zeroed value.
#
# Hardcoded for standard 2-level CSR: _sparse_arg_names = [:_pos1, :_crd1, :_nzval].
# Activated via sparse_mm!(...; skip_empty_rows=true).

_is_csr_like_for_guard(fmt::TensorFormat) =
    length(fmt.levels) == 2 &&
    fmt.levels[1].second isa Union{DenseLevel,BatchLevel} &&
    fmt.levels[2].second isa CompressedLevel &&
    is_unique(fmt.levels[2].second)

function _emit_spmm_body_csr_guarded(::Type{T}, n_col::Int, zero_beta::Bool) where T
    acc_inits  = Expr(:block, [:($(Symbol(:_acc_, c)) = $(zero(T))) for c in 1:n_col]...)
    leaf       = Expr(:block, [:($(Symbol(:_acc_, c)) += _nzval[_nnz_pos] * _B[_x_idx, $(c)]) for c in 1:n_col]...)
    acc_writes = if zero_beta
        Expr(:block, [:(_C[_tid, $(c)] = _alpha * $(Symbol(:_acc_, c))) for c in 1:n_col]...)
    else
        Expr(:block, [:(_C[_tid, $(c)] = _alpha * $(Symbol(:_acc_, c)) + _beta * _C[_tid, $(c)]) for c in 1:n_col]...)
    end
    quote
        _tid = @index(Global, Linear)
        if _tid <= _n_outer
            _lo = Int(_pos1[_tid])     - Int(_origin_off)
            _hi = Int(_pos1[_tid + 1]) - Int(_origin_off)
            if _lo < _hi
                $acc_inits
                for _i2 in (_lo + 1):_hi
                    _x_idx   = Int(_crd1[_i2]) - Int(_origin_off) + 1
                    _nnz_pos = _i2
                    $leaf
                end
                $acc_writes
            end
        end
    end
end

function _get_spmm_nf_beta0_guarded_kernel(fmt::TensorFormat, ::Type{T}, n_col::Int) where T
    key = (fmt, T, :spmm_nf0g, n_col)
    haskey(_emitter_cache, key) && return _emitter_cache[key]

    body      = _emit_spmm_body_csr_guarded(T, n_col, true)
    arg_names = [:_pos1, :_crd1, :_nzval, :_B, :_C, :_origin_off, :_n_outer, :_alpha]
    fname     = gensym(:ust_spmm_nf0g)
    kern = @eval begin
        @kernel inbounds=true function $fname($(arg_names...))
            $body
        end
        $fname
    end
    _emitter_cache[key] = kern
    return kern
end

# Guarded beta=1 kernel: accumulates sparse contribution only for non-empty rows.
# Used in pass 2 of _bbm_apply_diags! when a dense block already initialized C.
function _get_spmm_nf_guarded_kernel(fmt::TensorFormat, ::Type{T}, n_col::Int) where T
    key = (fmt, T, :spmm_nfg, n_col)
    haskey(_emitter_cache, key) && return _emitter_cache[key]

    body      = _emit_spmm_body_csr_guarded(T, n_col, false)
    arg_names = [:_pos1, :_crd1, :_nzval, :_B, :_C, :_origin_off, :_n_outer, :_alpha, :_beta]
    fname     = gensym(:ust_spmm_nfg)
    kern = @eval begin
        @kernel inbounds=true function $fname($(arg_names...))
            $body
        end
        $fname
    end
    _emitter_cache[key] = kern
    return kern
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
    _, lv = levels[lvl]
    _emit_spmm_tiled_lv(lv, levels, lvl, p_var, pc, cc, T, tile_k)
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

function _emit_spmm_tiled_lv(lv, args...)
    error("EmitterBackend tiled SpMM: unsupported level $(typeof(lv)); convert to CSR/DCSR first")
end

function _emit_spmm_body_tiled(fmt::TensorFormat, ::Type{T}, tile_k::Int) where T
    levels = fmt.levels
    _, lv1 = levels[1]

    acc_inits = Expr(:block, [:($(Symbol(:_acc_, c)) = $(zero(T))) for c in 1:tile_k]...)

    if lv1 isa Union{DenseLevel,BatchLevel}
        # CSR-like: thread → row; output row index is _tid
        pc = Ref(0); cc = Ref(0)
        inner = _emit_spmm_tiled_inner(levels, 2, :_tid, pc, cc, T, tile_k)
        acc_writes = Expr(:block, [:(_C[_tid, _tile_start + $(c-1)] = _alpha * $(Symbol(:_acc_, c)) + _beta * _C[_tid, _tile_start + $(c-1)]) for c in 1:tile_k]...)
        quote
            _tid = @index(Global, Linear)
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
            _tid = @index(Global, Linear)
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
        error("EmitterBackend tiled SpMM: unsupported outermost level $(typeof(lv1))")
    end
end

# Tiled kernel: cached per (fmt, T, :spmm_tiled).  Uses runtime _n_col for the
# tile loop bound; TILE_K accumulators are compile-time constants.
function _get_spmm_tiled_kernel(fmt::TensorFormat, ::Type{T}) where T
    key = (fmt, T, :spmm_tiled)
    haskey(_emitter_cache, key) && return _emitter_cache[key]

    body      = _emit_spmm_body_tiled(fmt, T, _SPMM_TILE_K)
    buf_names = _sparse_arg_names(fmt)
    arg_names = vcat(buf_names, [:_B, :_C, :_origin_off, :_n_outer, :_n_col, :_alpha, :_beta])
    fname     = gensym(:ust_spmm_tiled)

    kern = @eval begin
        @kernel inbounds=true function $fname($(arg_names...))
            $body
        end
        $fname
    end

    _emitter_cache[key] = kern
    return kern
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

    _, lv1 = fmt.levels[1]
    is_coo_like = lv1 isa CompressedLevel && !is_unique(lv1)

    # COO-like patterns use @atomic += and cannot scale existing C values.
    if is_coo_like
        iszero(T_beta)  || error("EmitterBackend sparse_mm!: beta ≠ 0 not supported for COO format")
        isone(T_alpha)  || error("EmitterBackend sparse_mm!: alpha ≠ 1 not yet supported for COO format")
        fill!(nonzeros(u_C), zero(T))
    end

    sparse_bufs = _sparse_args(u_A)

    if !is_coo_like && n_col <= _SPMM_NCOL_INLINE_THRESHOLD
        # NNZ-first: n_col baked in → compiler unrolls column loop, registers hold accumulators.
        # Beta=0 path: omits C read entirely, eliminating fill!+read overhead for accumulation.
        if iszero(T_beta)
            if skip_empty_rows && _is_csr_like_for_guard(fmt)
                # Guarded beta=0: skip empty-row writes; caller pre-zeroed those positions
                # (or a prior dense SpMM already wrote correct zeros there).
                kern = _get_spmm_nf_beta0_guarded_kernel(fmt, T, n_col)
            else
                kern = _get_spmm_nf_beta0_kernel(fmt, T, n_col)
            end
            all_args = (sparse_bufs..., nonzeros(u_B), nonzeros(u_C), off, n_outer, T_alpha)
            kernel_obj = Base.invokelatest(kern, ka, 64)
            Base.invokelatest(kernel_obj, all_args...; ndrange=Int(n_outer))
        else
            if skip_empty_rows && _is_csr_like_for_guard(fmt)
                # Guarded beta=1: accumulate only for non-empty rows; empty rows retain their
                # existing C value (written correctly by prior dense SpMM).
                kern = _get_spmm_nf_guarded_kernel(fmt, T, n_col)
            else
                kern = _get_spmm_nf_kernel(fmt, T, n_col)
            end
            all_args = (sparse_bufs..., nonzeros(u_B), nonzeros(u_C), off, n_outer, T_alpha, T_beta)
            kernel_obj = Base.invokelatest(kern, ka, 64)
            Base.invokelatest(kernel_obj, all_args...; ndrange=Int(n_outer))
        end
    elseif !is_coo_like && n_col % _SPMM_TILE_K == 0
        # Tiled NNZ-first: TILE_K accumulators per strip, n_col/TILE_K NNZ passes.
        kern = _get_spmm_tiled_kernel(fmt, T)
        all_args = (sparse_bufs..., nonzeros(u_B), nonzeros(u_C), off, n_outer, Int32(n_col), T_alpha, T_beta)
        kernel_obj = Base.invokelatest(kern, ka, 64)
        Base.invokelatest(kernel_obj, all_args...; ndrange=Int(n_outer))
    else
        # Column-first: runtime n_col loop (fallback for COO and non-divisible k).
        kern = _get_spmm_kernel(fmt, T)
        all_args = (sparse_bufs..., nonzeros(u_B), nonzeros(u_C), off, n_outer, Int32(n_col), T_alpha, T_beta)
        kernel_obj = Base.invokelatest(kern, ka, 64)
        Base.invokelatest(kernel_obj, all_args...; ndrange=Int(n_outer))
    end

    return u_C
end

# ─── Fused BBM scatter kernels ────────────────────────────────────────────────
#
# Replace T*nb_r tiny copyto! calls (48 for T=24, nb_r=2) with a single kernel.
# diag scatter: diag_out[r, t] → y[(t-1)*d_period + r]  (0-indexed t, r)
# ramp  scatter: ramp[r, t]     → y[(t-1)*d_period + ramp_off + r]

@kernel inbounds=true function _bbm_scatter_diag_kernel!(_y, @Const(_diag), _d_period, _n_diag)
    gid = @index(Global, Linear) - 1
    t   = gid ÷ _n_diag
    r   = gid % _n_diag
    _y[t * _d_period + r + 1] = _diag[r + 1, t + 1]
end

@kernel inbounds=true function _bbm_scatter_ramp_kernel!(_y, @Const(_ramp), _d_period, _ramp_off, _n_ramp)
    gid = @index(Global, Linear) - 1
    t   = gid ÷ _n_ramp
    r   = gid % _n_ramp
    _y[t * _d_period + _ramp_off + r + 1] = _ramp[r + 1, t + 1]
end

function JLUST._bbm_scatter_diag!(y, diag_out, d_period, n_diag, T)
    ka = KernelAbstractions.get_backend(y)
    _bbm_scatter_diag_kernel!(ka, 256)(y, diag_out, Int(d_period), Int(n_diag);
                                       ndrange=Int(n_diag) * Int(T))
end

function JLUST._bbm_scatter_ramp!(y, ramp, d_period, ramp_off, n_ramp, T_ramp)
    ka = KernelAbstractions.get_backend(y)
    _bbm_scatter_ramp_kernel!(ka, 256)(y, ramp, Int(d_period), Int(ramp_off), Int(n_ramp);
                                       ndrange=Int(n_ramp) * Int(T_ramp))
end
