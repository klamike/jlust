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
                _C[_tid, _col] = _acc
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
                    _C[_y_idx, _col] = _acc
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

function _emit_spmm_body_nnzfirst(fmt::TensorFormat, ::Type{T}, n_col::Int) where T
    pc = Ref(0); cc = Ref(0)
    _, lv1 = fmt.levels[1]
    na = lv1 isa CompressedLevel && !is_unique(lv1)
    _emit_spmm_level_nf(fmt.levels, 1, nothing, pc, cc, T, na, n_col)
end

function _emit_spmm_level_nf(levels, lvl, p_var, pc, cc, T, needs_atomic, n_col)
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
    _emit_spmm_lv_nf(lv, levels, lvl, p_var, pc, cc, T, needs_atomic, n_col)
end

function _emit_spmm_lv_nf(::Union{DenseLevel,BatchLevel}, levels, lvl, ::Nothing, pc, cc, T, _, n_col)
    inner = _emit_spmm_level_nf(levels, lvl + 1, :_tid, pc, cc, T, false, n_col)
    acc_inits  = Expr(:block, [:($(Symbol(:_acc_, c)) = $(zero(T))) for c in 1:n_col]...)
    acc_writes = Expr(:block, [:(_C[_tid, $(c)] = $(Symbol(:_acc_, c))) for c in 1:n_col]...)
    quote
        _tid = @index(Global, Linear)
        if _tid <= _n_outer
            $acc_inits
            $inner
            $acc_writes
        end
    end
end

function _emit_spmm_lv_nf(lv::CompressedLevel, levels, lvl, ::Nothing, pc, cc, T, _, n_col)
    pc[] += 1; ci = cc[] += 1
    cs = Symbol(:_crd, ci)
    if is_unique(lv)
        inner = _emit_spmm_level_nf(levels, lvl + 1, :_tid, pc, cc, T, false, n_col)
        acc_inits  = Expr(:block, [:($(Symbol(:_acc_, c)) = $(zero(T))) for c in 1:n_col]...)
        acc_writes = Expr(:block, [:(_C[_y_idx, $(c)] = $(Symbol(:_acc_, c))) for c in 1:n_col]...)
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

function _emit_spmm_lv_nf(::CompressedLevel, levels, lvl, p_var::Symbol, pc, cc, T, na, n_col)
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

function _emit_spmm_lv_nf(::SingletonLevel, levels, lvl, p_var::Symbol, pc, cc, T, na, n_col)
    ci = cc[] += 1
    cs = Symbol(:_crd, ci)
    inner = _emit_spmm_level_nf(levels, lvl + 1, p_var, pc, cc, T, na, n_col)
    quote
        _x_idx   = Int($cs[$p_var]) - Int(_origin_off) + 1
        _nnz_pos = $p_var
        $inner
    end
end

function _emit_spmm_lv_nf(::DeltaLevel, levels, lvl, p_var::Symbol, pc, cc, T, na, n_col)
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
function _emit_spmm_lv_nf(lv, levels, lvl, p_var, pc, cc, T, na, n_col)
    _emit_spmm_lv(lv, levels, lvl, p_var, pc, cc, T, na)
end

# ─── Kernel cache and launch ─────────────────────────────────────────────────

function _get_spmm_kernel(fmt::TensorFormat, ::Type{T}) where T
    key = (fmt, T, :spmm)
    haskey(_emitter_cache, key) && return _emitter_cache[key]

    body      = _emit_spmm_body(fmt, T)
    buf_names = _sparse_arg_names(fmt)
    arg_names = vcat(buf_names, [:_B, :_C, :_origin_off, :_n_outer, :_n_col])
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
    arg_names = vcat(buf_names, [:_B, :_C, :_origin_off, :_n_outer])
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
        acc_writes = Expr(:block, [:(_C[_tid, _tile_start + $(c-1)] = $(Symbol(:_acc_, c))) for c in 1:tile_k]...)
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
        acc_writes = Expr(:block, [:(_C[_y_idx, _tile_start + $(c-1)] = $(Symbol(:_acc_, c))) for c in 1:tile_k]...)
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
    arg_names = vcat(buf_names, [:_B, :_C, :_origin_off, :_n_outer, :_n_col])
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
                           alpha=one(eltype(u_A)), beta=zero(eltype(u_A)))
    beta  == 0 || error("EmitterBackend sparse_mm!: beta ≠ 0 not yet supported")
    alpha == 1 || error("EmitterBackend sparse_mm!: alpha ≠ 1 not yet supported")

    fmt     = format(u_A)
    T       = eltype(u_A)
    ka      = KernelAbstractions.get_backend(nonzeros(u_A))
    off     = Int32(index_origin(u_A) isa OneBased ? 1 : 0)
    n_outer = Int32(_spmv_ndrange(u_A))
    n_col   = Int(extents(u_B)[2])

    _, lv1 = fmt.levels[1]
    is_coo_like = lv1 isa CompressedLevel && !is_unique(lv1)

    # COO-like patterns use @atomic += so C must be pre-zeroed.
    if is_coo_like
        fill!(nonzeros(u_C), zero(T))
    end

    sparse_bufs = _sparse_args(u_A)

    if !is_coo_like && n_col <= _SPMM_NCOL_INLINE_THRESHOLD
        # NNZ-first: n_col baked in → compiler unrolls column loop, registers hold accumulators.
        kern = _get_spmm_nf_kernel(fmt, T, n_col)
        all_args = (sparse_bufs..., nonzeros(u_B), nonzeros(u_C), off, n_outer)
        kernel_obj = Base.invokelatest(kern, ka, 64)
        Base.invokelatest(kernel_obj, all_args...; ndrange=Int(n_outer))
    elseif !is_coo_like && n_col % _SPMM_TILE_K == 0
        # Tiled NNZ-first: TILE_K accumulators per strip, n_col/TILE_K NNZ passes.
        # 8× fewer NNZ bandwidth passes than column-first for k=64.
        kern = _get_spmm_tiled_kernel(fmt, T)
        all_args = (sparse_bufs..., nonzeros(u_B), nonzeros(u_C), off, n_outer, Int32(n_col))
        kernel_obj = Base.invokelatest(kern, ka, 64)
        Base.invokelatest(kernel_obj, all_args...; ndrange=Int(n_outer))
    else
        # Column-first: runtime n_col loop (fallback for COO and non-divisible k).
        kern = _get_spmm_kernel(fmt, T)
        all_args = (sparse_bufs..., nonzeros(u_B), nonzeros(u_C), off, n_outer, Int32(n_col))
        kernel_obj = Base.invokelatest(kern, ka, 64)
        Base.invokelatest(kernel_obj, all_args...; ndrange=Int(n_outer))
    end

    return u_C
end
