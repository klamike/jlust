# ─── EmitterBackend: SDDMM ───────────────────────────────────────────────────
#
# Sampled dense-dense matrix multiply:
#   C ← alpha * (A * B) ∘ sparsity(C) + beta * C
#
# C is sparse (the mask and result); A and B are dense.
# Parallelism: one thread per outer fiber of C (row for CSR, fiber for DCSR,
# NNZ for COO). The inner k-loop is sequential within each thread.
#
# Shares _sparse_arg_names / _sparse_args / _spmv_ndrange from spmv.jl.

function JLUST.supports_backend(::EmitterBackend, op::SDDMMOp)
    _is_dense_fmt(op.A) && _is_dense_fmt(op.B) && _is_emittable(op.C)
end

# ─── Body emitter ─────────────────────────────────────────────────────────────

function _emit_sddmm_body(fmt::TensorFormat, ::Type{T}) where T
    pc = Ref(0); cc = Ref(0)
    _emit_sddmm_level(fmt.levels, 1, nothing, pc, cc, T)
end

function _emit_sddmm_level(levels, lvl, p_var, pc, cc, T)
    if lvl > length(levels)
        return quote
            _dot = $(zero(T))
            for _k in 1:_n_inner
                _dot += _A[_row_idx, _k] * _B[_k, _col_idx]
            end
            _nzval[_nnz_pos] = _alpha * _dot + _beta * _nzval[_nnz_pos]
        end
    end
    _, lv = levels[lvl]
    _emit_sddmm_lv(lv, levels, lvl, p_var, pc, cc, T)
end

# DenseLevel (outermost) → thread = row
function _emit_sddmm_lv(::Union{DenseLevel,BatchLevel}, levels, lvl, ::Nothing, pc, cc, T)
    inner = _emit_sddmm_level(levels, lvl + 1, :_tid, pc, cc, T)
    quote
        _tid = @index(Global, Linear)
        if _tid <= _n_outer
            _row_idx = _tid
            $inner
        end
    end
end

# DenseLevel (non-outermost) → dense inner loop (uncommon for SDDMM)
function _emit_sddmm_lv(::Union{DenseLevel,BatchLevel}, levels, lvl, p_var::Symbol, pc, cc, T)
    sz   = Symbol(:_sz, lvl)
    lv2  = Symbol(:_i, lvl)
    inner = _emit_sddmm_level(levels, lvl + 1, lv2, pc, cc, T)
    quote
        for $lv2 in 1:$sz
            $inner
        end
    end
end

# CompressedLevel (outermost)
#   unique   → fiber-parallel (DCSR-like): one thread per non-empty row
#   non-unique → NNZ-parallel (COO-like): one thread per NNZ
function _emit_sddmm_lv(lv::CompressedLevel, levels, lvl, ::Nothing, pc, cc, T)
    pc[] += 1; ci = cc[] += 1
    cs = Symbol(:_crd, ci)
    if is_unique(lv)
        inner = _emit_sddmm_level(levels, lvl + 1, :_tid, pc, cc, T)
        quote
            _tid = @index(Global, Linear)
            if _tid <= _n_outer
                _row_idx = Int($cs[_tid]) - Int(_origin_off) + 1
                $inner
            end
        end
    else
        inner = _emit_sddmm_level(levels, lvl + 1, :_tid, pc, cc, T)
        quote
            _tid = @index(Global, Linear)
            if _tid <= _n_outer
                _row_idx = Int($cs[_tid]) - Int(_origin_off) + 1
                _nnz_pos = _tid
                $inner
            end
        end
    end
end

# CompressedLevel (non-outermost) → inner fiber loop (column scan for CSR)
function _emit_sddmm_lv(::CompressedLevel, levels, lvl, p_var::Symbol, pc, cc, T)
    pi = pc[] += 1; ci = cc[] += 1
    ps = Symbol(:_pos, pi); cs = Symbol(:_crd, ci)
    lvar = Symbol(:_i, lvl)
    inner = _emit_sddmm_level(levels, lvl + 1, lvar, pc, cc, T)
    quote
        _lo = Int($ps[$p_var])     - Int(_origin_off)
        _hi = Int($ps[$p_var + 1]) - Int(_origin_off)
        for $lvar in (_lo + 1):_hi
            _col_idx = Int($cs[$lvar]) - Int(_origin_off) + 1
            _nnz_pos = $lvar
            $inner
        end
    end
end

# SingletonLevel → one coordinate per position (COO col index)
function _emit_sddmm_lv(::SingletonLevel, levels, lvl, p_var::Symbol, pc, cc, T)
    ci = cc[] += 1
    cs = Symbol(:_crd, ci)
    inner = _emit_sddmm_level(levels, lvl + 1, p_var, pc, cc, T)
    quote
        _col_idx = Int($cs[$p_var]) - Int(_origin_off) + 1
        _nnz_pos = $p_var
        $inner
    end
end

function _emit_sddmm_lv(::RangeLevel, levels, lvl, _, pc, cc, T)
    error("EmitterBackend SDDMM: RangeLevel not supported. Convert C to CSR or COO first.")
end

function _emit_sddmm_lv(::DeltaLevel, levels, lvl, _, pc, cc, T)
    error("EmitterBackend SDDMM: DeltaLevel not supported. Convert C to CSR or COO first.")
end

# AbstractLevelFormat (custom inner level) → delegate to level_step hook.
function _emit_sddmm_lv(lv::AbstractLevelFormat, levels, lvl, p_var::Symbol, pc, cc, T)
    nz_sym = JLUST.level_has_nzval(lv) ? :_nzval : :nothing
    inner  = _emit_sddmm_level(levels, lvl + 1, p_var, pc, cc, T)
    quote
        _p1 = Int($p_var) - Int(_origin_off) + 1
        (_col_idx, _) = JLUST.level_step($lv, _p1, $nz_sym)
        _nnz_pos = $p_var
        $inner
    end
end

function _emit_sddmm_lv(lv::AbstractLevelFormat, levels, lvl, ::Nothing, pc, cc, T)
    error("EmitterBackend SDDMM: $(typeof(lv)) cannot be the outermost level; pair with DenseLevel.")
end

# ─── Kernel cache and launch ──────────────────────────────────────────────────

function _get_sddmm_kernel(fmt::TensorFormat, ::Type{T}) where T
    key = (fmt, T, :sddmm)
    haskey(_emitter_cache, key) && return _emitter_cache[key]

    body      = _emit_sddmm_body(fmt, T)
    buf_names = _sparse_arg_names(fmt)
    arg_names = vcat(buf_names, [:_A, :_B, :_alpha, :_beta, :_origin_off, :_n_outer, :_n_inner])
    fname     = gensym(:ust_sddmm)

    kern = @eval begin
        @kernel inbounds=true function $fname($(arg_names...))
            $body
        end
        $fname
    end

    _emitter_cache[key] = kern
    return kern
end

# ─── sparse_sddmm! ────────────────────────────────────────────────────────────

function JLUST.sparse_sddmm!(::EmitterBackend,
                               u_A::USTensor{T}, u_B::USTensor, u_C::USTensor;
                               alpha=one(T), beta=zero(T)) where T
    fmt     = format(u_C)
    ka      = KernelAbstractions.get_backend(nonzeros(u_C))
    off     = Int32(index_origin(u_C) isa OneBased ? 1 : 0)
    n_outer = Int32(_spmv_ndrange(u_C))
    n_inner = Int32(size(nonzeros(u_A), 2))   # cols of A = shared inner dimension

    kern = _get_sddmm_kernel(fmt, T)

    sparse_bufs = _sparse_args(u_C)
    all_args    = (sparse_bufs..., nonzeros(u_A), nonzeros(u_B),
                   T(alpha), T(beta), off, n_outer, n_inner)

    kernel_obj = Base.invokelatest(kern, ka, 64)
    Base.invokelatest(kernel_obj, all_args...; ndrange=Int(n_outer))

    return u_C
end

function JLUST.sparse_sddmm!(u_A::USTensor, u_B::USTensor, u_C::USTensor;
                               backend=EmitterBackend(), kw...)
    JLUST.sparse_sddmm!(backend, u_A, u_B, u_C; kw...)
end
