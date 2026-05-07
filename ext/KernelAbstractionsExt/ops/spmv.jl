# ─── Emitter: arg name/value collection ───────────────────────────────────────

# Symbolic arg names matching the order the emitter expects for the sparse A.
function _sparse_arg_names(fmt::TensorFormat)
    names = Symbol[]
    pc = Ref(0); cc = Ref(0)
    for (_, lv) in fmt.levels
        if lv isa Union{CompressedLevel,DeltaLevel}
            push!(names, Symbol(:_pos, pc[] += 1))
            push!(names, Symbol(:_crd, cc[] += 1))
        elseif lv isa SingletonLevel
            push!(names, Symbol(:_crd, cc[] += 1))
        end
    end
    push!(names, :_nzval)
    names
end

# Actual arrays in the same order as _sparse_arg_names.
# Outermost CompressedLevel (DCSR, COO) has no pos buffer; pass a typed empty
# array as a placeholder — the emitter never generates an access to _pos1.
function _sparse_args(u::USTensor{T,I,N,VA,VI,O}) where {T,I,N,VA,VI,O}
    dummy_pos = VI(undef, 0)
    args = AbstractArray[]
    for (lvl, (_, lv)) in enumerate(u.format.levels)
        if lv isa Union{CompressedLevel,DeltaLevel}
            pos = has_positions(u, lvl) ? positions(u, lvl) : dummy_pos
            push!(args, pos, coordinates(u, lvl))
        elseif lv isa SingletonLevel
            push!(args, coordinates(u, lvl))
        end
    end
    push!(args, nonzeros(u))
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
# p_var:      Symbol name of the current 1-based fiber position variable,
#             or nothing for the outermost level (becomes thread index).
# needs_atomic: true when y writes must use @atomic (e.g. COO-like patterns).

function _emit_spmv_body(fmt::TensorFormat, ::Type{T}) where T
    pc = Ref(0); cc = Ref(0)
    _, lv1 = fmt.levels[1]
    na = lv1 isa CompressedLevel && !is_unique(lv1)   # COO-like → atomic
    _emit_spmv_level(fmt.levels, 1, nothing, pc, cc, T, na)
end

function _emit_spmv_level(levels, lvl, p_var, pc, cc, T, needs_atomic) # TODO: separate atomic vs non-atomic?
    if lvl > length(levels)
        # Leaf: accumulate or atomic-write.
        return needs_atomic ?
            :(KernelAbstractions.@atomic _y[_y_idx] += _nzval[_nnz_pos] * _x[_x_idx]) :
            :(_acc += _nzval[_nnz_pos] * _x[_x_idx])
    end
    _, lv = levels[lvl]
    _emit_spmv_lv(lv, levels, lvl, p_var, pc, cc, T, needs_atomic)
end

# DenseLevel / BatchLevel (outermost) → thread = dense row index
function _emit_spmv_lv(::Union{DenseLevel,BatchLevel}, levels, lvl, ::Nothing, pc, cc, T, _)
    inner = _emit_spmv_level(levels, lvl + 1, :_tid, pc, cc, T, false)
    quote
        _tid = @index(Global, Linear)
        if _tid <= _n_outer
            _acc  = $(zero(T))
            $inner
            _y[_tid] = _acc
        end
    end
end

# DenseLevel / BatchLevel (non-outermost) → dense loop (uncommon for SpMV)
function _emit_spmv_lv(::Union{DenseLevel,BatchLevel}, levels, lvl, p_var::Symbol, pc, cc, T, na)
    sz  = Symbol(:_sz, lvl)
    lv2 = Symbol(:_i, lvl)
    inner = _emit_spmv_level(levels, lvl + 1, lv2, pc, cc, T, na)
    quote
        for $lv2 in 1:$sz
            $inner
        end
    end
end

# CompressedLevel (outermost)
#   unique   → fiber-parallel (DCSR-like): one thread per non-empty row, no atomics
#   non-unique → NNZ-parallel (COO-like): one thread per NNZ, atomic y update
function _emit_spmv_lv(lv::CompressedLevel, levels, lvl, ::Nothing, pc, cc, T, _)
    pc[] += 1; ci = cc[] += 1
    cs = Symbol(:_crd, ci)
    if is_unique(lv)
        inner = _emit_spmv_level(levels, lvl + 1, :_tid, pc, cc, T, false)
        quote
            _tid = @index(Global, Linear)
            if _tid <= _n_outer
                _y_idx = Int($cs[_tid]) - Int(_origin_off) + 1   # 1-based row
                _acc   = $(zero(T))
                $inner
                _y[_y_idx] = _acc
            end
        end
    else
        # Non-unique: thread = NNZ index within the single outer fiber.
        # Row index comes directly from crd (not from a fiber pos lookup).
        inner = _emit_spmv_level(levels, lvl + 1, :_tid, pc, cc, T, true)
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
function _emit_spmv_lv(::CompressedLevel, levels, lvl, p_var::Symbol, pc, cc, T, na)
    pi = pc[] += 1; ci = cc[] += 1
    ps = Symbol(:_pos, pi); cs = Symbol(:_crd, ci)
    lvar = Symbol(:_i, lvl)
    inner = _emit_spmv_level(levels, lvl + 1, lvar, pc, cc, T, na)
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
function _emit_spmv_lv(::SingletonLevel, levels, lvl, p_var::Symbol, pc, cc, T, na)
    ci = cc[] += 1
    cs = Symbol(:_crd, ci)
    inner = _emit_spmv_level(levels, lvl + 1, p_var, pc, cc, T, na)
    quote
        _x_idx   = Int($cs[$p_var]) - Int(_origin_off) + 1
        _nnz_pos = $p_var
        $inner
    end
end

# DeltaLevel (non-outermost) → accumulated delta decode + inner loop
function _emit_spmv_lv(::DeltaLevel, levels, lvl, p_var::Symbol, pc, cc, T, na)
    pi = pc[] += 1; ci = cc[] += 1
    ps = Symbol(:_pos, pi); cs = Symbol(:_crd, ci)
    lvar   = Symbol(:_i, lvl)
    corig  = Symbol(:_corig, lvl)
    inner  = _emit_spmv_level(levels, lvl + 1, lvar, pc, cc, T, na)
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

# RangeLevel: paired add/sub dimension reconstruction (DIA-like)
# Deferred to Phase 5 — throws a clear error for now.
function _emit_spmv_lv(::RangeLevel, levels, lvl, p_var, pc, cc, T, na)
    error("EmitterBackend SpMV: RangeLevel (DIA-style) kernels not yet emitted. " *
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
    key = (fmt, T, :spmv)
    haskey(_emitter_cache, key) && return _emitter_cache[key]

    body      = _emit_spmv_body(fmt, T)
    buf_names = _sparse_arg_names(fmt)
    arg_names = vcat(buf_names, [:_x, :_y, :_origin_off, :_n_outer])
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
                          alpha=one(eltype(u_A)), beta=zero(eltype(u_A)))
    beta  == 0 || error("EmitterBackend sparse_mv!: beta ≠ 0 not yet supported")
    alpha == 1 || error("EmitterBackend sparse_mv!: alpha ≠ 1 not yet supported")

    fmt     = format(u_A)
    T       = eltype(u_A)
    ka      = KernelAbstractions.get_backend(nonzeros(u_A))
    off     = Int32(index_origin(u_A) isa OneBased ? 1 : 0)

    # Dispatch: COO-like (sorted, non-unique outer) → chunked kernel to reduce
    # atomic contention.  All other formats → general recursive emitter.
    _, lv1 = fmt.levels[1]
    is_coo_like = lv1 isa CompressedLevel && !is_unique(lv1)

    if is_coo_like
        fill!(nonzeros(u_y), zero(T))
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
        n_outer     = Int32(_spmv_ndrange(u_A))
        sparse_bufs = _sparse_args(u_A)

        # Try CSR-vector hook: DenseLevel outermost + unique CompressedLevel inner.
        # CUDAExt overrides with a vector (multi-thread-per-row) kernel.
        dispatched = false
        if lv1 isa Union{DenseLevel,BatchLevel} && length(fmt.levels) >= 2
            lv2 = fmt.levels[2][2]
            if lv2 isa CompressedLevel && is_unique(lv2)
                dispatched = JLUST._csr_spmv_specialized!(
                    sparse_bufs[1], sparse_bufs[2],
                    nonzeros(u_A), nonzeros(u_x), nonzeros(u_y), off, n_outer)
            end
        end

        if !dispatched
            kern       = _get_spmv_kernel(fmt, T)
            all_args   = (sparse_bufs..., nonzeros(u_x), nonzeros(u_y), off, n_outer)
            kernel_obj = Base.invokelatest(kern, ka, 64)
            Base.invokelatest(kernel_obj, all_args...; ndrange=Int(n_outer))
        end
    end

    return u_y
end
