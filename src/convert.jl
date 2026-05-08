# ─── TensorDecomposer ─────────────────────────────────────────────────────────
# Traverses a USTensor and calls a functor for every stored non-zero element.
# Mirrors Python's TensorDecomposer._iterate (0-based level arithmetic throughout).

struct TensorDecomposer{F}
    u       :: USTensor
    functor :: F
end

function _decompose!(d::TensorDecomposer{F}, lvls::Vector{Int}, idx::Int, p::Int, lasta::Int) where {F}
    u   = d.u
    fmt = u.format
    nlevels = length(fmt.levels)

    if idx > nlevels
        d.functor(lvl2dim(fmt, lvls), u.val[p + 1])
        return
    end

    _, lv = fmt.levels[idx]
    origin_offset = index_origin(u) isa OneBased ? 1 : 0
    _decompose_level!(d, lvls, idx, p, lasta, origin_offset, lv)
end

function _decompose_level!(d::TensorDecomposer, lvls, idx, p, lasta, origin_offset, ::Union{DenseLevel,BatchLevel})
    fmt = d.u.format
    sz  = d.u.extents[_lvl_dim(fmt, idx)]
    for i in 0:(sz - 1)
        lvls[idx] = i
        _decompose!(d, lvls, idx + 1, p * sz + i, lasta)
    end
end

function _decompose_level!(d::TensorDecomposer, lvls, idx, p, lasta, origin_offset, ::CompressedLevel)
    u   = d.u
    fmt = u.format
    key, _ = fmt.levels[idx]
    new_lasta = (key isa LevelExpr && (key.op == :add || key.op == :sub)) ? idx : lasta
    pos = u.pos_buffers[idx]
    crd = u.crd_buffers[idx]
    lo  = Int(pos[p + 1]) - origin_offset
    hi  = Int(pos[p + 2]) - origin_offset
    for i in lo:(hi - 1)
        lvls[idx] = Int(crd[i + 1]) - origin_offset
        _decompose!(d, lvls, idx + 1, i, new_lasta)
    end
end

function _decompose_level!(d::TensorDecomposer, lvls, idx, p, lasta, origin_offset, ::SingletonLevel)
    crd = d.u.crd_buffers[idx]
    lvls[idx] = Int(crd[p + 1]) - origin_offset
    _decompose!(d, lvls, idx + 1, p, lasta)
end

function _decompose_level!(d::TensorDecomposer, lvls, idx, p, lasta, origin_offset, ::RangeLevel)
    u   = d.u
    fmt = u.format
    @assert lasta > 0 "RangeLevel encountered without prior add/sub level"
    add_key, _ = fmt.levels[lasta]
    key, _     = fmt.levels[idx]
    isI        = (key == add_key.rhs)
    expr_other = isI ? add_key.lhs : add_key.rhs
    di  = findfirst(==(key),        fmt.dimensions)
    dj  = findfirst(==(expr_other), fmt.dimensions)
    szi = u.extents[di]
    szj = u.extents[dj]
    lsz = lvls[lasta]
    if add_key.op == :add
        off = lsz
        lo  = max(0, off - szj + 1)
        hi  = min(szi, off + 1)
    else  # :sub
        off = isI ? -lsz : lsz
        lo  = max(0, off)
        hi  = min(szi, szj + off)
    end
    for i in lo:(hi - 1)
        lvls[idx] = i
        _decompose!(d, lvls, idx + 1, p * szi + i, 0)
    end
end

function _decompose_level!(d::TensorDecomposer, lvls, idx, p, lasta, origin_offset, ::DeltaLevel)
    u   = d.u
    pos = u.pos_buffers[idx]
    crd = u.crd_buffers[idx]
    lo  = Int(pos[p + 1]) - origin_offset
    hi  = Int(pos[p + 2]) - origin_offset
    corig = 0
    for i in lo:(hi - 1)
        corig += Int(crd[i + 1])
        lvls[idx] = corig
        _decompose!(d, lvls, idx + 1, i, lasta)
        corig += 1
    end
end

# AbstractLevelFormat: delegate to level_step for single-element-per-position levels.
function _decompose_level!(d::TensorDecomposer, lvls, idx, p, lasta, origin_offset, lv::AbstractLevelFormat)
    u  = d.u
    nz = level_has_nzval(lv) ? SparseArrays.nonzeros(u) : nothing
    col_1based, _ = level_step(lv, p + 1, nz)
    lvls[idx] = col_1based - 1   # 0-based for composer arithmetic
    _decompose!(d, lvls, idx + 1, p, lasta)
end

function run!(d::TensorDecomposer)
    u = d.u
    lvls = zeros(Int, length(u.format.levels))
    _decompose!(d, lvls, 1, 0, 0)
end

# ─── TensorComposer ───────────────────────────────────────────────────────────
# Two-pass builder: scan sizes, allocate, then fill.
# `indices` is (ndim × nse), `values` is (nse,), both on CPU.
# `is_sorted` means lexicographic on level coordinates (not dim coordinates).

mutable struct TensorComposer{I}
    fmt      :: TensorFormat
    extents  :: Vector{Int}   # target tensor extents, indexed by dimension
    indices  :: Matrix{I}     # (nlvl × nse) level-coordinate matrix
    vals     :: Vector        # nse values
    pos_sz   :: Vector{I}
    crd_sz   :: Vector{I}
    val_sz   :: Int
    pos_bufs :: Dict{Int,Vector{I}}
    crd_bufs :: Dict{Int,Vector{I}}
    val_buf  :: Vector
end

function TensorComposer(fmt::TensorFormat, extents::NTuple, indices::Matrix{I}, vals::Vector) where {I}
    nlvl = length(fmt.levels)
    TensorComposer{I}(
        fmt,
        collect(Int, extents),
        indices,
        vals,
        zeros(I, nlvl),
        zeros(I, nlvl),
        0,
        Dict{Int,Vector{I}}(),
        Dict{Int,Vector{I}}(),
        [],
    )
end

function _lvl_extent(c::TensorComposer, idx::Int)
    key, _ = c.fmt.levels[idx]
    dim    = _find_key_dim(c.fmt.dimensions, key)
    c.extents[dim]
end

function _get_prop(c::TensorComposer, idx::Int)
    _, lv = c.fmt.levels[idx]
    lv
end

function _append_pos!(c::TensorComposer, idx::Int, pos::Integer, repeat::Int, is_insert::Bool)
    if c.pos_sz[idx] == 0
        if is_insert
            c.pos_bufs[idx][1] = 0
        end
        c.pos_sz[idx] = 1
    end
    if is_insert
        start = Int(c.pos_sz[idx])
        for r in 1:repeat
            c.pos_bufs[idx][start + r] = pos
        end
    end
    c.pos_sz[idx] += repeat
end

function _append_crd!(c::TensorComposer, idx::Int, crd::Integer, repeat::Int, is_insert::Bool)
    if is_insert
        start = Int(c.crd_sz[idx])
        for r in 1:repeat
            c.crd_bufs[idx][start + r] = crd
        end
    end
    c.crd_sz[idx] += repeat
end

function _append_val!(c::TensorComposer, val, repeat::Int, is_insert::Bool)
    if is_insert
        start = c.val_sz
        for r in 1:repeat
            c.val_buf[start + r] = val
        end
    end
    c.val_sz += repeat
end

function _segment_builder!(c::TensorComposer, idx::Int, full::Int, repeat::Int, is_insert::Bool)
    nlvl = length(c.fmt.levels)
    if idx > nlvl
        _append_val!(c, zero(eltype(c.val_buf)), repeat, is_insert)
        return
    end
    _segment_builder_level!(c, idx, full, repeat, is_insert, _get_prop(c, idx))
end

function _segment_builder_level!(c::TensorComposer, idx, full, repeat, is_insert, ::Union{DenseLevel,BatchLevel,RangeLevel})
    sz = _lvl_extent(c, idx)
    @assert sz >= full
    _segment_builder!(c, idx + 1, 0, repeat * (sz - full), is_insert)
end

function _segment_builder_level!(c::TensorComposer, idx, full, repeat, is_insert, ::Union{CompressedLevel,DeltaLevel})
    _append_pos!(c, idx, c.crd_sz[idx], repeat, is_insert)
end

function _segment_builder_level!(c::TensorComposer, idx, full, repeat, is_insert, ::SingletonLevel)
    # nothing — singleton emits no segment padding
end

function _segment_builder_level!(c::TensorComposer, idx, full, repeat, is_insert, lv::AbstractLevelFormat)
    error("convert_format: cannot compose into custom level $(typeof(lv)) via COO intermediate. " *
          "Custom AbstractLevelFormat targets are not supported by TensorComposer.")
end

function _insert_builder!(c::TensorComposer, idx::Int, lo::Int, hi::Int, is_insert::Bool)
    nlvl = length(c.fmt.levels)
    if idx > nlvl
        @assert lo < hi
        _append_val!(c, c.vals[lo + 1], 1, is_insert)
        return
    end
    full   = 0
    seg_lo = lo
    while seg_lo < hi
        crd = c.indices[idx, seg_lo + 1]
        seg = seg_lo + 1
        lv  = _get_prop(c, idx)
        if is_unique(lv)
            while seg < hi && c.indices[idx, seg + 1] == crd
                seg += 1
            end
        end
        _insert_builder_level!(c, idx, crd, full, is_insert, lv)
        full   = Int(crd) + 1
        _insert_builder!(c, idx + 1, seg_lo, seg, is_insert)
        seg_lo = seg
    end
    _segment_builder!(c, idx, full, 1, is_insert)
end

function _insert_builder_level!(c::TensorComposer, idx, crd, full, is_insert, ::Union{DenseLevel,BatchLevel,RangeLevel})
    @assert crd >= full
    if crd > full
        _segment_builder!(c, idx + 1, 0, crd - full, is_insert)
    end
end

function _insert_builder_level!(c::TensorComposer, idx, crd, full, is_insert, ::Union{CompressedLevel,SingletonLevel})
    _append_crd!(c, idx, crd, 1, is_insert)
end

function _insert_builder_level!(c::TensorComposer, idx, crd, full, is_insert, lv::AbstractLevelFormat)
    error("convert_format: cannot insert into custom level $(typeof(lv)) via COO intermediate. " *
          "Custom AbstractLevelFormat targets are not supported by TensorComposer.")
end

function _insert_builder_level!(c::TensorComposer, idx, crd, full, is_insert, lv::DeltaLevel)
    mDelta = (1 << lv.bits) - 1
    delta  = Int(crd) - full
    while delta > mDelta
        _append_crd!(c, idx, mDelta, 1, is_insert)
        _append_val!(c, zero(eltype(c.val_buf)), 1, is_insert)
        delta -= mDelta + 1
    end
    _append_crd!(c, idx, delta, 1, is_insert)
end

function run!(c::TensorComposer{I}, val_type::Type) where {I}
    nlvl = length(c.fmt.levels)
    _, nse = size(c.indices)

    # Pass 1: scan to count sizes.
    _insert_builder!(c, 1, 0, nse, false)

    # Allocate buffers.
    for idx in 1:nlvl
        c.pos_bufs[idx] = Vector{I}(undef, Int(c.pos_sz[idx]))
        c.crd_bufs[idx] = Vector{I}(undef, Int(c.crd_sz[idx]))
        c.pos_sz[idx] = 0
        c.crd_sz[idx] = 0
    end
    c.val_buf = Vector{val_type}(undef, c.val_sz)
    c.val_sz  = 0

    # Pass 2: fill buffers.
    _insert_builder!(c, 1, 0, nse, true)

    return c.pos_bufs, c.crd_bufs, c.val_buf
end

# ─── convert_format ───────────────────────────────────────────────────────────

"""
    convert_format(u::USTensor, fmt::TensorFormat; index_type=nothing, value_type=nothing)

Convert `u` to `fmt` via a CPU COO intermediate (mirrors Python's TensorConverter).
The output tensor has the same index origin as the source.
"""
function convert_format(u::USTensor{T,I,N,VA,VI,O},
                        fmt::TensorFormat;
                        index_type::Union{Type,Nothing}=nothing,
                        value_type::Union{Type,Nothing}=nothing) where {T,I,N,VA,VI,O}
    Iout = something(index_type, I)
    Tout = something(value_type, T)

    ndim = N
    nse  = SparseArrays.nnz(u)
    dim_indices = Matrix{Iout}(undef, ndim, nse)
    vals        = Vector{Tout}(undef, nse)
    pos_ref     = Ref(0)

    function visit(dims::Vector{Int}, val)
        val == zero(T) && return   # drop explicit zeros
        p = pos_ref[]
        pos_ref[] = p + 1
        for d in 1:ndim
            dim_indices[d, p + 1] = Iout(dims[d])
        end
        vals[p + 1] = Tout(val)
    end

    run!(TensorDecomposer(u, visit))

    nse_actual = pos_ref[]
    lindices   = dim_indices[:, 1:nse_actual]
    lvals      = vals[1:nse_actual]

    # Translate dim→lvl coordinates (0-based).
    nlvl = length(fmt.levels)
    if !fmt.is_identity
        lvl_matrix = Matrix{Iout}(undef, nlvl, nse_actual)
        for j in 1:nse_actual
            res = dim2lvl(fmt, [Int(lindices[d, j]) for d in 1:ndim])
            for l in 1:nlvl
                lvl_matrix[l, j] = Iout(res[l])
            end
        end
        lindices   = lvl_matrix
        # Must re-sort since level layout may differ from dim layout.
        order      = _lexsort_cols(lindices)
        lindices   = lindices[:, order]
        lvals      = lvals[order]
    else
        # is_identity: level coords == dim coords in the same order, but
        # only sorted if source was ordered+identity.
        sformat = u.format
        if !(sformat.is_ordered && sformat.is_identity)
            order   = _lexsort_cols(lindices)
            lindices = lindices[:, order]
            lvals    = lvals[order]
        end
    end

    # Compose into target format.
    comp = TensorComposer(fmt, u.extents, lindices, lvals)
    pos_bufs, crd_bufs, val_buf = run!(comp, Tout)

    # Filter empty buffers according to format level types.
    final_pos = Dict{Int,Vector{Iout}}()
    final_crd = Dict{Int,Vector{Iout}}()
    for (idx, (_, lv)) in enumerate(fmt.levels)
        if lv isa CompressedLevel || lv isa DeltaLevel
            final_pos[idx] = pos_bufs[idx]
            final_crd[idx] = crd_bufs[idx]
        elseif lv isa SingletonLevel
            final_crd[idx] = crd_bufs[idx]
        end
        # DenseLevel, BatchLevel, RangeLevel: no buffers
    end

    # Composer always produces 0-based indices; adjust to OneBased if needed.
    if O === OneBased
        off = one(Iout)
        for buf in values(final_pos); buf .+= off; end
        for buf in values(final_crd); buf .+= off; end
    end

    USTensor{Tout,Iout,N,Vector{Tout},Vector{Iout},O}(
        u.extents,
        fmt,
        final_pos,
        final_crd,
        val_buf,
        nothing,
    )
end

# ─── convert_index_type / convert_value_type ─────────────────────────────────

convert_index_type(u::USTensor, ::Type{I2}) where {I2} =
    convert_format(u, u.format; index_type=I2)

convert_value_type(u::USTensor, ::Type{T2}) where {T2} =
    convert_format(u, u.format; value_type=T2)

# ─── Base.convert → SparseMatrixCSC ──────────────────────────────────────────

function Base.convert(::Type{SparseMatrixCSC{T,I}}, u::USTensor) where {T,I}
    u2 = convert_format(u, Formats.CSC; index_type=I, value_type=T)
    # CSC: pos[2]=colptr, crd[2]=rowind
    colptr = positions(u2, 2)
    rowind = coordinates(u2, 2)
    nzval  = SparseArrays.nonzeros(u2)
    # Ensure 1-based.
    off = index_origin(u2) isa OneBased ? I(0) : I(1)
    SparseMatrixCSC{T,I}(u2.extents[1], u2.extents[2],
                         colptr .+ off, rowind .+ off, nzval)
end

Base.convert(::Type{SparseMatrixCSC}, u::USTensor{T,I}) where {T,I} =
    convert(SparseMatrixCSC{T,I}, u)

# ─── Helpers ─────────────────────────────────────────────────────────────────

# Lexicographic sort on columns of a matrix (primary = row 1).
# Returns permutation vector.
function _lexsort_cols(A::Matrix)
    nrows, ncols = size(A)
    ncols == 0 && return Int[]
    keys = [ntuple(r -> A[r, j], nrows) for j in 1:ncols]
    sortperm(keys)
end

# materialize stub — concrete method added by CUDAExt
function materialize end
