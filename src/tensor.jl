# ─── Index origin ─────────────────────────────────────────────────────────────

abstract type AbstractIndexOrigin end
struct OneBased  <: AbstractIndexOrigin end
struct ZeroBased <: AbstractIndexOrigin end

# ─── Memory space ─────────────────────────────────────────────────────────────

abstract type AbstractMemorySpace end
struct CPUMemory <: AbstractMemorySpace end
struct GPUMemory <: AbstractMemorySpace end

# ─── Device descriptors ───────────────────────────────────────────────────────

struct CPUDevice end
struct CUDADevice
    id::Int
end
CUDADevice() = CUDADevice(0)

# ─── AbstractUSTensor ─────────────────────────────────────────────────────────

abstract type AbstractUSTensor{T,N} end

# ─── USTensor ─────────────────────────────────────────────────────────────────

# T: element type    I: index type
# N: ndims           VA: val array type   VI: idx array type   O: origin
struct USTensor{T, I, N,
                VA <: AbstractArray{T},
                VI <: AbstractArray{I},
                O  <: AbstractIndexOrigin,
                NL,
                FMT <: TensorFormat,
               } <: AbstractUSTensor{T,N}
    extents     :: NTuple{N,Int}
    format      :: FMT
    pos_buffers :: NTuple{NL, Union{Nothing, VI}}
    crd_buffers :: NTuple{NL, Union{Nothing, VI}}
    val         :: VA
    owner       :: Any   # GC anchor for zero-copy views
end

# ─── Convenience constructor: infer NL from format, accept Dict or Tuple ─────
#
# Existing call sites pass Dict{Int,VI}; the helper converts to NTuple at
# construction.  New call sites can pass NTuple directly to skip the conversion.

@inline _to_buf_tuple(t::NTuple, _NL::Int, ::Type) = t
function _to_buf_tuple(d::Dict, NL::Int, ::Type{VI}) where {VI}
    ntuple(i -> get(d, i, nothing), NL)::NTuple{NL, Union{Nothing, VI}}
end

# Build a length-NL pos/crd tuple with `buf` at level `at` and `nothing` elsewhere.
@inline _bufs_at(::Val{NL}, ::Type{VI}, at::Int, buf::VI) where {NL, VI} =
    ntuple(i -> i == at ? buf : nothing, Val(NL))::NTuple{NL, Union{Nothing, VI}}

# All-nothing pos/crd tuple of length NL (for dense formats).
@inline _no_bufs(::Val{NL}, ::Type{VI}) where {NL, VI} =
    ntuple(_ -> nothing, Val(NL))::NTuple{NL, Union{Nothing, VI}}

function USTensor{T,I,N,VA,VI,O}(extents::NTuple{N,Int}, format::FMT,
                                  pos_bufs, crd_bufs, val::VA, owner) where {T,I,N,VA,VI,O,FMT<:TensorFormat}
    NL  = length(format.levels)
    pos = _to_buf_tuple(pos_bufs, NL, VI)
    crd = _to_buf_tuple(crd_bufs, NL, VI)
    USTensor{T,I,N,VA,VI,O,NL,FMT}(extents, format, pos, crd, val, owner)
end

# 7-param convenience: NL specified, FMT inferred from format value.
function USTensor{T,I,N,VA,VI,O,NL}(extents::NTuple{N,Int}, format::FMT,
                                     pos_bufs, crd_bufs, val::VA, owner) where {T,I,N,VA,VI,O,NL,FMT<:TensorFormat}
    USTensor{T,I,N,VA,VI,O,NL,FMT}(extents, format, pos_bufs, crd_bufs, val, owner)
end

# ─── Memory space trait ───────────────────────────────────────────────────────

memory_space(::Type{<:Array}) = CPUMemory()
memory_space(u::USTensor{<:Any,<:Any,<:Any,VA}) where {VA} = memory_space(VA)

# ─── Accessors ────────────────────────────────────────────────────────────────

function positions(u::USTensor{T,I,N,VA,VI}, level::Int) where {T,I,N,VA,VI}
    1 <= level <= length(u.pos_buffers) ||
        throw(InvalidLevelAccess("level $level out of range"))
    p = u.pos_buffers[level]
    p === nothing &&
        throw(InvalidLevelAccess("level $level has no position buffer"))
    p::VI
end

function coordinates(u::USTensor{T,I,N,VA,VI}, level::Int) where {T,I,N,VA,VI}
    1 <= level <= length(u.crd_buffers) ||
        throw(InvalidLevelAccess("level $level out of range"))
    c = u.crd_buffers[level]
    c === nothing &&
        throw(InvalidLevelAccess("level $level has no coordinate buffer"))
    c::VI
end

# Extend SparseArrays functions to avoid name conflicts with Base.values / SparseArrays.nnz
SparseArrays.nonzeros(u::USTensor) = u.val
SparseArrays.nnz(u::USTensor)      = length(u.val)

has_positions(u::USTensor, level::Int) =
    1 <= level <= length(u.pos_buffers) && u.pos_buffers[level] !== nothing
has_coordinates(u::USTensor, level::Int) =
    1 <= level <= length(u.crd_buffers) && u.crd_buffers[level] !== nothing

format(u::USTensor)       = u.format
extents(u::USTensor)      = u.extents
index_origin(::USTensor{T,I,N,VA,VI,O}) where {T,I,N,VA,VI,O} = O()

Base.size(u::USTensor)          = u.extents
Base.size(u::USTensor, d::Int)  = u.extents[d]
Base.ndims(::USTensor{T,I,N})   where {T,I,N} = N
Base.eltype(::USTensor{T})      where {T}      = T
Base.length(u::USTensor)        = prod(u.extents)

# ─── copy / similar / copyto! ─────────────────────────────────────────────────

function Base.copy(u::USTensor{T,I,N,VA,VI,O,NL,FMT}) where {T,I,N,VA,VI,O,NL,FMT}
    _copy_or_nothing(b) = b === nothing ? nothing : copy(b)
    USTensor{T,I,N,VA,VI,O,NL,FMT}(
        u.extents,
        u.format,
        map(_copy_or_nothing, u.pos_buffers),
        map(_copy_or_nothing, u.crd_buffers),
        copy(u.val),
        nothing,
    )
end

# ─── getindex ─────────────────────────────────────────────────────────────────

# Walk the format levels to locate a coordinate; return val or zero(T).
# dim_indices are 1-based; stored indices are offset by index_origin.
function Base.getindex(u::USTensor{T,I,N,VA,VI,O}, idx::Vararg{Int,N}) where {T,I,N,VA,VI,O}
    origin_offset = O === OneBased ? one(I) : zero(I)
    lvl_idx = dim2lvl(u.format, collect(Int, idx) .- 1)   # convert to 0-based for algebra
    _locate(u, lvl_idx, origin_offset, 1, 1)
end

# Returns the value at the given level coordinates, or zero(T).
# p: current flat position (1-based, matching Julia array indexing throughout)
function _locate(u::USTensor{T,I,N,VA,VI,O}, lvl_idx, origin_offset, level::Int, p::Int) where {T,I,N,VA,VI,O}
    fmt = u.format
    nlevels = length(fmt.levels)
    if level > nlevels
        return u.val[p]
    end

    lv = fmt.levels[level]
    target = I(lvl_idx[level])
    stored_target = target + origin_offset   # stored coord value

    if lv isa DenseLevel || lv isa BatchLevel
        sz_stored = I(extents(u)[_lvl_dim(fmt, level)])
        next_p = (p - 1) * Int(sz_stored) + Int(target) + 1
        return _locate(u, lvl_idx, origin_offset, level + 1, next_p)

    elseif lv isa CompressedLevel || lv isa DeltaLevel
        pos = positions(u, level)
        crd = coordinates(u, level)
        off = Int(origin_offset)
        lo  = Int(pos[p])     - off
        hi  = Int(pos[p + 1]) - off
        if lv isa DeltaLevel
            corig = origin_offset
            for ii in lo:(hi - 1)
                corig += I(crd[ii + 1])
                if corig == stored_target
                    return _locate(u, lvl_idx, origin_offset, level + 1, ii + 1)
                end
                corig += one(I)
            end
        else
            for ii in lo:(hi - 1)
                if I(crd[ii + 1]) == stored_target
                    return _locate(u, lvl_idx, origin_offset, level + 1, ii + 1)
                end
            end
        end
        return zero(T)

    elseif lv isa SingletonLevel
        crd = coordinates(u, level)
        if I(crd[p]) == stored_target
            return _locate(u, lvl_idx, origin_offset, level + 1, p)
        else
            return zero(T)
        end

    elseif lv isa RangeLevel
        next_p = (p - 1) * Int(u.extents[_lvl_dim(fmt, level)]) + Int(target) + 1
        return _locate(u, lvl_idx, origin_offset, level + 1, next_p)

    else
        next_p = locate_level(lv, u, stored_target, origin_offset, level, p)
        next_p === nothing && return zero(T)
        return _locate(u, lvl_idx, origin_offset, level + 1, next_p)
    end
end

# Map level index → dimension index for Dense/Range size lookup
function _lvl_dim(fmt::TensorFormat, level::Int)
    _find_key_dim(fmt.dimensions, fmt.keys[level])
end

function _find_key_dim(dims, key)
    if key isa Dimension
        return findfirst(==(key), dims)
    elseif key isa LevelExpr
        return _find_key_dim(dims, key.lhs)
    end
    error("_find_key_dim: cannot find dimension for key $key")
end

# ─── show ─────────────────────────────────────────────────────────────────────

function Base.show(io::IO, u::USTensor{T,I,N,VA,VI,O}) where {T,I,N,VA,VI,O}
    origin_str = O === OneBased ? "OneBased" : "ZeroBased"
    device_str = memory_space(u) isa CPUMemory ? "cpu" : "gpu"
    nlevels    = length(u.format.levels)
    println(io, "---- Sparse Tensor<VAL=$T,IDX=$I,DIM=$N,LVL=$nlevels,ORIGIN=$origin_str>")
    println(io, "format   : $(u.format.name)  $(u.format.dimensions) -> $(u.format)")
    println(io, "device   : $device_str")
    println(io, "dim      : $(u.extents)")
    lvl_extents = isempty(u.format.dimensions) ? () :
        Tuple(dim2lvl(u.format, collect(Int, u.extents); as_size=true))
    println(io, "lvl      : $lvl_extents")
    println(io, "nnz      : $(SparseArrays.nnz(u))")
    for idx in eachindex(u.format.levels)
        if has_positions(u, idx)
            p = positions(u, idx)
            println(io, "pos[$idx]   : $(collect(p)) #$(length(p))   ($origin_str)")
        end
        if has_coordinates(u, idx)
            c = coordinates(u, idx)
            println(io, "crd[$idx]   : $(collect(c)) #$(length(c))")
        end
    end
    println(io, "values   : $(collect(u.val)) #$(SparseArrays.nnz(u))")
    print(io, "----")
end
