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
               } <: AbstractUSTensor{T,N}
    extents     :: NTuple{N,Int}
    format      :: TensorFormat
    pos_buffers :: Dict{Int,VI}
    crd_buffers :: Dict{Int,VI}
    val         :: VA
    owner       :: Any   # GC anchor for zero-copy views
end

# ─── Memory space trait ───────────────────────────────────────────────────────

memory_space(::Type{<:Array}) = CPUMemory()
memory_space(u::USTensor{<:Any,<:Any,<:Any,VA}) where {VA} = memory_space(VA)

# ─── Accessors ────────────────────────────────────────────────────────────────

function positions(u::USTensor{T,I,N,VA,VI}, level::Int) where {T,I,N,VA,VI}
    haskey(u.pos_buffers, level) ||
        throw(InvalidLevelAccess("level $level has no position buffer"))
    u.pos_buffers[level]
end

function coordinates(u::USTensor{T,I,N,VA,VI}, level::Int) where {T,I,N,VA,VI}
    haskey(u.crd_buffers, level) ||
        throw(InvalidLevelAccess("level $level has no coordinate buffer"))
    u.crd_buffers[level]
end

# Extend SparseArrays functions to avoid name conflicts with Base.values / SparseArrays.nnz
SparseArrays.nonzeros(u::USTensor) = u.val
SparseArrays.nnz(u::USTensor)      = length(u.val)

has_positions(u::USTensor, level::Int)    = haskey(u.pos_buffers, level)
has_coordinates(u::USTensor, level::Int)  = haskey(u.crd_buffers, level)

format(u::USTensor)       = u.format
extents(u::USTensor)      = u.extents
index_origin(::USTensor{T,I,N,VA,VI,O}) where {T,I,N,VA,VI,O} = O()

Base.size(u::USTensor)          = u.extents
Base.size(u::USTensor, d::Int)  = u.extents[d]
Base.ndims(::USTensor{T,I,N})   where {T,I,N} = N
Base.eltype(::USTensor{T})      where {T}      = T
Base.length(u::USTensor)        = prod(u.extents)

# ─── copy / similar / copyto! ─────────────────────────────────────────────────

function Base.copy(u::USTensor{T,I,N,VA,VI,O}) where {T,I,N,VA,VI,O}
    USTensor{T,I,N,VA,VI,O}(
        u.extents,
        u.format,
        Dict(k => copy(v) for (k,v) in u.pos_buffers),
        Dict(k => copy(v) for (k,v) in u.crd_buffers),
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

    _, lv = fmt.levels[level]
    target = I(lvl_idx[level])
    stored_target = target + origin_offset   # stored coord value

    if lv isa DenseLevel || lv isa BatchLevel
        sz_stored = I(extents(u)[_lvl_dim(fmt, level)])
        next_p = (p - 1) * Int(sz_stored) + Int(target) + 1
        return _locate(u, lvl_idx, origin_offset, level + 1, next_p)

    elseif lv isa CompressedLevel || lv isa DeltaLevel
        pos = u.pos_buffers[level]
        crd = u.crd_buffers[level]
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
        crd = u.crd_buffers[level]
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
    key, _ = fmt.levels[level]
    _find_key_dim(fmt.dimensions, key)
end

function _find_key_dim(dims, key)
    if key isa Dimension
        return findfirst(==(key), dims)
    elseif key isa LevelExpr
        return _find_key_dim(dims, key.lhs)
    end
    error("Cannot find dimension for key $key")
end

# ─── show ─────────────────────────────────────────────────────────────────────

function Base.show(io::IO, u::USTensor{T,I,N,VA,VI,O}) where {T,I,N,VA,VI,O}
    origin_str = O === OneBased ? "OneBased" : "ZeroBased"
    device_str = memory_space(u) isa CPUMemory ? "cpu" : "gpu"
    nlevels    = length(u.format.levels)
    println(io, "---- Sparse Tensor<VAL=$T,IDX=$I,DIM=$N,LVL=$nlevels,ORIGIN=$origin_str>")
    println(io, "format   : $(u.format.name)  $(u.format.dimensions) -> $(u.format.levels)")
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
