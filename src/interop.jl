# Zero-copy and manual construction helpers for USTensor.

# ─── SparseMatrixCSC zero-copy view ───────────────────────────────────────────

function ust(A::SparseMatrixCSC{T,I}) where {T,I}
    n, m = size(A)
    # CSC: levels are (j: dense, i: compressed)
    # pos[2] = colptr, crd[2] = rowval — level 2 is the compressed j-over-i level
    pos = Dict{Int,Vector{I}}(2 => A.colptr)
    crd = Dict{Int,Vector{I}}(2 => A.rowval)
    USTensor{T,I,2,Vector{T},Vector{I},OneBased}(
        (n, m),
        Formats.CSC,
        pos,
        crd,
        A.nzval,
        A,   # owner keeps A alive
    )
end

# ─── SparseVector zero-copy view ─────────────────────────────────────────────

function ust(A::SparseVector{T,I}) where {T,I}
    pos = Dict{Int,Vector{I}}()
    crd = Dict{Int,Vector{I}}(1 => A.nzind)
    USTensor{T,I,1,Vector{T},Vector{I},OneBased}(
        (length(A),),
        Formats.SparseVector,
        pos,
        crd,
        A.nzval,
        A,
    )
end

# ─── Dense AbstractArray zero-copy view ───────────────────────────────────────

function ust(A::AbstractArray{T,N}) where {T,N}
    dims = Formats.DensedRight(N)
    pos = Dict{Int,Vector{Int}}()
    crd = Dict{Int,Vector{Int}}()
    USTensor{T,Int,N,typeof(A),Vector{Int},OneBased}(
        size(A),
        dims,
        pos,
        crd,
        A,
        A,
    )
end

# ─── Manual constructors ──────────────────────────────────────────────────────

"""
    csr_tensor(rowptr, colind, nzval, dims; origin=ZeroBased())

Build a CSR USTensor from pre-allocated buffers. Buffers are not copied.
`dims` is `(nrows, ncols)`. Index convention is set by `origin`.
"""
function csr_tensor(rowptr::VI, colind::VI, nzval::VA,
                    dims::Tuple{Int,Int};
                    origin::O=ZeroBased()) where {T, I,
                                                   VA <: AbstractArray{T},
                                                   VI <: AbstractArray{I},
                                                   O  <: AbstractIndexOrigin}
    pos = Dict{Int,VI}(2 => rowptr)
    crd = Dict{Int,VI}(2 => colind)
    USTensor{T,I,2,VA,VI,O}(dims, Formats.CSR, pos, crd, nzval, nothing)
end

"""
    csc_tensor(colptr, rowind, nzval, dims; origin=OneBased())

Build a CSC USTensor from pre-allocated buffers.
"""
function csc_tensor(colptr::VI, rowind::VI, nzval::VA,
                    dims::Tuple{Int,Int};
                    origin::O=OneBased()) where {T, I,
                                                  VA <: AbstractArray{T},
                                                  VI <: AbstractArray{I},
                                                  O  <: AbstractIndexOrigin}
    pos = Dict{Int,VI}(2 => colptr)
    crd = Dict{Int,VI}(2 => rowind)
    USTensor{T,I,2,VA,VI,O}(dims, Formats.CSC, pos, crd, nzval, nothing)
end

"""
    coo_tensor(rows, cols, nzval, dims; origin=ZeroBased(), sorted=false)

Build a COO USTensor. `rows` becomes the compressed level coordinate buffer,
`cols` becomes the singleton level coordinate buffer.
"""
function coo_tensor(rows::VI, cols::VI, nzval::VA,
                    dims::Tuple{Int,Int};
                    origin::O=ZeroBased(),
                    sorted::Bool=false) where {T, I,
                                               VA <: AbstractArray{T},
                                               VI <: AbstractArray{I},
                                               O  <: AbstractIndexOrigin}
    _ = sorted  # consumed by user; not stored (no canonical ordering guarantee)
    pos = Dict{Int,VI}()
    crd = Dict{Int,VI}(1 => rows, 2 => cols)
    USTensor{T,I,2,VA,VI,O}(dims, Formats.COO, pos, crd, nzval, nothing)
end
