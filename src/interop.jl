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
#
# Shared empty dicts used as the positions/coordinates placeholders for dense
# tensors.  DensedRight levels have no pos/crd buffers; these dicts are never
# accessed or mutated, so sharing them across all ust(::AbstractArray) calls is
# safe and avoids 2 Dict allocations per wrapping (8 per BlockSparseMatrix mul!).
const _DENSE_POS = Dict{Int,Vector{Int}}()
const _DENSE_CRD = Dict{Int,Vector{Int}}()

function ust(A::AbstractArray{T,N}) where {T,N}
    USTensor{T,Int,N,typeof(A),Vector{Int},OneBased}(
        size(A),
        Formats.DensedRight(N),
        _DENSE_POS,
        _DENSE_CRD,
        A,
        A,
    )
end

# ─── Manual constructors ──────────────────────────────────────────────────────

"""
    csr_tensor(rowptr, colind, nzval, dims; origin=OneBased())
    csr_tensor(rowptr, colind, nzval; m, n, origin=OneBased())

Build a CSR USTensor from pre-allocated buffers. Buffers are not copied.
`dims` is `(nrows, ncols)`; keyword form accepts `m` and `n` separately.
"""
function csr_tensor(rowptr::VI, colind::VI, nzval::VA,
                    dims::Tuple{Int,Int};
                    origin::O=OneBased()) where {T, I,
                                                  VA <: AbstractArray{T},
                                                  VI <: AbstractArray{I},
                                                  O  <: AbstractIndexOrigin}
    pos = Dict{Int,VI}(2 => rowptr)
    crd = Dict{Int,VI}(2 => colind)
    USTensor{T,I,2,VA,VI,O}(dims, Formats.CSR, pos, crd, nzval, nothing)
end

function csr_tensor(rowptr::VI, colind::VI, nzval::VA;
                    m::Int, n::Int,
                    origin::O=OneBased()) where {T, I,
                                                  VA <: AbstractArray{T},
                                                  VI <: AbstractArray{I},
                                                  O  <: AbstractIndexOrigin}
    csr_tensor(rowptr, colind, nzval, (m, n); origin=origin)
end

"""
    csr_tensor(A::SparseMatrixCSC{T}; device=identity)

Build a CSR USTensor from a Julia SparseMatrixCSC, using Int32 index buffers.
Transposes via `sparse(A')` to convert CSC→CSR layout, then applies `device`
to each buffer so the result can live on any accelerator.

    csr_tensor(A)                       # CPU, shares no buffers with A
    csr_tensor(A; device=CUDA.CuArray)  # GPU-resident CSR tensor
"""
function csr_tensor(A::SparseMatrixCSC{T}; device=identity) where T
    m, n = size(A)
    At   = sparse(A')
    csr_tensor(device(Int32.(At.colptr)), device(Int32.(At.rowval)),
               device(T.(At.nzval)); m=m, n=n)
end

"""
    csc_tensor(colptr, rowind, nzval, dims; origin=OneBased())
    csc_tensor(colptr, rowind, nzval; m, n, origin=OneBased())

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

function csc_tensor(colptr::VI, rowind::VI, nzval::VA;
                    m::Int, n::Int,
                    origin::O=OneBased()) where {T, I,
                                                  VA <: AbstractArray{T},
                                                  VI <: AbstractArray{I},
                                                  O  <: AbstractIndexOrigin}
    csc_tensor(colptr, rowind, nzval, (m, n); origin=origin)
end

"""
    coo_tensor(rows, cols, nzval, dims; origin=OneBased(), sorted=false)
    coo_tensor(rows, cols, nzval; m, n, origin=OneBased(), sorted=false)

Build a COO USTensor. `rows` becomes the compressed level coordinate buffer,
`cols` becomes the singleton level coordinate buffer.
"""
function coo_tensor(rows::VI, cols::VI, nzval::VA,
                    dims::Tuple{Int,Int};
                    origin::O=OneBased(),
                    sorted::Bool=false) where {T, I,
                                               VA <: AbstractArray{T},
                                               VI <: AbstractArray{I},
                                               O  <: AbstractIndexOrigin}
    _ = sorted  # consumed by user; not stored (no canonical ordering guarantee)
    pos = Dict{Int,VI}()
    crd = Dict{Int,VI}(1 => rows, 2 => cols)
    USTensor{T,I,2,VA,VI,O}(dims, Formats.COO, pos, crd, nzval, nothing)
end

function coo_tensor(rows::VI, cols::VI, nzval::VA;
                    m::Int, n::Int,
                    origin::O=OneBased(),
                    sorted::Bool=false) where {T, I,
                                               VA <: AbstractArray{T},
                                               VI <: AbstractArray{I},
                                               O  <: AbstractIndexOrigin}
    coo_tensor(rows, cols, nzval, (m, n); origin=origin, sorted=sorted)
end

"""
    dcsr_tensor(A::SparseMatrixCSC{T}; device=identity)

Build a DCSR USTensor from a Julia SparseMatrixCSC, using Int32 index buffers.
Only non-empty rows are stored; the outer coordinate array records their indices,
giving an iteration cost proportional to the number of active rows rather than
the total row count.  Applies `device` to each buffer for accelerator placement.

    dcsr_tensor(A)                       # CPU
    dcsr_tensor(A; device=CUDA.CuArray)  # GPU-resident DCSR tensor
"""
function dcsr_tensor(A::SparseMatrixCSC{T}; device=identity) where T
    m, n  = size(A)
    At    = sparse(A')
    rowptr = At.colptr        # 1-based, length m+1

    active    = filter(i -> rowptr[i+1] > rowptr[i], 1:m)
    outer_crd = Int32.(active)
    inner_pos = Int32.(rowptr[[active; m+1]])   # 1-based offsets into inner_crd
    inner_crd = Int32.(At.rowval)
    nzval     = T.(At.nzval)

    dcsr_tensor(device(outer_crd), device(inner_pos), device(inner_crd), device(nzval); m=m, n=n)
end

"""
    dcsr_tensor(outer_crd, inner_pos, inner_crd, nzval; m, n, origin=OneBased())

Build a DCSR (doubly-compressed sparse row) USTensor from pre-allocated buffers.
`outer_crd` holds the row indices of non-empty rows; `inner_pos`/`inner_crd`
are the standard CSR pos/crd arrays for the compressed column dimension.
"""
function dcsr_tensor(outer_crd::VI, inner_pos::VI, inner_crd::VI, nzval::VA;
                     m::Int, n::Int,
                     origin::O=OneBased()) where {T, I,
                                                   VA <: AbstractArray{T},
                                                   VI <: AbstractArray{I},
                                                   O  <: AbstractIndexOrigin}
    pos = Dict{Int,VI}(2 => inner_pos)
    crd = Dict{Int,VI}(1 => outer_crd, 2 => inner_crd)
    USTensor{T,I,2,VA,VI,O}((m, n), Formats.DCSR, pos, crd, nzval, nothing)
end
