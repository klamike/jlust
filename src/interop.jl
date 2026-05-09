# Zero-copy and manual construction helpers for USTensor.

# ─── SparseMatrixCSC zero-copy view ───────────────────────────────────────────

function ust(A::SparseMatrixCSC{T,I}) where {T,I}
    n, m = size(A)
    VI = Vector{I}
    USTensor{T,I,2,Vector{T},VI,OneBased,2}(
        (n, m),
        Formats.CSC,
        _bufs_at(Val(2), VI, 2, A.colptr),
        _bufs_at(Val(2), VI, 2, A.rowval),
        A.nzval,
        A,
    )
end

# ─── SparseVector zero-copy view ─────────────────────────────────────────────

function ust(A::SparseVector{T,I}) where {T,I}
    VI = Vector{I}
    USTensor{T,I,1,Vector{T},VI,OneBased,1}(
        (length(A),),
        Formats.SparseVector,
        _no_bufs(Val(1), VI),
        _bufs_at(Val(1), VI, 1, A.nzind),
        A.nzval,
        A,
    )
end

# ─── Dense AbstractArray zero-copy view ───────────────────────────────────────
# DensedRight has N levels (all DenseLevel), all pos/crd entries are nothing.
# _no_bufs returns a stack-allocated NTuple, so no per-wrapping heap allocation.

function ust(A::AbstractArray{T,N}) where {T,N}
    VI = Vector{Int}
    USTensor{T,Int,N,typeof(A),VI,OneBased,N}(
        size(A),
        Formats.DensedRight(N),
        _no_bufs(Val(N), VI),
        _no_bufs(Val(N), VI),
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
    USTensor{T,I,2,VA,VI,O,2}(dims, Formats.CSR,
        _bufs_at(Val(2), VI, 2, rowptr),
        _bufs_at(Val(2), VI, 2, colind),
        nzval, nothing)
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
    selector_tensor(cols, vals, dims; origin=OneBased())
    selector_tensor(cols, vals; m, n, origin=OneBased())

Build a `(DenseLevel, SingletonLevel)` USTensor — *exactly one* nonzero per row
at the column given by `cols[r]` with value `vals[r]`.  No rowptr is stored.

The generic emitter walker generates a single-load-per-row SpMV kernel for this
format with no pos indirection — equivalent to a hand-written selector kernel,
just produced automatically from the level structure.

Useful for: gen-to-bus incidence (1 generator → 1 bus), ramp coupling matrices
(R = sparse(1:n_gen, 1:n_gen, …)), permutation maps, scatter operators.

Distinct from `(Dense, Compressed)` CSR, which stores a rowptr to allow rows
with 0 or many nnz; the selector format trades that flexibility for a faster
kernel and 4n_rows fewer index bytes.
"""
function selector_tensor(cols::VI, vals::VA, dims::Tuple{Int,Int};
                          origin::O=OneBased()) where {T, I,
                                                        VA <: AbstractArray{T},
                                                        VI <: AbstractArray{I},
                                                        O  <: AbstractIndexOrigin}
    m, n = dims
    length(cols) == m ||
        error("selector_tensor: length(cols)=$(length(cols)) ≠ n_rows=$m")
    length(vals) == m ||
        error("selector_tensor: length(vals)=$(length(vals)) ≠ n_rows=$m")
    USTensor{T,I,2,VA,VI,O,2}(dims, Formats.SelectorRow,
        _no_bufs(Val(2), VI),
        _bufs_at(Val(2), VI, 2, cols),
        vals, nothing)
end

function selector_tensor(cols::VI, vals::VA;
                          m::Int, n::Int,
                          origin::O=OneBased()) where {T, I,
                                                        VA <: AbstractArray{T},
                                                        VI <: AbstractArray{I},
                                                        O  <: AbstractIndexOrigin}
    selector_tensor(cols, vals, (m, n); origin=origin)
end

"""
    diagonal_tensor(d::AbstractVector; n_cols=length(d), device=identity)

Build a diagonal USTensor — special case of `selector_tensor` with
`cols = 1:length(d)`.  When `n_cols > length(d)` the trailing columns are
implicitly zero (the matrix is `length(d) × n_cols`).
"""
function diagonal_tensor(d::AbstractVector{T};
                          n_cols::Int=length(d),
                          device=identity) where T
    m  = length(d)
    cols = device(Int32.(1:m))
    vals = device(T.(d))
    selector_tensor(cols, vals; m=m, n=n_cols)
end

"""
    shifted_diag_tensor(::Type{T}, n_rows, n_cols; shift=0, val=one(T), device=identity)

Build a structural shifted scaled-identity USTensor — every row `r` has exactly
one nonzero at column `r + shift` with constant value `val`, and there are no
pos / crd / nzval arrays (both shift and val live in the format type, baked in
as kernel literals).  When used as a `BlockSparseMatrix` block the BSM compile
path inlines the row-wise contribution as a per-row scalar add — no CSR
replication, no indirect-load chain.

Compatible with the standard SpMV walker (`Dense + ShiftedDiag` shape) so it
also runs as a standalone tensor.
"""
function shifted_diag_tensor(::Type{T}, n_rows::Integer, n_cols::Integer;
                              shift::Integer=0, val=one(T),
                              device=identity) where T
    fmt        = Formats.ShiftedDiag(T; shift=shift, val=T(val))
    empty_val  = device(Vector{T}(undef, 0))
    empty_crd  = device(Vector{Int32}(undef, 0))
    VA         = typeof(empty_val)
    VI         = typeof(empty_crd)
    USTensor{T,Int32,2,VA,VI,OneBased,2}(
        (Int(n_rows), Int(n_cols)), fmt,
        _no_bufs(Val(2), VI),
        _no_bufs(Val(2), VI),
        empty_val, nothing)
end

"""
    periodic_csr_tensor(block::USTensor, T_per::Int) -> USTensor

Wrap a CSR-formatted block in a `(PeriodicLevel{T_per, n_cols}, Dense{block_rows},
Compressed)` USTensor that represents a T-fold block-diagonal replica of `block`
along both dims.  The block's pos / crd / nzval are *shared* — no replication —
and the walker reconstructs each period's per-row contribution by adjusting
the column offset at thread granularity.

The resulting tensor is a regular USTensor: standard SpMV / mul! / `*` work
through the same emitter pipeline, and the walker handles the periodic shape
generically.  Equivalent to `kron(I(T_per), block)` but in O(one-block) space.

The block format must be CSR (`(Dense, Compressed)`).  For periodic
replication of other block formats, use `periodic_tensor` (general path).
"""
function periodic_csr_tensor(block::USTensor{T,I,2,VA,VI,O,2}, T_per::Integer) where {T,I,VA,VI,O}
    fmt_old = format(block)
    fmt_old == Formats.CSR ||
        error("periodic_csr_tensor: block must be CSR-formatted (got $fmt_old)")
    block_rows = Int(extents(block)[1])
    block_cols = Int(extents(block)[2])
    fmt        = Formats.PeriodicCSR(Int(T_per), block_rows, block_cols)
    new_extents = (Int(T_per) * block_rows, Int(T_per) * block_cols)
    pos_buf = positions(block, 2)
    crd_buf = coordinates(block, 2)
    new_pos = _bufs_at(Val(3), VI, 3, pos_buf)
    new_crd = _bufs_at(Val(3), VI, 3, crd_buf)
    USTensor{T,I,2,VA,VI,O,3,typeof(fmt),typeof(block)}(
        new_extents, fmt, new_pos, new_crd, block.val, block)
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
    USTensor{T,I,2,VA,VI,O,2}(dims, Formats.CSC,
        _bufs_at(Val(2), VI, 2, colptr),
        _bufs_at(Val(2), VI, 2, rowind),
        nzval, nothing)
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
    USTensor{T,I,2,VA,VI,O,2}(dims, Formats.COO,
        _no_bufs(Val(2), VI),
        (rows, cols),
        nzval, nothing)
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
    USTensor{T,I,2,VA,VI,O,2}((m, n), Formats.DCSR,
        _bufs_at(Val(2), VI, 2, inner_pos),
        (outer_crd, inner_crd),
        nzval, nothing)
end
