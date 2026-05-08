import LinearAlgebra

# ─── Accept AbstractMatrix in sparse_mm! ─────────────────────────────────────
#
# Wraps raw matrices as dense USTensors (DensedRight(2) format) so callers
# can pass plain Matrix / CuMatrix without constructing USTensors manually.

function sparse_mm!(be::AbstractUSTBackend, A::USTensor,
                    B::AbstractMatrix, C::AbstractMatrix; kw...)
    sparse_mm!(be, A, ust(B), ust(C); kw...)
end

function sparse_mm!(A::USTensor, B::AbstractMatrix, C::AbstractMatrix; kw...)
    sparse_mm!(A, ust(B), ust(C); kw...)
end

# ─── Accept AbstractVector in sparse_mv! ─────────────────────────────────────
#
# No-backend overload lives in KernelAbstractionsExt (default backend is resolved
# there). The explicit-backend and handle overloads below belong here so they
# don't depend on any extension being loaded.

function sparse_mv!(be::AbstractUSTBackend, A::USTensor,
                    x::AbstractVector, y::AbstractVector; kw...)
    sparse_mv!(be, A, ust(x), ust(y); kw...)
end

# Handle variant (EmitterSpMVHandle, CUSPARSESpMVHandle, etc.)
function sparse_mv!(h, A::USTensor, x::AbstractVector, y::AbstractVector; kw...)
    sparse_mv!(h, A, ust(x), ust(y); kw...)
end

# ─── LinearAlgebra compatibility for USTensor ─────────────────────────────────
#
# mul!(y, A, x) and A*x delegate to sparse_mv! with the default backend.
# The default backend is resolved at call time via the KernelAbstractionsExt
# method `sparse_mv!(A::USTensor, x::USTensor, y::USTensor; backend=...)`.

function LinearAlgebra.mul!(y::AbstractVector, A::USTensor, x::AbstractVector)
    sparse_mv!(A, x, y)
    return y
end

function Base.:*(A::USTensor, x::AbstractVector)
    y = similar(nonzeros(A), size(A, 1))
    LinearAlgebra.mul!(y, A, x)
    return y
end

# ─── make_tensor ─────────────────────────────────────────────────────────────
#
# Convenience constructor for custom level formats (DiagonalLevel, RampLevel, …)
# that carry no position or coordinate arrays.  Removes the boilerplate type
# annotation gymnastics from user-written tensor constructors.
#
#   diagonal_tensor(diag; n) = make_tensor(DiagonalFmt, diag; m=n, n=n)
#   ramp_tensor(ref; m, n, sign) = make_tensor(fmt, similar(ref, T, 0); m, n)

"""
    make_tensor(fmt::TensorFormat, nzval::AbstractArray; m::Int, n::Int,
                index_type::Type{I}=Int32, origin::O=OneBased()) → USTensor

Build a 2-D USTensor with no position or coordinate arrays.  Intended for custom
`AbstractLevelFormat` subtypes whose buffers are fully implicit (DiagonalLevel,
RampLevel, etc.).  `nzval` may be a zero-length array for formats with no stored
values.
"""
function make_tensor(fmt::TensorFormat, nzval::VA;
                     m::Int, n::Int,
                     index_type::Type{I}=Int32,
                     origin::O=OneBased()) where {T, VA<:AbstractArray{T},
                                                   I<:Integer,
                                                   O<:AbstractIndexOrigin}
    VI = typeof(similar(nzval, I, 0))
    USTensor{T, I, 2, VA, VI, O}((m, n), fmt, Dict{Int,VI}(), Dict{Int,VI}(), nzval, nothing)
end

# ─── BlockSparseMatrix ────────────────────────────────────────────────────────
#
# A block-structured sparse matrix whose entries are USTensors or nothing (zero
# blocks).  Each block row may contain any mix of formats; blocks in the same
# block-column must have the same column count, and blocks in the same block-row
# must have the same row count.
#
# Construction:
#   A = BlockSparseMatrix([u_Cg   nothing  u_negB;
#                          u_PTDF u_negI   nothing;
#                          nothing nothing  u_E   ])
#
# The matrix-literal syntax produces a Matrix{Any} in Julia when elements have
# mixed types (USTensor subtypes and Nothing), which is exactly what the
# BlockSparseMatrix constructor expects.
#
# Usage:
#   y = A * x                  # allocates y
#   mul!(y, A, x)              # in-place
#   mul!(y, A, x; backend=be)  # explicit backend

struct BlockSparseMatrix{T}
    blocks    :: Matrix{Any}   # nb_rows × nb_cols; entries: USTensor or Nothing
    row_sizes :: Vector{Int}   # number of matrix rows per block row
    col_sizes :: Vector{Int}   # number of matrix cols per block col
    _row_off  :: Vector{Int}   # cumulative row offsets (length nb_rows+1)
    _col_off  :: Vector{Int}   # cumulative col offsets (length nb_cols+1)
end

function BlockSparseMatrix(blocks::Matrix)
    nb_r, nb_c = size(blocks)
    row_sizes = zeros(Int, nb_r)
    col_sizes = zeros(Int, nb_c)
    ref_T = nothing
    first_block = nothing

    for i in 1:nb_r, j in 1:nb_c
        b = blocks[i, j]
        b === nothing && continue
        b isa AbstractUSTensor || error("BlockSparseMatrix: block ($i,$j) must be a USTensor or nothing, got $(typeof(b))")

        rs, cs = size(b, 1), size(b, 2)
        if row_sizes[i] == 0
            row_sizes[i] = rs
        elseif row_sizes[i] != rs
            error("BlockSparseMatrix: block row $i has inconsistent sizes ($(row_sizes[i]) vs $rs at col $j)")
        end
        if col_sizes[j] == 0
            col_sizes[j] = cs
        elseif col_sizes[j] != cs
            error("BlockSparseMatrix: block col $j has inconsistent sizes ($(col_sizes[j]) vs $cs at row $i)")
        end

        T_b = eltype(b)
        if ref_T === nothing
            ref_T = T_b
            first_block = b
        elseif ref_T != T_b
            error("BlockSparseMatrix: mixed element types ($ref_T vs $T_b at block ($i,$j))")
        end
    end

    any(==(0), row_sizes) && error("BlockSparseMatrix: block row(s) $(findall(==(0), row_sizes)) are entirely nothing — cannot infer size")
    any(==(0), col_sizes) && error("BlockSparseMatrix: block col(s) $(findall(==(0), col_sizes)) are entirely nothing — cannot infer size")
    ref_T === nothing && error("BlockSparseMatrix: all blocks are nothing")

    BlockSparseMatrix{ref_T}(Matrix{Any}(blocks), row_sizes, col_sizes,
                              cumsum([0; row_sizes]), cumsum([0; col_sizes]))
end

Base.size(A::BlockSparseMatrix)         = (sum(A.row_sizes), sum(A.col_sizes))
Base.size(A::BlockSparseMatrix, d::Int) = sum(d == 1 ? A.row_sizes : A.col_sizes)
Base.eltype(::BlockSparseMatrix{T}) where T = T

# ─── BlockSparseMatrix: fused CPU SpMV ───────────────────────────────────────
#
# When all non-null blocks are CPU-resident CSR tensors and no explicit backend
# is requested, we skip the per-block KA dispatch loop and run a single fused
# Julia pass over all blocks.  This closes the performance gap with Julia's
# monolithic SparseMatrixCSC (which does one pass over all nnz).
#
# Falls back to the standard KA path for GPU/Metal arrays, non-CSR formats
# (DiagonalLevel, DCSR, …), or when an explicit backend is provided.

_is_cpu_array(::Array)      = true
_is_cpu_array(x::SubArray)  = _is_cpu_array(parent(x))
_is_cpu_array(::AbstractArray) = false


function _cpu_csr_accumulate!(y::AbstractVector{T}, rp, ci, nz, x::AbstractVector{T},
                               y_lo::Int, x_lo::Int, m::Int) where T
    @inbounds for row in 1:m
        acc = zero(T)
        for k in Int(rp[row]) : Int(rp[row+1]) - 1
            acc += nz[k] * x[x_lo + Int(ci[k])]
        end
        y[y_lo + row] += acc
    end
end

# Custom inner level formats opt into the fused CPU path in two ways:
#   1. High-level: define JLUST.level_step(lv, i, nz) → (col, val).
#      _is_cpu_fusable_level auto-detects it via hasmethod; _cpu_level_accumulate!
#      loops over level_step calls automatically.  No extra methods needed.
#   2. Low-level: define _is_cpu_fusable_level(::MyLevel) = true and implement
#      _cpu_level_accumulate! directly (for non-diagonal access patterns).
_is_cpu_fusable_level(lv::AbstractLevelFormat) =
    hasmethod(JLUST.level_step, Tuple{typeof(lv), Int, Nothing}) ||
    hasmethod(JLUST.level_step, Tuple{typeof(lv), Int, AbstractVector})
_is_cpu_fusable_level(::DenseLevel)      = true
_is_cpu_fusable_level(::CompressedLevel) = true  # inner: standard CSR column spans

# Outer level determines iteration strategy.
# CSR:  DenseLevel outer  → iterate all m rows (outer_crd not needed).
# DCSR: CompressedLevel outer → iterate only active rows via crd[1].
function _is_cpu_fusable(b::AbstractUSTensor)
    length(b.format.levels) == 2 || return false
    _is_cpu_array(nonzeros(b))   || return false
    outer_lv = b.format.levels[1][2]
    inner_lv = b.format.levels[2][2]
    (outer_lv isa DenseLevel && _is_cpu_fusable_level(inner_lv)) ||
    (outer_lv isa CompressedLevel && inner_lv isa CompressedLevel)
end

# Dispatch on outer level type so CSR and DCSR take different code paths.
function _cpu_block_accumulate!(b::AbstractUSTensor, y, x, y_lo, x_lo, m)
    outer_lv = b.format.levels[1][2]
    if outer_lv isa DenseLevel
        _cpu_level_accumulate!(b.format.levels[2][2], b, y, x, y_lo, x_lo, m)
    else  # CompressedLevel outer (DCSR): iterate active rows via crd[1]
        _cpu_dcsr_accumulate!(y, coordinates(b, 1), positions(b, 2), coordinates(b, 2),
                              nonzeros(b), x, y_lo, x_lo)
    end
end

function _cpu_level_accumulate!(::CompressedLevel, b, y, x, y_lo, x_lo, m)
    _cpu_csr_accumulate!(y, positions(b, 2), coordinates(b, 2), nonzeros(b), x, y_lo, x_lo, m)
end

function _cpu_level_accumulate!(lv::AbstractLevelFormat, b, y, x, y_lo, x_lo, m)
    nz = level_has_nzval(lv) ? nonzeros(b) : nothing
    @inbounds for i in 1:m
        col, val = level_step(lv, i, nz)
        y[y_lo + i] += val * x[x_lo + col]
    end
end

function _cpu_dcsr_accumulate!(y::AbstractVector{T}, outer_crd, inner_pos, inner_crd,
                                nz, x::AbstractVector{T}, y_lo::Int, x_lo::Int) where T
    @inbounds for k in 1:length(outer_crd)
        row = Int(outer_crd[k])
        acc = zero(T)
        for p in Int(inner_pos[k]):Int(inner_pos[k+1])-1
            acc += nz[p] * x[x_lo + Int(inner_crd[p])]
        end
        y[y_lo + row] += acc
    end
end

# ─── Fused CPU SpMM ──────────────────────────────────────────────────────────
#
# Multi-RHS variant of the fused CPU SpMV path.  Each NNZ is loaded once and
# scattered across all n_col output columns in the innermost loop, which the
# compiler can vectorize.  This avoids T separate SpMV kernel dispatches for
# the multi-stage OPF and gives bandwidth ~ T× better arithmetic intensity.

function _cpu_csr_accumulate_mm!(Y::AbstractMatrix{T}, rp, ci, nz,
                                   X::AbstractMatrix{T}, y_lo::Int, x_lo::Int, m::Int) where T
    n_col = size(X, 2)
    @inbounds for row in 1:m
        for k in Int(rp[row]):Int(rp[row+1])-1
            val = nz[k]; col = Int(ci[k])
            for t in 1:n_col
                Y[y_lo+row, t] += val * X[x_lo+col, t]
            end
        end
    end
end

function _cpu_dcsr_accumulate_mm!(Y::AbstractMatrix{T}, outer_crd, inner_pos, inner_crd,
                                   nz, X::AbstractMatrix{T}, y_lo::Int, x_lo::Int) where T
    n_col = size(X, 2)
    @inbounds for k in 1:length(outer_crd)
        row = Int(outer_crd[k])
        for p in Int(inner_pos[k]):Int(inner_pos[k+1])-1
            val = nz[p]; col = Int(inner_crd[p])
            for t in 1:n_col
                Y[y_lo+row, t] += val * X[x_lo+col, t]
            end
        end
    end
end

function _cpu_level_accumulate_mm!(lv::AbstractLevelFormat, b, Y, X, y_lo, x_lo, m)
    nz = level_has_nzval(lv) ? nonzeros(b) : nothing
    n_col = size(X, 2)
    @inbounds for i in 1:m
        col, val = level_step(lv, i, nz)
        for t in 1:n_col
            Y[y_lo+i, t] += val * X[x_lo+col, t]
        end
    end
end

function _cpu_block_accumulate_mm!(b::AbstractUSTensor, Y, X, y_lo, x_lo, m)
    outer_lv = b.format.levels[1][2]
    if outer_lv isa DenseLevel
        inner_lv = b.format.levels[2][2]
        if inner_lv isa CompressedLevel
            _cpu_csr_accumulate_mm!(Y, positions(b, 2), coordinates(b, 2), nonzeros(b),
                                     X, y_lo, x_lo, m)
        else
            _cpu_level_accumulate_mm!(inner_lv, b, Y, X, y_lo, x_lo, m)
        end
    else  # DCSR: CompressedLevel outer
        _cpu_dcsr_accumulate_mm!(Y, coordinates(b, 1), positions(b, 2), coordinates(b, 2),
                                  nonzeros(b), X, y_lo, x_lo)
    end
end

function _try_fused_cpu_mul_mm!(Y::AbstractMatrix, A::BlockSparseMatrix, X::AbstractMatrix)
    nb_r, nb_c = size(A.blocks)
    for i in 1:nb_r, j in 1:nb_c
        b = A.blocks[i, j]; b === nothing && continue
        _is_cpu_fusable(b) || return false
    end
    fill!(Y, zero(eltype(A)))
    for i in 1:nb_r
        y_lo = A._row_off[i]
        for j in 1:nb_c
            b = A.blocks[i, j]; b === nothing && continue
            _cpu_block_accumulate_mm!(b, Y, X, y_lo, A._col_off[j], size(b, 1))
        end
    end
    return true
end

function _try_fused_cpu_mul!(y::AbstractVector, A::BlockSparseMatrix, x::AbstractVector)
    nb_r, nb_c = size(A.blocks)
    for i in 1:nb_r, j in 1:nb_c
        b = A.blocks[i, j]
        b === nothing && continue
        _is_cpu_fusable(b) || return false
    end
    fill!(y, zero(eltype(A)))
    for i in 1:nb_r
        y_lo = A._row_off[i]
        for j in 1:nb_c
            b = A.blocks[i, j]
            b === nothing && continue
            _cpu_block_accumulate!(b, y, x, y_lo, A._col_off[j], size(b, 1))
        end
    end
    return true
end

function LinearAlgebra.mul!(y::AbstractVector, A::BlockSparseMatrix, x::AbstractVector;
                             backend::Union{AbstractUSTBackend,Nothing}=nothing)
    # Fast path: fused Julia loop for CPU-resident CSR blocks.
    if backend === nothing && _is_cpu_array(y) && _is_cpu_array(x)
        _try_fused_cpu_mul!(y, A, x) && return y
    end

    # General path: one kernel call per non-null block.
    nb_r, nb_c = size(A.blocks)
    fill!(y, zero(eltype(A)))

    for i in 1:nb_r
        y_sl      = view(y, A._row_off[i]+1 : A._row_off[i+1])
        first_col = true

        for j in 1:nb_c
            b = A.blocks[i, j]
            b === nothing && continue
            x_sl = view(x, A._col_off[j]+1 : A._col_off[j+1])
            β    = first_col ? false : true
            if backend === nothing
                sparse_mv!(b, x_sl, y_sl; beta=β)
            else
                sparse_mv!(backend, b, x_sl, y_sl; beta=β)
            end
            first_col = false
        end
    end

    return y
end

function Base.:*(A::BlockSparseMatrix{T}, x::AbstractVector{T}) where T
    y = similar(x, sum(A.row_sizes))
    LinearAlgebra.mul!(y, A, x)
    return y
end

# ─── BlockSparseMatrix: matrix multiply (SpMM) ───────────────────────────────
#
# mul!(Y, A, X) computes Y = A * X where X and Y are matrices.
# Each block contributes via sparse_mm! with beta=1 for accumulation across
# blocks in the same block-row.  Y is pre-zeroed so the first block can use
# beta=1 without a separate beta=0 path.

function LinearAlgebra.mul!(Y::AbstractMatrix, A::BlockSparseMatrix, X::AbstractMatrix;
                             backend::Union{AbstractUSTBackend,Nothing}=nothing)
    # Fast path: fused Julia loop for CPU-resident blocks (mirrors SpMV fused path).
    if backend === nothing && _is_cpu_array(Y) && _is_cpu_array(X)
        _try_fused_cpu_mul_mm!(Y, A, X) && return Y
    end

    # General path: one SpMM kernel per non-null block.
    nb_r, nb_c = size(A.blocks)
    fill!(Y, zero(eltype(A)))

    for i in 1:nb_r
        Y_sl = view(Y, A._row_off[i]+1 : A._row_off[i+1], :)
        for j in 1:nb_c
            b = A.blocks[i, j]
            b === nothing && continue
            X_sl = view(X, A._col_off[j]+1 : A._col_off[j+1], :)
            if backend === nothing
                sparse_mm!(b, X_sl, Y_sl; beta=one(eltype(A)))
            else
                sparse_mm!(backend, b, X_sl, Y_sl; beta=one(eltype(A)))
            end
        end
    end

    return Y
end

function Base.:*(A::BlockSparseMatrix{T}, X::AbstractMatrix{T}) where T
    Y = similar(X, sum(A.row_sizes), size(X, 2))
    LinearAlgebra.mul!(Y, A, X)
    return Y
end

# ─── BlockSparseMatrix: selective value updates ───────────────────────────────
#
# In batched or receding-horizon applications some blocks have constant
# structure (sparsity pattern, index buffers) but variable values — e.g. the
# PTDF matrix changes when a line trips, while the +1/-1 incidence stays fixed.
#
# update_block_values!(BM, i, j, new_nzval) swaps the nonzero value buffer of
# block (i,j) without touching its index buffers.  new_nzval must have the same
# length as the current buffer.  The block's USTensor is replaced atomically so
# the change is visible to the next mul! or batch_mul! call.

_swap_val(b::USTensor{T,I,N,VA,VI,O}, new_val) where {T,I,N,VA,VI,O} =
    USTensor{eltype(new_val),I,N,typeof(new_val),VI,O}(
        b.extents, b.format, b.pos_buffers, b.crd_buffers, new_val, nothing)

function update_block_values!(A::BlockSparseMatrix, i::Int, j::Int, new_nzval::AbstractVector)
    b = A.blocks[i, j]
    b isa AbstractUSTensor || error("update_block_values!: block ($i,$j) is not a USTensor (got $(typeof(b)))")
    length(new_nzval) == nnz(b) ||
        error("update_block_values!: new_nzval length $(length(new_nzval)) ≠ nnz $(nnz(b))")
    # Same concrete array type: update values in-place without reconstructing the struct.
    # This avoids one allocation and keeps A.blocks[i,j] pointing to the same USTensor.
    if typeof(new_nzval) === typeof(nonzeros(b))
        copyto!(nonzeros(b), new_nzval)
    else
        A.blocks[i, j] = _swap_val(b, new_nzval)
    end
    return A
end

# ─── BlockSparseMatrix: batched multiply ─────────────────────────────────────
#
# batch_mul!(Y, BM, X) computes Y[:,k] = BM * X[:,k] for each column k.
# Y and X are matrices stored column-major.  Each column call is identical to
# mul! on a vector; no extra allocation beyond views into Y and X.
#
# For the common case where only some blocks change between batch elements, call
# update_block_values! for those blocks between batch_mul! calls.

function batch_mul!(Y::AbstractMatrix, A::BlockSparseMatrix, X::AbstractMatrix;
                    backend::Union{AbstractUSTBackend,Nothing}=nothing)
    n_batch = size(X, 2)
    size(Y, 2) == n_batch || error("batch_mul!: Y has $(size(Y,2)) columns, X has $n_batch")
    size(Y, 1) == size(A, 1) || error("batch_mul!: Y rows $(size(Y,1)) ≠ BM rows $(size(A,1))")
    size(X, 1) == size(A, 2) || error("batch_mul!: X rows $(size(X,1)) ≠ BM cols $(size(A,2))")

    for k in 1:n_batch
        LinearAlgebra.mul!(view(Y, :, k), A, view(X, :, k); backend=backend)
    end
    return Y
end

function batch_mul(A::BlockSparseMatrix{T}, X::AbstractMatrix{T}) where T
    Y = similar(X, sum(A.row_sizes), size(X, 2))
    batch_mul!(Y, A, X)
    return Y
end

# ─── LinearAlgebra.mul! for USTensor (alpha/beta and matrix variants) ────────
#
# The plain mul!(y, A::USTensor, x) is defined above (line 39).
# These variants add alpha/beta scaling and matrix-matrix support, enabling
# BlockBandedMatrix to dispatch uniformly regardless of block type.

function LinearAlgebra.mul!(y::AbstractVector, A::USTensor, x::AbstractVector,
                             alpha::Number, beta::Number)
    sparse_mv!(A, x, y; alpha=alpha, beta=beta)
    return y
end

function LinearAlgebra.mul!(C::AbstractMatrix, A::USTensor, B::AbstractMatrix)
    sparse_mm!(A, ust(B), ust(C))
    return C
end

function LinearAlgebra.mul!(C::AbstractMatrix, A::USTensor, B::AbstractMatrix,
                             alpha::Number, beta::Number)
    sparse_mm!(A, ust(B), ust(C); alpha=alpha, beta=beta)
    return C
end

# ─── BlockBandedMatrix ────────────────────────────────────────────────────────
#
# General block-banded matrix for multi-period problems (OPF, MPC, stochastic
# programming).  Row layout interleaves diagonal and coupling rows:
#
#   ┌ D[1]                                   ┐   ← n_diag_rows rows (period 1)
#   │ C[1][1].neg  C[1][1].pos               │   ← n_off_rows[1] rows (bw=1, t=1→2)
#   │ C[2][1].neg           C[2][1].pos      │   ← n_off_rows[2] rows (bw=2, t=1→3)
#   │             D[2]                        │   ← n_diag_rows rows (period 2)
#   │             C[1][2].neg  C[1][2].pos   │   ← n_off_rows[1] rows (bw=1, t=2→3)
#   └                          D[3]           ┘   ← n_diag_rows rows (period 3)
#
# `diags` — either a single AbstractMatrix (repeated, enables batch SpMM) or an
#           AbstractVector of T distinct matrices (time-varying).
# `off_diags` — length-bw vector; off_diags[k] is either:
#   • Tuple{neg, pos}              — same coupling matrices for all T-k transitions
#   • AbstractVector of Tuples     — distinct coupling pair per transition
#
# Julia's parametric type system specializes mul! on the concrete block types.

struct BlockBandedMatrix{D, O<:AbstractVector, BUF, SBUFS}
    diags       :: D
    off_diags   :: O              # length bw
    T           :: Int
    bw          :: Int
    n_diag_rows :: Int
    n_off_rows  :: Vector{Int}    # length bw; rows per coupling level
    n_cols      :: Int
    _buf        :: BUF            # n_diag_rows×T staging buffer, or nothing
    _spmm_bufs  :: SBUFS         # per-block GPU buffers for BlockSparseMatrix SpMM; Nothing on CPU
end

# ── Constructors ──────────────────────────────────────────────────────────────

"""
    BlockBandedMatrix(diags, off_diags, T, bw, n_diag_rows, n_off_rows, n_cols)

General constructor.  `diags` is a single shared `AbstractMatrix` or a length-T
vector of per-period matrices.  `off_diags` is a length-`bw` vector where each
element is either a `(neg, pos)` tuple (shared across all transitions at that
bandwidth) or a vector of `(neg, pos)` tuples (one per transition).
The staging buffer is allocated automatically on the same device as the first
coupling matrix.
"""
function BlockBandedMatrix(diags::D, off_diags::O,
                            T::Int, bw::Int,
                            n_diag_rows::Int, n_off_rows::AbstractVector{Int},
                            n_cols::Int) where {D, O<:AbstractVector}
    n_off_int = collect(Int, n_off_rows)
    buf   = _bbm_alloc_buf(diags, off_diags, n_diag_rows, T)
    sbufs = _combine_gpu_bufs(
        _bbm_alloc_spmm_bufs(diags, buf, T),
        _bbm_alloc_ramp_bufs(off_diags, buf, bw, n_off_int, T))
    BlockBandedMatrix{D, O, typeof(buf), typeof(sbufs)}(
        diags, off_diags, T, bw, n_diag_rows, n_off_int, n_cols, buf, sbufs)
end

# Backward-compatible bw=1 convenience constructor (repeated coupling pair)
function BlockBandedMatrix(diag, neg_off, pos_off,
                            T::Int, n_diag_rows::Int, n_off_rows::Int, n_cols::Int)
    BlockBandedMatrix(diag, [(neg_off, pos_off)], T, 1, n_diag_rows, [n_off_rows], n_cols)
end

# ── Interface ─────────────────────────────────────────────────────────────────

function Base.size(M::BlockBandedMatrix)
    nrows = M.T * M.n_diag_rows +
            sum((M.T - k) * M.n_off_rows[k] for k in 1:M.bw; init=0)
    (nrows, M.T * M.n_cols)
end
Base.size(M::BlockBandedMatrix, d::Int) = size(M)[d]

_bbm_eltype(A::AbstractMatrix) = eltype(A)
_bbm_eltype(A::AbstractVector) = eltype(first(A))
Base.eltype(M::BlockBandedMatrix) = _bbm_eltype(M.diags)

# ── mul! ──────────────────────────────────────────────────────────────────────

function LinearAlgebra.mul!(y::AbstractVector, M::BlockBandedMatrix, x::AbstractVector)
    (; diags, off_diags, T, bw, n_diag_rows, n_off_rows, n_cols, _buf, _spmm_bufs) = M

    # cum_off[k+1] = total off-diagonal rows for levels 1..k; used for row offsets
    cum_off  = cumsum([0; n_off_rows])
    d_period = n_diag_rows + cum_off[end]  # period stride in y (diag + all off rows)

    d_starts = _bbm_diag_starts(n_diag_rows, cum_off, T, bw)
    _bbm_apply_diags!(y, diags, x, d_starts, n_diag_rows, n_cols, _buf, _spmm_bufs)

    # Fused scatter: diag_out[:, t] → y at (t-1)*d_period for each t.
    # One kernel launch instead of T*nb_r tiny copyto!s (48 → 1 for T=24, nb_r=2).
    diag_out = _bbm_diag_out(_spmm_bufs)
    diag_out === nothing || _bbm_scatter_diag!(y, diag_out, d_period, n_diag_rows, T)

    rbufs = _bbm_ramp_bufs(_spmm_bufs)

    for k in 1:bw
        rbuf_k = rbufs === nothing ? nothing : rbufs[k]
        if rbuf_k !== nothing && off_diags[k] isa Tuple
            # Batched ramp SpMM: 2 SpMMs + 1 scatter instead of 2*(T-k) SpMVs.
            neg, pos = off_diags[k]
            T_ramp    = T - k
            x_lo_mat  = reshape(view(x, 1:T_ramp*n_cols), n_cols, T_ramp)
            x_hi_mat  = reshape(view(x, k*n_cols+1:T*n_cols), n_cols, T_ramp)
            LinearAlgebra.mul!(rbuf_k, neg, x_lo_mat)
            LinearAlgebra.mul!(rbuf_k, pos, x_hi_mat, true, true)
            _bbm_scatter_ramp!(y, rbuf_k, d_period, n_diag_rows + cum_off[k], n_off_rows[k], T_ramp)
        else
            for t in 1:T-k
                off_row  = d_starts[t] + n_diag_rows + cum_off[k]
                y_off    = view(y, off_row:off_row+n_off_rows[k]-1)
                x_lo     = view(x, (t-1)*n_cols+1:t*n_cols)
                x_hi     = view(x, (t+k-1)*n_cols+1:(t+k)*n_cols)
                neg, pos = _bbm_get_level(off_diags[k], t)
                LinearAlgebra.mul!(y_off, neg, x_lo)
                LinearAlgebra.mul!(y_off, pos, x_hi, true, true)
            end
        end
    end

    return y
end

# ── Internal helpers ──────────────────────────────────────────────────────────

function _bbm_diag_starts(n_diag_rows, cum_off, T, bw)
    starts = Vector{Int}(undef, T)
    starts[1] = 1
    for t in 1:T-1
        k_max = min(bw, T - t)
        starts[t+1] = starts[t] + n_diag_rows + cum_off[k_max + 1]
    end
    starts
end

# Access coupling pair: Tuple (repeated) or AbstractVector (per-transition)
_bbm_get_level(level::Tuple, t) = level
_bbm_get_level(level::AbstractVector, t) = level[t]

# BlockSparseMatrix, CPU: batch_mul! (contiguous 1D column views, avoids SubArray issue)
function _bbm_apply_diags!(y, diag::BlockSparseMatrix, x,
                            d_starts, n_diag_rows, n_cols, buf::AbstractMatrix, ::Nothing)
    T = size(buf, 2)
    batch_mul!(buf, diag, reshape(x, n_cols, T))
    for t in 1:T
        copyto!(view(y, d_starts[t]:d_starts[t]+n_diag_rows-1), view(buf, :, t))
    end
end

# Returns true when the outermost level is a unique CompressedLevel (DCSR-like).
# Exposed as JLUST._is_row_compressed for profiling utilities.
_is_row_compressed(b::USTensor) =
    let lv1 = b.format.levels[1].second; lv1 isa CompressedLevel && is_unique(lv1); end
_is_row_compressed(_) = false

# Returns true when b is a 2-level CSR USTensor (DenseLevel + unique CompressedLevel)
# with NNZ < n_rows — meaning some rows are empty and the NNZ-first beta=0 kernel
# would waste bandwidth writing zeros for those rows.
# Used by _bbm_apply_diags! to decide: pre-zero row_buf + use guarded SpMM kernel.
_needs_row_guard(b::USTensor) =
    let levels = b.format.levels
        length(levels) == 2 &&
        levels[1].second isa Union{DenseLevel,BatchLevel} &&
        levels[2].second isa CompressedLevel &&
        is_unique(levels[2].second) &&
        length(nonzeros(b)) < extents(b)[1]
    end
_needs_row_guard(_) = false

# BlockSparseMatrix, GPU: EmitterBackend SpMM per non-null block.
# col_bufs eliminated: X2 SubArray views passed directly as B (no gather kernel needed).
# L40S 96 MB L2 absorbs the inter-period stride difference; zero bandwidth penalty.
# row_bufs are SubArray views into diag_out → EmitterBackend dispatched (not cuSPARSE).
#
# Two-pass ordering for mixed dense+sparse rows (e.g. DCOPF row block 1: Bbus+Cg):
#   Pass 1 (dense first, beta=0 then beta=1): writes all rows including zeros for
#           any empty rows — this correctly initializes C for the sparse pass.
#   Pass 2 (sparse, guarded beta=1): accumulates only non-empty rows onto pass-1 result;
#           rows empty in the sparse matrix retain the dense block's correctly-zeroed value.
# Benefit: eliminates fill! kernel AND the dense-block C-read (beta=1 → beta=0),
# saving the dominant Bbus C-read bandwidth (~5.76 MB, ~15 μs per mul! call).
function _bbm_apply_diags!(y, diag::BlockSparseMatrix, x,
                            d_starts, n_diag_rows, n_cols, buf, sbufs)
    T_periods = length(d_starts)
    X2 = reshape(x, n_cols, T_periods)
    nb_r, nb_c = size(diag.blocks)
    (; row_bufs) = sbufs
    for i in 1:nb_r
        # Pass 1: dense blocks (not guard-eligible).  First gets beta=0; rest beta=1.
        first_dense = true
        for j in 1:nb_c
            b = diag.blocks[i, j]; b === nothing && continue
            _needs_row_guard(b) && continue  # handled in pass 2
            col_view = view(X2, diag._col_off[j]+1:diag._col_off[j+1], :)
            sparse_mm!(b, col_view, row_bufs[i];
                       beta = first_dense ? zero(eltype(row_bufs[i])) : one(eltype(row_bufs[i])))
            first_dense = false
        end
        # Pass 2: sparse blocks with guarded accumulation.
        for j in 1:nb_c
            b = diag.blocks[i, j]; b === nothing && continue
            !_needs_row_guard(b) && continue  # handled in pass 1
            col_view = view(X2, diag._col_off[j]+1:diag._col_off[j+1], :)
            if first_dense
                # No dense block ran yet — pre-zero and use guarded beta=0.
                fill!(row_bufs[i], zero(eltype(row_bufs[i])))
                sparse_mm!(b, col_view, row_bufs[i]; beta=0.0, skip_empty_rows=true)
                first_dense = false
            else
                # Dense block already wrote C — use guarded beta=1 to accumulate.
                sparse_mm!(b, col_view, row_bufs[i]; beta=1.0, skip_empty_rows=true)
            end
        end
        # All-null row block: zero the buffer so scatter writes correct zeros.
        first_dense && fill!(row_bufs[i], zero(eltype(row_bufs[i])))
    end
    # Scatter is done by _bbm_scatter_diag! in mul! (one fused kernel vs T*nb_r copyto!s).
end

# USTensor: sparse_mm! supports 2D matrix operands directly.
function _bbm_apply_diags!(y, diag::USTensor, x,
                            d_starts, n_diag_rows, n_cols, buf::AbstractMatrix, sbufs)
    T = size(buf, 2)
    LinearAlgebra.mul!(buf, diag, reshape(x, n_cols, T))
    for t in 1:T
        copyto!(view(y, d_starts[t]:d_starts[t]+n_diag_rows-1), view(buf, :, t))
    end
end

# Repeated matrix (generic): T separate mul! calls
function _bbm_apply_diags!(y, diag::AbstractMatrix, x, d_starts, n_diag_rows, n_cols, buf, sbufs)
    for t in eachindex(d_starts)
        LinearAlgebra.mul!(view(y, d_starts[t]:d_starts[t]+n_diag_rows-1),
                           diag, view(x, (t-1)*n_cols+1:t*n_cols))
    end
end

# Time-varying diagonals: dispatch on each block independently
function _bbm_apply_diags!(y, diags::AbstractVector, x, d_starts, n_diag_rows, n_cols, buf, sbufs)
    for t in eachindex(diags)
        LinearAlgebra.mul!(view(y, d_starts[t]:d_starts[t]+n_diag_rows-1),
                           diags[t], view(x, (t-1)*n_cols+1:t*n_cols))
    end
end

# Buffer allocation: only needed for batchable repeated-diagonal types
function _bbm_alloc_buf(diag::Union{BlockSparseMatrix, USTensor}, off_diags, n_diag_rows, T)
    similar(_bbm_nzref(off_diags[1]), n_diag_rows, T)
end
_bbm_alloc_buf(diag, off_diags, n_diag_rows, T) = nothing

# Per-block GPU buffers for BlockSparseMatrix SpMM.
# Allocates a single stacked diag_out [n_diag, T] buffer; row_bufs are views into it.
# SpMM reads x directly via SubArray views (no gather step), writes to row_bufs which
# are views into diag_out, then the fused scatter kernel copies diag_out → y in one launch.
# No col_bufs needed: L40S 96 MB L2 absorbs the larger stride, zero bandwidth penalty.
# Returns nothing on CPU (batch_mul! is used instead).
function _bbm_alloc_spmm_bufs(diag::BlockSparseMatrix, buf::AbstractMatrix, T)
    _is_cpu_array(buf) && return nothing
    nb_r = size(diag.blocks, 1)
    n_diag   = sum(diag.row_sizes)
    diag_out = similar(buf, n_diag, T)
    row_bufs = [view(diag_out, diag._row_off[i]+1:diag._row_off[i]+diag.row_sizes[i], :)
                for i in 1:nb_r]
    (row_bufs = row_bufs, diag_out = diag_out)
end
_bbm_alloc_spmm_bufs(diag, buf, T) = nothing

# Pre-allocated ramp output buffers (n_off_rows[k] × (T-k)) for batch ramp SpMM.
# Allocated only on GPU when off_diags[k] is a repeated Tuple (batchable).
function _bbm_alloc_ramp_bufs(off_diags, buf, bw, n_off_rows, T)
    buf === nothing && return nothing
    _is_cpu_array(buf) && return nothing
    [
        (off_diags[k] isa Tuple) ? similar(buf, n_off_rows[k], T - k) : nothing
        for k in 1:bw
    ]
end

# Merge diagonal-block GPU bufs (or nothing) with ramp bufs (or nothing).
_combine_gpu_bufs(::Nothing, ::Nothing) = nothing
_combine_gpu_bufs(sbufs::NamedTuple, ::Nothing) = sbufs
_combine_gpu_bufs(::Nothing, rbufs::Vector) = (ramp_bufs = rbufs,)
_combine_gpu_bufs(sbufs::NamedTuple, rbufs::Vector) = merge(sbufs, (ramp_bufs = rbufs,))

# Extract ramp_bufs from sbufs NamedTuple (nothing if absent or if sbufs is nothing).
_bbm_ramp_bufs(::Nothing) = nothing
_bbm_ramp_bufs(sbufs::NamedTuple{names}) where {names} =
    :ramp_bufs ∈ names ? sbufs.ramp_bufs : nothing

# Extract diag_out (the stacked [n_diag, T] scatter source) from sbufs.
_bbm_diag_out(::Nothing) = nothing
_bbm_diag_out(sbufs::NamedTuple{names}) where {names} =
    :diag_out ∈ names ? sbufs.diag_out : nothing

# Fused GPU scatter kernels: implemented in KernelAbstractionsExt.
# _bbm_scatter_diag! scatters diag_out[:, t] → y at (t-1)*d_period + 1 for each t.
# _bbm_scatter_ramp! scatters ramp[:, t] → y at (t-1)*d_period + ramp_off + 1 for each t.
function _bbm_scatter_diag! end
function _bbm_scatter_ramp! end

# Extract a reference array (for device-aware similar()) from a coupling level
_bbm_nzref(level::Tuple) = _bbm_nzref_mat(first(level))
_bbm_nzref(level::AbstractVector) = _bbm_nzref(first(level))
_bbm_nzref_mat(A::USTensor) = nonzeros(A)
_bbm_nzref_mat(A::AbstractMatrix) = A
