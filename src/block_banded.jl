import LinearAlgebra

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
#   • Tuple{neg, pos}          — same coupling matrices for all T-k transitions
#   • AbstractVector of Tuples — distinct coupling pair per transition

struct BlockBandedMatrix{D, O<:AbstractVector, BUF, SBUFS}
    diags       :: D
    off_diags   :: O              # length bw
    T           :: Int
    bw          :: Int
    n_diag_rows :: Int
    n_off_rows  :: Vector{Int}    # length bw; rows per coupling level
    n_cols      :: Int
    _buf        :: BUF            # n_diag_rows×T staging buffer, or nothing
    _spmm_bufs  :: SBUFS         # per-block GPU buffers; nothing on CPU
end

# ── Constructors ──────────────────────────────────────────────────────────────

"""
    BlockBandedMatrix(diags, off_diags, T, bw, n_diag_rows, n_off_rows, n_cols)

General constructor.  `diags` is a single shared `AbstractMatrix` or a length-T
vector of per-period matrices.  `off_diags` is a length-`bw` vector where each
element is either a `(neg, pos)` tuple (shared across all transitions at that
bandwidth) or a vector of `(neg, pos)` tuples (one per transition).
"""
function BlockBandedMatrix(diags::D, off_diags::O,
                            T::Int, bw::Int,
                            n_diag_rows::Int, n_off_rows::AbstractVector{Int},
                            n_cols::Int) where {D, O<:AbstractVector}
    n_off_int = collect(Int, n_off_rows)
    buf   = _bbm_alloc_buf(diags, off_diags, n_diag_rows, T)
    sbufs = _combine_gpu_bufs(
        _bbm_alloc_spmm_bufs(diags, buf, T),
        _bbm_alloc_off_bufs(off_diags, buf, bw, n_off_int, T))
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

# ── BBMSpMV: execute(BBMSpMVOp, M, x, y) ─────────────────────────────────────
# `mul!(y, M::BlockBandedMatrix, x)` is a thin wrapper around `execute`.

function execute(::Type{<:Op{:BBMSpMV}}, M::BlockBandedMatrix,
                 x::AbstractVector, y::AbstractVector;
                 backend::Union{AbstractUSTBackend,Nothing}=nothing)
    (; diags, off_diags, T, bw, n_diag_rows, n_off_rows, n_cols, _buf, _spmm_bufs) = M

    cum_off  = cumsum([0; n_off_rows])
    d_period = n_diag_rows + cum_off[end]  # period stride in y

    d_starts = _bbm_diag_starts(n_diag_rows, cum_off, T, bw)
    _bbm_apply_diags!(y, diags, x, d_starts, n_diag_rows, n_cols, _buf, _spmm_bufs)

    # Fused scatter: diag_out[:, t] → y at (t-1)*d_period for each t.
    diag_out = _bbm_diag_out(_spmm_bufs)
    diag_out === nothing || _bbm_scatter_diag!(y, diag_out, d_period, n_diag_rows, T)

    off_bufs = _bbm_off_bufs(_spmm_bufs)

    for k in 1:bw
        off_buf_k = off_bufs === nothing ? nothing : off_bufs[k]
        if off_buf_k !== nothing && off_diags[k] isa Tuple
            # Batched off-diagonal SpMM: 2 SpMMs + 1 scatter instead of 2*(T-k) SpMVs.
            # Only available when coupling matrices are shared (Tuple) across all transitions.
            neg, pos  = off_diags[k]
            T_off     = T - k
            x_lo_mat  = reshape(view(x, 1:T_off*n_cols), n_cols, T_off)
            x_hi_mat  = reshape(view(x, k*n_cols+1:T*n_cols), n_cols, T_off)
            LinearAlgebra.mul!(off_buf_k, neg, x_lo_mat)
            LinearAlgebra.mul!(off_buf_k, pos, x_hi_mat, true, true)
            _bbm_scatter_off!(y, off_buf_k, d_period, n_diag_rows + cum_off[k], n_off_rows[k], T_off)
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

LinearAlgebra.mul!(y::AbstractVector, M::BlockBandedMatrix, x::AbstractVector;
                    backend::Union{AbstractUSTBackend,Nothing}=nothing) =
    execute(BBMSpMVOp, M, x, y; backend=backend)

# ── Internal helpers ──────────────────────────────────────────────────────────

function _bbm_diag_starts(n_diag_rows, cum_off, T, bw)
    starts    = Vector{Int}(undef, T)
    starts[1] = 1
    for t in 1:T-1
        k_max        = min(bw, T - t)
        starts[t+1]  = starts[t] + n_diag_rows + cum_off[k_max + 1]
    end
    starts
end

# Access coupling pair: Tuple (repeated) or AbstractVector (per-transition)
_bbm_get_level(level::Tuple,          t) = level
_bbm_get_level(level::AbstractVector, t) = level[t]

# ── _bbm_apply_diags! dispatch ────────────────────────────────────────────────
#
# One method per diagonal block type.  GPU BlockSparseMatrix uses a two-pass
# ordering for mixed dense+sparse row blocks: dense blocks first (beta=0 then
# beta=1), then sparse blocks with guarded beta=1 for rows the dense block
# already initialized.  See needs_row_guard for the predicate that gates pass 2.

# BlockSparseMatrix, CPU: batch_mul! into staging buffer, then scatter.
function _bbm_apply_diags!(y, diag::BlockSparseMatrix, x,
                            d_starts, n_diag_rows, n_cols, buf::AbstractMatrix, ::Nothing)
    T = size(buf, 2)
    batch_mul!(buf, diag, reshape(x, n_cols, T))
    for t in 1:T
        copyto!(view(y, d_starts[t]:d_starts[t]+n_diag_rows-1), view(buf, :, t))
    end
end

# BlockSparseMatrix, GPU: EmitterBackend SpMM per non-null block.
#
# Two-pass ordering avoids a fill! while correctly handling blocks where some
# rows are dense and others are sparse:
#   Pass 1 (non-guarded blocks, beta=0 then beta=1): initialises all output rows,
#           writing zeros for rows empty in the dense block.
#   Pass 2 (guarded blocks, beta=1): accumulates onto the pass-1 result; rows
#           absent from the sparse block keep the dense block's value unchanged.
function _bbm_apply_diags!(y, diag::BlockSparseMatrix, x,
                            d_starts, n_diag_rows, n_cols, buf, sbufs)
    T_periods  = length(d_starts)
    X2         = reshape(x, n_cols, T_periods)
    nb_r, nb_c = size(diag.blocks)
    (; row_bufs) = sbufs
    for i in 1:nb_r
        first_dense = true
        # Pass 1: blocks that do not need a row guard.
        for j in 1:nb_c
            b = diag.blocks[i, j]; b === nothing && continue
            needs_row_guard(b) && continue
            col_view = view(X2, diag._col_off[j]+1:diag._col_off[j+1], :)
            execute(SpMMOp, b, ust(col_view), ust(row_bufs[i]);
                       beta = first_dense ? zero(eltype(row_bufs[i])) : one(eltype(row_bufs[i])))
            first_dense = false
        end
        # Pass 2: blocks that need a row guard (sparse rows).
        for j in 1:nb_c
            b = diag.blocks[i, j]; b === nothing && continue
            needs_row_guard(b) || continue
            col_view = view(X2, diag._col_off[j]+1:diag._col_off[j+1], :)
            if first_dense
                fill!(row_bufs[i], zero(eltype(row_bufs[i])))
                execute(SpMMOp, b, ust(col_view), ust(row_bufs[i]); beta=0.0, skip_empty_rows=true)
                first_dense = false
            else
                execute(SpMMOp, b, ust(col_view), ust(row_bufs[i]); beta=1.0, skip_empty_rows=true)
            end
        end
        first_dense && fill!(row_bufs[i], zero(eltype(row_bufs[i])))
    end
end

# USTensor diagonal: SpMM into staging buffer, then scatter.
function _bbm_apply_diags!(y, diag::USTensor, x,
                            d_starts, n_diag_rows, n_cols, buf::AbstractMatrix, sbufs)
    T = size(buf, 2)
    LinearAlgebra.mul!(buf, diag, reshape(x, n_cols, T))
    for t in 1:T
        copyto!(view(y, d_starts[t]:d_starts[t]+n_diag_rows-1), view(buf, :, t))
    end
end

# Repeated matrix (generic): T separate mul! calls.
function _bbm_apply_diags!(y, diag::AbstractMatrix, x, d_starts, n_diag_rows, n_cols, buf, sbufs)
    for t in eachindex(d_starts)
        LinearAlgebra.mul!(view(y, d_starts[t]:d_starts[t]+n_diag_rows-1),
                           diag, view(x, (t-1)*n_cols+1:t*n_cols))
    end
end

# Time-varying diagonals: dispatch on each block independently.
function _bbm_apply_diags!(y, diags::AbstractVector, x, d_starts, n_diag_rows, n_cols, buf, sbufs)
    for t in eachindex(diags)
        LinearAlgebra.mul!(view(y, d_starts[t]:d_starts[t]+n_diag_rows-1),
                           diags[t], view(x, (t-1)*n_cols+1:t*n_cols))
    end
end

# ── Buffer allocation ─────────────────────────────────────────────────────────

# Staging buffer: only needed for batchable repeated-diagonal types.
function _bbm_alloc_buf(diag::Union{BlockSparseMatrix, USTensor}, off_diags, n_diag_rows, T)
    similar(_bbm_nzref(off_diags[1]), n_diag_rows, T)
end
_bbm_alloc_buf(diag, off_diags, n_diag_rows, T) = nothing

# Per-block GPU buffers for BlockSparseMatrix SpMM.
# Allocates a single stacked diag_out [n_diag, T] buffer; row_bufs are views into it.
# SpMM reads x directly via SubArray views; scatter kernel copies diag_out → y in one launch.
function _bbm_alloc_spmm_bufs(diag::BlockSparseMatrix, buf::AbstractMatrix, T)
    _is_cpu_array(buf) && return nothing
    nb_r     = size(diag.blocks, 1)
    n_diag   = sum(diag.row_sizes)
    diag_out = similar(buf, n_diag, T)
    row_bufs = [view(diag_out, diag._row_off[i]+1:diag._row_off[i]+diag.row_sizes[i], :)
                for i in 1:nb_r]
    (row_bufs = row_bufs, diag_out = diag_out)
end
_bbm_alloc_spmm_bufs(diag, buf, T) = nothing

# Pre-allocated off-diagonal output buffers (n_off_rows[k] × (T-k)) for batched SpMM.
# Allocated on GPU only when off_diags[k] is a repeated Tuple (same matrices for all
# transitions at bandwidth k) — otherwise per-transition SpMVs are used instead.
function _bbm_alloc_off_bufs(off_diags, buf, bw, n_off_rows, T)
    buf === nothing && return nothing
    _is_cpu_array(buf) && return nothing
    [(off_diags[k] isa Tuple) ? similar(buf, n_off_rows[k], T - k) : nothing for k in 1:bw]
end

# Merge diagonal-block GPU bufs and off-diagonal bufs into a single NamedTuple (or nothing).
_combine_gpu_bufs(::Nothing,         ::Nothing)     = nothing
_combine_gpu_bufs(sbufs::NamedTuple, ::Nothing)     = sbufs
_combine_gpu_bufs(::Nothing,         bufs::Vector)  = (off_bufs = bufs,)
_combine_gpu_bufs(sbufs::NamedTuple, bufs::Vector)  = merge(sbufs, (off_bufs = bufs,))

# Accessors for optional fields of the sbufs NamedTuple.
_bbm_off_bufs(::Nothing) = nothing
_bbm_off_bufs(sbufs::NamedTuple{names}) where {names} =
    :off_bufs ∈ names ? sbufs.off_bufs : nothing

_bbm_diag_out(::Nothing) = nothing
_bbm_diag_out(sbufs::NamedTuple{names}) where {names} =
    :diag_out ∈ names ? sbufs.diag_out : nothing

# Reference array for device-aware similar() from a coupling level.
_bbm_nzref(level::Tuple)          = _bbm_nzref_mat(first(level))
_bbm_nzref(level::AbstractVector) = _bbm_nzref(first(level))
_bbm_nzref_mat(A::USTensor)       = nonzeros(A)
_bbm_nzref_mat(A::AbstractMatrix) = A

# ── Fused GPU scatter kernels (implemented in KernelAbstractionsExt) ──────────
#
# _bbm_scatter_diag! scatters diag_out[:, t] → y at (t-1)*d_period + 1 for each t.
# _bbm_scatter_off!  scatters off_buf[:, t]  → y at (t-1)*d_period + off_start + 1.
function _bbm_scatter_diag! end
function _bbm_scatter_off! end
