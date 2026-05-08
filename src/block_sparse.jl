import LinearAlgebra

# ─── CPU array predicate ─────────────────────────────────────────────────────
#
# Used by both BlockSparseMatrix (fused CPU path) and BlockBandedMatrix
# (buffer allocation decisions).

_is_cpu_array(::Array)         = true
_is_cpu_array(x::SubArray)     = _is_cpu_array(parent(x))
_is_cpu_array(::AbstractArray) = false

# ─── BlockSparseMatrix ────────────────────────────────────────────────────────
#
# A block-structured sparse matrix whose entries are USTensors or nothing (zero
# blocks).  Each block row may contain any mix of formats; blocks in the same
# block-column must have the same column count, and blocks in the same block-row
# must have the same row count.
#
# Construction:
#   A = BlockSparseMatrix([u_Cg   nothing  u_negB;
#                          u_PTDF u_negI   nothing])
#
# The matrix-literal syntax produces a Matrix{Any} in Julia when elements have
# mixed types (USTensor subtypes and Nothing), which is exactly what the
# constructor expects.

struct BlockSparseMatrix{T}
    blocks    :: Matrix{Union{Nothing, AbstractUSTensor}}
    row_sizes :: Vector{Int}   # matrix rows per block row
    col_sizes :: Vector{Int}   # matrix cols per block col
    _row_off  :: Vector{Int}   # cumulative row offsets (length nb_rows+1)
    _col_off  :: Vector{Int}   # cumulative col offsets (length nb_cols+1)
end

function BlockSparseMatrix(blocks::Matrix)
    nb_r, nb_c = size(blocks)
    row_sizes = zeros(Int, nb_r)
    col_sizes = zeros(Int, nb_c)
    ref_T     = nothing

    for i in 1:nb_r, j in 1:nb_c
        b = blocks[i, j]
        b === nothing && continue
        b isa AbstractUSTensor ||
            error("BlockSparseMatrix: block ($i,$j) must be a USTensor or nothing, got $(typeof(b))")

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
        elseif ref_T != T_b
            error("BlockSparseMatrix: mixed element types ($ref_T vs $T_b at block ($i,$j))")
        end
    end

    any(==(0), row_sizes) &&
        error("BlockSparseMatrix: block row(s) $(findall(==(0), row_sizes)) are entirely nothing")
    any(==(0), col_sizes) &&
        error("BlockSparseMatrix: block col(s) $(findall(==(0), col_sizes)) are entirely nothing")
    ref_T === nothing && error("BlockSparseMatrix: all blocks are nothing")

    BlockSparseMatrix{ref_T}(
        Matrix{Union{Nothing, AbstractUSTensor}}(blocks),
        row_sizes, col_sizes,
        cumsum([0; row_sizes]), cumsum([0; col_sizes]))
end

Base.size(A::BlockSparseMatrix)         = (sum(A.row_sizes), sum(A.col_sizes))
Base.size(A::BlockSparseMatrix, d::Int) = sum(d == 1 ? A.row_sizes : A.col_sizes)
Base.eltype(::BlockSparseMatrix{T}) where T = T

# ─── Fused CPU SpMV ──────────────────────────────────────────────────────────
#
# When all non-null blocks are CPU-resident and fusable (CSR, DCSR, or any
# 2-level format whose inner level implements level_step), skip the per-block
# KA dispatch and run a single fused Julia pass over all blocks.
#
# Falls back to the standard KA path for GPU arrays, unfusable formats,
# or when an explicit backend is provided.

function _cpu_csr_accumulate!(y::AbstractVector{T}, rp, ci, nz,
                               x::AbstractVector{T}, y_lo::Int, x_lo::Int, m::Int) where T
    @inbounds for row in 1:m
        acc = zero(T)
        for k in Int(rp[row]) : Int(rp[row+1]) - 1
            acc += nz[k] * x[x_lo + Int(ci[k])]
        end
        y[y_lo + row] += acc
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

# Custom inner level formats opt into the fused CPU path by defining level_step.
# _is_cpu_fusable_level detects this via hasmethod; _cpu_level_accumulate! loops
# over level_step calls.  No extra methods needed.
_is_cpu_fusable_level(lv::AbstractLevelFormat) =
    hasmethod(JLUST.level_step, Tuple{typeof(lv), Int, Nothing}) ||
    hasmethod(JLUST.level_step, Tuple{typeof(lv), Int, AbstractVector})
_is_cpu_fusable_level(::DenseLevel)      = true
_is_cpu_fusable_level(::CompressedLevel) = true

function _is_cpu_fusable(b::AbstractUSTensor)
    length(b.format.levels) == 2 || return false
    _is_cpu_array(nonzeros(b))   || return false
    outer_lv = b.format.levels[1]
    inner_lv = b.format.levels[2]
    (outer_lv isa DenseLevel && _is_cpu_fusable_level(inner_lv)) ||
    (outer_lv isa CompressedLevel && inner_lv isa CompressedLevel)
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

function _cpu_block_accumulate!(b::AbstractUSTensor, y, x, y_lo, x_lo, m)
    outer_lv = b.format.levels[1]
    if outer_lv isa DenseLevel
        _cpu_level_accumulate!(b.format.levels[2], b, y, x, y_lo, x_lo, m)
    else  # CompressedLevel outer (DCSR)
        _cpu_dcsr_accumulate!(y, coordinates(b, 1), positions(b, 2), coordinates(b, 2),
                              nonzeros(b), x, y_lo, x_lo)
    end
end

function _try_fused_cpu_mul!(y::AbstractVector, A::BlockSparseMatrix, x::AbstractVector)
    nb_r, nb_c = size(A.blocks)
    for i in 1:nb_r, j in 1:nb_c
        b = A.blocks[i, j]; b === nothing && continue
        _is_cpu_fusable(b) || return false
    end
    fill!(y, zero(eltype(A)))
    for i in 1:nb_r
        y_lo = A._row_off[i]
        for j in 1:nb_c
            b = A.blocks[i, j]; b === nothing && continue
            _cpu_block_accumulate!(b, y, x, y_lo, A._col_off[j], size(b, 1))
        end
    end
    return true
end

# ─── Fused CPU SpMM ──────────────────────────────────────────────────────────
#
# Multi-RHS variant: each NNZ is loaded once and scattered across all n_col
# output columns in the innermost loop, which the compiler can vectorize.

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
    outer_lv = b.format.levels[1]
    if outer_lv isa DenseLevel
        inner_lv = b.format.levels[2]
        if inner_lv isa CompressedLevel
            _cpu_csr_accumulate_mm!(Y, positions(b, 2), coordinates(b, 2), nonzeros(b),
                                     X, y_lo, x_lo, m)
        else
            _cpu_level_accumulate_mm!(inner_lv, b, Y, X, y_lo, x_lo, m)
        end
    else  # DCSR
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

# ─── Backend dispatch helpers ─────────────────────────────────────────────────
#
# Resolve `nothing` backend via default dispatch (resolved in KernelAbstractionsExt)
# vs explicit backend via positional arg.  Avoids an if/else inside the inner loop.

_bsm_mv!(::Nothing, b, x, y, β) = sparse_mv!(b, x, y; beta=β)
_bsm_mv!(be, b, x, y, β)        = sparse_mv!(be, b, x, y; beta=β)

_bsm_mm!(::Nothing, b, X, Y, β) = sparse_mm!(b, X, Y; beta=β)
_bsm_mm!(be, b, X, Y, β)        = sparse_mm!(be, b, X, Y; beta=β)

# ─── SpMV: mul!(y, A, x) ─────────────────────────────────────────────────────

function LinearAlgebra.mul!(y::AbstractVector, A::BlockSparseMatrix, x::AbstractVector;
                             backend::Union{AbstractUSTBackend,Nothing}=nothing)
    # Fast path: fused Julia loop for CPU-resident fusable blocks.
    if backend === nothing && _is_cpu_array(y) && _is_cpu_array(x)
        _try_fused_cpu_mul!(y, A, x) && return y
    end

    # General path: one kernel per non-null block.
    # No fill! needed: the first non-null block per row uses beta=false (ZERO_BETA),
    # which initialises empty rows to 0.
    nb_r, nb_c = size(A.blocks)
    for i in 1:nb_r
        y_sl      = view(y, A._row_off[i]+1 : A._row_off[i+1])
        first_col = true
        for j in 1:nb_c
            b = A.blocks[i, j]; b === nothing && continue
            x_sl = view(x, A._col_off[j]+1 : A._col_off[j+1])
            _bsm_mv!(backend, b, x_sl, y_sl, first_col ? false : true)
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

# ─── SpMM: mul!(Y, A, X) ─────────────────────────────────────────────────────

function LinearAlgebra.mul!(Y::AbstractMatrix, A::BlockSparseMatrix, X::AbstractMatrix;
                             backend::Union{AbstractUSTBackend,Nothing}=nothing)
    if backend === nothing && _is_cpu_array(Y) && _is_cpu_array(X)
        _try_fused_cpu_mul_mm!(Y, A, X) && return Y
    end

    nb_r, nb_c = size(A.blocks)
    fill!(Y, zero(eltype(A)))
    for i in 1:nb_r
        Y_sl = view(Y, A._row_off[i]+1 : A._row_off[i+1], :)
        for j in 1:nb_c
            b = A.blocks[i, j]; b === nothing && continue
            X_sl = view(X, A._col_off[j]+1 : A._col_off[j+1], :)
            _bsm_mm!(backend, b, X_sl, Y_sl, one(eltype(A)))
        end
    end
    return Y
end

function Base.:*(A::BlockSparseMatrix{T}, X::AbstractMatrix{T}) where T
    Y = similar(X, sum(A.row_sizes), size(X, 2))
    LinearAlgebra.mul!(Y, A, X)
    return Y
end

# ─── Selective value updates ──────────────────────────────────────────────────
#
# update_block_values!(BM, i, j, new_nzval) swaps the nonzero value buffer of
# block (i,j) without touching its index buffers.  new_nzval must have the same
# length as the current buffer.
#
# Same concrete array type → copyto! in-place (pointer unchanged; cached CUDA
# graphs remain valid).  Different type → reconstruct USTensor with new pointer
# (CUDAExt overrides this method to also clear the graph cache).

_swap_val(b::USTensor{T,I,N,VA,VI,O}, new_val) where {T,I,N,VA,VI,O} =
    USTensor{eltype(new_val),I,N,typeof(new_val),VI,O}(
        b.extents, b.format, b.pos_buffers, b.crd_buffers, new_val, nothing)

function update_block_values!(A::BlockSparseMatrix, i::Int, j::Int, new_nzval::AbstractVector)
    b = A.blocks[i, j]
    b isa AbstractUSTensor ||
        error("update_block_values!: block ($i,$j) is not a USTensor (got $(typeof(b)))")
    length(new_nzval) == nnz(b) ||
        error("update_block_values!: new_nzval length $(length(new_nzval)) ≠ nnz $(nnz(b))")
    if typeof(new_nzval) === typeof(nonzeros(b))
        copyto!(nonzeros(b), new_nzval)
    else
        A.blocks[i, j] = _swap_val(b, new_nzval)
    end
    return A
end

# ─── Batched multiply ─────────────────────────────────────────────────────────
#
# batch_mul!(Y, BM, X) computes Y[:,k] = BM * X[:,k] for each column k.
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
