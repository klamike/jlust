module Formats

using ..JLUST: Dimension, LevelExpr, dims, TensorFormat, @tensor_format,
               DenseLevel, BatchLevel, CompressedLevel, SingletonLevel, RangeLevel, DeltaLevel,
               dim2lvl, lvl2dim, InvalidTensorFormat

# ─── Scalar ───────────────────────────────────────────────────────────────────

const Scalar = TensorFormat((), (); name=:Scalar)

# ─── Vector formats ───────────────────────────────────────────────────────────

@tensor_format DenseVector  (i,) -> (i : dense,)
@tensor_format SparseVector (i,) -> (i : compressed,)

# ─── Dense matrix formats ─────────────────────────────────────────────────────

@tensor_format DenseMatrixRight (i, j) -> (i : dense, j : dense)
@tensor_format DenseMatrixLeft  (i, j) -> (j : dense, i : dense)

# ─── Standard sparse matrix formats ──────────────────────────────────────────

@tensor_format COO  (i, j) -> (i : compressed(nonunique), j : singleton)
@tensor_format CSR  (i, j) -> (i : dense, j : compressed)
@tensor_format CSC  (i, j) -> (j : dense, i : compressed)
@tensor_format DCSR (i, j) -> (i : compressed, j : compressed)
@tensor_format DCSC (i, j) -> (j : compressed, i : compressed)
@tensor_format CROW (i, j) -> (i : compressed, j : dense)
@tensor_format CCOL (i, j) -> (j : compressed, i : dense)

# ─── Diagonal formats ─────────────────────────────────────────────────────────

@tensor_format DIAI    (i, j) -> ((j - i) : compressed, i : range)
@tensor_format DIAJ    (i, j) -> ((j - i) : compressed, j : range)
@tensor_format SkewDIAI (i, j) -> ((i + j) : compressed, i : range)
@tensor_format SkewDIAJ (i, j) -> ((i + j) : compressed, j : range)

# ─── Batched formats ──────────────────────────────────────────────────────────

@tensor_format BatchedCSR (batch, i, j) -> (batch : dense, i : dense, j : compressed)
@tensor_format BatchedDIAINonUniform (batch, i, j) -> (batch : dense, (j - i) : compressed, i : range)
@tensor_format BatchedDIAIUniform    (batch, i, j) -> ((j - i) : compressed, batch : dense, i : range)

# ─── Parameterized format builders ────────────────────────────────────────────

function BSRRight(blocksize::Tuple{Int,Int})
    b1, b2 = blocksize
    i, j = dims(:i, :j)
    TensorFormat(
        [i, j],
        [i÷b1 => DenseLevel(), j÷b2 => CompressedLevel(),
         i%b1 => DenseLevel(), j%b2 => DenseLevel()];
        name   = Symbol("BSRRight$(b1)x$(b2)"),
        family = :BSR,
    )
end

function BSRLeft(blocksize::Tuple{Int,Int})
    b1, b2 = blocksize
    i, j = dims(:i, :j)
    TensorFormat(
        [i, j],
        [i÷b1 => DenseLevel(), j÷b2 => CompressedLevel(),
         j%b2 => DenseLevel(), i%b1 => DenseLevel()];
        name   = Symbol("BSRLeft$(b1)x$(b2)"),
        family = :BSR,
    )
end

function BSCRight(blocksize::Tuple{Int,Int})
    b1, b2 = blocksize
    i, j = dims(:i, :j)
    TensorFormat(
        [i, j],
        [j÷b2 => DenseLevel(), i÷b1 => CompressedLevel(),
         i%b1 => DenseLevel(), j%b2 => DenseLevel()];
        name   = Symbol("BSCRight$(b1)x$(b2)"),
        family = :BSC,
    )
end

function BSCLeft(blocksize::Tuple{Int,Int})
    b1, b2 = blocksize
    i, j = dims(:i, :j)
    TensorFormat(
        [i, j],
        [j÷b2 => DenseLevel(), i÷b1 => CompressedLevel(),
         j%b2 => DenseLevel(), i%b1 => DenseLevel()];
        name   = Symbol("BSCLeft$(b1)x$(b2)"),
        family = :BSC,
    )
end

function DELTA(delta::Int)
    i, j = dims(:i, :j)
    TensorFormat(
        [i, j],
        [i => DenseLevel(), j => DeltaLevel(delta)];
        name   = Symbol("Delta$(delta)"),
        family = :Delta,
    )
end

function BSR3(blocksize::Tuple{Int,Int,Int})
    b1, b2, b3 = blocksize
    i, j, k = dims(:i, :j, :k)
    TensorFormat(
        [i, j, k],
        [i÷b1 => DenseLevel(), j÷b2 => CompressedLevel(), k÷b3 => CompressedLevel(),
         i%b1 => DenseLevel(), j%b2 => DenseLevel(),       k%b3 => DenseLevel()];
        name   = Symbol("BSR3$(b1)x$(b2)x$(b3)"),
        family = :BSR,
    )
end

function BlockVector(blocksize::Int)
    i, = dims(:i)
    TensorFormat(
        [i],
        [i÷blocksize => CompressedLevel(), i%blocksize => DenseLevel()];
        name   = Symbol("BlockVector$(blocksize)"),
        family = :BlockVector,
    )
end

# ─── SELL (Sliced ELL) ────────────────────────────────────────────────────────
# Rows grouped into slices of `slice_size`; columns padded to max nnz per slice.
# Buffers: sellSliceOffsets[nslices+1], sellColInd[padded], sellValues[padded].
# cuSPARSE: cusparseCreateSlicedEll; SpMV only (CUSPARSE_SPMV_SELL_ALG1).
# No CuSparseMatrixSELL type in CUDA.jl — raw buffer assembly required.

function SELL(slice_size::Int)
    i, j = dims(:i, :j)
    TensorFormat(
        [i, j],
        [i÷slice_size => CompressedLevel(), j => DenseLevel(), i%slice_size => DenseLevel()];
        name   = Symbol("SELL$(slice_size)"),
        family = :SELL,
    )
end

# ─── Blocked ELL ──────────────────────────────────────────────────────────────
# Rows grouped into blocks of `block_size`; each row-block has ellCols column-block
# entries (padded).  Buffers: ellColInd[nblocks, ellCols], ellValue[nrows, ellCols*B].
# cuSPARSE: cusparseCreateBlockedEll; SpMM only (CUSPARSE_SPMM_BLOCKED_ELL_ALG1).
# No CuSparseMatrixBlockedELL type in CUDA.jl — raw buffer assembly required.

function BlockedELL(block_size::Int)
    i, j = dims(:i, :j)
    TensorFormat(
        [i, j],
        [i÷block_size => DenseLevel(), j÷block_size => DenseLevel(),
         i%block_size => DenseLevel(), j%block_size => DenseLevel()];
        name   = Symbol("BlockedELL$(block_size)"),
        family = :BlockedELL,
    )
end

function DensedRight(dim::Int)
    ds = [Dimension(Symbol(Char(Int('i') + d))) for d in 0:dim-1]
    TensorFormat(
        ds,
        [d => DenseLevel() for d in ds];
        name   = Symbol("Dense$(dim)Right"),
        family = :Dense,
    )
end

function DensedLeft(dim::Int)
    ds = [Dimension(Symbol(Char(Int('i') + d))) for d in 0:dim-1]
    TensorFormat(
        ds,
        [ds[dim-d] => DenseLevel() for d in 0:dim-1];
        name   = Symbol("Dense$(dim)Left"),
        family = :Dense,
    )
end

function COOd(sparse_dim::Int, dense_dim::Int=0)
    dim = sparse_dim + dense_dim
    ds  = [Dimension(Symbol(Char(Int('i') + d))) for d in 0:dim-1]
    lvls = [
        ds[d+1] => (
            d == 0 ? (sparse_dim == 1 ? CompressedLevel() : CompressedLevel(unique=false)) :
            d < sparse_dim ? SingletonLevel() :
            DenseLevel()
        )
        for d in 0:dim-1
    ]
    name_sym = dense_dim == 0 ? (dim == 2 ? :COO : Symbol("COO$(dim)")) : nothing
    TensorFormat(
        ds, lvls;
        name   = isnothing(name_sym) ? Symbol(repr(lvls)) : name_sym,
        family = :COO,
    )
end

function CSRd(batch_dim::Int=0, dense_dim::Int=0)
    dim = batch_dim + 2 + dense_dim
    ds  = [Dimension(Symbol(Char(Int('i') + d))) for d in 0:dim-1]
    lvls = [
        ds[d+1] => (
            d < batch_dim          ? BatchLevel() :
            d == batch_dim + 1     ? CompressedLevel() :
                                     DenseLevel()
        )
        for d in 0:dim-1
    ]
    name_str = "CSR" *
               (batch_dim > 0 ? "-b$(batch_dim)" : "") *
               (dense_dim > 0 ? "-d$(dense_dim)" : "")
    TensorFormat(ds, lvls; name=Symbol(name_str), family=:CSR)
end

function CSCd(batch_dim::Int=0, dense_dim::Int=0)
    dim = batch_dim + 2 + dense_dim
    ds  = [Dimension(Symbol(Char(Int('i') + d))) for d in 0:dim-1]
    lvls = [
        ds[d == batch_dim ? d+2 : d == batch_dim+1 ? d : d+1] => (
            d < batch_dim      ? BatchLevel() :
            d == batch_dim + 1 ? CompressedLevel() :
                                 DenseLevel()
        )
        for d in 0:dim-1
    ]
    name_str = "CSC" *
               (batch_dim > 0 ? "-b$(batch_dim)" : "") *
               (dense_dim > 0 ? "-d$(dense_dim)" : "")
    TensorFormat(ds, lvls; name=Symbol(name_str), family=:CSC)
end

function BSRRightd(blocksize::Tuple{Int,Int}, batch_dim::Int=0, dense_dim::Int=0)
    b1, b2 = blocksize
    dim    = batch_dim + 2 + dense_dim
    ds     = [Dimension(Symbol(Char(Int('i') + d))) for d in 0:dim-1]
    bi, bj = ds[batch_dim+1], ds[batch_dim+2]
    batch_pairs = [ds[d+1] => BatchLevel() for d in 0:batch_dim-1]
    block_pairs = [bi÷b1 => DenseLevel(), bj÷b2 => CompressedLevel(),
                   bi%b1 => DenseLevel(), bj%b2 => DenseLevel()]
    dense_pairs = [ds[d+1] => DenseLevel() for d in batch_dim+2:dim-1]
    name_str = "BSRRight$(b1)x$(b2)" *
               (batch_dim > 0 ? "-b$(batch_dim)" : "") *
               (dense_dim > 0 ? "-d$(dense_dim)" : "")
    TensorFormat(ds, [batch_pairs; block_pairs; dense_pairs]; name=Symbol(name_str), family=:BSR)
end

function BSCRightd(blocksize::Tuple{Int,Int}, batch_dim::Int=0, dense_dim::Int=0)
    b1, b2 = blocksize
    dim    = batch_dim + 2 + dense_dim
    ds     = [Dimension(Symbol(Char(Int('i') + d))) for d in 0:dim-1]
    bi, bj = ds[batch_dim+1], ds[batch_dim+2]
    batch_pairs = [ds[d+1] => BatchLevel() for d in 0:batch_dim-1]
    block_pairs = [bj÷b2 => DenseLevel(), bi÷b1 => CompressedLevel(),
                   bi%b1 => DenseLevel(), bj%b2 => DenseLevel()]
    dense_pairs = [ds[d+1] => DenseLevel() for d in batch_dim+2:dim-1]
    name_str = "BSCRight$(b1)x$(b2)" *
               (batch_dim > 0 ? "-b$(batch_dim)" : "") *
               (dense_dim > 0 ? "-d$(dense_dim)" : "")
    TensorFormat(ds, [batch_pairs; block_pairs; dense_pairs]; name=Symbol(name_str), family=:BSC)
end

function CSFd(dim::Int)
    ds = [Dimension(Symbol(Char(Int('i') + d))) for d in 0:dim-1]
    TensorFormat(
        ds,
        [d => CompressedLevel() for d in ds];
        name   = Symbol("CSF$(dim)"),
        family = :CSF,
    )
end

end  # module Formats
