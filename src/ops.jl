# ─── Op types ─────────────────────────────────────────────────────────────────
#
# Capability queries for multi-operand operations require the full operand
# signature, not just a single format.  Structured op types carry that
# information and allow dispatch to specialize on both op class and backend.
#
# Dense operands use the appropriate Formats.DensedRight(N) format — dense is
# not a special case, just an all-DenseLevel TensorFormat.
#
# Vector operands (1-D dense) use Formats.DensedRight(1).

abstract type AbstractUSTOp end

# ─── Compute ops ──────────────────────────────────────────────────────────────

"""
    SpVVOp(x_fmt, y_fmt)

Sparse vector dot product: result = xᵀ·y.
cuSPARSE: `cusparseDotEx` / `cusparseSpVV`.
"""
struct SpVVOp <: AbstractUSTOp
    x::TensorFormat
    y::TensorFormat
end

"""
    SpMVOp(A_fmt, x_fmt, y_fmt)

Sparse matrix–dense vector product y = α·A·x + β·y.
cuSPARSE defaults: x and y are `Formats.DensedRight(1)`, but specialized
backends may impose different layout constraints on either operand.
cuSPARSE: `cusparseSpMV`.
"""
struct SpMVOp <: AbstractUSTOp
    A::TensorFormat
    x::TensorFormat
    y::TensorFormat
end

"""
    SpMMOp(A_fmt, B_fmt, C_fmt)

Sparse × dense matrix product C = α·A·B + β·C.
For cuSPARSE SpMM, B and C are `Formats.DensedRight(2)`.
cuSPARSE: `cusparseSpMM`.
"""
struct SpMMOp <: AbstractUSTOp
    A::TensorFormat
    B::TensorFormat
    C::TensorFormat
end

"""
    BatchedSpMMOp(A_fmt, B_fmt, C_fmt)

Batched sparse × dense matrix product.
cuSPARSE: `cusparseSpMM` on `CuSparseArrayCSR` (N-D batched CSR).
"""
struct BatchedSpMMOp <: AbstractUSTOp
    A::TensorFormat
    B::TensorFormat
    C::TensorFormat
end

"""
    SpGEMMOp(A_fmt, B_fmt, C_fmt)

Sparse × sparse matrix product C = α·A·B + β·C.
cuSPARSE: `cusparseSpGEMM` (CSR×CSR→CSR) or `cusparseSpGEMMreuse`.
"""
struct SpGEMMOp <: AbstractUSTOp
    A::TensorFormat
    B::TensorFormat
    C::TensorFormat
end

"""
    SpSVOp(A_fmt, b_fmt, x_fmt)

Sparse triangular solve: solve A·x = α·b for x (single RHS).
cuSPARSE defaults: b and x are `Formats.DensedRight(1)`.
cuSPARSE: `cusparseSpSV`. A must be CSR or CSC (CSC ≡ transposed CSR).
"""
struct SpSVOp <: AbstractUSTOp
    A::TensorFormat
    b::TensorFormat
    x::TensorFormat
end

"""
    SpSMOp(A_fmt, B_fmt, C_fmt)

Sparse triangular solve with multiple RHS: solve A·X = α·B for X.
B and C are dense; cuSPARSE requires `Formats.DensedRight(2)`.
cuSPARSE: `cusparseSpSM`.
"""
struct SpSMOp <: AbstractUSTOp
    A::TensorFormat
    B::TensorFormat
    C::TensorFormat
end

"""
    SDDMMOp(A_fmt, B_fmt, C_fmt)

Sampled dense–dense matrix multiply: C = α·(A∘(B·Dᵀ)) + β·C.
C is sparse (the sampling mask); A and B are dense.
cuSPARSE: `cusparseSDDMM`. C must be CSR or COO.
"""
struct SDDMMOp <: AbstractUSTOp
    A::TensorFormat   # dense
    B::TensorFormat   # dense
    C::TensorFormat   # sparse result/mask
end

# ─── Format conversion ops ────────────────────────────────────────────────────

"""
    SparseToDenseOp(src_fmt)

Convert a sparse tensor to a dense tensor.
cuSPARSE: `cusparseSparseToDense`.
"""
struct SparseToDenseOp <: AbstractUSTOp
    src::TensorFormat
end

"""
    DenseToSparseOp(dst_fmt)

Convert a dense tensor to a sparse tensor.
cuSPARSE: `cusparseDenseToSparse`.
"""
struct DenseToSparseOp <: AbstractUSTOp
    dst::TensorFormat
end

# ─── Sparse vector ops ────────────────────────────────────────────────────────

"""
    GatherOp(fmt)

Gather: y[idx] = x_dense for a sparse index vector.
cuSPARSE: `cusparseGather`.
"""
struct GatherOp <: AbstractUSTOp
    fmt::TensorFormat
end

"""
    ScatterOp(fmt)

Scatter: x_dense[idx] = y for a sparse index vector.
cuSPARSE: `cusparseScatter`.
"""
struct ScatterOp <: AbstractUSTOp
    fmt::TensorFormat
end

"""
    AxpbyOp(x_fmt, y_fmt)

Sparse vector scale-and-add: y = α·x + β·y.
cuSPARSE: `cusparseAxpby`.
"""
struct AxpbyOp <: AbstractUSTOp
    x::TensorFormat   # sparse
    y::TensorFormat   # dense
end

"""
    RotOp(x_fmt, y_fmt)

Givens rotation applied to sparse x and dense y.
cuSPARSE: `cusparseRot`.
"""
struct RotOp <: AbstractUSTOp
    x::TensorFormat   # sparse
    y::TensorFormat   # dense
end
