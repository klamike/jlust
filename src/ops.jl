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

# ─── Operation function stubs ─────────────────────────────────────────────────
#
# Declared here so the full operation API is visible in one place.
# Backend extensions (KernelAbstractionsExt, CUDAExt) add concrete methods.

# Generic function stubs — methods are added by backend extensions
function materialize end

"""
    apply_values!(f, u::USTensor; backend) -> u

Apply `f` element-wise to every stored non-zero value of `u` in-place.
Returns `u`. Does not change the sparsity structure.
"""
function apply_values! end

"""
    sparse_mv!(A, x, y; backend, alpha=1, beta=0) -> y

Compute `y = alpha * A * x + beta * y` where `A` is sparse and `x`, `y` are dense.
Backend extensions add concrete methods dispatching on `backend` as a positional arg.
"""
function sparse_mv! end

"""
    sparse_mm!(A, B, C; backend, transa='N', transb='N', alpha=1, beta=0) -> C

Compute `C = alpha * op(A) * op(B) + beta * C` where `A` is sparse, `B` and `C` are dense.
"""
function sparse_mm! end

"""
    sparse_vv(x, y; backend) -> Number

Sparse vector dot product. `x` is sparse (COO-style), `y` is dense.
"""
function sparse_vv end

"""
    sparse_sv!(A, b, x; backend, uplo='L', diag='N', transa='N', alpha=1) -> x

Sparse triangular solve: `A * x = alpha * b`. `b` and `x` are dense vectors.
"""
function sparse_sv! end

"""
    sparse_sm!(A, B, C; backend, uplo='L', diag='N', transa='N', alpha=1) -> C

Sparse triangular solve with multiple RHS: solve `A * X = alpha * B`. `B`, `C` dense.
"""
function sparse_sm! end

"""
    sparse_sddmm!(A, B, C; backend, transa='N', transb='N', alpha=1, beta=0) -> C

Sampled dense-dense matmul: `C = alpha * (op(A) * op(B)) ∘ sparsity(C) + beta * C`.
`C` is sparse (mask); `A`, `B` are dense.
"""
function sparse_sddmm! end

"""
    sparse_gemm!(A, B, C; backend, transa='N', transb='N', alpha=1, beta=0) -> USTensor

Compute sparse-sparse matrix product. Because SpGEMM may produce a result with a
different sparsity structure than the input `C`, the returned USTensor may not share
buffers with `C`. For beta=0 (default) `C` is used only for sizing; for beta≠0 `C`
must already carry the exact sparsity pattern of op(A)*op(B).
"""
function sparse_gemm! end

"""
    sparse_to_dense(u; backend) -> USTensor{dense}

Convert a sparse `USTensor` to a dense `USTensor`.
"""
function sparse_to_dense end

"""
    dense_to_sparse(u, fmt; backend) -> USTensor{sparse}

Convert a dense `USTensor` to a sparse `USTensor` with format `fmt`.
"""
function dense_to_sparse end

"""
    sparse_gather!(y, x_dense; backend) -> y

Gather: `y[idx] = x_dense[y.indices]` for a sparse vector `y`.
"""
function sparse_gather! end

"""
    sparse_scatter!(x_dense, y; backend) -> x_dense

Scatter: `x_dense[y.indices] = y.values` for a sparse vector `y`.
"""
function sparse_scatter! end

"""
    sparse_axpby!(alpha, x, beta, y; backend) -> y

Scale-and-add: `y = alpha * x + beta * y` where `x` is sparse, `y` is dense.
"""
function sparse_axpby! end

"""
    sparse_rot!(x, y, c, s; backend) -> (x, y)

Apply Givens rotation to sparse `x` and dense `y`.
"""
function sparse_rot! end

"""
    prepare(backend, OpType, u_A; kwargs...) -> handle

Pre-build a reusable execution handle for repeated `OpType` operations on sparse matrix
`u_A`. Allocates workspace, builds backend-specific descriptors, and (where the backend
supports it) runs format analysis so subsequent calls pay no allocation or analysis cost.

`OpType` is the operation *type* used as a dispatch tag — e.g. `SpMVOp`, `SpMMOp`,
`SpSVOp`. Each backend extension adds one `prepare` method per supported op.

    h = prepare(CUSPARSEBackend(), SpMVOp, u_A)
    h = prepare(CUSPARSEBackend(), SpMMOp, u_A; n_cols=k)
"""
function prepare end

"""
    update_values!(handle, u_A) -> handle

Update the non-zero values of a prepared handle's sparse matrix descriptor without
rebuilding the descriptor or workspace. The sparsity structure must remain identical
to what was used in `prepare`.
"""
function update_values! end
