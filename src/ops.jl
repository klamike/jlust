# ─── Op type ──────────────────────────────────────────────────────────────────
#
# A single parametric singleton subsumes every operation tag.  The op is
# identified by:
#
#   `Tag::Symbol`               — what kind of operation (`:SpMV`, `:Cholesky`, …)
#   `Formats <: Tuple`          — tuple of operand TensorFormat *types*, in operand order
#
# Capability queries dispatch on the Op type alone — no value-level field
# access, no per-op struct.  Adding a new op (Cholesky, LDLT, LU, BiCGStab, …)
# is one const alias plus the backend's `supports_backend` and execution
# methods; no boilerplate struct.
#
# Operand TYPES, not values, drive dispatch.  Because format identity (level
# structure + family) is now in the TensorFormat type, two formats with the
# same structural shape always resolve to the same Op type — capability
# decisions are shared structurally without extra plumbing.
#
# Dense operands use `Formats.DensedRight(N)` — dense is not a special case,
# just a TensorFormat whose levels are all `DenseLevel`.

abstract type AbstractUSTOp end

struct Op{Tag, Formats <: Tuple} <: AbstractUSTOp end

# Construct an op instance from runtime format values.  The operand format
# types are captured into the type parameter; the resulting `Op{...}()` is a
# singleton whose type alone carries all dispatch-relevant information.
@inline function Op{Tag}(formats::TensorFormat...) where {Tag}
    Op{Tag, Tuple{(typeof(f) for f in formats)...}}()
end

# Convenience: extract operand format types from an Op type or instance.
operand_format_types(::Op{Tag, Formats})       where {Tag, Formats} = (Formats.parameters...,)
operand_format_types(::Type{<:Op{Tag, F}})     where {Tag, F}       = (F.parameters...,)
op_tag(::Op{Tag})                              where {Tag}          = Tag
op_tag(::Type{<:Op{Tag}})                      where {Tag}          = Tag

# ─── Op aliases ───────────────────────────────────────────────────────────────
#
# Each named op is a UnionAll alias `Op{:Tag}` — instantiated as
# `SpMVOp(A_fmt, x_fmt, y_fmt)` produces a singleton of the concrete subtype.

"""    SpVVOp(x_fmt, y_fmt)
Sparse vector dot product: result = xᵀ·y.  cuSPARSE: `cusparseDotEx`/`cusparseSpVV`. """
const SpVVOp           = Op{:SpVV}

"""    SpMVOp(A_fmt, x_fmt, y_fmt)
Sparse matrix-dense vector product y = α·A·x + β·y.  cuSPARSE: `cusparseSpMV`. """
const SpMVOp           = Op{:SpMV}

"""    SpMMOp(A_fmt, B_fmt, C_fmt)
Sparse × dense matrix product C = α·A·B + β·C.  cuSPARSE: `cusparseSpMM`. """
const SpMMOp           = Op{:SpMM}

"""    BatchedSpMMOp(A_fmt, B_fmt, C_fmt)
Batched sparse × dense matrix product.  cuSPARSE: `cusparseSpMM` on `CuSparseArrayCSR`. """
const BatchedSpMMOp    = Op{:BatchedSpMM}

"""    SpGEMMOp(A_fmt, B_fmt, C_fmt)
Sparse × sparse matrix product C = α·A·B + β·C.  cuSPARSE: `cusparseSpGEMM`. """
const SpGEMMOp         = Op{:SpGEMM}

"""    SpSVOp(A_fmt, b_fmt, x_fmt)
Sparse triangular solve: A·x = α·b (single RHS).  cuSPARSE: `cusparseSpSV`. """
const SpSVOp           = Op{:SpSV}

"""    SpSMOp(A_fmt, B_fmt, C_fmt)
Sparse triangular solve with multiple RHS: A·X = α·B.  cuSPARSE: `cusparseSpSM`. """
const SpSMOp           = Op{:SpSM}

"""    SDDMMOp(A_fmt, B_fmt, C_fmt)
Sampled dense-dense matrix multiply: C = α·(A·B) ∘ sparsity(C) + β·C.  cuSPARSE: `cusparseSDDMM`. """
const SDDMMOp          = Op{:SDDMM}

"""    SparseToDenseOp(src_fmt)
Convert sparse tensor to dense.  cuSPARSE: `cusparseSparseToDense`. """
const SparseToDenseOp  = Op{:SparseToDense}

"""    DenseToSparseOp(dst_fmt)
Convert dense tensor to sparse.  cuSPARSE: `cusparseDenseToSparse`. """
const DenseToSparseOp  = Op{:DenseToSparse}

"""    GatherOp(fmt)
Gather: `y[idx] = x_dense` for a sparse index vector.  cuSPARSE: `cusparseGather`. """
const GatherOp         = Op{:Gather}

"""    ScatterOp(fmt)
Scatter: `x_dense[idx] = y` for a sparse index vector.  cuSPARSE: `cusparseScatter`. """
const ScatterOp        = Op{:Scatter}

"""    AxpbyOp(x_fmt, y_fmt)
Sparse vector scale-and-add: y = α·x + β·y.  cuSPARSE: `cusparseAxpby`. """
const AxpbyOp          = Op{:Axpby}

"""    RotOp(x_fmt, y_fmt)
Givens rotation applied to sparse x and dense y.  cuSPARSE: `cusparseRot`. """
const RotOp            = Op{:Rot}

# Future ops follow the same pattern:
#   const CholeskyOp = Op{:Cholesky}     # A = L·L'  (factorization)
#   const LDLTOp     = Op{:LDLT}         # A = L·D·L'
#   const LUOp       = Op{:LU}           # A = P·L·U

# ─── Operation function declarations ─────────────────────────────────────────
#
# Declared here so the full operation API is visible in one place.
# Backend extensions (KernelAbstractionsExt, CUDAExt) add concrete methods.

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
