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

# ─── Aggregate / block-structure ops ─────────────────────────────────────────
# Composite operations over aggregate matrices (BlockSparseMatrix,
# BlockBandedMatrix).  Each is an Op tag like the atomic ones; `execute`
# dispatches on the tag and the matrix's concrete type.

"""    BlockSpMVOp(A_BlockSparseMatrix, x_fmt, y_fmt)
Block sparse matrix × dense vector. """
const BlockSpMVOp      = Op{:BlockSpMV}

"""    BlockSpMMOp(A_BlockSparseMatrix, B_fmt, C_fmt)
Block sparse matrix × dense matrix (multi-RHS / batched-mul). """
const BlockSpMMOp      = Op{:BlockSpMM}

"""    BBMSpMVOp(M_BlockBandedMatrix, x_fmt, y_fmt)
Block-banded matrix × dense vector (multi-period DCOPF / MPC structure). """
const BBMSpMVOp        = Op{:BBMSpMV}

# Future ops follow the same pattern:
#   const CholeskyOp = Op{:Cholesky}     # A = L·L'  (factorization)
#   const LDLTOp     = Op{:LDLT}         # A = L·D·L'
#   const LUOp       = Op{:LU}           # A = P·L·U

# ─── Unified execute entry point ─────────────────────────────────────────────
#
# `execute(backend_or_handle, op_or_op_type, args...; kw...)` is the single
# canonical dispatch surface for op execution.  Backend extensions implement
# *one* method per supported (backend, op) pair:
#
#     execute(::EmitterBackend, ::Op{:SpMV, Tuple{A,X,Y}}, u_A, u_x, u_y; kw...) where {A,X,Y} = ...
#     execute(::CUSPARSEBackend, ::Op{:SpMM, Tuple{A,B,C}}, u_A, u_B, u_C; kw...) where {A,B,C} = ...
#
# Per-op named functions (sparse_mv!, sparse_mm!, …) are user-facing aliases —
# they build the Op instance, resolve the backend via `default_backend`, and
# delegate to `execute`.  Adding a new op is a Tag, an `execute` method per
# backend, and a one-line wrapper.
#
# Handle paths use the same entry: `execute(handle, args...; kw...)` is the
# overload backends define for amortized-analysis execution.
function execute end

# ─── Auxiliary function declarations ─────────────────────────────────────────
#
# All op execution flows through `execute` (declared above) — there are no
# per-op named functions.  The two functions below are non-Op operations that
# don't fit the (op, operands...) shape:
#
#   `apply_values!` is a sparsity-preserving in-place map (single tensor + a
#   function).  No formats to capture in an Op tag.
#
#   `materialize` moves a tensor to a different device / changes index origin.
#   It's a tensor transformation, not an algebraic op.

function materialize end

"""
    apply_values!(f, u::USTensor; backend) -> u

Apply `f` element-wise to every stored non-zero value of `u` in-place.
Returns `u`. Does not change the sparsity structure.
"""
function apply_values! end

# ─── Handle / preparation API ────────────────────────────────────────────────

"""
    prepare(backend, OpType, u_A; kwargs...) -> handle

Pre-build a reusable execution handle for repeated `OpType` operations on sparse matrix
`u_A`. Allocates workspace, builds backend-specific descriptors, and (where the backend
supports it) runs format analysis so subsequent calls pay no allocation or analysis cost.

`OpType` is the operation *type* used as a dispatch tag — e.g. `SpMVOp`, `SpMMOp`,
`SpSVOp`. Each backend extension adds one `prepare` method per supported op.

    h = prepare(CUSPARSEBackend(), SpMVOp, u_A)
    h = prepare(CUSPARSEBackend(), SpMMOp, u_A; n_cols=k)

Use `execute(handle, args...; kw...)` to invoke the prepared op.
"""
function prepare end

"""
    update_values!(handle, u_A) -> handle

Update the non-zero values of a prepared handle's sparse matrix descriptor without
rebuilding the descriptor or workspace. The sparsity structure must remain identical
to what was used in `prepare`.
"""
function update_values! end
