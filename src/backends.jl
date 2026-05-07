# ─── Backend abstraction ──────────────────────────────────────────────────────

abstract type AbstractUSTBackend end

# ─── Capability predicates ────────────────────────────────────────────────────

"""
    supports_backend(backend, op) -> Bool

Return `true` if `backend` can execute `op` with the given operand formats.
Default is `false`; each backend extension overrides for its supported triples.

Use structured `AbstractUSTOp` values rather than bare `Symbol`s so that
multi-operand capability (which depends on all of A, B, C formats) is
expressed without losing operand role information.
"""
supports_backend(::AbstractUSTBackend, ::AbstractUSTOp) = false

"""
    supports_convert(backend, src, dst) -> Bool

Return `true` if `backend` has a direct (vendor-accelerated) path for
converting a tensor from format `src` to format `dst`. Default is `false`.

Distinct from `supports_backend` because conversion capability depends on
both source and destination; encoding the destination in an op name
(e.g. `:convertToCOO`) is fragile and produces a large symbol namespace.
When false, callers fall through to the CPU COO-intermediate path.
"""
supports_convert(::AbstractUSTBackend, ::TensorFormat, ::TensorFormat) = false

# ─── Storage validation ───────────────────────────────────────────────────────

"""
    validate_storage(u, backend; op) -> Nothing

Check that `u`'s buffers satisfy the invariants required to execute `op` on
`backend`. Throws `NonCanonicalStorage` on failure.

`backend` is a positional argument so backend extensions can specialize via
dispatch. The base implementation checks format-agnostic invariants; extensions
call `invoke` to run the base check before adding their own.
"""
# ─── Extension hooks ──────────────────────────────────────────────────────────

# Default: not specialized.  CUDAExt overrides with CuVector dispatch to run
# the warp-level segmented-reduce COO kernel; returns true if handled.
function _coo_spmv_specialized!(row_crd, col_crd, nzval, x, y, off, n_nnz)
    return false
end

# Default: not specialized.  CUDAExt overrides with CuVector dispatch to run
# a vector (multi-thread-per-row) CSR kernel; returns true if handled.
function _csr_spmv_specialized!(pos, crd, nzval, x, y, off, n_outer)
    return false
end

# ─── Storage validation ───────────────────────────────────────────────────────

function validate_storage(u::USTensor, backend::AbstractUSTBackend; op = :unknown)
    nze = SparseArrays.nnz(u)
    length(SparseArrays.nonzeros(u)) == nze ||
        throw(NonCanonicalStorage(
            "val buffer length $(length(SparseArrays.nonzeros(u))) ≠ nnz $nze " *
            "(backend=$(typeof(backend)), op=$op)"))
    return nothing
end
