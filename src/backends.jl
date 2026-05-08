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
# beta: scale factor for y accumulation (y ← A*x + beta*y).
# Val{ZERO_BETA}: true eliminates the y-read on the GPU when beta=0.
function _csr_spmv_specialized!(pos, crd, nzval, x, y, off, n_outer, beta)
    return false
end

# ─── Concrete backend types ───────────────────────────────────────────────────
#
# Defined in core (not in extensions) so that:
#   - Users can write EmitterBackend() / CUSPARSEBackend() without importing extensions
#   - Extensions add methods to these types; they don't define the types themselves

"""    EmitterBackend <: AbstractUSTBackend
KernelAbstractions-based JIT emitter. Emits a specialized @kernel per sparse format
at first call; subsequent calls reuse the cached kernel.
Loaded when KernelAbstractions is available (JLUST loads KAExt automatically).
"""
struct EmitterBackend <: AbstractUSTBackend end

"""    CUSPARSEBackend <: AbstractUSTBackend
NVIDIA cuSPARSE vendor backend. Methods added by CUDAExt when CUDA.jl is loaded.
"""
struct CUSPARSEBackend <: AbstractUSTBackend end

"""
    default_backend(u::USTensor, ::Type{<:AbstractUSTOp}) -> AbstractUSTBackend

Backend selected by no-backend convenience wrappers when the user does not
specify one. Extensions add methods to specialize on element type, storage,
or operation. Core default: `EmitterBackend` (works on CPU and GPU via KA).

Encoding the policy here — rather than via wrapper-override order across
extensions — means there is exactly one method per (USTensor type, Op) pair
and the choice can dispatch on storage (e.g. CuArray-backed → CUSPARSEBackend
for SpMM, but stay on EmitterBackend for SpMV).
"""
default_backend(::USTensor, ::Type{<:AbstractUSTOp}) = EmitterBackend()

# ─── Custom level format hooks ────────────────────────────────────────────────
#
# Extend these four functions for custom AbstractLevelFormat subtypes to plug
# into EmitterBackend's @kernel code generator.  See the JLUST tour for
# DiagonalLevel and RampLevel examples.
#
#   level_has_nzval(lv)                   → Bool (default: true)
#       Return false if the level encodes values implicitly (no nzval array).
#       When any level returns false, :_nzval is omitted from the @kernel signature.
#
#   level_arg_names(lv, pc, cc)           → Vector{Symbol} (default: Symbol[])
#       @kernel argument names contributed by this level.
#       Increment pc[] for each pos array used, cc[] for each crd array.
#
#   level_args(lv, u, lvl)                → Vector{AbstractArray} (default: [])
#       Runtime arrays in the same order as level_arg_names.
#
#   emit_spmv_lv(lv, p_var, input_fn_sym) → Expr
#       Inner kernel body for a leaf level.  p_var is the Symbol of the fiber
#       position variable; input_fn_sym is the per-element transform Symbol.
#       The body should accumulate into _acc using _nzval / _x / _origin_off.

level_has_nzval(::AbstractLevelFormat) = true
level_arg_names(::AbstractLevelFormat, pc::Ref, cc::Ref) = Symbol[]
level_args(::AbstractLevelFormat, u::AbstractUSTensor, lvl::Int) = AbstractArray[]

"""
    level_step(lv, i::Int, nz) → (col::Int, val)

High-level hook for "diagonal-like" inner levels (col = f(row), val = g(nz, row)).
Implement this instead of `emit_spmv_lv` + `_cpu_level_accumulate!` — JLUST
generates both the GPU @kernel body and the fused CPU loop from it automatically.

`i` is the 1-based thread/row index.  `nz` is `nonzeros(tensor)` when
`level_has_nzval(lv)` is true, or `nothing` when false (no element storage).

    # Diagonal: y[i] += nz[i] * x[i]
    JLUST.level_step(::DiagonalLevel, i::Int, nz) = (i, nz[i])

    # Scaled identity with no stored values:
    JLUST.level_has_nzval(::RampLevel) = false
    JLUST.level_step(lv::RampLevel, i::Int, ::Nothing) = (i, lv.sign)
"""
function level_step end

# Default emit_spmv_lv for custom levels: call level_step from within the @kernel.
# lv is captured as a compile-time constant (baked into the kernel at generation time).
# LLVM constant-folds and inlines level_step, so there is no dispatch overhead at runtime.
function emit_spmv_lv(lv::AbstractLevelFormat, p_var::Symbol, input_fn_sym::Symbol)
    nz_sym = level_has_nzval(lv) ? :_nzval : :nothing
    quote
        _p1 = Int($p_var) - Int(_origin_off) + 1
        (_x_idx, _val) = JLUST.level_step($lv, _p1, $nz_sym)
        _acc += _val * $input_fn_sym(_x[_x_idx])
    end
end

# ─── Getindex hook ────────────────────────────────────────────────────────────

"""
    locate_level(lv, u, stored_target, origin_offset, level, p) → Union{Int,Nothing}

Hook for custom `AbstractLevelFormat` subtypes in `Base.getindex`.

Given the current parent fiber position `p` (1-based) and the stored coordinate
`stored_target` being sought, return the 1-based child fiber position, or `nothing`
for a structural zero.

`level` is the 1-based level index; use `positions(u, level)` / `coordinates(u, level)`
to access format buffers.
"""
function locate_level end

# ─── Sparse-row predicate ─────────────────────────────────────────────────────

"""
    needs_row_guard(b::AbstractUSTensor) → Bool

Return `true` when `b` has a dense outer level over a sparse inner level and
significantly fewer NNZ than rows.  When true, the EmitterBackend SpMM kernel
uses a guarded beta=1 path that skips empty rows rather than writing zeros.

The built-in method for `USTensor` checks for a 2-level `DenseLevel`/`BatchLevel`
outer + unique `CompressedLevel` inner with nnz < n_rows.  Define additional methods
for custom formats with the same property.
"""
needs_row_guard(::AbstractUSTensor) = false

function needs_row_guard(b::USTensor)
    levels = b.format.levels
    length(levels) == 2                                   || return false
    levels[1] isa Union{DenseLevel,BatchLevel}            || return false
    lv2 = levels[2]
    lv2 isa CompressedLevel                                || return false
    is_unique(lv2)                                         || return false
    length(nonzeros(b)) < extents(b)[1]
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
