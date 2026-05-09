# ─── Backend abstraction ──────────────────────────────────────────────────────

abstract type AbstractUSTBackend end

# ─── Backend traits ───────────────────────────────────────────────────────────
#
# Format-agnostic capability flags consulted by the emitter walker.  An
# extension declares "my backend supports X" by overriding the trait for its
# concrete backend type — applies to every USTensor format the walker handles,
# without per-format specialization.
#
# `_supports_ldg(ka)` — does this backend's device kernels have a separate
# read-only data cache (CUDA's LDG)?  When true the walker wraps x in
# `Base.Experimental.Const`, so x[col] reads use the read-only path on
# backends that have it.  Default false; overridden by CUDAExt.

@inline _supports_ldg(::Any) = false

# `_default_workgroup_size(ka)` — block / work-group size the walker uses by
# default for emitter-driven kernel launches.  The right value is hardware-
# dependent (NVIDIA SMs prefer 256-thread blocks; OpenCL / POCL devices vary).
# Backends override for their specific hardware; default is 64 (a safe value
# that maps to 2 warps on NVIDIA, 1 wavefront on AMD).

@inline _default_workgroup_size(::Any) = 64

# `_supports_warp_vector(ka)` — does this backend's device kernels support
# warp-level shuffle reductions (CUDA's shfl_*_sync, AMD's __shfl_*)?  When
# true the walker can emit "warp-vector" SpMV kernels: VS threads cooperate
# on each row's inner loop, then warp-shuffle-reduce into a single accumulator
# before lane 0 writes y[row].  Format-agnostic: applies to any (Dense outer,
# Compressed-unique inner) structure the walker emits, not just CSR.
# Default false.

@inline _supports_warp_vector(::Any) = false

# `_warp_reduce_sum_down(val, mask, ::Val{VS})` — warp-level sum reduction
# across VS consecutive lanes within a warp.  Emitted by the walker when
# `vs > 1`.  Each backend with `_supports_warp_vector === true` MUST provide
# a method using its native shuffle intrinsic (shfl_down_sync on CUDA).
# Declared here so the walker can reference it without coupling to a backend.

function _warp_reduce_sum_down end

# `_warp_seg_reduce_sum_down(val, row, mask) -> (val_after, is_head)` —
# segmented warp-level sum reduction.  Each lane holds a (value, row-key)
# pair; after this call, the first lane of each contiguous same-row run
# holds the sum across that run, and `is_head=true` for those lanes only.
# Emitted by the walker when the outermost level is non-unique-sorted
# Compressed (COO-style row list) and the backend supports warp shuffles.
# Backends with `_supports_warp_vector === true` MUST provide a method
# (CUDA uses log2(32) shfl_down_sync rounds + a single shfl_up_sync for
# segment-head detection).

function _warp_seg_reduce_sum_down end

# `_bbm_periodic_spmv_launch!(ka, …)` and `_bbm_periodic_selector_launch!(ka, …)`
# — backend-agnostic launchers for the BlockBandedMatrix periodic SpMV path
# (one diagonal block CSR replicated across T periods + one off-diagonal CSR
# pair for T-1 transitions).  Implemented in KernelAbstractionsExt with KA
# `@kernel` bodies so every KA-targetable backend (CUDA, ROCm, CPU, POCL,
# oneAPI) runs the same structurally-aware kernel.  Used by BlockBandedMatrix
# `mul!` in CUDAExt and (eventually) any other backend's BBM glue.

function _bbm_periodic_spmv_launch! end
function _bsm_with_patches_spmv_launch! end
# (`_bbm_periodic_selector_launch!` was retired — selector off-diag is now a
# regular `(Dense, ShiftedDiag)`-formatted USTensor in the off-diag pair, and
# the periodic walker handles it via the standard nzval-less custom-level
# leaf path.  No separate kernel needed.)

# Block-level "selector patch": describes a (Dense, ShiftedDiag)-shaped block
# embedded in a BlockSparseMatrix or BlockBandedMatrix.  For BSM rows in
# `[row_start, row_end]`, the patch contributes
#   acc_r += val * x[(r - row_start + 1) + col_offset (+ optional period_offset)]
# A `Tuple{SelectorPatch...}` is part of the compiled BSM/BBM state so the
# kernel can walk patches with a Julia-unrolled loop (no runtime length, no
# array indirection).  Used to skip CSR replication for blocks that are
# constant-scaled identities — the kernel reads `val` as a literal and avoids
# pos / crd / nzval traffic for the patched rows entirely.

struct SelectorPatch{T}
    row_start  :: Int32   # 1-based
    row_end    :: Int32   # 1-based, inclusive
    col_offset :: Int32   # 0-based: col_1b = (r - row_start + 1) + col_offset
    val        :: T
end

# ─── Handle abstraction ───────────────────────────────────────────────────────
#
# A `KernelHandle` represents a *prepared* op: descriptors built, workspace
# allocated, symbolic analysis done.  Every backend's prepared kernel state
# subtypes `AbstractKernelHandle` so that the convenience `execute(::Type{<:Op},
# h, args...)` shim can route through Julia dispatch without colliding with the
# all-USTensor convenience method.

abstract type AbstractKernelHandle end

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
#
# (`_csr_spmv_specialized!` and `_coo_spmv_specialized!` were both retired —
# their functionality moved into the generic walker, gated by the
# `_supports_warp_vector(ka)` + `_warp_reduce_sum_down` /
# `_warp_seg_reduce_sum_down` traits so that every (Dense/Batch outer +
# Compressed-unique inner) and every (Compressed-non-unique outer) format on
# CUDA gets the warp-shuffle treatment, not just CSR / COO.)

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

# ShiftedDiagLevel: structural row-wise selector with no element storage.
# col = row + Shift, val = Val (both type params, baked into the kernel by the
# walker as compile-time literals).  Empty arg lists — no per-level buffers.
@inline level_has_nzval(::ShiftedDiagLevel) = false
@inline level_step(::ShiftedDiagLevel{S, V}, i::Int, ::Nothing) where {S, V} = (i + S, V)

# PeriodicLevel: pure structure, no per-level buffers.  T_per and n_cols are
# both type params so the walker emits them as kernel literals.  `level_args`
# returns nothing (no buffer to upload); the walker dispatches on the type.
# nzval/pos/crd live one level deeper (in the inner block's levels).
@inline level_has_nzval(::PeriodicLevel) = true   # passes through to inner block

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
