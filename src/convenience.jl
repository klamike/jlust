import LinearAlgebra

# ─── Generic execute dispatcher ──────────────────────────────────────────────
#
# `execute(OpType, args...; backend, kw...)` is the single user-facing entry.
# It builds the op instance (capturing operand format types), resolves the
# backend via `default_backend`, and delegates to the backend's
# `execute(::Backend, ::Op{:Tag, ...}, args...)` method.
#
# Adding a new op needs only:
#   const NewOp = Op{:NewName}
#   default_backend(::USTensor, ::Type{NewOp}) = ...   (optional override)
#   execute(::Backend, ::Op{:NewName, ...}, args...) = ...   (per backend)

@inline function execute(::Type{OT}, args::USTensor...; backend=nothing, kw...) where {OT<:Op}
    be = something(backend, default_backend(args[1], OT))
    execute(be, OT(format.(args)...), args...; kw...)
end

# SDDMM: A and B are dense, C is the sparse mask.  Backend selection follows
# C's storage, not A's.  This is the only op where the "primary" tensor is
# not the first argument.
@inline function execute(::Type{OT}, A::USTensor, B::USTensor, C::USTensor;
                          backend=nothing, kw...) where {OT<:Op{:SDDMM}}
    be = something(backend, default_backend(C, OT))
    execute(be, OT(format(A), format(B), format(C)), A, B, C; kw...)
end

# Accept raw AbstractMatrix / AbstractVector by wrapping as dense USTensors.
@inline execute(::Type{OT}, A::USTensor, x::AbstractVector, y::AbstractVector; kw...) where {OT<:Op} =
    execute(OT, A, ust(x), ust(y); kw...)
@inline execute(::Type{OT}, A::USTensor, B::AbstractMatrix, C::AbstractMatrix; kw...) where {OT<:Op} =
    execute(OT, A, ust(B), ust(C); kw...)

# Op-tag shim for handle-driven calls: `execute(SpMVOp, h, x, y; …)` routes to
# the handle's own `execute(h, x, y; …)`.  The op tag is informational here —
# the handle already encodes both op and operand formats — so it is dropped.
@inline execute(::Type{<:Op}, h::AbstractKernelHandle, args...; kw...) =
    execute(h, args...; kw...)

# ─── LinearAlgebra integration ────────────────────────────────────────────────
#
# `mul!` and `*` route through execute so users get the unified API even when
# calling through Base/LinearAlgebra interfaces.

LinearAlgebra.mul!(y::AbstractVector, A::USTensor, x::AbstractVector) =
    (execute(SpMVOp, A, ust(x), ust(y)); y)
LinearAlgebra.mul!(y::AbstractVector, A::USTensor, x::AbstractVector, alpha::Number, beta::Number) =
    (execute(SpMVOp, A, ust(x), ust(y); alpha=alpha, beta=beta); y)
LinearAlgebra.mul!(C::AbstractMatrix, A::USTensor, B::AbstractMatrix) =
    (execute(SpMMOp, A, ust(B), ust(C)); C)
LinearAlgebra.mul!(C::AbstractMatrix, A::USTensor, B::AbstractMatrix, alpha::Number, beta::Number) =
    (execute(SpMMOp, A, ust(B), ust(C); alpha=alpha, beta=beta); C)

function Base.:*(A::USTensor, x::AbstractVector)
    y = similar(nonzeros(A), size(A, 1))
    LinearAlgebra.mul!(y, A, x)
    return y
end

# ─── make_tensor ─────────────────────────────────────────────────────────────
#
# Convenience constructor for custom level formats (DiagonalLevel, RampLevel, …)
# that carry no position or coordinate arrays.
#
#   diagonal_tensor(diag; n) = make_tensor(DiagonalFmt, diag; m=n, n=n)
#   ramp_tensor(ref; m, n) = make_tensor(fmt, similar(ref, T, 0); m, n)

"""
    make_tensor(fmt::TensorFormat, nzval::AbstractArray; m::Int, n::Int,
                index_type::Type{I}=Int32, origin::O=OneBased()) → USTensor

Build a 2-D USTensor with no position or coordinate arrays.  Intended for custom
`AbstractLevelFormat` subtypes whose buffers are fully implicit (DiagonalLevel,
RampLevel, etc.).  `nzval` may be a zero-length array for formats with no stored
values.
"""
function make_tensor(fmt::TensorFormat, nzval::VA;
                     m::Int, n::Int,
                     index_type::Type{I}=Int32,
                     origin::O=OneBased()) where {T, VA<:AbstractArray{T},
                                                   I<:Integer,
                                                   O<:AbstractIndexOrigin}
    VI = typeof(similar(nzval, I, 0))
    NL = length(fmt.levels)
    USTensor{T, I, 2, VA, VI, O}((m, n), fmt,
        ntuple(_ -> nothing, NL),
        ntuple(_ -> nothing, NL),
        nzval, nothing)
end
