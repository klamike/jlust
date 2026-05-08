import LinearAlgebra

# ─── Accept AbstractMatrix in sparse_mm! ─────────────────────────────────────
#
# Wraps raw matrices as dense USTensors (DensedRight(2) format) so callers
# can pass plain Matrix / CuMatrix without constructing USTensors manually.

function sparse_mm!(be::AbstractUSTBackend, A::USTensor,
                    B::AbstractMatrix, C::AbstractMatrix; kw...)
    sparse_mm!(be, A, ust(B), ust(C); kw...)
end

function sparse_mm!(A::USTensor, B::AbstractMatrix, C::AbstractMatrix; kw...)
    sparse_mm!(A, ust(B), ust(C); kw...)
end

# ─── Accept AbstractVector in sparse_mv! ─────────────────────────────────────
#
# The no-backend overload lives in KernelAbstractionsExt (default backend is
# resolved there).  The explicit-backend and handle overloads below belong
# here so they don't depend on any extension being loaded.

function sparse_mv!(be::AbstractUSTBackend, A::USTensor,
                    x::AbstractVector, y::AbstractVector; kw...)
    sparse_mv!(be, A, ust(x), ust(y); kw...)
end

function sparse_mv!(h, A::USTensor, x::AbstractVector, y::AbstractVector; kw...)
    sparse_mv!(h, A, ust(x), ust(y); kw...)
end

# ─── LinearAlgebra compatibility for USTensor ─────────────────────────────────
#
# mul!(y, A, x) and A*x delegate to sparse_mv! with the default backend.
# The default backend is resolved at call time via the KernelAbstractionsExt
# method `sparse_mv!(A::USTensor, x::USTensor, y::USTensor; backend=...)`.

function LinearAlgebra.mul!(y::AbstractVector, A::USTensor, x::AbstractVector)
    sparse_mv!(A, x, y)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, A::USTensor, x::AbstractVector,
                             alpha::Number, beta::Number)
    sparse_mv!(A, x, y; alpha=alpha, beta=beta)
    return y
end

function LinearAlgebra.mul!(C::AbstractMatrix, A::USTensor, B::AbstractMatrix)
    sparse_mm!(A, ust(B), ust(C))
    return C
end

function LinearAlgebra.mul!(C::AbstractMatrix, A::USTensor, B::AbstractMatrix,
                             alpha::Number, beta::Number)
    sparse_mm!(A, ust(B), ust(C); alpha=alpha, beta=beta)
    return C
end

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
    USTensor{T, I, 2, VA, VI, O}((m, n), fmt, Dict{Int,VI}(), Dict{Int,VI}(), nzval, nothing)
end
