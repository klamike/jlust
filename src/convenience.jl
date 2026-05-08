import LinearAlgebra

# ─── No-backend convenience wrappers ─────────────────────────────────────────
#
# Each op's no-backend wrapper consults `default_backend(u, OpTag)` to pick
# the right backend.  Extensions override `default_backend` per (storage, op)
# combination — e.g. CUDAExt selects CUSPARSEBackend for SpMM on CuArray-backed
# tensors but stays on EmitterBackend for SpMV.  This is the single point of
# truth for backend-selection policy; extensions never override these wrappers.

function sparse_mv!(A::USTensor, x::USTensor, y::USTensor; backend=nothing, kw...)
    be = something(backend, default_backend(A, SpMVOp))
    sparse_mv!(be, A, x, y; kw...)
end

function sparse_mm!(A::USTensor, B::USTensor, C::USTensor; backend=nothing, kw...)
    be = something(backend, default_backend(A, SpMMOp))
    sparse_mm!(be, A, B, C; kw...)
end

function sparse_gemm!(A::USTensor, B::USTensor, C::USTensor; backend=nothing, kw...)
    be = something(backend, default_backend(A, SpGEMMOp))
    sparse_gemm!(be, A, B, C; kw...)
end

function sparse_sv!(A::USTensor, b::USTensor, x::USTensor; backend=nothing, kw...)
    be = something(backend, default_backend(A, SpSVOp))
    sparse_sv!(be, A, b, x; kw...)
end

function sparse_sm!(A::USTensor, B::USTensor, C::USTensor; backend=nothing, kw...)
    be = something(backend, default_backend(A, SpSMOp))
    sparse_sm!(be, A, B, C; kw...)
end

function sparse_sddmm!(A::USTensor, B::USTensor, C::USTensor; backend=nothing, kw...)
    be = something(backend, default_backend(C, SDDMMOp))   # C is the sparse arg
    sparse_sddmm!(be, A, B, C; kw...)
end

function sparse_to_dense(u::USTensor; backend=nothing, kw...)
    be = something(backend, default_backend(u, SparseToDenseOp))
    sparse_to_dense(be, u; kw...)
end

# ─── Accept AbstractMatrix / AbstractVector ─────────────────────────────────
#
# Wraps raw arrays as dense USTensors (DensedRight(N) format) so callers can
# pass plain Matrix / CuMatrix / Vector / CuVector without manual wrapping.

function sparse_mm!(be::AbstractUSTBackend, A::USTensor,
                    B::AbstractMatrix, C::AbstractMatrix; kw...)
    sparse_mm!(be, A, ust(B), ust(C); kw...)
end
function sparse_mm!(A::USTensor, B::AbstractMatrix, C::AbstractMatrix; kw...)
    sparse_mm!(A, ust(B), ust(C); kw...)
end

function sparse_mv!(be::AbstractUSTBackend, A::USTensor,
                    x::AbstractVector, y::AbstractVector; kw...)
    sparse_mv!(be, A, ust(x), ust(y); kw...)
end
function sparse_mv!(A::USTensor, x::AbstractVector, y::AbstractVector; kw...)
    sparse_mv!(A, ust(x), ust(y); kw...)
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
    NL = length(fmt.levels)
    USTensor{T, I, 2, VA, VI, O}((m, n), fmt,
        ntuple(_ -> nothing, NL),
        ntuple(_ -> nothing, NL),
        nzval, nothing)
end
