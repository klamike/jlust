module KernelAbstractionsExt

using JLUST, KernelAbstractions, SparseArrays
import JLUST:
    USTensor, TensorFormat, AbstractUSTBackend, Op,
    supports_backend, format, extents, index_origin, OneBased,
    positions, coordinates, nonzeros, has_positions, has_coordinates,
    DenseLevel, BatchLevel, CompressedLevel, SingletonLevel, RangeLevel, DeltaLevel,
    ShiftedDiagLevel,
    AbstractLevelFormat, is_unique, is_ordered, format_family, Formats,
    apply_values!, execute,
    csr_tensor, prepare,
    EmitterBackend, level_has_nzval, level_arg_names, level_args, emit_spmv_lv,
    needs_row_guard,
    _bbm_scatter_diag!, _bbm_scatter_off!

# Type-level dispatch on the LevelTypes tuple: a format is "dense" iff every
# level type is Dense or Batch.  Hot-path predicates resolve to compile-time
# constants since LevelTypes is in the type system.
@generated function _is_dense_fmt(::Type{<:TensorFormat{LT}}) where {LT}
    all(L <: Union{DenseLevel,BatchLevel} for L in LT.parameters) ? true : false
end
_is_dense_fmt(fmt::TensorFormat) = _is_dense_fmt(typeof(fmt))

_is_emittable(::Type{T}) where {T<:TensorFormat} = !_is_dense_fmt(T)
_is_emittable(fmt::TensorFormat) = !_is_dense_fmt(fmt)

@inline _is_csr_type(::Type{<:TensorFormat{LT, :CSR}}) where {LT} = true
@inline _is_csr_type(::Type)                                       = false

function JLUST.supports_backend(::EmitterBackend, ::Op{:SpMV, Tuple{A, X, Y}}) where {A, X, Y}
    _is_emittable(A) && _is_dense_fmt(X) && _is_dense_fmt(Y)
end

function JLUST.supports_backend(::EmitterBackend, ::Op{:SpMM, Tuple{A, B, C}}) where {A, B, C}
    _is_emittable(A) && _is_dense_fmt(B) && _is_dense_fmt(C)
end

function JLUST.supports_backend(::EmitterBackend, ::Op{:SpGEMM, Tuple{A, B, C}}) where {A, B, C}
    _is_csr_type(A) && _is_csr_type(B) && _is_csr_type(C)
end

# ─── apply_values! ────────────────────────────────────────────────────────────

@kernel inbounds=true function _apply_values_kernel!(values, f)
    i = @index(Global, Linear)
    values[i] = f(values[i])
end

function JLUST.apply_values!(f, u::USTensor; backend::EmitterBackend=EmitterBackend())
    vals = nonzeros(u)
    ka   = KernelAbstractions.get_backend(vals)
    _apply_values_kernel!(ka, 64)(vals, f; ndrange=length(vals))
    return u
end

# ─── SpMV / SpMM / SpGEMM / SpSV / SpSM / SDDMM ─────────────────────────────
# No-backend convenience wrappers live in src/convenience.jl and consult
# `default_backend` — extensions only override `default_backend` and the
# explicit-backend execution methods.

include("ops/_walker.jl")
include("ops/spmv.jl")
include("ops/spmm.jl")
include("ops/spgemm.jl")
include("ops/spsv.jl")
include("ops/spsm.jl")
include("ops/sddmm.jl")
include("ops/conversion.jl")
include("ops/block_periodic.jl")

end # module KernelAbstractionsExt
