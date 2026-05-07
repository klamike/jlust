module KernelAbstractionsExt

using JLUST, KernelAbstractions, SparseArrays
import JLUST:
    USTensor, TensorFormat, AbstractUSTBackend,
    SpMVOp, SpMMOp, SpSVOp, SpSMOp, SDDMMOp, SparseToDenseOp,
    supports_backend, format, extents, index_origin, OneBased,
    positions, coordinates, nonzeros, has_positions, has_coordinates,
    DenseLevel, BatchLevel, CompressedLevel, SingletonLevel, RangeLevel, DeltaLevel,
    is_unique, is_ordered,
    apply_values!, sparse_mv!, sparse_mm!,
    sparse_sv!, sparse_sm!, sparse_sddmm!, sparse_to_dense, dense_to_sparse

# ─── EmitterBackend ───────────────────────────────────────────────────────────

struct EmitterBackend <: AbstractUSTBackend end

export EmitterBackend

_is_dense_fmt(fmt::TensorFormat) =
    all(lv isa Union{DenseLevel,BatchLevel} for (_, lv) in fmt.levels)

_is_emittable(fmt::TensorFormat) = !_is_dense_fmt(fmt)

function JLUST.supports_backend(::EmitterBackend, op::SpMVOp)
    _is_emittable(op.A) && _is_dense_fmt(op.x) && _is_dense_fmt(op.y)
end

function JLUST.supports_backend(::EmitterBackend, op::SpMMOp)
    _is_emittable(op.A) && _is_dense_fmt(op.B) && _is_dense_fmt(op.C)
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

# ─── Convenience wrappers (default backend; overridden by CUDA ext when loaded) ─

function JLUST.sparse_mv!(u_A::USTensor, u_x::USTensor, u_y::USTensor;
                           backend=EmitterBackend(), kw...)
    JLUST.sparse_mv!(backend, u_A, u_x, u_y; kw...)
end

function JLUST.sparse_mm!(u_A::USTensor, u_B::USTensor, u_C::USTensor;
                           backend=EmitterBackend(), kw...)
    JLUST.sparse_mm!(backend, u_A, u_B, u_C; kw...)
end

# ─── SpMV / SpMM ──────────────────────────────────────────────────────────────

include("ops/spmv.jl")
include("ops/spmm.jl")
include("ops/spsv.jl")
include("ops/spsm.jl")
include("ops/sddmm.jl")
include("ops/conversion.jl")

end # module KernelAbstractionsExt
