module KernelAbstractionsExt

using JLUST, KernelAbstractions, SparseArrays
import JLUST:
    USTensor, TensorFormat, AbstractUSTBackend,
    SpMVOp, SpMMOp, SpGEMMOp, SpSVOp, SpSMOp, SDDMMOp, SparseToDenseOp,
    supports_backend, format, extents, index_origin, OneBased,
    positions, coordinates, nonzeros, has_positions, has_coordinates,
    DenseLevel, BatchLevel, CompressedLevel, SingletonLevel, RangeLevel, DeltaLevel,
    AbstractLevelFormat, is_unique, is_ordered, format_family, Formats,
    apply_values!, sparse_mv!, sparse_mm!, sparse_gemm!,
    sparse_sv!, sparse_sm!, sparse_sddmm!, sparse_to_dense, dense_to_sparse,
    csr_tensor, prepare,
    EmitterBackend, level_has_nzval, level_arg_names, level_args, emit_spmv_lv,
    _bbm_scatter_diag!, _bbm_scatter_ramp!

_is_dense_fmt(fmt::TensorFormat) =
    all(lv isa Union{DenseLevel,BatchLevel} for (_, lv) in fmt.levels)

_is_emittable(fmt::TensorFormat) = !_is_dense_fmt(fmt)

function JLUST.supports_backend(::EmitterBackend, op::SpMVOp)
    _is_emittable(op.A) && _is_dense_fmt(op.x) && _is_dense_fmt(op.y)
end

function JLUST.supports_backend(::EmitterBackend, op::SpMMOp)
    _is_emittable(op.A) && _is_dense_fmt(op.B) && _is_dense_fmt(op.C)
end

function JLUST.supports_backend(::EmitterBackend, op::SpGEMMOp)
    op.A == Formats.CSR && op.B == Formats.CSR && op.C == Formats.CSR
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

# Default-backend overload for raw AbstractVector operands.
# Pairs with the explicit-backend overload in src/convenience.jl.
function JLUST.sparse_mv!(u_A::USTensor, x::AbstractVector, y::AbstractVector;
                           backend=EmitterBackend(), kw...)
    JLUST.sparse_mv!(backend, u_A, x, y; kw...)
end

function JLUST.sparse_mm!(u_A::USTensor, u_B::USTensor, u_C::USTensor;
                           backend=EmitterBackend(), kw...)
    JLUST.sparse_mm!(backend, u_A, u_B, u_C; kw...)
end

# ─── SpMV / SpMM / SpGEMM / SpSV / SpSM / SDDMM ─────────────────────────────

include("ops/spmv.jl")
include("ops/spmm.jl")
include("ops/spgemm.jl")
include("ops/spsv.jl")
include("ops/spsm.jl")
include("ops/sddmm.jl")
include("ops/conversion.jl")

end # module KernelAbstractionsExt
