module CUDAExt

using CUDA, CUDA.CUSPARSE
using JLUST, JLUST.Formats
using SparseArrays
import SparseArrays: nnz, nonzeros, rowvals
import Adapt
import JLUST:
    USTensor, TensorFormat, AbstractUSTBackend,
    format, extents, index_origin, OneBased, ZeroBased, format_family,
    positions, coordinates,
    SpMVOp, SpMMOp, SpGEMMOp, SpSVOp, SpSMOp,
    sparse_mv!, sparse_mm!, sparse_gemm!, sparse_sv!, sparse_sm!, sparse_sddmm!,
    sparse_to_dense, dense_to_sparse,
    prepare, update_values!

import CUDA.CUSPARSE:
    CuSparseMatrixDescriptor, CuDenseVectorDescriptor, CuDenseMatrixDescriptor,
    CuSpGEMMDescriptor,
    handle,
    cusparseSpMV_bufferSize, cusparseSpMV_preprocess, cusparseSpMV,
    cusparseSpMM_bufferSize, cusparseSpMM_preprocess, cusparseSpMM,
    cusparseDnVecSetValues, cusparseDnMatSetValues,
    cusparseSpMatSetValues, cusparseSpMatSetAttribute,
    cusparseSpMatGetSize, cusparseCsrSetPointers,
    CUSPARSE_SPMV_ALG_DEFAULT, CUSPARSE_SPMM_ALG_DEFAULT,
    cusparseSpMVAlg_t, cusparseSpMMAlg_t,
    CuSparseSpSVDescriptor, CuSparseSpSMDescriptor,
    cusparseSpSV_bufferSize, cusparseSpSV_analysis, cusparseSpSV_solve,
    cusparseSpSM_bufferSize, cusparseSpSM_analysis, cusparseSpSM_solve,
    CUSPARSE_SPSV_ALG_DEFAULT, CUSPARSE_SPSM_ALG_DEFAULT,
    cusparseSpSVAlg_t, cusparseSpSMAlg_t,
    cusparseFillMode_t, cusparseDiagType_t,
    CUSPARSE_FILL_MODE_LOWER, CUSPARSE_FILL_MODE_UPPER,
    CUSPARSE_DIAG_TYPE_NON_UNIT, CUSPARSE_DIAG_TYPE_UNIT,
    CUSPARSE_SPMAT_FILL_MODE, CUSPARSE_SPMAT_DIAG_TYPE,
    cusparseSpGEMMAlg_t, CUSPARSE_SPGEMM_DEFAULT,
    cusparseSpGEMMreuse_workEstimation, cusparseSpGEMMreuse_nnz,
    cusparseSpGEMMreuse_copy, cusparseSpGEMMreuse_compute,
    cusparseSDDMMAlg_t, CUSPARSE_SDDMM_ALG_DEFAULT,
    cusparseSDDMM_bufferSize, cusparseSDDMM_preprocess, cusparseSDDMM

# ─── Memory space trait ───────────────────────────────────────────────────────

JLUST.memory_space(::Type{<:CuArray}) = GPUMemory()

# ─── Dense GPU array adapter ──────────────────────────────────────────────────

function JLUST.ust(A::CuArray{T,N}) where {T,N}
    fmt = Formats.DensedRight(N)
    # No pos/crd buffers for an all-dense tensor; VI=CuArray{Int32,1} by convention
    pos = Dict{Int,CuArray{Int32,1}}()
    crd = Dict{Int,CuArray{Int32,1}}()
    USTensor{T,Int32,N,CuArray{T,N},CuArray{Int32,1},OneBased}(
        size(A), fmt, pos, crd, A, A,
    )
end

# ─── CuSparseMatrixCSR adapter ────────────────────────────────────────────────

function JLUST.ust(A::CuSparseMatrixCSR{T,Ti}) where {T,Ti}
    # rowPtr is OneBased (length = nrows+1); colVal is OneBased (length = nnz)
    pos = Dict{Int,CuVector{Ti}}(2 => A.rowPtr)
    crd = Dict{Int,CuVector{Ti}}(2 => A.colVal)
    USTensor{T,Ti,2,CuVector{T},CuVector{Ti},OneBased}(
        size(A), Formats.CSR, pos, crd, nonzeros(A), A,
    )
end

# ─── CuSparseMatrixCSC adapter ────────────────────────────────────────────────

function JLUST.ust(A::CuSparseMatrixCSC{T,Ti}) where {T,Ti}
    # colPtr is OneBased (length = ncols+1); rowVal is OneBased (length = nnz)
    pos = Dict{Int,CuVector{Ti}}(2 => A.colPtr)
    crd = Dict{Int,CuVector{Ti}}(2 => rowvals(A))
    USTensor{T,Ti,2,CuVector{T},CuVector{Ti},OneBased}(
        size(A), Formats.CSC, pos, crd, nonzeros(A), A,
    )
end

# ─── CuSparseMatrixBSR adapter ────────────────────────────────────────────────

function JLUST.ust(A::CuSparseMatrixBSR{T,Ti}) where {T,Ti}
    bsz = (Int(A.blockDim), Int(A.blockDim))
    # dir = 'R': row-major blocks → BSRRight; 'C': col-major → BSRLeft
    fmt = A.dir == 'R' ? Formats.BSRRight(bsz) : Formats.BSRLeft(bsz)
    # rowPtr: block-level, 1-based, length = nrows/blockDim + 1
    # colVal: block-level, 1-based, length = nnzb
    # nzVal:  flat, length = nnzb * blockDim^2
    pos = Dict{Int,CuVector{Ti}}(2 => A.rowPtr)
    crd = Dict{Int,CuVector{Ti}}(2 => A.colVal)
    USTensor{T,Ti,2,CuVector{T},CuVector{Ti},OneBased}(
        size(A), fmt, pos, crd, nonzeros(A), A,
    )
end

# ─── CuSparseMatrixCOO adapter ────────────────────────────────────────────────

function JLUST.ust(A::CuSparseMatrixCOO{T,Ti}) where {T,Ti}
    # rowInd and colInd are OneBased.  COO level 1 (compressed nonunique) has no
    # synthetic pos buffer here — same convention as coo_tensor().
    pos = Dict{Int,CuVector{Ti}}()
    crd = Dict{Int,CuVector{Ti}}(1 => A.rowInd, 2 => A.colInd)
    USTensor{T,Ti,2,CuVector{T},CuVector{Ti},OneBased}(
        size(A), Formats.COO, pos, crd, nonzeros(A), A,
    )
end

# ─── Adapt.adapt_structure ────────────────────────────────────────────────────

function Adapt.adapt_structure(adaptor, u::USTensor{T,I,N,VA,VI,O}) where {T,I,N,VA,VI,O}
    new_val = Adapt.adapt(adaptor, u.val)
    VA2 = typeof(new_val)

    # Determine VI2 by adapting a reference index buffer (if any exist)
    ref_buf = if !isempty(u.pos_buffers)
        first(values(u.pos_buffers))
    elseif !isempty(u.crd_buffers)
        first(values(u.crd_buffers))
    else
        nothing
    end

    VI2 = ref_buf === nothing ? VI : typeof(Adapt.adapt(adaptor, ref_buf))
    new_pos = Dict{Int,VI2}(k => Adapt.adapt(adaptor, v) for (k, v) in u.pos_buffers)
    new_crd = Dict{Int,VI2}(k => Adapt.adapt(adaptor, v) for (k, v) in u.crd_buffers)

    USTensor{T,I,N,VA2,VI2,O}(u.extents, u.format, new_pos, new_crd, new_val, nothing)
end

# ─── materialize ─────────────────────────────────────────────────────────────

function JLUST.materialize(u::USTensor{T,I,N,VA,VI,O};
                            device,
                            origin::AbstractIndexOrigin = index_origin(u)) where {T,I,N,VA,VI,O}
    adaptor = _adaptor(device)
    u_dev   = Adapt.adapt(adaptor, u)      # buffers moved to target device; O preserved

    O2 = typeof(origin)
    O2 === O && return u_dev

    # Origin change: `.+` produces new arrays (no in-place mutation risk)
    shift   = I(O2 === OneBased ? 1 : -1)
    _shift_origin(u_dev, shift, O2)
end

# Reconstruct with shifted index buffers and a new origin type parameter.
# Dispatches on the concrete u_dev type so VI2 is statically known.
function _shift_origin(u::USTensor{T,I,N,VA,VI,O}, shift::I, ::Type{O2}) where {T,I,N,VA,VI,O,O2}
    new_pos = Dict{Int,VI}(k => v .+ shift for (k, v) in u.pos_buffers)
    new_crd = Dict{Int,VI}(k => v .+ shift for (k, v) in u.crd_buffers)
    USTensor{T,I,N,VA,VI,O2}(u.extents, u.format, new_pos, new_crd, u.val, nothing)
end

_adaptor(::CPUDevice)  = Array
_adaptor(::CUDADevice) = CuArray

# ─── Backend capability ───────────────────────────────────────────────────────

struct CUSPARSEBackend <: AbstractUSTBackend end

# Generic-API formats (cusparseSpMV / cusparseSpMM / cusparseSpSV / cusparseSpSM).
const _CUSPARSE_GENERIC_FORMATS = Set([Formats.CSR, Formats.CSC, Formats.COO])
# Triangular-solve formats (generic SpSV / SpSM).
const _CUSPARSE_TRISV_FORMATS   = Set([Formats.CSR, Formats.CSC])

const _dense1 = Formats.DensedRight(1)
const _dense2 = Formats.DensedRight(2)

# Format family predicates — use the typed :family field, not name strings.
_is_bsr(fmt::TensorFormat)  = format_family(fmt) == :BSR
_is_sell(fmt::TensorFormat) = format_family(fmt) == :SELL
_is_bell(fmt::TensorFormat) = format_family(fmt) == :BlockedELL

# SpVV: x sparse (CSR/COO-style vector), y dense.
# cuSPARSE: cusparseSpVV.
function JLUST.supports_backend(::CUSPARSEBackend, op::SpVVOp)
    op.x in _CUSPARSE_GENERIC_FORMATS && op.y == _dense1
end

# SpMV: A sparse, x and y dense vectors.
# Generic API: CSR, CSC, COO.  Legacy bsrmv: BSR.  SELL_ALG1: SELL.
function JLUST.supports_backend(::CUSPARSEBackend, op::SpMVOp)
    (op.A in _CUSPARSE_GENERIC_FORMATS || _is_bsr(op.A) || _is_sell(op.A)) &&
    op.x == _dense1 && op.y == _dense1
end

# SpMM: A sparse, B and C dense matrices.
# Generic API: CSR, CSC, COO.  Generic ≥12.5.1: BSR.  BLOCKED_ELL_ALG1: BlockedELL.
function JLUST.supports_backend(::CUSPARSEBackend, op::SpMMOp)
    (op.A in _CUSPARSE_GENERIC_FORMATS || _is_bsr(op.A) || _is_bell(op.A)) &&
    op.B == _dense2 && op.C == _dense2
end

# BatchedSpMM: CuSparseArrayCSR (N-D); B and C dense.
# cuSPARSE: cusparseSpMM on batched CuSparseArrayCSR.
function JLUST.supports_backend(::CUSPARSEBackend, op::BatchedSpMMOp)
    op.A == Formats.CSR && op.B == _dense2 && op.C == _dense2
end

# SpGEMM: cusparseSpGEMM requires CSR×CSR→CSR.
function JLUST.supports_backend(::CUSPARSEBackend, op::SpGEMMOp)
    op.A == Formats.CSR && op.B == Formats.CSR && op.C == Formats.CSR
end

# SpSV: sparse triangular solve, single RHS.
# Generic API: CSR, CSC.  Legacy bsrsv2: BSR.
function JLUST.supports_backend(::CUSPARSEBackend, op::SpSVOp)
    (op.A in _CUSPARSE_TRISV_FORMATS || _is_bsr(op.A)) &&
    op.b == _dense1 && op.x == _dense1
end

# SpSM: sparse triangular solve, multiple RHS.
# Generic API: CSR, CSC; B and C dense.  BSR not supported by generic SpSM.
function JLUST.supports_backend(::CUSPARSEBackend, op::SpSMOp)
    op.A in _CUSPARSE_TRISV_FORMATS && op.B == _dense2 && op.C == _dense2
end

# SDDMM: C (sparse mask/result) must be CSR, COO, or BSR (≥12.1.0).
function JLUST.supports_backend(::CUSPARSEBackend, op::SDDMMOp)
    op.A == _dense2 && op.B == _dense2 &&
    (op.C == Formats.CSR || op.C == Formats.COO || _is_bsr(op.C))
end

# SparseToDense: CSR, CSC, COO (generic API); BSR (type-specific path with reinterpret).
function JLUST.supports_backend(::CUSPARSEBackend, op::SparseToDenseOp)
    op.src in _CUSPARSE_GENERIC_FORMATS || _is_bsr(op.src)
end

# DenseToSparse: CSR, CSC, COO only (generic cusparseDenseToSparse).
function JLUST.supports_backend(::CUSPARSEBackend, op::DenseToSparseOp)
    op.dst in _CUSPARSE_GENERIC_FORMATS
end

# Sparse vector ops: COO-style (index + value arrays).
function JLUST.supports_backend(::CUSPARSEBackend, op::GatherOp)
    op.fmt == Formats.COO
end

function JLUST.supports_backend(::CUSPARSEBackend, op::ScatterOp)
    op.fmt == Formats.COO
end

function JLUST.supports_backend(::CUSPARSEBackend, op::AxpbyOp)
    op.x == Formats.COO && op.y == _dense1
end

function JLUST.supports_backend(::CUSPARSEBackend, op::RotOp)
    op.x == Formats.COO && op.y == _dense1
end

# Direct format conversions (vendor-accelerated).
# CSR↔CSC, CSR↔COO: generic API.  BSR↔CSR: csr2bsr / bsr2csr legacy API.
function JLUST.supports_convert(::CUSPARSEBackend, src::TensorFormat, dst::TensorFormat)
    (src == Formats.CSR  && dst == Formats.CSC) ||
    (src == Formats.CSC  && dst == Formats.CSR) ||
    (src == Formats.CSR  && dst == Formats.COO) ||
    (src == Formats.COO  && dst == Formats.CSR) ||
    (src == Formats.CSR  && _is_bsr(dst))       ||   # csr2bsr
    (_is_bsr(src)        && dst == Formats.CSR)       # bsr2csr
end

function JLUST.validate_storage(u::USTensor, backend::CUSPARSEBackend; op = :unknown)
    invoke(JLUST.validate_storage, Tuple{USTensor, AbstractUSTBackend}, u, backend; op)
    # cuSPARSE-specific buffer checks (sorted coords, block-size divisibility, etc.)
    # are added in Phase 4 alongside _prepare.
    return nothing
end

export CUSPARSEBackend

# ─── cuSPARSE execution paths ─────────────────────────────────────────────────

include("ops/common.jl")
include("ops/spmv.jl")
include("ops/spmm.jl")
include("ops/spgemm.jl")
include("ops/spsv.jl")
include("ops/spsm.jl")
include("ops/sddmm.jl")
include("ops/conversion.jl")

end # module CUDAExt
