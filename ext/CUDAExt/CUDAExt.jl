module CUDAExt

using CUDA, CUDA.CUSPARSE
using JLUST, JLUST.Formats
using SparseArrays
import SparseArrays: nonzeros, rowvals
import Adapt
import JLUST:
    USTensor, TensorFormat, AbstractUSTBackend,
    format, extents, index_origin, OneBased, ZeroBased, format_family,
    positions, coordinates,
    SpMVOp, SpMMOp, SpGEMMOp, SpSVOp, SpSMOp,
    sparse_mv!, sparse_mm!, sparse_gemm!, sparse_sv!, sparse_sm!, sparse_sddmm!,
    sparse_to_dense, dense_to_sparse,
    prepare, update_values!,
    CUSPARSEBackend, EmitterBackend

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
    VI = CuArray{Int32,1}
    USTensor{T,Int32,N,CuArray{T,N},VI,OneBased,N}(
        size(A), Formats.DensedRight(N),
        JLUST._no_bufs(Val(N), VI),
        JLUST._no_bufs(Val(N), VI),
        A, A,
    )
end

# ─── CuSparseMatrixCSR adapter ────────────────────────────────────────────────

function JLUST.ust(A::CuSparseMatrixCSR{T,Ti}) where {T,Ti}
    VI = CuVector{Ti}
    USTensor{T,Ti,2,CuVector{T},VI,OneBased,2}(
        size(A), Formats.CSR,
        JLUST._bufs_at(Val(2), VI, 2, A.rowPtr),
        JLUST._bufs_at(Val(2), VI, 2, A.colVal),
        nonzeros(A), A,
    )
end

# ─── CuSparseMatrixCSC adapter ────────────────────────────────────────────────

function JLUST.ust(A::CuSparseMatrixCSC{T,Ti}) where {T,Ti}
    VI = CuVector{Ti}
    USTensor{T,Ti,2,CuVector{T},VI,OneBased,2}(
        size(A), Formats.CSC,
        JLUST._bufs_at(Val(2), VI, 2, A.colPtr),
        JLUST._bufs_at(Val(2), VI, 2, rowvals(A)),
        nonzeros(A), A,
    )
end

# ─── CuSparseMatrixBSR adapter ────────────────────────────────────────────────

function JLUST.ust(A::CuSparseMatrixBSR{T,Ti}) where {T,Ti}
    bsz = (Int(A.blockDim), Int(A.blockDim))
    # dir = 'R': row-major blocks → BSRRight; 'C': col-major → BSRLeft
    fmt = A.dir == 'R' ? Formats.BSRRight(bsz) : Formats.BSRLeft(bsz)
    # BSR has 4 levels: i÷b, j÷b, i%b, j%b — pos/crd live at level 2.
    VI = CuVector{Ti}
    USTensor{T,Ti,2,CuVector{T},VI,OneBased,4}(
        size(A), fmt,
        JLUST._bufs_at(Val(4), VI, 2, A.rowPtr),
        JLUST._bufs_at(Val(4), VI, 2, A.colVal),
        nonzeros(A), A,
    )
end

# ─── CuSparseMatrixCOO adapter ────────────────────────────────────────────────

function JLUST.ust(A::CuSparseMatrixCOO{T,Ti}) where {T,Ti}
    # rowInd and colInd are OneBased.  COO level 1 (compressed nonunique) has no
    # synthetic pos buffer here — same convention as coo_tensor().
    VI = CuVector{Ti}
    USTensor{T,Ti,2,CuVector{T},VI,OneBased,2}(
        size(A), Formats.COO,
        JLUST._no_bufs(Val(2), VI),
        (A.rowInd, A.colInd),
        nonzeros(A), A,
    )
end

# ─── Adapt.adapt_structure ────────────────────────────────────────────────────

function Adapt.adapt_structure(adaptor, u::USTensor{T,I,N,VA,VI,O,NL}) where {T,I,N,VA,VI,O,NL}
    new_val = Adapt.adapt(adaptor, u.val)
    VA2 = typeof(new_val)

    # Determine VI2 by adapting the first non-nothing buffer encountered.
    ref_buf = nothing
    for b in u.pos_buffers
        b !== nothing && (ref_buf = b; break)
    end
    if ref_buf === nothing
        for b in u.crd_buffers
            b !== nothing && (ref_buf = b; break)
        end
    end
    VI2 = ref_buf === nothing ? VI : typeof(Adapt.adapt(adaptor, ref_buf))

    _adapt(b) = b === nothing ? nothing : Adapt.adapt(adaptor, b)::VI2
    new_pos = map(_adapt, u.pos_buffers)::NTuple{NL, Union{Nothing, VI2}}
    new_crd = map(_adapt, u.crd_buffers)::NTuple{NL, Union{Nothing, VI2}}

    USTensor{T,I,N,VA2,VI2,O,NL}(u.extents, u.format, new_pos, new_crd, new_val, nothing)
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
function _shift_origin(u::USTensor{T,I,N,VA,VI,O,NL}, shift::I, ::Type{O2}) where {T,I,N,VA,VI,O,O2,NL}
    _shifted(b) = b === nothing ? nothing : b .+ shift
    new_pos = map(_shifted, u.pos_buffers)::NTuple{NL, Union{Nothing, VI}}
    new_crd = map(_shifted, u.crd_buffers)::NTuple{NL, Union{Nothing, VI}}
    USTensor{T,I,N,VA,VI,O2,NL}(u.extents, u.format, new_pos, new_crd, u.val, nothing)
end

_adaptor(::CPUDevice)  = Array
_adaptor(::CUDADevice) = CuArray

# ─── Backend capability ───────────────────────────────────────────────────────

# CUSPARSEBackend is defined in JLUST core (src/backends.jl); imported above.

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
# Generic API: CSR, CSC, COO.
function JLUST.supports_backend(::CUSPARSEBackend, op::SpMVOp)
    op.A in _CUSPARSE_GENERIC_FORMATS && op.x == _dense1 && op.y == _dense1
end

# SpMM: A sparse, B and C dense matrices.
# Generic API: CSR, CSC, COO.
function JLUST.supports_backend(::CUSPARSEBackend, op::SpMMOp)
    op.A in _CUSPARSE_GENERIC_FORMATS && op.B == _dense2 && op.C == _dense2
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
# Generic API: CSR, CSC.
function JLUST.supports_backend(::CUSPARSEBackend, op::SpSVOp)
    op.A in _CUSPARSE_TRISV_FORMATS && op.b == _dense1 && op.x == _dense1
end

# SpSM: sparse triangular solve, multiple RHS.
# Generic API: CSR, CSC; B and C dense.  BSR not supported by generic SpSM.
function JLUST.supports_backend(::CUSPARSEBackend, op::SpSMOp)
    op.A in _CUSPARSE_TRISV_FORMATS && op.B == _dense2 && op.C == _dense2
end

# SDDMM: C (sparse mask/result) must be CSR or COO.
function JLUST.supports_backend(::CUSPARSEBackend, op::SDDMMOp)
    op.A == _dense2 && op.B == _dense2 &&
    (op.C == Formats.CSR || op.C == Formats.COO)
end

# SparseToDense: CSR, CSC, COO (generic API).
function JLUST.supports_backend(::CUSPARSEBackend, op::SparseToDenseOp)
    op.src in _CUSPARSE_GENERIC_FORMATS
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
    return nothing
end

# ─── cuSPARSE execution paths ─────────────────────────────────────────────────

include("ops/common.jl")
include("ops/spmv.jl")
include("ops/spmm.jl")
include("ops/spgemm.jl")
include("ops/spsv.jl")
include("ops/spsm.jl")
include("ops/sddmm.jl")
include("ops/conversion.jl")
include("ops/block_mul.jl")

end # module CUDAExt
