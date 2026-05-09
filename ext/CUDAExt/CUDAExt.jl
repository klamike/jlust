module CUDAExt

using CUDA, CUDA.CUSPARSE
using JLUST, JLUST.Formats
using SparseArrays
import SparseArrays: nonzeros, rowvals
import Adapt
import JLUST:
    USTensor, TensorFormat, AbstractUSTBackend, Op,
    format, extents, index_origin, OneBased, ZeroBased, format_family,
    positions, coordinates,
    execute,
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

# ─── Backend traits (opt into emitter optimisations) ────────────────────────
#
# CUDA's PTX has the LDG (load-via-read-only-cache) instruction; the emitter
# walker wraps x in `Const` when this is true, so x[col] reads use the
# read-only data cache.  Format-agnostic — applies to every USTensor format
# the walker emits, not just CSR/COO.

@inline JLUST._supports_ldg(::CUDABackend) = true

# 256 = 8 warps/block — sweet spot on Ampere/Hopper/Ada SMs (4 schedulers × 2
# warps).  Walker-emitted scalar kernels were running at the conservative 64
# default, leaving threads on the table for medium-large matrices.
@inline JLUST._default_workgroup_size(::CUDABackend) = 256

# Warp-shuffle reductions: NVIDIA SMs do these in a single PTX instruction
# (shfl_down_sync).  The walker emits stride-VS inner loops + reduce when
# `_supports_warp_vector(ka) === true` and the format admits warp-vector mode.
@inline JLUST._supports_warp_vector(::CUDABackend) = true

# log2(VS)-step warp-shuffle sum reduction across VS lanes.  VS must be a
# power of two ≤ 32 — the @generated body emits exactly log2(VS) shfl_down
# instructions, all hoisted to compile time.
@generated function JLUST._warp_reduce_sum_down(val, mask::UInt32, ::Val{VS}) where VS
    @assert VS isa Integer && VS > 0 && (VS & (VS - 1)) == 0 && VS <= 32 "VS must be a power of 2 ≤ 32"
    body = Expr(:block)
    δ = 1
    while δ < VS
        push!(body.args, :(val += CUDA.shfl_down_sync(mask, val, Int32($δ))))
        δ <<= 1
    end
    push!(body.args, :(return val))
    body
end

# Segmented warp-level sum reduction (5-step shfl_down across the full warp,
# guarded by row-key match) + segment-head detection via one shfl_up.  Each
# of the 32 lanes holds (val, row); after this call the leftmost lane of each
# contiguous same-row run holds the run's sum and is_head=true.  Used by the
# walker for sorted-COO SpMV: 1 thread per NNZ, atomic add only at heads.
#
# IMPORTANT: CUDA's `shfl_down_sync` returns the caller's own value when the
# source lane is out of range (lane + δ >= 32).  Without an explicit `lane +
# δ < 32` guard, a lane at the warp tail would see `peer_row == orig_row`
# trivially and double-count itself — a bug the previous hand-written CUDA
# COO kernel also had but never tripped on the limited test inputs.
@inline function JLUST._warp_seg_reduce_sum_down(val::T, row::Int32, mask::UInt32) where T
    orig_row = row
    lane     = (threadIdx().x - Int32(1)) % Int32(32)
    for δ in (Int32(1), Int32(2), Int32(4), Int32(8), Int32(16))
        peer_val = CUDA.shfl_down_sync(mask, val, δ)
        peer_row = CUDA.shfl_down_sync(mask, orig_row, δ)
        if (lane + δ < Int32(32)) & (peer_row == orig_row) & (orig_row >= Int32(0))
            val += peer_val
        end
    end
    prev_row = CUDA.shfl_up_sync(mask, orig_row, UInt32(1))
    is_head  = (lane == Int32(0)) | (prev_row != orig_row)
    return (val, is_head)
end

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

# ─── Default backend policy for CUDA-backed tensors ─────────────────────────
#
# Per-op selection: SpMV stays on EmitterBackend (warp-shuffle CSR kernel beats
# cuSPARSE on the L40S benchmark for the matrix sizes we target).  All other ops
# default to CUSPARSEBackend on CuArray-backed tensors — vendor-tuned kernels
# outperform the emitter for SpMM / SpGEMM / SpSV / SpSM / SDDMM.

const _CuUST = USTensor{T,I,N,VA} where {T,I,N,VA<:CuArray}

JLUST.default_backend(::_CuUST, ::Type{<:Op{:SpMM}})           = CUSPARSEBackend()
JLUST.default_backend(::_CuUST, ::Type{<:Op{:BatchedSpMM}})    = CUSPARSEBackend()
JLUST.default_backend(::_CuUST, ::Type{<:Op{:SpGEMM}})         = CUSPARSEBackend()
JLUST.default_backend(::_CuUST, ::Type{<:Op{:SpSV}})           = CUSPARSEBackend()
JLUST.default_backend(::_CuUST, ::Type{<:Op{:SpSM}})           = CUSPARSEBackend()
JLUST.default_backend(::_CuUST, ::Type{<:Op{:SDDMM}})          = CUSPARSEBackend()
JLUST.default_backend(::_CuUST, ::Type{<:Op{:SparseToDense}})  = CUSPARSEBackend()
JLUST.default_backend(::_CuUST, ::Type{<:Op{:DenseToSparse}})  = CUSPARSEBackend()

# ─── Backend capability ───────────────────────────────────────────────────────
#
# All capability checks are pure type-level dispatch.  Operand format types are
# in the Op's type parameters; their family / level structure / etc. is in the
# TensorFormat type system.  No value comparisons, no Dict lookups, no runtime
# field reads — capability resolves through Julia's method table.

# CUSPARSEBackend is defined in JLUST core (src/backends.jl); imported above.

# Generic-API families (cusparseSpMV / cusparseSpMM / cusparseSpSV / cusparseSpSM).
const _CUSPARSE_GENERIC_FAMILIES = (:CSR, :CSC, :COO)
const _CUSPARSE_TRISV_FAMILIES   = (:CSR, :CSC)

# Type-level predicates dispatched on the format's Family parameter.
@inline _is_csr(::Type{<:TensorFormat{LT, :CSR}}) where {LT} = true
@inline _is_csr(::Type)                                       = false
@inline _is_dense(::Type{<:TensorFormat{LT, :Dense}}) where {LT} = true
@inline _is_dense(::Type{<:TensorFormat{LT, :DenseVector}}) where {LT} = true
@inline _is_dense(::Type)                                              = false
@inline _is_bsr(::Type{<:TensorFormat{LT, :BSR}}) where {LT} = true
@inline _is_bsr(::Type)                                       = false
@inline _is_sell(::Type{<:TensorFormat{LT, :SELL}}) where {LT} = true
@inline _is_sell(::Type)                                       = false
@inline _is_bell(::Type{<:TensorFormat{LT, :BlockedELL}}) where {LT} = true
@inline _is_bell(::Type)                                              = false
@inline _is_generic(F)  = format_family(F) in _CUSPARSE_GENERIC_FAMILIES
@inline _is_trisv_fmt(F) = format_family(F) in _CUSPARSE_TRISV_FAMILIES

# `nlevels(::Type)` — number of levels in a TensorFormat type.
@inline nlevels(::Type{<:TensorFormat{LT}}) where {LT} = length(LT.parameters)

# `_is_dense_n(F, N)` — F is a dense format with N levels.
@inline _is_dense_n(F, N::Int) = _is_dense(F) && nlevels(F) == N

# SpVV: x sparse, y dense (1D).
function JLUST.supports_backend(::CUSPARSEBackend, ::Op{:SpVV, Tuple{X, Y}}) where {X, Y}
    _is_generic(X) && _is_dense_n(Y, 1)
end

# SpMV: A sparse, x and y 1D dense.
function JLUST.supports_backend(::CUSPARSEBackend, ::Op{:SpMV, Tuple{A, X, Y}}) where {A, X, Y}
    _is_generic(A) && _is_dense_n(X, 1) && _is_dense_n(Y, 1)
end

# SpMM: A sparse, B and C 2D dense.
function JLUST.supports_backend(::CUSPARSEBackend, ::Op{:SpMM, Tuple{A, B, C}}) where {A, B, C}
    _is_generic(A) && _is_dense_n(B, 2) && _is_dense_n(C, 2)
end

# BatchedSpMM: only CSR with 2D dense B/C; cuSPARSE batched SpMM via CuSparseArrayCSR.
function JLUST.supports_backend(::CUSPARSEBackend, ::Op{:BatchedSpMM, Tuple{A, B, C}}) where {A, B, C}
    _is_csr(A) && _is_dense_n(B, 2) && _is_dense_n(C, 2)
end

# SpGEMM: CSR×CSR→CSR.
function JLUST.supports_backend(::CUSPARSEBackend, ::Op{:SpGEMM, Tuple{A, B, C}}) where {A, B, C}
    _is_csr(A) && _is_csr(B) && _is_csr(C)
end

# SpSV: sparse triangular solve (single RHS); CSR or CSC, 1D dense b/x.
function JLUST.supports_backend(::CUSPARSEBackend, ::Op{:SpSV, Tuple{A, B, X}}) where {A, B, X}
    _is_trisv_fmt(A) && _is_dense_n(B, 1) && _is_dense_n(X, 1)
end

# SpSM: sparse triangular solve (multi-RHS); CSR or CSC, 2D dense B/C.
function JLUST.supports_backend(::CUSPARSEBackend, ::Op{:SpSM, Tuple{A, B, C}}) where {A, B, C}
    _is_trisv_fmt(A) && _is_dense_n(B, 2) && _is_dense_n(C, 2)
end

# SDDMM: A,B 2D dense; C sparse (CSR or COO).
function JLUST.supports_backend(::CUSPARSEBackend, ::Op{:SDDMM, Tuple{A, B, C}}) where {A, B, C}
    _is_dense_n(A, 2) && _is_dense_n(B, 2) &&
    (format_family(C) === :CSR || format_family(C) === :COO)
end

# SparseToDense / DenseToSparse: CSR, CSC, COO (generic API).
JLUST.supports_backend(::CUSPARSEBackend, ::Op{:SparseToDense, Tuple{S}}) where {S} = _is_generic(S)
JLUST.supports_backend(::CUSPARSEBackend, ::Op{:DenseToSparse, Tuple{D}}) where {D} = _is_generic(D)

# Sparse vector ops: COO-style (index + value arrays).
JLUST.supports_backend(::CUSPARSEBackend, ::Op{:Gather,  Tuple{F}}) where {F} = format_family(F) === :COO
JLUST.supports_backend(::CUSPARSEBackend, ::Op{:Scatter, Tuple{F}}) where {F} = format_family(F) === :COO
function JLUST.supports_backend(::CUSPARSEBackend, ::Op{:Axpby, Tuple{X, Y}}) where {X, Y}
    format_family(X) === :COO && _is_dense_n(Y, 1)
end
function JLUST.supports_backend(::CUSPARSEBackend, ::Op{:Rot, Tuple{X, Y}}) where {X, Y}
    format_family(X) === :COO && _is_dense_n(Y, 1)
end

# Direct format conversions (vendor-accelerated).
# CSR↔CSC, CSR↔COO: generic API.  BSR↔CSR: csr2bsr / bsr2csr legacy API.
function JLUST.supports_convert(::CUSPARSEBackend, src::TensorFormat, dst::TensorFormat)
    fs = format_family(src)
    fd = format_family(dst)
    (fs === :CSR && fd === :CSC) ||
    (fs === :CSC && fd === :CSR) ||
    (fs === :CSR && fd === :COO) ||
    (fs === :COO && fd === :CSR) ||
    (fs === :CSR && fd === :BSR) ||   # csr2bsr
    (fs === :BSR && fd === :CSR)      # bsr2csr
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
