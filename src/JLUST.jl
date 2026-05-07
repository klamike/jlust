module JLUST

using SparseArrays
import SparseArrays: nnz, nonzeros   # extended for USTensor in tensor.jl

include("errors.jl")
include("formats.jl")
include("named_formats.jl")
include("tensor.jl")
include("ops.jl")
include("backends.jl")
include("interop.jl")
include("convert.jl")

export
    # Errors
    InvalidTensorFormat, InvalidLevelAccess,
    UnsupportedFormat, UnsupportedIndexType, UnsupportedValueType,
    IndexOriginMismatch, DeviceMismatch, IncompatibleExtents, NonCanonicalStorage,

    # Level formats
    AbstractLevelFormat,
    DenseLevel, BatchLevel, CompressedLevel, SingletonLevel, RangeLevel, DeltaLevel,
    is_ordered, is_unique,

    # Format DSL
    Dimension, LevelExpr, dims,
    TensorFormat, dim2lvl, lvl2dim, format_family,

    # Macro
    @tensor_format,

    # Named formats
    Formats,

    # Index origin / memory space / device
    AbstractIndexOrigin, OneBased, ZeroBased,
    AbstractMemorySpace, CPUMemory, GPUMemory,
    CPUDevice, CUDADevice,

    # Backend API
    AbstractUSTBackend,
    AbstractUSTOp,
    SpVVOp, SpMVOp, SpMMOp, BatchedSpMMOp, SpGEMMOp,
    SpSVOp, SpSMOp, SDDMMOp,
    SparseToDenseOp, DenseToSparseOp,
    GatherOp, ScatterOp, AxpbyOp, RotOp,
    supports_backend, supports_convert, validate_storage,

    # Tensor
    AbstractUSTensor, USTensor,
    positions, coordinates, has_positions, has_coordinates,
    format, extents, index_origin, memory_space,
    # nnz and nonzeros come from SparseArrays (extended for USTensor)

    # Interop
    ust, csr_tensor, csc_tensor, coo_tensor,

    # Convert
    convert_format, convert_index_type, convert_value_type,
    TensorDecomposer, TensorComposer, run!,
    materialize,

    # Handle types (exported from backend extensions)
    # CUSPARSESpMVHandle, CUSPARSESpMMHandle, CUSPARSESpSVHandle, CUSPARSESpSMHandle,
    # CUSPARSESDDMMHandle, CUSPARSESpGEMMHandle  (exported from CUDAExt)

    # Hooks for backend-specialized SpMV; overridden by CUDAExt for GPU kernels
    _coo_spmv_specialized!,
    _csr_spmv_specialized!,

    # Execution (methods added by backend extensions)
    apply_values!,
    sparse_mv!, sparse_mm!, sparse_gemm!,
    sparse_vv, sparse_sv!, sparse_sm!, sparse_sddmm!,
    sparse_to_dense, dense_to_sparse,
    sparse_gather!, sparse_scatter!, sparse_axpby!, sparse_rot!,
    prepare, update_values!

end  # module JLUST
