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
include("convenience.jl")
include("block_sparse.jl")
include("block_banded.jl")

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
    AbstractKernelHandle,
    AbstractUSTOp, Op,
    SpVVOp, SpMVOp, SpMMOp, BatchedSpMMOp, SpGEMMOp,
    SpSVOp, SpSMOp, SDDMMOp,
    SparseToDenseOp, DenseToSparseOp,
    GatherOp, ScatterOp, AxpbyOp, RotOp,
    supports_backend, supports_convert, validate_storage,
    EmitterBackend, CUSPARSEBackend,
    level_has_nzval, level_arg_names, level_args, emit_spmv_lv, level_step,
    locate_level, needs_row_guard,

    # Tensor
    AbstractUSTensor, USTensor,
    positions, coordinates, has_positions, has_coordinates,
    format, extents, index_origin, memory_space,
    # nnz and nonzeros come from SparseArrays (extended for USTensor)

    # Interop / constructors
    ust, csr_tensor, csc_tensor, coo_tensor, dcsr_tensor,
    selector_tensor, diagonal_tensor,

    # Convert
    convert_format, convert_index_type, convert_value_type,
    TensorDecomposer, TensorComposer, run!,
    materialize,

    # Handle types (exported from backend extensions)
    # CUSPARSESpMVHandle, CUSPARSESpMMHandle, CUSPARSESpSVHandle, CUSPARSESpSMHandle,
    # CUSPARSESDDMMHandle, CUSPARSESpGEMMHandle  (exported from CUDAExt)

    # Execution
    #   `execute(OpType, args...; backend, kw...)` — canonical entry point.
    #   `execute(handle, args...; kw...)` — prepared-handle path.
    execute,
    apply_values!,
    prepare, update_values!,

    # Custom level format constructor helper
    make_tensor,

    # Block matrix
    BlockSparseMatrix,
    update_block_values!,
    batch_mul!, batch_mul,
    BlockBandedMatrix

end  # module JLUST
