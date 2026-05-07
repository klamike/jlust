# ─── Shared helpers for CUSPARSEBackend ops ──────────────────────────────────

# cuSPARSE index-base character
_cusparse_index(::OneBased)  = 'O'
_cusparse_index(::ZeroBased) = 'Z'
_cusparse_index(u::USTensor) = _cusparse_index(index_origin(u))

# Element types natively supported by the cuSPARSE generic API
const _CUSPARSE_ELTYPES = Union{Float32, Float64, ComplexF32, ComplexF64}

# Build a CUDA.jl sparse matrix from a 2-D USTensor (CSR, CSC, or COO only).
function _to_cuspmat(u::USTensor{T,Ti}) where {T,Ti}
    fmt  = format(u)
    dims = (Int(extents(u)[1]), Int(extents(u)[2]))
    nz   = nonzeros(u)

    if fmt == Formats.CSR
        CuSparseMatrixCSR{T,Ti}(positions(u, 2), coordinates(u, 2), nz, dims)
    elseif fmt == Formats.CSC
        CuSparseMatrixCSC{T,Ti}(positions(u, 2), coordinates(u, 2), nz, dims)
    elseif fmt == Formats.COO
        CuSparseMatrixCOO{T,Ti}(coordinates(u, 1), coordinates(u, 2), nz, dims)
    else
        error("CUSPARSEBackend: _to_cuspmat not implemented for format $(format_family(fmt))")
    end
end
