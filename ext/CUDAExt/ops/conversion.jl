# ─── cuSPARSE format conversions ─────────────────────────────────────────────

# sparse_to_dense: CSR, CSC, or COO → dense CuMatrix wrapped as a USTensor.
function JLUST.sparse_to_dense(::CUSPARSEBackend, u::USTensor{T}) where {T<:_CUSPARSE_ELTYPES}
    cusA = _to_cuspmat(u)
    idx  = _cusparse_index(u)
    dense_mat = CUSPARSE.sparsetodense(cusA, idx)
    return ust(dense_mat)
end

# dense_to_sparse: dense CuMatrix USTensor → sparse USTensor in fmt (CSR, CSC, or COO).
# The input u must be a 2-D all-dense USTensor whose val field is a CuMatrix.
function JLUST.dense_to_sparse(::CUSPARSEBackend, u::USTensor{T}, fmt::TensorFormat) where {T<:_CUSPARSE_ELTYPES}
    idx = _cusparse_index(u)
    sym = if fmt == Formats.CSR
        :csr
    elseif fmt == Formats.CSC
        :csc
    elseif fmt == Formats.COO
        :coo
    else
        error("dense_to_sparse: unsupported target format $(format_family(fmt)) for CUSPARSEBackend")
    end
    result = CUSPARSE.densetosparse(nonzeros(u), sym, idx)
    return ust(result)
end
