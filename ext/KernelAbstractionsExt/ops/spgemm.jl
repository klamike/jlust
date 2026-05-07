# ─── EmitterBackend SpGEMM (scatter-sort-reduce) ─────────────────────────────
#
# C ← alpha * A * B  (beta=0 only; CSR×CSR→CSR)
#
# Five-phase algorithm:
#   1. Count total products per row of C.
#   2. Scatter (key, val) pairs: key = (row-1)*n + (col-1), val = a*b.
#   3. Sort pairs by key → groups same (row,col) entries adjacently.
#   4. Mark segment heads; cumsum heads → positions in C.
#   5. Scatter-reduce vals into C_nzVal; extract row/col from keys → C_rowPtr/colInd.

# ── Phase 1: count products per row ──────────────────────────────────────────

@kernel inbounds=true function _emitter_spgemm_count!(
        A_rowPtr, A_colInd, B_rowPtr, prod_count, n_outer, off)
    i = @index(Global, Linear)
    if i <= n_outer
        lo = Int(A_rowPtr[i])     - Int(off)
        hi = Int(A_rowPtr[i + 1]) - Int(off)
        cnt = 0
        for p = lo + 1:hi
            j = Int(A_colInd[p]) - Int(off) + 1   # 1-based row of B
            cnt += Int(B_rowPtr[j + 1]) - Int(B_rowPtr[j])
        end
        prod_count[i] = cnt
    end
end

# ── Phase 2: scatter products as (key, val) pairs ────────────────────────────
#
# prod_offset[i] is the 1-based start position (in keys/vals) for row i.
# Key encodes the 0-based (row, col) pair as  (i-1)*n + col_0based.

@kernel inbounds=true function _emitter_spgemm_scatter!(
        A_rowPtr, A_colInd, A_nzVal,
        B_rowPtr, B_colInd, B_nzVal,
        prod_offset, keys, vals, n_outer, n, off)
    i = @index(Global, Linear)
    if i <= n_outer
        lo_a = Int(A_rowPtr[i])     - Int(off)
        hi_a = Int(A_rowPtr[i + 1]) - Int(off)
        pos  = Int(prod_offset[i])
        for p_a = lo_a + 1:hi_a
            j   = Int(A_colInd[p_a]) - Int(off) + 1   # 1-based row of B
            a_v = A_nzVal[p_a]
            lo_b = Int(B_rowPtr[j])     - Int(off)
            hi_b = Int(B_rowPtr[j + 1]) - Int(off)
            for p_b = lo_b + 1:hi_b
                col_0 = Int(B_colInd[p_b]) - Int(off)   # 0-based col of B/C
                keys[pos] = Int64(i - 1) * Int64(n) + Int64(col_0)
                vals[pos] = a_v * B_nzVal[p_b]
                pos      += 1
            end
        end
    end
end

# ── Phase 3: mark segment heads (after external sort by key) ─────────────────

@kernel inbounds=true function _emitter_spgemm_mark_heads!(keys_sorted, heads, total_products)
    i = @index(Global, Linear)
    if i <= total_products
        heads[i] = (i == 1) | (keys_sorted[i] != keys_sorted[i - 1])
    end
end

# ── Phase 4: scatter-reduce into C nzVal + copy unique keys ──────────────────
#
# head_pos[i] = 1-based position in C for entry i (cumsum of heads array).
# Initialise C_nzVal to zero before calling.

@kernel inbounds=true function _emitter_spgemm_reduce!(
        keys_sorted, vals_sorted, heads, head_pos, C_keys, C_nzVal, total_products)
    i = @index(Global, Linear)
    if i <= total_products
        out = head_pos[i]
        if heads[i]
            C_keys[out] = keys_sorted[i]
        end
        KernelAbstractions.@atomic C_nzVal[out] += vals_sorted[i]
    end
end

# ── Phase 5: extract row/col from C_keys → build C_rowPtr and C_colInd ───────

@kernel inbounds=true function _emitter_spgemm_build!(C_keys, row_count, C_colInd, nnzC, n, off)
    i = @index(Global, Linear)
    if i <= nnzC
        key   = C_keys[i]
        row   = Int(key ÷ Int64(n)) + 1          # 1-based row
        col_0 = Int(key % Int64(n))               # 0-based col
        C_colInd[i] = eltype(C_colInd)(col_0 + Int(off))   # restore index base
        KernelAbstractions.@atomic row_count[row] += one(eltype(row_count))
    end
end

# ─── sparse_gemm! ─────────────────────────────────────────────────────────────

function JLUST.sparse_gemm!(::EmitterBackend,
                              u_A::USTensor{T,Ti}, u_B::USTensor;
                              transa::Char='N', transb::Char='N',
                              alpha=one(T), beta=zero(T)) where {T, Ti}
    (transa == 'N' && transb == 'N') ||
        error("EmitterBackend sparse_gemm!: transposed operands not yet supported")
    beta == zero(T) ||
        error("EmitterBackend sparse_gemm!: beta ≠ 0 not yet supported")
    (format(u_A) == Formats.CSR && format(u_B) == Formats.CSR) ||
        error("EmitterBackend sparse_gemm!: only CSR×CSR supported")

    A_rowPtr = positions(u_A, 2)
    A_colInd = coordinates(u_A, 2)
    A_nzVal  = nonzeros(u_A)
    B_rowPtr = positions(u_B, 2)
    B_colInd = coordinates(u_B, 2)
    B_nzVal  = nonzeros(u_B)

    m   = Int(extents(u_A)[1])
    n   = Int(extents(u_B)[2])
    off = Int32(index_origin(u_A) isa OneBased ? 1 : 0)

    ka = KernelAbstractions.get_backend(A_nzVal)

    # ── Step 1: count products per row ─────────────────────────────────────────
    prod_count = similar(A_rowPtr, Int64, m)
    _emitter_spgemm_count!(ka, 256)(
        A_rowPtr, A_colInd, B_rowPtr, prod_count, Int32(m), off;
        ndrange = m)
    KernelAbstractions.synchronize(ka)

    # ── Step 2: prefix-sum → scatter positions ─────────────────────────────────
    prod_offset = similar(A_rowPtr, Int64, m + 1)
    fill!(prod_offset, Int64(0))
    accumulate!(+, view(prod_offset, 2:m + 1), prod_count)
    prod_offset .+= Int64(1)

    total_products = Int(prod_offset[m + 1]) - 1

    if total_products == 0
        C_rowPtr = fill!(similar(A_rowPtr, Ti, m + 1), Ti(off))
        C_colInd = similar(A_colInd, Ti, 0)
        C_nzVal  = similar(A_nzVal, T, 0)
        orig = index_origin(u_A)
        return csr_tensor(C_rowPtr, C_colInd, C_nzVal, (m, n); origin=orig)
    end

    # ── Step 3: scatter (key, val) pairs ───────────────────────────────────────
    keys = similar(A_rowPtr, Int64, total_products)
    vals = similar(A_nzVal,  total_products)
    _emitter_spgemm_scatter!(ka, 256)(
        A_rowPtr, A_colInd, A_nzVal,
        B_rowPtr, B_colInd, B_nzVal,
        prod_offset, keys, vals, Int32(m), Int32(n), off;
        ndrange = m)
    KernelAbstractions.synchronize(ka)

    # ── Step 4: sort by key ────────────────────────────────────────────────────
    perm        = sortperm(keys)
    keys_sorted = keys[perm]
    vals_sorted = vals[perm]

    alpha_T = T(alpha)
    if alpha_T != one(T)
        vals_sorted .*= alpha_T
    end

    # ── Step 5: mark segment heads ─────────────────────────────────────────────
    heads = similar(A_rowPtr, Bool, total_products)
    _emitter_spgemm_mark_heads!(ka, 256)(keys_sorted, heads, Int32(total_products);
                                          ndrange = total_products)
    KernelAbstractions.synchronize(ka)

    # cumsum(heads) → 1-based output position for each product
    head_pos = similar(A_rowPtr, Int64, total_products)
    accumulate!(+, head_pos, heads)
    nnzC = Int(head_pos[end])

    # ── Step 6: scatter-reduce into C values ───────────────────────────────────
    C_keys   = similar(A_rowPtr, Int64, nnzC)
    C_nzVal  = fill!(similar(A_nzVal, nnzC), zero(T))
    _emitter_spgemm_reduce!(ka, 256)(
        keys_sorted, vals_sorted, heads, head_pos, C_keys, C_nzVal, Int32(total_products);
        ndrange = total_products)
    KernelAbstractions.synchronize(ka)

    # ── Step 7: build C_rowPtr and C_colInd ───────────────────────────────────
    row_count = fill!(similar(A_rowPtr, Ti, m), zero(Ti))
    C_colInd  = similar(A_colInd, Ti, nnzC)
    _emitter_spgemm_build!(ka, 256)(
        C_keys, row_count, C_colInd, Int32(nnzC), Int32(n), off;
        ndrange = nnzC)
    KernelAbstractions.synchronize(ka)

    C_rowPtr = similar(A_rowPtr, Ti, m + 1)
    fill!(C_rowPtr, zero(Ti))
    accumulate!(+, view(C_rowPtr, 2:m + 1), row_count)
    C_rowPtr .+= Ti(off)

    orig = index_origin(u_A)
    return csr_tensor(C_rowPtr, C_colInd, C_nzVal, (m, n); origin=orig)
end
