# ─── EmitterBackend SpGEMM (scatter-sort-reduce) ─────────────────────────────
#
# C ← alpha * A * B  (beta=0 only; CSR×CSR→CSR)
#
# Five-phase algorithm (direct path):
#   1. Count total products per row of C.
#   2. Scatter (key, val) pairs; key encodes 0-based (row, col).
#   3. Sort pairs by key → groups same (row,col) entries adjacently.
#   4. Mark segment heads; cumsum heads → positions in C.
#   5. Scatter-reduce vals into C_nzVal; extract row/col from keys → C_rowPtr/colInd.
#
# Handle path (EmitterSpGEMMHandle):
#   prepare() caches sorted a_idx, b_idx, c_pos, is_head from symbolic analysis.
#   sparse_gemm!(h, ...) uses product-parallel kernels — one thread per product:
#     Fast path (nnzC == total_products): direct non-atomic write to C_nzVal.
#     General path: assign-heads pass then atomic add-nonheads pass.
#
# Key encoding — chosen based on matrix dimensions at runtime:
#   UInt32 (m ≤ 65536 && n ≤ 65536): bit-packed as row<<16 | col.
#     Halves the GPU radix sort cost (4 passes vs 8) and replaces multiply with shift.
#   Int64 (otherwise): row * n + col (safe for any size).

# ── Encode / decode helpers (inlined into kernels) ────────────────────────────

@inline _spgemm_encode(::Type{UInt32}, row_0::Int, col_0::Int, _n) =
    UInt32(row_0) << UInt32(16) | UInt32(col_0)

@inline _spgemm_encode(::Type{Int64}, row_0::Int, col_0::Int, n) =
    Int64(row_0) * Int64(n) + Int64(col_0)

@inline _spgemm_decode(key::UInt32, _n) =
    (Int(key >> UInt32(16)) + 1, Int(key & UInt32(0xFFFF)))   # (1-based row, 0-based col)

@inline _spgemm_decode(key::Int64, n) =
    (Int(key ÷ Int64(n)) + 1, Int(key % Int64(n)))

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
# Key type K is UInt32 (bit-packed) or Int64 (linear) based on dimensions.

@kernel inbounds=true function _emitter_spgemm_scatter!(
        A_rowPtr, A_colInd, A_nzVal,
        B_rowPtr, B_colInd, B_nzVal,
        prod_offset, keys::AbstractArray{K}, vals, n_outer, n, off) where K
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
                keys[pos] = _spgemm_encode(K, i - 1, col_0, n)
                vals[pos] = a_v * B_nzVal[p_b]
                pos      += 1
            end
        end
    end
end

# ── Phase 2b: fill per-product source indices (handle path only) ──────────────
#
# Fills a_idx[pos] = 1-based A nzVal index and b_idx[pos] = 1-based B nzVal
# index for each product at scatter position pos.  Called once in prepare();
# after sorting with perm these become a_idx_sorted / b_idx_sorted that the
# product-parallel numeric kernel uses to recompute A[a]*B[b] without storing vals.

@kernel inbounds=true function _emitter_spgemm_fill_indices!(
        A_rowPtr, A_colInd, B_rowPtr,
        prod_offset, a_idx, b_idx, n_outer, off)
    i = @index(Global, Linear)
    if i <= n_outer
        lo_a = Int(A_rowPtr[i])     - Int(off)
        hi_a = Int(A_rowPtr[i + 1]) - Int(off)
        pos  = Int(prod_offset[i])
        for p_a = lo_a + 1:hi_a
            j   = Int(A_colInd[p_a]) - Int(off) + 1
            lo_b = Int(B_rowPtr[j])     - Int(off)
            hi_b = Int(B_rowPtr[j + 1]) - Int(off)
            for p_b = lo_b + 1:hi_b
                a_idx[pos] = Int32(p_a)
                b_idx[pos] = Int32(p_b)
                pos += 1
            end
        end
    end
end

# ── Phase 3: mark segment heads (after external sort by key) ─────────────────

@kernel inbounds=true function _emitter_spgemm_mark_heads!(keys_sorted, heads, total_products)
    i = @index(Global, Linear)
    if i <= total_products
        # Branch on i==1 to avoid out-of-bounds access at keys_sorted[i-1] when i=1.
        # Using `|` (bitwise) would evaluate the OOB index unconditionally on GPU.
        if i == 1
            heads[i] = true
        else
            heads[i] = keys_sorted[i] != keys_sorted[i - 1]
        end
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

@kernel inbounds=true function _emitter_spgemm_build!(C_keys::AbstractArray{K}, row_count, C_colInd, nnzC, n, off) where K
    i = @index(Global, Linear)
    if i <= nnzC
        row, col_0 = _spgemm_decode(C_keys[i], n)
        C_colInd[i] = eltype(C_colInd)(col_0 + Int(off))
        KernelAbstractions.@atomic row_count[row] += one(eltype(row_count))
    end
end

# ── Handle numeric kernels (product-parallel, one thread per product) ─────────
#
# Fast path: when nnzC == total_products every product maps to a unique C element.
# c_pos[k] == k so we write C_nzVal[k] directly — no atomics, no fill!, perfect
# sequential writes.  Reads from a_idx / b_idx are coalesced; A_nzVal / B_nzVal
# random reads fit in L2 for the matrix sizes we target.

@kernel inbounds=true function _emitter_spgemm_direct_write!(
        A_nzVal, B_nzVal, a_idx, b_idx, C_nzVal, alpha, n_products)
    k = @index(Global, Linear)
    if k <= n_products
        C_nzVal[k] = alpha * A_nzVal[a_idx[k]] * B_nzVal[b_idx[k]]
    end
end

# General path pass 1: assign head products to C_nzVal (non-atomic, initialises
# every C output position exactly once before pass 2 runs).

@kernel inbounds=true function _emitter_spgemm_assign_heads!(
        A_nzVal, B_nzVal, a_idx, b_idx, c_pos, is_head, C_nzVal, alpha, n_products)
    k = @index(Global, Linear)
    if k <= n_products && is_head[k]
        C_nzVal[c_pos[k]] = alpha * A_nzVal[a_idx[k]] * B_nzVal[b_idx[k]]
    end
end

# General path pass 2: atomically accumulate non-head products.  Safe to run
# after pass 1 completes (synchronize between passes) because head assignments
# are already visible.

@kernel inbounds=true function _emitter_spgemm_add_nonheads!(
        A_nzVal, B_nzVal, a_idx, b_idx, c_pos, is_head, C_nzVal, alpha, n_products)
    k = @index(Global, Linear)
    if k <= n_products && !is_head[k]
        KernelAbstractions.@atomic C_nzVal[c_pos[k]] += alpha * A_nzVal[a_idx[k]] * B_nzVal[b_idx[k]]
    end
end

# Small-n path: single atomic kernel (used after fill!(C_nzVal,0)).
# Avoids the second kernel launch overhead of the two-pass path.  Preferred
# when total_products ≤ 1_000_000 because fill! cost (~nnzC*8 bytes) is much
# smaller than the launch overhead saved.

@kernel inbounds=true function _emitter_spgemm_all_atomic!(
        A_nzVal, B_nzVal, a_idx, b_idx, c_pos, C_nzVal, alpha, n_products)
    k = @index(Global, Linear)
    if k <= n_products
        KernelAbstractions.@atomic C_nzVal[c_pos[k]] += alpha * A_nzVal[a_idx[k]] * B_nzVal[b_idx[k]]
    end
end

# ── Large-n warp-per-row + register accumulation ─────────────────────────────
#
# For each C row (= A row), a group of WGRP=32 threads cooperates:
#   Each thread owns C columns {lid, lid+W} (where W=32 and lid is the 0-based
#   lane index within the group).  All 32 threads iterate over the SAME A-row
#   entries in lockstep (SIMT broadcast — all threads read the same addresses,
#   so effective bandwidth is 1× not 32×).  For each A×B product, every thread
#   binary-searches C_colInd for the output column c_local, and only the thread
#   whose lid equals c_local accumulates the product into its private register
#   (acc0 for lid, acc1 for lid+W).
#
# Why no atomics or shared memory:
#   Each thread's accumulator is thread-private (in registers).  Two threads
#   never write to the same register, so no synchronisation is needed.
#
# Constraints:
#   • max C row nnz must not exceed 2 * WGRP = 64.  Checked in prepare().
#   • ndrange = m * WGRP (m rows × 32 threads each).

@inline function _spgemm_bsearch(colind, lo::Int, hi::Int, target::Int)
    while lo <= hi
        mid = (lo + hi) >>> 1
        c   = Int(colind[mid])
        c == target && return mid
        lo  = c < target ? mid + 1 : lo
        hi  = c < target ? hi      : mid - 1
    end
    return lo   # unreachable when C structure is correct
end

const _SPGEMM_WGRP         = Int32(32)   # threads per row group (= warp size)
const _SPGEMM_SMEM_PER_GRP = Int32(64)   # max C row nnz this path handles (2 × WGRP)

@kernel inbounds=true function _emitter_spgemm_warp_row_reg!(
        A_rowPtr, A_colInd, A_nzVal::AbstractArray{T},
        B_rowPtr, B_colInd, B_nzVal,
        C_rowPtr, C_colInd, C_nzVal,
        alpha, m, off) where {T}

    W   = Int(_SPGEMM_WGRP)

    gI  = @index(Global, Linear)
    lI  = @index(Local,  Linear)
    lid = (lI - 1) % W           # 0-based lane within group
    row = (gI - 1) ÷ W + 1       # 1-based A/C row

    valid = row <= m
    c_lo  = valid ? Int(C_rowPtr[row])     - Int(off) : 0
    c_hi  = valid ? Int(C_rowPtr[row + 1]) - Int(off) : 0
    c_len = c_hi - c_lo

    # Private register accumulators: this thread owns C positions lid and lid+W.
    acc0 = zero(T)   # for C position c_lo + lid
    acc1 = zero(T)   # for C position c_lo + lid + W (if c_len > W)

    if valid
        lo_a = Int(A_rowPtr[row]) - Int(off)
        hi_a = Int(A_rowPtr[row + 1]) - Int(off)
        for p_a = lo_a + 1:hi_a
            a_val = A_nzVal[p_a] * alpha
            j     = Int(A_colInd[p_a]) - Int(off) + 1   # 1-based B row
            lo_b  = Int(B_rowPtr[j])     - Int(off)
            hi_b  = Int(B_rowPtr[j + 1]) - Int(off)
            for p_b = lo_b + 1:hi_b
                b_col   = Int(B_colInd[p_b])
                c_local = _spgemm_bsearch(C_colInd, c_lo + 1, c_hi, b_col) - (c_lo + 1)
                prod    = a_val * B_nzVal[p_b]
                if c_local == lid
                    acc0 += prod
                end
                if c_local == lid + W
                    acc1 += prod
                end
            end
        end
    end

    # Write owned C columns to global memory.
    if valid && lid < c_len
        C_nzVal[c_lo + lid + 1] = acc0
    end
    if valid && lid + W < c_len
        C_nzVal[c_lo + lid + W + 1] = acc1
    end
end

# ── EmitterSpGEMMHandle ───────────────────────────────────────────────────────
#
# prepare() caches the symbolic analysis once:
#   a_idx, b_idx  — sorted product source indices into A_nzVal / B_nzVal
#   c_pos         — sorted product → C output position (Int32); empty when fast path
#   is_head       — true if product k is first in its C output group; empty when fast path
#   nnzC          — number of C non-zeros (== total_products triggers fast path)
#
# Subsequent sparse_gemm!(h, u_A, u_B) calls recompute values from a_idx/b_idx
# without any sort, scatter, or malloc.

mutable struct EmitterSpGEMMHandle{K, T, Ti, ORIG<:AbstractIndexOrigin}
    C_rowPtr::AbstractVector{Ti}
    C_colInd::AbstractVector{Ti}
    C_nzVal::AbstractVector{T}
    a_idx::AbstractVector{Int32}    # sorted product A nzVal index (1-based)
    b_idx::AbstractVector{Int32}    # sorted product B nzVal index (1-based)
    c_pos::AbstractVector{Int32}    # sorted product C output position; empty if fast path
    is_head::AbstractVector{Bool}   # segment head flags; empty if fast path
    nnzC::Int
    total_products::Int
    m::Int; n::Int; off::Int32
    orig::ORIG
end

export EmitterSpGEMMHandle

# ── prepare (symbolic analysis + first numeric result) ───────────────────────

function JLUST.prepare(::EmitterBackend, ::Type{<:Op{:SpGEMM}},
                        u_A::USTensor{T,Ti}, u_B::USTensor;
                        transa::Char='N', transb::Char='N',
                        alpha=one(T), beta=zero(T)) where {T, Ti}
    (transa == 'N' && transb == 'N') ||
        error("EmitterSpGEMMHandle: transposed operands not supported")
    beta == zero(T) ||
        error("EmitterSpGEMMHandle: beta ≠ 0 not supported")
    (format(u_A) == Formats.CSR && format(u_B) == Formats.CSR) ||
        error("EmitterSpGEMMHandle: only CSR×CSR supported")

    A_rowPtr = positions(u_A, 2); A_colInd = coordinates(u_A, 2); A_nzVal = nonzeros(u_A)
    B_rowPtr = positions(u_B, 2); B_colInd = coordinates(u_B, 2); B_nzVal = nonzeros(u_B)

    m   = Int(extents(u_A)[1])
    n   = Int(extents(u_B)[2])
    off = Int32(index_origin(u_A) isa OneBased ? 1 : 0)
    ka  = KernelAbstractions.get_backend(A_nzVal)

    prod_count = similar(A_rowPtr, Int64, m)
    _emitter_spgemm_count!(ka, 256)(
        A_rowPtr, A_colInd, B_rowPtr, prod_count, Int32(m), off; ndrange = m)
    KernelAbstractions.synchronize(ka)

    total_products = Int(sum(prod_count))

    if total_products == 0
        C_rowPtr = fill!(similar(A_rowPtr, Ti, m + 1), Ti(off))
        C_colInd = similar(A_colInd, Ti, 0)
        C_nzVal  = similar(A_nzVal,  T,  0)
        a_idx    = similar(A_rowPtr, Int32, 0)
        b_idx    = similar(A_rowPtr, Int32, 0)
        c_pos    = similar(A_rowPtr, Int32, 0)
        is_head  = similar(A_rowPtr, Bool,  0)
        orig = index_origin(u_A)
        return EmitterSpGEMMHandle{UInt32, T, Ti, typeof(orig)}(
            C_rowPtr, C_colInd, C_nzVal, a_idx, b_idx, c_pos, is_head,
            0, 0, m, n, off, orig)
    end

    prod_offset = similar(A_rowPtr, Int64, m + 1)
    fill!(prod_offset, Int64(0))
    accumulate!(+, view(prod_offset, 2:m + 1), prod_count)
    prod_offset .+= Int64(1)

    # Scatter (key, val) pairs — vals used for first-call C_nzVal via reduce!
    K       = (m <= 0x10000 && n <= 0x10000) ? UInt32 : Int64
    keys    = similar(A_rowPtr, K, total_products)
    vals_ws = similar(A_nzVal, total_products)
    _emitter_spgemm_scatter!(ka, 256)(
        A_rowPtr, A_colInd, A_nzVal,
        B_rowPtr, B_colInd, B_nzVal,
        prod_offset, keys, vals_ws, Int32(m), Int32(n), off; ndrange = m)
    KernelAbstractions.synchronize(ka)

    perm        = sortperm(keys)
    keys_sorted = keys[perm]
    vals_sorted = vals_ws[perm]

    alpha_T = T(alpha)
    if alpha_T != one(T)
        vals_sorted .*= alpha_T
    end

    heads = similar(A_rowPtr, Bool, total_products)
    _emitter_spgemm_mark_heads!(ka, 256)(
        keys_sorted, heads, Int32(total_products); ndrange = total_products)
    KernelAbstractions.synchronize(ka)

    nnzC     = Int(sum(heads))
    head_pos = similar(A_rowPtr, Int64, total_products)
    accumulate!(+, head_pos, heads)

    # Reduce to get first C_nzVal + C_keys for building C_colInd.
    C_keys  = similar(A_rowPtr, K, nnzC)
    C_nzVal = fill!(similar(A_nzVal, nnzC), zero(T))
    _emitter_spgemm_reduce!(ka, 256)(
        keys_sorted, vals_sorted, heads, head_pos, C_keys, C_nzVal, Int32(total_products);
        ndrange = total_products)
    KernelAbstractions.synchronize(ka)

    row_count = fill!(similar(A_rowPtr, Ti, m), zero(Ti))
    C_colInd  = similar(A_colInd, Ti, nnzC)
    _emitter_spgemm_build!(ka, 256)(
        C_keys, row_count, C_colInd, Int32(nnzC), Int32(n), off; ndrange = nnzC)
    KernelAbstractions.synchronize(ka)

    C_rowPtr = similar(A_rowPtr, Ti, m + 1)
    fill!(C_rowPtr, zero(Ti))
    accumulate!(+, view(C_rowPtr, 2:m + 1), row_count)
    C_rowPtr .+= Ti(off)

    # Fill per-product source indices; sort by same perm used for keys.
    a_idx_ws = similar(A_rowPtr, Int32, total_products)
    b_idx_ws = similar(A_rowPtr, Int32, total_products)
    _emitter_spgemm_fill_indices!(ka, 256)(
        A_rowPtr, A_colInd, B_rowPtr,
        prod_offset, a_idx_ws, b_idx_ws, Int32(m), off; ndrange = m)
    KernelAbstractions.synchronize(ka)
    a_idx = a_idx_ws[perm]
    b_idx = b_idx_ws[perm]

    if nnzC == total_products
        c_pos_stored   = similar(A_rowPtr, Int32, 0)
        is_head_stored = similar(A_rowPtr, Bool,  0)
    else
        c_pos_stored   = similar(A_rowPtr, Int32, total_products)
        c_pos_stored  .= head_pos
        is_head_stored = copy(heads)
    end

    orig = index_origin(u_A)
    return EmitterSpGEMMHandle{K, T, Ti, typeof(orig)}(
        C_rowPtr, C_colInd, C_nzVal, a_idx, b_idx, c_pos_stored, is_head_stored,
        nnzC, total_products, m, n, off, orig)
end

# ── Handle path — product-parallel numeric; no sort, no malloc ────────────────
#
# Fast path (nnzC == total_products): one non-atomic write per thread, sequential
# C_nzVal writes, no fill! needed.
# General path: assign-heads pass initialises every C position, then atomic
# add-nonheads pass accumulates remaining products.  Two kernel launches with
# a synchronise between them guarantees no race between the two passes.

function JLUST.execute(h::EmitterSpGEMMHandle{K, T, Ti},
                        u_A::USTensor, u_B::USTensor;
                        alpha=one(T), beta=zero(T)) where {K, T, Ti}
    beta == zero(T) ||
        error("EmitterSpGEMMHandle sparse_gemm!: beta ≠ 0 not supported")

    h.total_products == 0 &&
        return csr_tensor(h.C_rowPtr, h.C_colInd, h.C_nzVal, (h.m, h.n); origin=h.orig)

    A_nzVal = nonzeros(u_A)
    B_nzVal = nonzeros(u_B)
    ka      = KernelAbstractions.get_backend(A_nzVal)
    alpha_T = T(alpha)
    np      = Int32(h.total_products)

    if h.nnzC == h.total_products
        # Fast path: all products unique → direct sequential write, no atomics, no fill!.
        _emitter_spgemm_direct_write!(ka, 256)(
            A_nzVal, B_nzVal, h.a_idx, h.b_idx, h.C_nzVal, alpha_T, np;
            ndrange = h.total_products)
        KernelAbstractions.synchronize(ka)
    elseif h.total_products <= 1_000_000
        # Small-n path: fill! + single atomic kernel.  One kernel launch instead of
        # two; fill! cost (nnzC*8 bytes) is negligible at these sizes.
        fill!(h.C_nzVal, zero(T))
        _emitter_spgemm_all_atomic!(ka, 256)(
            A_nzVal, B_nzVal, h.a_idx, h.b_idx, h.c_pos, h.C_nzVal, alpha_T, np;
            ndrange = h.total_products)
        KernelAbstractions.synchronize(ka)
    else
        # Large-n two-pass: pass 1 assigns head products (no fill! needed), pass 2
        # atomically adds the rare non-head products.
        _emitter_spgemm_assign_heads!(ka, 256)(
            A_nzVal, B_nzVal, h.a_idx, h.b_idx, h.c_pos, h.is_head, h.C_nzVal, alpha_T, np;
            ndrange = h.total_products)
        KernelAbstractions.synchronize(ka)
        _emitter_spgemm_add_nonheads!(ka, 256)(
            A_nzVal, B_nzVal, h.a_idx, h.b_idx, h.c_pos, h.is_head, h.C_nzVal, alpha_T, np;
            ndrange = h.total_products)
        KernelAbstractions.synchronize(ka)
    end

    return csr_tensor(h.C_rowPtr, h.C_colInd, h.C_nzVal, (h.m, h.n); origin=h.orig)
end

# ─── sparse_gemm! ─────────────────────────────────────────────────────────────
#
# The direct path IS prepare-then-extract-result.  `prepare` runs the full
# pipeline (count → scatter → sort → mark heads → reduce → build) and stores
# everything in the handle, including a valid first-call C_nzVal.  The direct
# path returns a USTensor view onto those buffers — no second pass over the
# data.  Re-running the numeric pipeline against new (A, B) values is what the
# handle path does via `execute(handle, A_new, B_new)`.

function JLUST.execute(::EmitterBackend, ::Op{:SpGEMM, F},
                       u_A::USTensor, u_B::USTensor, u_C::USTensor=u_B;
                       transa::Char='N', transb::Char='N',
                       alpha=one(eltype(u_A)), beta=zero(eltype(u_A))) where {F}
    h = JLUST.prepare(EmitterBackend(), Op{:SpGEMM}, u_A, u_B;
                      transa=transa, transb=transb, alpha=alpha, beta=beta)
    return csr_tensor(h.C_rowPtr, h.C_colInd, h.C_nzVal, (h.m, h.n); origin=h.orig)
end
