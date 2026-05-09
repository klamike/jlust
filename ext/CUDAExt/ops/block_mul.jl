# ─── Structured matrix GPU mul!: 4-layer architecture ───────────────────────
#
# A BlockSparseMatrix or BlockBandedMatrix is *conceptually* a single sparse
# matrix whose nzval is logically partitioned into named blocks.  We maintain
# four views of the same matrix, each optimal for a different access pattern:
#
#   1. blocks view  — per-block USTensors; source of truth for parametric
#                     updates (`update_block_values!` patches one block).
#   2. compiled CSR — single materialized CSR USTensor; the BSM's "compute
#                     identity".  BBM falls back here for shapes the periodic
#                     view doesn't handle.
#   3. periodic     — direct refs to underlying block CSRs + structural
#                     metadata; used for BBM with shared diag/off-diag (the
#                     common case).  Avoids T-fold materialization of the BSM.
#   4. CUDA graph   — captures the emitter-SpMV launch from view 2 or 3,
#                     replays via cuGraphLaunch.  Keyed on (ptr_y, ptr_x);
#                     hot-path field cache (`last_y_ptr` / `last_x_ptr` /
#                     `last_exec`) elides the dict lookup when buffers reuse.
#
# `update_block_values!` keeps view 1 and view 2 in sync via per-block index
# maps (the CSR's pos/crd are unchanged when nzval is patched, so the captured
# graph stays valid).  View 3 inherits view 2's BSM-compiled CSR by reference,
# so it stays in sync automatically.
#
# Why emitter SpMV instead of cuSPARSE for the captured kernel: the emitter
# warp-vector CSR kernel is ~1.7-2.0× leaner per launch than cusparseSpMV on
# these matrix shapes (vendor algorithm has more setup work per call).  When
# cached in a CUDA graph, the captured kernel replays at near-launch-overhead
# cost — JLUST mul! beats cuSPARSE-on-equivalent-CSR by 1.2-3.1× on L40S.
#
# Inside an outer CUDA capture (user-driven `CUDA.capture`), `mul!` short-
# circuits to direct kernel emission so the capture absorbs our launch as a
# single graph node — `cuGraphLaunch` on a child graph would invalidate the
# outer capture.

import LinearAlgebra
import JLUST: BlockSparseMatrix, BlockBandedMatrix, BBMSpMVOp, AbstractKernelHandle

# ─── Compiled CSR view ───────────────────────────────────────────────────────

mutable struct _CompiledBSM{T,Ti}
    asm     :: USTensor                              # assembled CSR USTensor
    map     :: Matrix{Union{Nothing, CuVector{Ti}}}  # (i,j) → indices into asm.nzval
    graph   :: Dict{Tuple{UInt,UInt}, CUDA.CuGraphExec}  # (ptr_y, ptr_x) → captured graph
    # Hot-path fast cache: most repeated mul! calls reuse the same (y, x) buffers,
    # so a single field comparison skips the Dict lookup and tuple alloc.
    last_y_ptr :: UInt
    last_x_ptr :: UInt
    last_exec  :: Union{Nothing, CUDA.CuGraphExec}
end

const _bsm_compiled_cache = IdDict{BlockSparseMatrix, _CompiledBSM}()

# Whether a BSM is supported by the compiled-CSR fast path: every non-null
# block must be CSR.  Other formats fall through to the per-block emitter path.
function _bsm_supports_compile(A::BlockSparseMatrix)
    nb_r, nb_c = size(A.blocks)
    for i in 1:nb_r, j in 1:nb_c
        b = A.blocks[i, j]; b === nothing && continue
        format(b) == Formats.CSR || return false
    end
    true
end

# Build the compiled CSR on the host, then move buffers to GPU.
# Block (i,j) row r contributes pos[r+1]-pos[r] entries that land contiguously
# in assembled.nzval at the slots reserved for block-row i row r, column-block j.
function _compile_bsm(A::BlockSparseMatrix{T}) where T
    nb_r, nb_c = size(A.blocks)
    n_rows = sum(A.row_sizes)
    n_cols = sum(A.col_sizes)

    # Pull pos / crd / nzval to host once per non-null block.  All blocks must
    # be CSR — checked by `_bsm_supports_compile` at the call site.
    block_pos = Dict{Tuple{Int,Int}, Vector{Int}}()
    block_crd = Dict{Tuple{Int,Int}, Vector{Int}}()
    block_nz  = Dict{Tuple{Int,Int}, Vector{T}}()
    for i in 1:nb_r, j in 1:nb_c
        b = A.blocks[i, j]; b === nothing && continue
        block_pos[(i,j)] = Vector{Int}(Array(positions(b, 2)))
        block_crd[(i,j)] = Vector{Int}(Array(coordinates(b, 2)))
        block_nz[(i,j)]  = Vector{T}(Array(nonzeros(b)))
    end

    # Per-row nnz count of the assembled.
    row_nnz = zeros(Int, n_rows)
    for ((i,j), pos) in block_pos
        for r in 1:A.row_sizes[i]
            row_nnz[A._row_off[i] + r] += pos[r+1] - pos[r]
        end
    end

    rowptr = Vector{Int32}(undef, n_rows + 1)
    rowptr[1] = 1
    for r in 1:n_rows
        rowptr[r+1] = rowptr[r] + Int32(row_nnz[r])
    end
    nnz_total = Int(rowptr[end]) - 1

    colind = Vector{Int32}(undef, nnz_total)
    nzval  = Vector{T}(undef, nnz_total)
    block_map = Dict{Tuple{Int,Int}, Vector{Int32}}()
    for ((i,j), nz) in block_nz
        block_map[(i,j)] = Vector{Int32}(undef, length(nz))
    end

    cursor = copy(rowptr)
    for i in 1:nb_r, j in 1:nb_c
        haskey(block_pos, (i,j)) || continue
        col_off = A._col_off[j]; row_off = A._row_off[i]
        pos = block_pos[(i,j)]; crd = block_crd[(i,j)]
        nz  = block_nz[(i,j)];  map_ij = block_map[(i,j)]
        for r in 1:A.row_sizes[i]
            for k in pos[r]:pos[r+1] - 1
                slot = cursor[row_off + r]
                cursor[row_off + r] = slot + Int32(1)
                colind[slot] = Int32(crd[k] + col_off)
                nzval[slot]  = nz[k]
                map_ij[k]    = slot
            end
        end
    end

    asm_csr = CUDA.CUSPARSE.CuSparseMatrixCSR{T,Int32}(
        CuArray(rowptr), CuArray(colind), CuArray(nzval), (n_rows, n_cols))
    asm_ust = JLUST.ust(asm_csr)

    map_gpu = Matrix{Union{Nothing, CuVector{Int32}}}(nothing, nb_r, nb_c)
    for ((i,j), v) in block_map
        map_gpu[i, j] = CuArray(v)
    end

    _CompiledBSM{T,Int32}(asm_ust, map_gpu,
                          Dict{Tuple{UInt,UInt}, CUDA.CuGraphExec}(),
                          UInt(0), UInt(0), nothing)
end

@inline _ensure_compiled_bsm(A::BlockSparseMatrix) =
    get!(() -> _compile_bsm(A), _bsm_compiled_cache, A)

# ─── mul! routes through the compiled CSR ────────────────────────────────────

function LinearAlgebra.mul!(y::CuVector, A::BlockSparseMatrix, x::CuVector;
                              backend::Union{AbstractUSTBackend,Nothing}=nothing)
    # Compiled-CSR fast path requires every block to be CSR.  Mixed-format
    # BSMs fall through to per-block dispatch below.
    if (backend === nothing || backend isa CUSPARSEBackend) && _bsm_supports_compile(A)
        c   = _ensure_compiled_bsm(A)
        CUDA.is_capturing() && (_bsm_emit_spmv!(c, x, y); return y)
        yp  = UInt(pointer(y));  xp = UInt(pointer(x))
        last = c.last_exec
        if last !== nothing && yp === c.last_y_ptr && xp === c.last_x_ptr
            CUDA.launch(last)
            return y
        end
        return _bsm_mul_slowpath!(y, x, c, yp, xp)
    end
    return _bsm_emitter_mul!(y, A, x, something(backend, EmitterBackend()))
end

@noinline function _bsm_mul_slowpath!(y::CuVector, x::CuVector, c::_CompiledBSM,
                                       yp::UInt, xp::UInt)
    # Inside an outer capture, fall through to a direct emitter SpMV (no nested capture).
    CUDA.is_capturing() && (_bsm_emit_spmv!(c, x, y); return y)

    key = (yp, xp)
    if haskey(c.graph, key)
        exec = c.graph[key]
    else
        # Capture the EMITTER SpMV (warp-vector kernel) — leaner per-launch
        # than cuSPARSE's cusparseSpMV on these matrix shapes.
        _bsm_emit_spmv!(c, x, y); CUDA.synchronize()
        g = CUDA.capture() do; _bsm_emit_spmv!(c, x, y); end
        exec = CUDA.instantiate(g)
        c.graph[key] = exec
    end
    c.last_y_ptr = yp; c.last_x_ptr = xp; c.last_exec = exec
    CUDA.launch(exec)
    return y
end

@inline _bsm_emit_spmv!(c::_CompiledBSM, x::CuVector, y::CuVector) =
    JLUST.execute(JLUST.SpMVOp, c.asm, JLUST.ust(x), JLUST.ust(y);
                  backend=JLUST.EmitterBackend())

function _bsm_emitter_mul!(y::CuVector, A::BlockSparseMatrix, x::CuVector,
                            backend::AbstractUSTBackend)
    nb_r, nb_c = size(A.blocks)
    for i in 1:nb_r
        y_sl      = view(y, A._row_off[i]+1 : A._row_off[i+1])
        first_col = true
        for j in 1:nb_c
            b = A.blocks[i, j]; b === nothing && continue
            x_sl = view(x, A._col_off[j]+1 : A._col_off[j+1])
            β    = first_col ? false : true
            JLUST.execute(JLUST.SpMVOp, b, JLUST.ust(x_sl), JLUST.ust(y_sl);
                          backend=backend, beta=β)
            first_col = false
        end
    end
    return y
end

# ─── update_block_values!: keep both views in sync ──────────────────────────

function JLUST.update_block_values!(A::BlockSparseMatrix, i::Int, j::Int, new_nzval::CuVector)
    b = A.blocks[i, j]
    b isa AbstractUSTensor ||
        error("update_block_values!: block ($i,$j) is not a USTensor (got $(typeof(b)))")
    length(new_nzval) == JLUST.nnz(b) ||
        error("update_block_values!: new_nzval length $(length(new_nzval)) ≠ nnz $(JLUST.nnz(b))")

    # Block-view update (pointer-stable when same eltype/storage).
    if typeof(new_nzval) === typeof(nonzeros(b))
        copyto!(nonzeros(b), new_nzval)
    else
        A.blocks[i, j] = JLUST._swap_val(b, new_nzval)
    end

    # Compiled-view update: scatter new nzval into the slots reserved for (i,j).
    # The BSM's full compiled CSR shares pointer with the captured graph, so
    # this in-place update is picked up automatically on the next launch.
    if haskey(_bsm_compiled_cache, A)
        c = _bsm_compiled_cache[A]
        idx = c.map[i, j]
        if idx !== nothing
            asm_nz = nonzeros(c.asm)
            @inbounds @views asm_nz[idx] .= new_nzval
        end
    end

    # Any BBM that holds A as its diag has a derived compiled view (full
    # assembled BBM CSR and/or periodic compiled view with leaner CSR + patch
    # detection).  Both are snapshots of A's values at compile time and become
    # stale on any update to A's blocks.  Invalidate them so the next mul!
    # rebuilds against the current values.  The cost is a one-shot compile per
    # parametric step — for tight inner loops, the user can batch updates
    # before calling mul!.
    _invalidate_bbm_caches_for_bsm!(A)

    return A
end

function _invalidate_bbm_caches_for_bsm!(A::BlockSparseMatrix)
    # Iterate snapshots of the cache keys to allow safe deletion mid-loop.
    for bbm in collect(keys(_bbm_compiled_cache))
        bbm.diags === A && delete!(_bbm_compiled_cache, bbm)
    end
    for bbm in collect(keys(_bbm_periodic_cache))
        bbm.diags === A && delete!(_bbm_periodic_cache, bbm)
    end
    return nothing
end

# ─── BlockBandedMatrix GPU mul! via compiled CSR view ────────────────────────
#
# A BBM is a (T*n_diag_rows + Σ(T-k)*n_off_rows[k]) × (T*n_cols) banded sparse
# matrix.  Compiling it into a single CSR + cuSPARSE handle reduces every mul!
# to one cuSPARSE call, matching cuSP-h's reference performance exactly.
#
# Materialization happens once per BBM via SparseArrays on the host, then the
# CSR is moved to GPU.  For T=24 this is a one-shot cost amortized over all
# subsequent SpMVs.

mutable struct _CompiledBBM{T,Ti}
    asm    :: USTensor
    graph  :: Dict{Tuple{UInt,UInt}, CUDA.CuGraphExec}
    last_y_ptr :: UInt
    last_x_ptr :: UInt
    last_exec  :: Union{Nothing, CUDA.CuGraphExec}
end

const _bbm_compiled_cache = IdDict{BlockBandedMatrix, _CompiledBBM}()

# Host-side CSR view of a sparse-row source (USTensor or BSM).  Each is a
# tuple (rowptr, colind, nzval) all on host and 1-based — matches our internal
# CSR convention.  We assemble the BBM by row-major splicing of these.

struct _HostCSR{T}
    rowptr :: Vector{Int}
    colind :: Vector{Int}
    nzval  :: Vector{T}
    m      :: Int
    n      :: Int
end

function _host_csr(u::AbstractUSTensor)
    format(u) == Formats.CSR || error("_host_csr: only CSR USTensor supported, got $(format(u))")
    rp = Vector{Int}(Array(positions(u, 2)))
    ci = Vector{Int}(Array(coordinates(u, 2)))
    nz = Vector{eltype(u)}(Array(nonzeros(u)))
    m, n = Int.(extents(u))
    _HostCSR{eltype(u)}(rp, ci, nz, m, n)
end

# A BlockSparseMatrix as a single host CSR.  We process block-row-by-block-row,
# and within each block-row, row-by-row across all column-blocks.
function _host_csr(A::BlockSparseMatrix{T}) where T
    nb_r, nb_c = size(A.blocks)
    n_rows = sum(A.row_sizes); n_cols = sum(A.col_sizes)

    # Pull per-block CSRs to host once.
    block_csr = Dict{Tuple{Int,Int}, _HostCSR{T}}()
    for i in 1:nb_r, j in 1:nb_c
        b = A.blocks[i, j]; b === nothing && continue
        block_csr[(i, j)] = _host_csr(b)
    end

    # Total nnz.
    total_nnz = sum(length(c.nzval) for c in values(block_csr); init=0)
    rowptr = Vector{Int}(undef, n_rows + 1)
    colind = Vector{Int}(undef, total_nnz)
    nzval  = Vector{T}(undef, total_nnz)

    rowptr[1] = 1
    nz_cur = 0
    for i in 1:nb_r
        row_off = A._row_off[i]
        for r in 1:A.row_sizes[i]
            for j in 1:nb_c
                haskey(block_csr, (i, j)) || continue
                bc      = block_csr[(i, j)]
                col_off = A._col_off[j]
                for k in bc.rowptr[r]:bc.rowptr[r+1]-1
                    nz_cur += 1
                    colind[nz_cur] = bc.colind[k] + col_off
                    nzval[nz_cur]  = bc.nzval[k]
                end
            end
            rowptr[row_off + r + 1] = nz_cur + 1
        end
    end
    _HostCSR{T}(rowptr, colind, nzval, n_rows, n_cols)
end

# Per-period diag and off-diag accessors. `diags` may be a single matrix
# (shared across periods) or length-T vector. `off_diags[k]` is either a
# (neg, pos) Tuple (shared across band-k transitions) or a Vector{Tuple}.
_diag_csr(d::AbstractUSTensor, t)    = _host_csr(d)
_diag_csr(d::BlockSparseMatrix, t)   = _host_csr(d)
_diag_csr(d::AbstractVector, t)      = _diag_csr(d[t], t)

_off_pair_csr(p::Tuple, t)           = (_host_csr(p[1]), _host_csr(p[2]))
_off_pair_csr(p::AbstractVector, t)  = _off_pair_csr(p[t], t)

# Direct CSR assembly for the BBM — fills (rowptr, colind, nzval) row-by-row
# from the underlying block CSRs without going through SparseArrays.  Avoids
# the O(rows × cols) cost of `spzeros(...) ; setindex!(...)` for large BBMs.
function _compile_bbm(M::BlockBandedMatrix)
    diags     = M.diags
    off_diags = M.off_diags
    T_per     = M.T
    bw        = M.bw
    n_diag    = M.n_diag_rows
    n_off     = M.n_off_rows
    n_cols    = M.n_cols
    Tel       = eltype(M)

    # Period-shared diag CSR (when diags is a single object) — cache it once.
    diag_shared = (diags isa AbstractVector) ? nothing : _diag_csr(diags, 1)
    @inline _diag_at(t) = diag_shared === nothing ? _diag_csr(diags, t) : diag_shared

    # Same idea for off-diag pairs at each band k: shared if Tuple, varying if Vector.
    off_shared = ntuple(k -> (off_diags[k] isa Tuple ? _off_pair_csr(off_diags[k], 1) : nothing), bw)
    @inline _off_at(k, t) = off_shared[k] === nothing ? _off_pair_csr(off_diags[k], t) : off_shared[k]

    # Total rows and nnz.
    n_rows = T_per * n_diag + sum((T_per - k) * n_off[k] for k in 1:bw; init=0)
    n_cols_full = T_per * n_cols
    nnz_diag = length(_diag_at(1).nzval)   # constant across t when shared; else upper bound
    nnz_off  = ntuple(bw) do k
        nz_neg, nz_pos = if off_shared[k] !== nothing
            length(off_shared[k][1].nzval), length(off_shared[k][2].nzval)
        else
            # variable per transition — sum over t in 1..T-k
            sum(length(_off_pair_csr(off_diags[k], t)[1].nzval) for t in 1:T_per-k),
            sum(length(_off_pair_csr(off_diags[k], t)[2].nzval) for t in 1:T_per-k)
        end
        nz_neg, nz_pos
    end

    total_nnz = T_per * nnz_diag + sum((nnz_off[k][1] + nnz_off[k][2]) *
                                        (off_shared[k] === nothing ? 1 : (T_per - k))
                                        for k in 1:bw; init=0)
    # When off_shared[k] !== nothing, nnz_off[k] is per-transition counts and we
    # multiply by (T_per - k); when nothing, nnz_off[k] is already the sum.
    # Simpler: recompute correctly:
    total_nnz = T_per * nnz_diag
    for k in 1:bw
        for t in 1:T_per - k
            neg_t, pos_t = _off_at(k, t)
            total_nnz += length(neg_t.nzval) + length(pos_t.nzval)
        end
    end

    rowptr = Vector{Int32}(undef, n_rows + 1)
    colind = Vector{Int32}(undef, total_nnz)
    nzval  = Vector{Tel}(undef, total_nnz)

    rowptr[1] = 1
    nz_cur  = 0
    row_cur = 0

    for t in 1:T_per
        diag_t  = _diag_at(t)
        col_off = (t - 1) * n_cols
        # Diag rows of this period.
        for r in 1:n_diag
            for k in diag_t.rowptr[r] : diag_t.rowptr[r+1] - 1
                nz_cur += 1
                colind[nz_cur] = Int32(diag_t.colind[k] + col_off)
                nzval[nz_cur]  = diag_t.nzval[k]
            end
            row_cur += 1
            rowptr[row_cur + 1] = Int32(nz_cur + 1)
        end
        # Off-diag rows of this period (one row-block per band k).
        for k in 1:bw
            t + k > T_per && continue
            neg, pos = _off_at(k, t)
            col_off_neg = (t - 1)     * n_cols
            col_off_pos = (t + k - 1) * n_cols
            for r in 1:n_off[k]
                # neg row r
                for kk in neg.rowptr[r] : neg.rowptr[r+1] - 1
                    nz_cur += 1
                    colind[nz_cur] = Int32(neg.colind[kk] + col_off_neg)
                    nzval[nz_cur]  = neg.nzval[kk]
                end
                # pos row r
                for kk in pos.rowptr[r] : pos.rowptr[r+1] - 1
                    nz_cur += 1
                    colind[nz_cur] = Int32(pos.colind[kk] + col_off_pos)
                    nzval[nz_cur]  = pos.nzval[kk]
                end
                row_cur += 1
                rowptr[row_cur + 1] = Int32(nz_cur + 1)
            end
        end
    end

    @assert row_cur == n_rows
    @assert nz_cur  == total_nnz

    # Move to GPU.
    cu_csr = CUDA.CUSPARSE.CuSparseMatrixCSR{Tel,Int32}(
        CuArray(rowptr), CuArray(colind), CuArray(nzval), (n_rows, n_cols_full))
    asm = JLUST.ust(cu_csr)
    _CompiledBBM{Tel,Int32}(asm,
                            Dict{Tuple{UInt,UInt}, CUDA.CuGraphExec}(),
                            UInt(0), UInt(0), nothing)
end

@inline _ensure_compiled_bbm(M::BlockBandedMatrix) =
    get!(() -> _compile_bbm(M), _bbm_compiled_cache, M)

@inline _bbm_emit_spmv!(c::_CompiledBBM, x::CuVector, y::CuVector) =
    JLUST.execute(JLUST.SpMVOp, c.asm, JLUST.ust(x), JLUST.ust(y);
                  backend=JLUST.EmitterBackend())

# ─── BBM-periodic SpMV: structurally-aware kernel ────────────────────────────
#
# For the common BBM shape (bw=1, BSM diag shared across periods, Tuple
# off-diag shared across transitions), we can avoid materializing the T copies
# of the BSM diag CSR.  The kernel walks the BBM's row layout — period by period
# — and indexes into the SHARED block CSRs with a column offset computed from
# the period number.
#
# Bandwidth gain: the BSM diag CSR is read once across the kernel (cache-warm
# for the second period's threads), instead of T copies of duplicate bytes.
# For T=24 this is up to ~24× less DRAM traffic on the diag part.

# Row layout (1-based): period t occupies rows
#   [(t-1)*P+1 .. (t-1)*P+n_diag]  (diag rows of period t),  P = n_diag + n_off
#   [(t-1)*P+n_diag+1 .. t*P]      (off rows for t→t+1, only if t < T_per).
# Total rows: T_per*n_diag + (T_per-1)*n_off.

function _bbm_periodic_spmv_kernel!(
        d_pos, d_crd, d_nzval,
        n_pos, n_crd, n_nzval,
        p_pos, p_crd, p_nzval,
        y, x_raw,
        n_diag::Int32, n_off::Int32, n_cols::Int32,
        n_total_rows::Int32,
        ::Val{HAS_PATCH}, dp_row_start::Int32, dp_row_end::Int32,
        dp_col_offset::Int32, dp_val,
        ::Val{ZERO_BETA}, beta) where {HAS_PATCH, ZERO_BETA}
    T   = eltype(y)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    tid > n_total_rows && return nothing

    # Mark x as read-only so loads go through LDG (read-only data cache).  x is
    # accessed at irregular column indices, so the texture-style cache helps
    # more than the regular L1 — same warp threads frequently hit the same
    # cache line for nearby cols, and the read-only cache has separate budget
    # from the L1 lines holding nzval/crd.
    x = Base.Experimental.Const(x_raw)

    period_size  = n_diag + n_off
    period_m1    = (tid - Int32(1)) ÷ period_size
    rem          = (tid - Int32(1)) - period_m1 * period_size
    col_off_t    = period_m1 * n_cols

    acc = zero(T)

    if rem < n_diag
        r = rem + Int32(1)
        @inbounds begin
            lo = d_pos[r]
            hi = d_pos[r + Int32(1)] - Int32(1)
            for k in lo:hi
                col  = d_crd[k] + col_off_t
                acc += d_nzval[k] * x[col]
            end
            if HAS_PATCH && r >= dp_row_start && r <= dp_row_end
                pcol = (r - dp_row_start + Int32(1)) + dp_col_offset + col_off_t
                acc += dp_val * x[pcol]
            end
        end
    else
        r = rem - n_diag + Int32(1)
        col_off_next = (period_m1 + Int32(1)) * n_cols
        @inbounds begin
            lo = n_pos[r]
            hi = n_pos[r + Int32(1)] - Int32(1)
            for k in lo:hi
                col  = n_crd[k] + col_off_t
                acc += n_nzval[k] * x[col]
            end
            lo = p_pos[r]
            hi = p_pos[r + Int32(1)] - Int32(1)
            for k in lo:hi
                col  = p_crd[k] + col_off_next
                acc += p_nzval[k] * x[col]
            end
        end
    end

    @inbounds y[tid] = ZERO_BETA ? acc : acc + beta * y[tid]
    return nothing
end

# ─── BBM-periodic w/ selector off-diag: zero-indirection coupling ────────────
#
# When the off-diagonal (neg, pos) pair is a selector matrix — exactly one nnz
# per row at col = row, with constant nzval — the kernel can drop pos/crd/nzval
# reads entirely.  Each off-diag row in period t becomes simply:
#
#     y[off_row] = neg_val * x[r + col_off_t] + pos_val * x[r + col_off_next]
#
# For a typical multi-period ramp (R = sparse(1:n_gen, 1:n_gen, ones, n_gen,
# n_var); off_diag = (-R, R)) this saves 8 indirect loads per off-diag row —
# 6 μs at the §3 13659 case.  Detector also handles n_off ≤ n_cols (the
# selector lives in the leading n_off columns of an n_cols-wide block).

function _bbm_periodic_selector_kernel!(
        d_pos, d_crd, d_nzval,
        y, x_raw,
        n_diag::Int32, n_off::Int32, n_cols::Int32,
        n_total_rows::Int32,
        neg_val, pos_val,
        ::Val{HAS_PATCH}, dp_row_start::Int32, dp_row_end::Int32,
        dp_col_offset::Int32, dp_val,
        ::Val{ZERO_BETA}, beta) where {HAS_PATCH, ZERO_BETA}
    T   = eltype(y)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    tid > n_total_rows && return nothing

    x = Base.Experimental.Const(x_raw)

    period_size  = n_diag + n_off
    period_m1    = (tid - Int32(1)) ÷ period_size
    rem          = (tid - Int32(1)) - period_m1 * period_size
    col_off_t    = period_m1 * n_cols

    if rem < n_diag
        # Diag rows: identical to the generic periodic kernel.
        r = rem + Int32(1)
        acc = zero(T)
        @inbounds begin
            lo = d_pos[r]
            hi = d_pos[r + Int32(1)] - Int32(1)
            for k in lo:hi
                col  = d_crd[k] + col_off_t
                acc += d_nzval[k] * x[col]
            end
            if HAS_PATCH && r >= dp_row_start && r <= dp_row_end
                pcol = (r - dp_row_start + Int32(1)) + dp_col_offset + col_off_t
                acc += dp_val * x[pcol]
            end
            y[tid] = ZERO_BETA ? acc : acc + beta * y[tid]
        end
    else
        # Off-diag selector path: 0 indirect loads, 2 x reads, 2 muls.
        r = rem - n_diag + Int32(1)        # 1..n_off, lives in cols [1..n_off]
        col_off_next = (period_m1 + Int32(1)) * n_cols
        @inbounds begin
            acc = T(neg_val) * x[r + col_off_t] + T(pos_val) * x[r + col_off_next]
            y[tid] = ZERO_BETA ? acc : acc + beta * y[tid]
        end
    end
    return nothing
end

# Detect "constant-value selector" shape:  the matrix has exactly 1 nnz per
# row at column = row (i.e. the leading n_rows×n_rows block is a scaled
# identity, with all-zero columns to the right).  Returns `(true, val)` if
# detected, else `(false, zero)`.
function _selector_value(u::AbstractUSTensor)
    fmt = format(u)
    fmt == Formats.CSR || return (false, zero(eltype(u)))
    m = Int(extents(u)[1])
    pos_h = Vector{Int}(Array(positions(u, 2)))
    length(pos_h) == m + 1 || return (false, zero(eltype(u)))
    pos_h[end] - pos_h[1] == m || return (false, zero(eltype(u)))
    for r in 1:m
        pos_h[r+1] - pos_h[r] == 1 || return (false, zero(eltype(u)))
    end
    crd_h = Vector{Int}(Array(coordinates(u, 2)))
    nz_h  = Array(nonzeros(u))
    @inbounds for r in 1:m
        crd_h[pos_h[r]] == r || return (false, zero(eltype(u)))
    end
    val = nz_h[1]
    all(v -> v == val, nz_h) || return (false, zero(eltype(u)))
    return (true, val)
end

# Find a single BSM block (i, j) that's a constant-value selector and report it
# as a "diagonal patch": rows in [row_start..row_end] of the BSM contribute one
# inline term `val * x[col_off + (r - row_start)]` instead of going through the
# CSR.  Returns `nothing` if no such block is found.  Supports a single patch
# for now — first selector wins.  Multi-patch is a straightforward extension.
function _bsm_find_diag_patch(A::BlockSparseMatrix{T}) where T
    nb_r, nb_c = size(A.blocks)
    for i in 1:nb_r, j in 1:nb_c
        b = A.blocks[i, j]; b === nothing && continue
        is_sel, val = _selector_value(b)
        is_sel || continue
        # row r of the block (1..A.row_sizes[i]) maps to BSM row r + row_off,
        # and the selector's nnz lands at BSM col r + col_off.  So in kernel:
        #   for BSM row R in [row_off+1, row_off+row_sizes[i]],
        #   x col within the period = (R - (row_off+1)) + col_off + 1.
        return (Int32(A._row_off[i] + 1),
                Int32(A._row_off[i] + A.row_sizes[i]),
                Int32(A._col_off[j]),
                T(val))
    end
    return nothing
end

# Build a leaner CSR from the BSM compiled CSR by removing the nnz that fall
# inside a diagonal patch's row range and at the patch's predicted column.
# Returns (new_pos, new_crd, new_nzval) on host (Vectors).
function _strip_diag_patch_csr(asm::USTensor, row_start::Int32, row_end::Int32,
                                 col_offset::Int32, ::T) where T
    pos_h = Vector{Int32}(Array(positions(asm, 2)))
    crd_h = Vector{Int32}(Array(coordinates(asm, 2)))
    nz_h  = Vector{T}(Array(nonzeros(asm)))
    n_rows = length(pos_h) - 1
    n_remove_total = Int(row_end - row_start + 1)

    new_pos = Vector{Int32}(undef, n_rows + 1)
    new_crd = Vector{Int32}(undef, length(crd_h) - n_remove_total)
    new_nz  = Vector{T}(undef, length(nz_h) - n_remove_total)

    new_pos[1] = 1
    nz_cur = 0
    @inbounds for r in 1:n_rows
        lo = pos_h[r]; hi = pos_h[r + 1] - 1
        in_patch = (Int32(r) >= row_start) & (Int32(r) <= row_end)
        # Column we expect to remove if in patch: r within block + col_offset (1-based).
        target_col = in_patch ? Int32(r - row_start + 1) + col_offset : Int32(0)
        for k in lo:hi
            if in_patch && crd_h[k] == target_col
                continue   # skip the patch entry
            end
            nz_cur += 1
            new_crd[nz_cur] = crd_h[k]
            new_nz[nz_cur]  = nz_h[k]
        end
        new_pos[r + 1] = Int32(nz_cur + 1)
    end
    @assert nz_cur == length(new_crd)
    return new_pos, new_crd, new_nz
end

# Periodic compiled view — only created when the BBM matches the supported
# shape (bw=1, BSM diag, single Tuple off-diag pair).  Holds direct refs to
# the source block CSRs, no T-fold replication.
mutable struct _CompiledBBMPeriodic{T,Ti}
    bsm_compiled :: _CompiledBSM{T,Ti}      # diag's compiled CSR (reused, with selector blocks removed if any)
    # Off-diag CSR storage (only populated when not in selector mode):
    n_pos        :: CuVector{Int32}          # off-diag neg CSR rowptr
    n_crd        :: CuVector{Int32}
    n_nzval      :: CuVector{T}
    p_pos        :: CuVector{Int32}          # off-diag pos CSR rowptr
    p_crd        :: CuVector{Int32}
    p_nzval      :: CuVector{T}
    # Off-diag selector mode: both off-diag matrices are scaled identities.
    selector     :: Bool
    neg_val      :: T
    pos_val      :: T
    # Optional in-diag selector patch.  When the BSM has a block at (i, j)
    # whose CSR is a constant-value selector (1 nnz/row at col=row), we exclude
    # it from the BSM compiled CSR and inline `acc += val * x[col_off + r_off
    # + (r - row_start)]` in the kernel — saves one indirect-load chain per
    # patched row.  `has_diag_patch=false` disables this path (CSR retains all
    # entries).
    has_diag_patch    :: Bool
    diag_row_start    :: Int32     # 1-based; first BSM row in the patch
    diag_row_end      :: Int32     # 1-based; last BSM row in the patch (inclusive)
    diag_col_offset   :: Int32     # 0-based shift within the period
    diag_val          :: T
    n_diag       :: Int32
    n_off        :: Int32
    n_cols       :: Int32
    n_total_rows :: Int32
    graph        :: Dict{Tuple{UInt,UInt}, CUDA.CuGraphExec}
    last_y_ptr   :: UInt
    last_x_ptr   :: UInt
    last_exec    :: Union{Nothing, CUDA.CuGraphExec}
end

const _bbm_periodic_cache = IdDict{BlockBandedMatrix, _CompiledBBMPeriodic}()

@inline function _bbm_supports_periodic(M::BlockBandedMatrix)
    M.bw == 1 || return false
    M.diags isa BlockSparseMatrix || return false
    od = M.off_diags[1]
    od isa Tuple || return false
    od[1] isa AbstractUSTensor && od[2] isa AbstractUSTensor
end

# Pull the compiled BSM's CSR buffers + off-diag CSR buffers into a periodic
# launcher.  No full BBM materialization.  If both off-diag matrices are
# constant-value selectors, skip uploading their CSR buffers and let the
# kernel use the closed-form selector access pattern instead.
function _compile_bbm_periodic(M::BlockBandedMatrix{D,O}) where {D,O}
    bsm = M.diags::BlockSparseMatrix
    bsm_c = _ensure_compiled_bsm(bsm)

    neg, pos = M.off_diags[1]
    Tel = eltype(M)
    n_diag = Int32(M.n_diag_rows)
    n_off  = Int32(M.n_off_rows[1])
    n_cols = Int32(M.n_cols)
    n_total_rows = Int32(M.T * M.n_diag_rows + (M.T - 1) * M.n_off_rows[1])

    # Detect a single in-diag selector patch (e.g., DCOPF's negI block).  If
    # found, REPLACE bsm_c with a leaner compiled BSM whose CSR omits the patch
    # entries.  The kernel re-adds them via the inline path.
    patch = _bsm_find_diag_patch(bsm)
    if patch !== nothing
        rs, re, co, val = patch
        new_pos, new_crd, new_nz = _strip_diag_patch_csr(bsm_c.asm, rs, re, co, val)
        m, n = Int.(extents(bsm_c.asm))
        cu = CUDA.CUSPARSE.CuSparseMatrixCSR{Tel,Int32}(
            CuArray(new_pos), CuArray(new_crd), CuArray(new_nz), (m, n))
        leaner_asm = JLUST.ust(cu)
        # Wrap the leaner CSR in a fresh _CompiledBSM so the kernel reads the
        # stripped pos/crd/nzval; this leaves the BSM's full compiled CSR (used
        # by §2 BSM mul!) untouched.
        bsm_c = _CompiledBSM{Tel,Int32}(leaner_asm, bsm_c.map,
            Dict{Tuple{UInt,UInt}, CUDA.CuGraphExec}(),
            UInt(0), UInt(0), nothing)
        has_patch = true; dp_rs = rs; dp_re = re; dp_co = co; dp_v = val
    else
        has_patch = false; dp_rs = Int32(0); dp_re = Int32(0); dp_co = Int32(0); dp_v = zero(Tel)
    end

    n_is_sel, n_v = _selector_value(neg)
    p_is_sel, p_v = _selector_value(pos)
    if n_is_sel && p_is_sel
        empty_iv = CuVector{Int32}(undef, 0)
        empty_v  = CuVector{Tel}(undef, 0)
        return _CompiledBBMPeriodic{Tel,Int32}(
            bsm_c,
            empty_iv, empty_iv, empty_v,
            empty_iv, empty_iv, empty_v,
            true, Tel(n_v), Tel(p_v),
            has_patch, dp_rs, dp_re, dp_co, dp_v,
            n_diag, n_off, n_cols, n_total_rows,
            Dict{Tuple{UInt,UInt}, CUDA.CuGraphExec}(),
            UInt(0), UInt(0), nothing)
    end

    n_pos, n_crd, n_nz = positions(neg, 2), coordinates(neg, 2), nonzeros(neg)
    p_pos, p_crd, p_nz = positions(pos, 2), coordinates(pos, 2), nonzeros(pos)
    _CompiledBBMPeriodic{Tel,Int32}(
        bsm_c,
        CuVector{Int32}(n_pos), CuVector{Int32}(n_crd), CuVector{Tel}(n_nz),
        CuVector{Int32}(p_pos), CuVector{Int32}(p_crd), CuVector{Tel}(p_nz),
        false, zero(Tel), zero(Tel),
        has_patch, dp_rs, dp_re, dp_co, dp_v,
        n_diag, n_off, n_cols, n_total_rows,
        Dict{Tuple{UInt,UInt}, CUDA.CuGraphExec}(),
        UInt(0), UInt(0), nothing)
end

@inline _ensure_periodic_bbm(M::BlockBandedMatrix) =
    get!(() -> _compile_bbm_periodic(M), _bbm_periodic_cache, M)

@inline function _bbm_periodic_launch!(c::_CompiledBBMPeriodic, x::CuVector, y::CuVector,
                                         beta::Real=zero(eltype(y)))
    bsm_asm = c.bsm_compiled.asm
    d_pos = positions(bsm_asm, 2)
    d_crd = coordinates(bsm_asm, 2)
    d_nz  = nonzeros(bsm_asm)
    threads = 256
    blocks  = cld(Int(c.n_total_rows), threads)
    zero_beta = Val(iszero(beta))
    has_patch = Val(c.has_diag_patch)
    if c.selector
        CUDA.@cuda threads=threads blocks=blocks _bbm_periodic_selector_kernel!(
            d_pos, d_crd, d_nz,
            y, x,
            c.n_diag, c.n_off, c.n_cols, c.n_total_rows,
            c.neg_val, c.pos_val,
            has_patch, c.diag_row_start, c.diag_row_end, c.diag_col_offset, c.diag_val,
            zero_beta, eltype(y)(beta))
    else
        CUDA.@cuda threads=threads blocks=blocks _bbm_periodic_spmv_kernel!(
            d_pos, d_crd, d_nz,
            c.n_pos, c.n_crd, c.n_nzval,
            c.p_pos, c.p_crd, c.p_nzval,
            y, x,
            c.n_diag, c.n_off, c.n_cols, c.n_total_rows,
            has_patch, c.diag_row_start, c.diag_row_end, c.diag_col_offset, c.diag_val,
            zero_beta, eltype(y)(beta))
    end
    return y
end

# ─── BBM mul! dispatch ───────────────────────────────────────────────────────

function LinearAlgebra.mul!(y::CuVector, A::BlockBandedMatrix, x::CuVector;
                              backend::Union{AbstractUSTBackend,Nothing}=nothing)
    if backend === nothing || backend isa CUSPARSEBackend
        # Periodic fast path: bw=1, BSM diag, Tuple off-diag — runs the
        # structurally-aware kernel that reads each block CSR once and applies
        # it across all T periods via column offset (no T-fold materialisation).
        if _bbm_supports_periodic(A)
            cp = _ensure_periodic_bbm(A)
            CUDA.is_capturing() && (_bbm_periodic_launch!(cp, x, y); return y)
            yp = UInt(pointer(y));  xp = UInt(pointer(x))
            last = cp.last_exec
            if last !== nothing && yp === cp.last_y_ptr && xp === cp.last_x_ptr
                CUDA.launch(last)
                return y
            end
            return _bbm_periodic_slowpath!(y, x, cp, yp, xp)
        end

        # Compiled-CSR fast path: requires every contributing block to be a CSR
        # USTensor (or BSM whose blocks are CSR).  For any other shape — dense
        # diags, COO/DCSR USTensors, mixed formats — fall through to the
        # generic per-block dispatch via `JLUST.execute(BBMSpMVOp, ...)`, which
        # routes each per-period SpMV through the emitter walker and handles
        # arbitrary level formats.
        if _bbm_supports_compile(A)
            c    = _ensure_compiled_bbm(A)
            CUDA.is_capturing() && (_bbm_emit_spmv!(c, x, y); return y)
            yp   = UInt(pointer(y));  xp = UInt(pointer(x))
            last = c.last_exec
            if last !== nothing && yp === c.last_y_ptr && xp === c.last_x_ptr
                CUDA.launch(last)
                return y
            end
            return _bbm_mul_slowpath!(y, x, c, yp, xp)
        end
    end
    JLUST.execute(BBMSpMVOp, A, x, y; backend=backend)
end

# Compile-time check: every block contributing to this BBM must be a CSR
# USTensor (or BSM with all-CSR blocks).  Mixed / dense / unusual-format BBMs
# fall through to the generic emitter dispatch.
function _bbm_supports_compile(M::BlockBandedMatrix)
    _bbm_diag_supports(M.diags) || return false
    for k in 1:M.bw
        _bbm_off_supports(M.off_diags[k]) || return false
    end
    true
end

@inline _bbm_diag_supports(d::AbstractUSTensor)   = format(d) == Formats.CSR
@inline _bbm_diag_supports(d::BlockSparseMatrix)  = _bsm_supports_compile(d)
@inline _bbm_diag_supports(d::AbstractVector)     = all(_bbm_diag_supports, d)
@inline _bbm_diag_supports(::Any)                 = false   # incl. AbstractMatrix

@inline _bbm_off_supports(p::Tuple)               = _bbm_diag_supports(p[1]) && _bbm_diag_supports(p[2])
@inline _bbm_off_supports(p::AbstractVector)      = all(_bbm_off_supports, p)
@inline _bbm_off_supports(::Any)                  = false

@noinline function _bbm_periodic_slowpath!(y::CuVector, x::CuVector,
                                              cp::_CompiledBBMPeriodic, yp::UInt, xp::UInt)
    CUDA.is_capturing() && (_bbm_periodic_launch!(cp, x, y); return y)
    key = (yp, xp)
    if haskey(cp.graph, key)
        exec = cp.graph[key]
    else
        _bbm_periodic_launch!(cp, x, y); CUDA.synchronize()
        g = CUDA.capture() do; _bbm_periodic_launch!(cp, x, y); end
        exec = CUDA.instantiate(g)
        cp.graph[key] = exec
    end
    cp.last_y_ptr = yp; cp.last_x_ptr = xp; cp.last_exec = exec
    CUDA.launch(exec)
    return y
end

@noinline function _bbm_mul_slowpath!(y::CuVector, x::CuVector, c::_CompiledBBM,
                                        yp::UInt, xp::UInt)
    CUDA.is_capturing() && (_bbm_emit_spmv!(c, x, y); return y)
    key = (yp, xp)
    if haskey(c.graph, key)
        exec = c.graph[key]
    else
        _bbm_emit_spmv!(c, x, y); CUDA.synchronize()
        g = CUDA.capture() do; _bbm_emit_spmv!(c, x, y); end
        exec = CUDA.instantiate(g)
        c.graph[key] = exec
    end
    c.last_y_ptr = yp; c.last_x_ptr = xp; c.last_exec = exec
    CUDA.launch(exec)
    return y
end
