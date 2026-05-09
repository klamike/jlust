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

mutable struct _CompiledBSM{T,Ti,P<:Tuple}
    asm     :: USTensor                              # assembled CSR USTensor (CSR blocks only)
    map     :: Matrix{Union{Nothing, CuVector{Ti}}}  # (i,j) → indices into asm.nzval (CSR blocks)
    patches :: P                                     # NTuple of SelectorPatch{T} — ShiftedDiag blocks
    graph   :: Dict{Tuple{UInt,UInt}, CUDA.CuGraphExec}  # (ptr_y, ptr_x) → captured graph
    # Hot-path fast cache: most repeated mul! calls reuse the same (y, x) buffers,
    # so a single field comparison skips the Dict lookup and tuple alloc.
    last_y_ptr :: UInt
    last_x_ptr :: UInt
    last_exec  :: Union{Nothing, CUDA.CuGraphExec}
end

const _bsm_compiled_cache = IdDict{BlockSparseMatrix, _CompiledBSM}()

# Per-block compile classification.  Returns one of:
#   :null          — empty (skip)
#   :csr           — CSR-formatted USTensor; assemble into the CSR
#   :shifted_diag  — (Dense, ShiftedDiag) USTensor; extract as a SelectorPatch
#   :unsupported   — fall through to per-block dispatch
@inline function _block_compile_kind(b)
    b === nothing && return :null
    if b isa AbstractUSTensor
        format(b) == Formats.CSR && return :csr
        levels = format(b).levels
        length(levels) == 2 && levels[1] isa DenseLevel && levels[2] isa ShiftedDiagLevel &&
            return :shifted_diag
    end
    :unsupported
end

# Whether a BSM is supported by the compiled-CSR fast path: every non-null
# block must be either CSR or (Dense, ShiftedDiag).  Other formats fall
# through to the per-block emitter path.
function _bsm_supports_compile(A::BlockSparseMatrix)
    nb_r, nb_c = size(A.blocks)
    for i in 1:nb_r, j in 1:nb_c
        b = A.blocks[i, j]; b === nothing && continue
        k = _block_compile_kind(b)
        (k === :csr || k === :shifted_diag) || return false
    end
    true
end

# Detect "constant-value selector" shape on a CSR block: exactly 1 nnz per
# row at column = row (i.e. the leading n_rows×n_rows is a scaled identity).
# Returns `(true, val)` if detected, else `(false, zero)`.  Used as back-
# compat: a CSR-encoded selector block is treated as if it were
# (Dense, ShiftedDiag) — same patch path, same kernel literal.
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

# Build a SelectorPatch from a (Dense, ShiftedDiag)-formatted block at BSM
# position (i, j).  The block's row range is `[row_off+1, row_off+row_sizes[i]]`
# in the assembled coords; the col formula `(r - row_start + 1) + col_offset`
# in the kernel must hit BSM column `col_off + (r_local + shift)` where
# `r_local = r - row_start + 1` is 1-based within-block.  So col_offset =
# col_off + shift.
@inline function _shifted_diag_patch(A::BlockSparseMatrix{T}, i::Int, j::Int, b::AbstractUSTensor) where T
    lv2     = format(b).levels[2]
    rs      = Int32(A._row_off[i] + 1)
    re      = Int32(A._row_off[i] + A.row_sizes[i])
    co      = Int32(A._col_off[j] + diag_shift(lv2))
    val     = T(diag_val(lv2))
    SelectorPatch{T}(rs, re, co, val)
end

# Build the compiled CSR on the host, then move buffers to GPU.
# CSR blocks (i,j) row r contribute pos[r+1]-pos[r] entries that land
# contiguously in assembled.nzval at the slots reserved for block-row i row r,
# column-block j.  ShiftedDiag blocks are extracted as patches and skipped.
function _compile_bsm(A::BlockSparseMatrix{T}) where T
    nb_r, nb_c = size(A.blocks)
    n_rows = sum(A.row_sizes)
    n_cols = sum(A.col_sizes)

    # Classify blocks; collect patches for the ShiftedDiag ones AND for any
    # CSR-encoded constant-value selector blocks (back-compat with old code
    # that built `negI` via `sparse(I, n, n) * -1` rather than the explicit
    # ShiftedDiag format).  Both shapes go through the same patch path so
    # multi-selector BSMs work uniformly.
    patches = SelectorPatch{T}[]
    csr_blocks = Tuple{Int,Int}[]
    for i in 1:nb_r, j in 1:nb_c
        b = A.blocks[i, j]; b === nothing && continue
        kind = _block_compile_kind(b)
        if kind === :shifted_diag
            push!(patches, _shifted_diag_patch(A, i, j, b))
        elseif kind === :csr
            is_sel, val = _selector_value(b)
            if is_sel
                push!(patches, SelectorPatch{T}(
                    Int32(A._row_off[i] + 1),
                    Int32(A._row_off[i] + A.row_sizes[i]),
                    Int32(A._col_off[j]),   # detected pattern has within-block shift = 0
                    T(val)))
            else
                push!(csr_blocks, (i, j))
            end
        else
            error("_compile_bsm: block ($i,$j) has unsupported format $(format(b))")
        end
    end

    # Pull pos / crd / nzval to host once per CSR block.
    block_pos = Dict{Tuple{Int,Int}, Vector{Int}}()
    block_crd = Dict{Tuple{Int,Int}, Vector{Int}}()
    block_nz  = Dict{Tuple{Int,Int}, Vector{T}}()
    for (i, j) in csr_blocks
        b = A.blocks[i, j]
        block_pos[(i,j)] = Vector{Int}(Array(positions(b, 2)))
        block_crd[(i,j)] = Vector{Int}(Array(coordinates(b, 2)))
        block_nz[(i,j)]  = Vector{T}(Array(nonzeros(b)))
    end

    # Per-row nnz count of the assembled (CSR contributions only).
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
    for (i, j) in csr_blocks
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

    patches_tuple = (patches...,)
    _CompiledBSM{T,Int32,typeof(patches_tuple)}(
        asm_ust, map_gpu, patches_tuple,
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

@inline function _bsm_emit_spmv!(c::_CompiledBSM, x::CuVector, y::CuVector)
    if isempty(c.patches)
        # No selector patches → standard SpMV through the assembled CSR.
        # Walker's warp-vector kernel handles this with no extra overhead.
        JLUST.execute(JLUST.SpMVOp, c.asm, JLUST.ust(x), JLUST.ust(y);
                      backend=JLUST.EmitterBackend())
    else
        # Fused CSR + patch SpMV: the kernel walks the leaner CSR and adds
        # each patch's per-row contribution as a literal `val * x[col]` term.
        # Patches tuple is type-stable so the inner loop unrolls; saves CSR
        # replication and 3-buffer indirect loads per patched row.
        ka = CUDABackend()
        JLUST._bsm_with_patches_spmv_launch!(ka,
            positions(c.asm, 2), coordinates(c.asm, 2), nonzeros(c.asm),
            y, x,
            Int32(extents(c.asm)[1]), c.patches,
            true, zero(eltype(y)))
    end
end

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

# ─── BBM-periodic SpMV: backend-agnostic kernels live in KAExt ──────────────
#
# The two structurally-aware kernels (generic periodic + selector off-diag
# fast path) used to be CUDA-only @cuda functions in this file.  They moved
# to KernelAbstractionsExt as `@kernel` bodies so every KA-targetable backend
# (CUDA, ROCm, CPU, POCL, oneAPI) runs the same code.  CUDA's read-only data
# cache (LDG) is opted into via the `_supports_ldg(ka)` trait — backends
# without it fall through to a regular global load, no kernel duplication.
#
# Launch entry points: `JLUST._bbm_periodic_spmv_launch!` /
# `JLUST._bbm_periodic_selector_launch!` (declared in src/backends.jl,
# implemented in ext/KernelAbstractionsExt/ops/block_periodic.jl).

# Periodic compiled view — only created when the BBM matches the supported
# shape (bw=1, BSM diag, single Tuple off-diag pair).  Holds direct refs to
# the source block CSRs (no T-fold replication) plus the BSM's already-extracted
# selector patches (carried straight through from `bsm_compiled.patches`).
mutable struct _CompiledBBMPeriodic{T,Ti,P<:Tuple}
    bsm_compiled :: _CompiledBSM{T,Ti,P}     # diag's compiled CSR + extracted patches
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
# launcher.  No full BBM materialization.  Selector blocks inside the BSM diag
# are already extracted as `bsm_c.patches` by `_compile_bsm`; the periodic
# kernel walks them alongside the per-period CSR walk.  If both off-diagonal
# matrices are constant-value selectors, the kernel uses the closed-form
# selector access pattern for off-diag rows and skips uploading their CSR.
function _compile_bbm_periodic(M::BlockBandedMatrix{D,O}) where {D,O}
    bsm = M.diags::BlockSparseMatrix
    bsm_c = _ensure_compiled_bsm(bsm)
    Pty   = typeof(bsm_c.patches)

    neg, pos = M.off_diags[1]
    Tel = eltype(M)
    n_diag = Int32(M.n_diag_rows)
    n_off  = Int32(M.n_off_rows[1])
    n_cols = Int32(M.n_cols)
    n_total_rows = Int32(M.T * M.n_diag_rows + (M.T - 1) * M.n_off_rows[1])

    n_is_sel, n_v = _selector_value(neg)
    p_is_sel, p_v = _selector_value(pos)
    if n_is_sel && p_is_sel
        empty_iv = CuVector{Int32}(undef, 0)
        empty_v  = CuVector{Tel}(undef, 0)
        return _CompiledBBMPeriodic{Tel,Int32,Pty}(
            bsm_c,
            empty_iv, empty_iv, empty_v,
            empty_iv, empty_iv, empty_v,
            true, Tel(n_v), Tel(p_v),
            n_diag, n_off, n_cols, n_total_rows,
            Dict{Tuple{UInt,UInt}, CUDA.CuGraphExec}(),
            UInt(0), UInt(0), nothing)
    end

    n_pos, n_crd, n_nz = positions(neg, 2), coordinates(neg, 2), nonzeros(neg)
    p_pos, p_crd, p_nz = positions(pos, 2), coordinates(pos, 2), nonzeros(pos)
    _CompiledBBMPeriodic{Tel,Int32,Pty}(
        bsm_c,
        CuVector{Int32}(n_pos), CuVector{Int32}(n_crd), CuVector{Tel}(n_nz),
        CuVector{Int32}(p_pos), CuVector{Int32}(p_crd), CuVector{Tel}(p_nz),
        false, zero(Tel), zero(Tel),
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
    ka    = CUDABackend()
    patches = c.bsm_compiled.patches   # NTuple{N, SelectorPatch{T}} from BSM compile
    if c.selector
        JLUST._bbm_periodic_selector_launch!(ka,
            d_pos, d_crd, d_nz,
            y, x,
            c.n_diag, c.n_off, c.n_cols, c.n_total_rows,
            eltype(y)(c.neg_val), eltype(y)(c.pos_val),
            patches,
            iszero(beta), eltype(y)(beta))
    else
        JLUST._bbm_periodic_spmv_launch!(ka,
            d_pos, d_crd, d_nz,
            c.n_pos, c.n_crd, c.n_nzval,
            c.p_pos, c.p_crd, c.p_nzval,
            y, x,
            c.n_diag, c.n_off, c.n_cols, c.n_total_rows,
            patches,
            iszero(beta), eltype(y)(beta))
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
# USTensor (or BSM with all-CSR blocks).  Mixed / dense / unusual-format /
# ShiftedDiag-block BBMs fall through to the generic emitter dispatch (or, for
# the periodic special case, to the patch-aware periodic path which handles
# ShiftedDiag blocks via the BSM compile's `patches` tuple).
function _bbm_supports_compile(M::BlockBandedMatrix)
    _bbm_diag_supports(M.diags) || return false
    for k in 1:M.bw
        _bbm_off_supports(M.off_diags[k]) || return false
    end
    true
end

# CSR-only predicate.  `_compile_bbm` materialises the BBM into a single
# assembled CSR via `_host_csr`, which only knows how to read CSR-formatted
# blocks — so a BSM containing a ShiftedDiag block is NOT compilable for the
# full BBM path (the periodic path handles it instead via patches).
@inline _bsm_all_csr(A::BlockSparseMatrix) = begin
    nb_r, nb_c = size(A.blocks)
    for i in 1:nb_r, j in 1:nb_c
        b = A.blocks[i, j]; b === nothing && continue
        format(b) == Formats.CSR || return false
    end
    true
end

@inline _bbm_diag_supports(d::AbstractUSTensor)   = format(d) == Formats.CSR
@inline _bbm_diag_supports(d::BlockSparseMatrix)  = _bsm_all_csr(d)
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
