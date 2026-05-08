# ─── Multi-stage SpMM component profiler ─────────────────────────────────────
#
# Isolates the time budget of each component in BlockBandedMatrix mul!.
# Run on the GPU server after the main tour benchmarks to guide optimization.
#
# Usage:
#   cd ~/bench/JLUST && julia --project=benchmark benchmark/profile_multistage.jl

using Printf, Downloads
using JLUST, JLUST.Formats
using SparseArrays, LinearAlgebra
using BenchmarkTools

const HAS_CUDA  = try; using CUDA; CUDA.functional(); catch; false; end
HAS_CUDA || error("This profiler requires CUDA. Aborting.")

to_device(x) = CUDA.CuArray(x)
const FloatType = Float64
sync() = CUDA.synchronize()

# ─── Data setup ───────────────────────────────────────────────────────────────

const _PGLIB_URL   = "https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/"
const _PGLIB_CACHE = joinpath(dirname(abspath(@__FILE__)), "..", ".cache", "pglib")

function _pglib_file(case)
    mkpath(_PGLIB_CACHE)
    path = joinpath(_PGLIB_CACHE, case * ".m")
    isfile(path) && return path
    Downloads.download(_PGLIB_URL * case * ".m", path)
    path
end

function _mp_table(txt, name)
    m = match(Regex("mpc\\.$name\\s*=\\s*\\[(.*?)\\]\\s*;", "s"), txt)
    m === nothing && return zeros(0, 0)
    rows = [Float64.(filter(!isnothing, tryparse.(Float64, split(strip(s))))
                    ) for s in split(m.captures[1], ';') if !isempty(strip(s))]
    filter!(!isempty, rows)
    isempty(rows) && return zeros(0, 0)
    nc = maximum(length, rows); M = zeros(length(rows), nc)
    for (i, r) in enumerate(rows); M[i, 1:length(r)] .= r; end; M
end

function load_pglib(case)
    txt = replace(read(_pglib_file(case), String), r"%[^\n]*" => "")
    bus_d = _mp_table(txt, "bus"); gen_d = _mp_table(txt, "gen"); br_d = _mp_table(txt, "branch")
    br = br_d[br_d[:, 11] .== 1, :]; gen = gen_d[gen_d[:, 8] .== 1, :]
    n_bus = size(bus_d, 1); n_line = size(br, 1); n_gen = size(gen, 1)
    id2i = Dict(Int(bus_d[i, 1]) => i for i in 1:n_bus)
    from_b = [id2i[Int(b)] for b in br[:, 1]]; to_b = [id2i[Int(b)] for b in br[:, 2]]
    b_k = ifelse.(abs.(br[:, 4]) .< 1e-10, 0.0, 1.0 ./ br[:, 4])
    Bf   = sparse([1:n_line; 1:n_line], [from_b; to_b], [b_k; -b_k], n_line, n_bus)
    gen_b = [id2i[Int(b)] for b in gen[:, 1]]
    Cg    = sparse(gen_b, 1:n_gen, ones(n_gen), n_bus, n_gen)
    d_b   = zeros(n_bus)
    for k in 1:n_line; d_b[from_b[k]] += b_k[k]; d_b[to_b[k]] += b_k[k]; end
    Bbus  = sparse([from_b; to_b; 1:n_bus], [to_b; from_b; 1:n_bus], [-b_k; -b_k; d_b], n_bus, n_bus)
    I_l   = sparse(1.0I, n_line, n_line)
    (; Cg, Bf, Bbus, negI=-I_l, n_bus, n_line, n_gen)
end

print("Loading pglib_opf_case30000_goc ..."); bm = load_pglib("pglib_opf_case30000_goc")
println(" $(bm.n_bus) buses  $(bm.n_line) lines  $(bm.n_gen) gens")

n_bus = bm.n_bus; n_gen = bm.n_gen; n_line = bm.n_line

u_Cg      = csr_tensor(bm.Cg;    device=to_device)
u_negBbus = csr_tensor(-bm.Bbus; device=to_device)
u_negI    = csr_tensor(bm.negI;  device=to_device)
u_Bf      = csr_tensor(bm.Bf;    device=to_device)

BM = BlockSparseMatrix([u_Cg nothing u_negBbus; nothing u_negI u_Bf])

T_ms  = 24
n_con = n_bus + n_line; n_var = n_gen + n_line + n_bus; n_rmp = n_gen

R_cpu  = sparse(1:n_gen, 1:n_gen, ones(n_gen), n_gen, n_var)
u_posR = csr_tensor(R_cpu;  device=to_device)
u_negR = csr_tensor(-R_cpu; device=to_device)

BM_ms  = BlockBandedMatrix(BM, u_negR, u_posR, T_ms, n_con, n_rmp, n_var)
x_ms   = CUDA.randn(Float64, n_var * T_ms)
y_ms   = CUDA.zeros(Float64, T_ms * n_con + (T_ms-1) * n_rmp)
mul!(y_ms, BM_ms, x_ms); sync()
println("Warm-up done.")

# ─── Introspect internals ─────────────────────────────────────────────────────

(; diags, off_diags, _spmm_bufs) = BM_ms
(; row_bufs, diag_out, off_bufs) = _spmm_bufs
nb_r, nb_c = size(diags.blocks)
cum_off  = cumsum([0; BM_ms.n_off_rows])
d_period = BM_ms.n_diag_rows + cum_off[end]
d_starts = JLUST._bbm_diag_starts(BM_ms.n_diag_rows, cum_off, T_ms, BM_ms.bw)
n_cols   = BM_ms.n_cols
X2       = reshape(x_ms, n_cols, T_ms)
neg_R, pos_R = off_diags[1]
off_buf = off_bufs[1]  # pre-allocated (n_off × (T-1))

# Recreate col_bufs locally for old-gather-path comparison only
col_bufs = [CUDA.zeros(Float64, diags.col_sizes[j], T_ms) for j in 1:nb_c]

# ─── Benchmark ────────────────────────────────────────────────────────────────

N = 300
println("\n" * "═"^62)
println("Component timing  (T=$T_ms, $(N) samples, NVIDIA L40S)")
println("═"^62)

t_full = @belapsed(begin mul!($y_ms, $BM_ms, $x_ms); CUDA.synchronize() end,
                   samples=N, evals=1) * 1e6
@printf("  full mul!  (ungraphed)    : %7.1f μs  (%.1f μs/period)\n", t_full, t_full/T_ms)

graph_ms = CUDA.capture() do; mul!(y_ms, BM_ms, x_ms); end
exec_ms  = CUDA.instantiate(graph_ms)
CUDA.launch(exec_ms); sync()
t_graph = @belapsed(begin CUDA.launch($exec_ms); CUDA.synchronize() end, samples=N, evals=1) * 1e6
@printf("  graphed mul!              : %7.1f μs  (%.1f μs/period)\n", t_graph, t_graph/T_ms)

# cuSPARSE serial reference
A_full = vcat(hcat(bm.Cg, spzeros(n_bus, n_line), -bm.Bbus),
              hcat(spzeros(n_line, n_gen), bm.negI, bm.Bf))
A_full_cu = CUDA.CUSPARSE.CuSparseMatrixCSR(A_full)
x_full_cu = CUDA.randn(Float64, size(A_full, 2))
y_full_cu = CUDA.zeros(Float64, size(A_full, 1))
mul!(y_full_cu, A_full_cu, x_full_cu); sync()
t_cusparse = @belapsed(begin
    for _ in 1:$T_ms; mul!($y_full_cu, $A_full_cu, $x_full_cu); end; CUDA.synchronize()
end, samples=30, evals=1) * 1e6
@printf("  cuSPARSE serial ×%d       : %7.1f μs  (%.1f μs/period)\n",
        T_ms, t_cusparse, t_cusparse/T_ms)
@printf("  speedup vs cuSPARSE       :     %.2f×  (graphed)\n", t_cusparse / t_graph)

println("─"^62)
println("  Component breakdown:")

# Gather (OLD path — eliminated; col_bufs created locally for comparison only)
t_gather = @belapsed(begin
    for j in 1:$nb_c
        copyto!($col_bufs[j], view($X2, $diags._col_off[j]+1:$diags._col_off[j+1], :))
    end; CUDA.synchronize()
end, samples=N, evals=1) * 1e6
mb_gather = sum(diags.col_sizes) * T_ms * 8 / 1e6
@printf("    gather   (%d copyto!, ELIMINATED): %7.1f μs  [%.1f MB, %.0f GB/s]\n",
        nb_c, t_gather, mb_gather, mb_gather/t_gather*1e3)

# SpMM (actual path: two-pass — dense blocks first beta=0/1, sparse blocks guarded beta=1)
# No fill! needed: dense first block (Bbus) beta=0 writes 0 for its empty rows,
# giving Cg's guarded beta=1 a correctly-zeroed base.
_run_bbm_spmm! = (X2, row_bufs, diags, nb_r, nb_c) -> begin
    for i in 1:nb_r
        first_dense = true
        for j in 1:nb_c  # pass 1: dense
            b = diags.blocks[i, j]; b === nothing && continue
            JLUST.needs_row_guard(b) && continue
            col_view = view(X2, diags._col_off[j]+1:diags._col_off[j+1], :)
            JLUST.execute(SpMMOp, b, col_view, row_bufs[i];
                             beta=first_dense ? 0.0 : 1.0)
            first_dense = false
        end
        for j in 1:nb_c  # pass 2: sparse (guarded)
            b = diags.blocks[i, j]; b === nothing && continue
            !JLUST.needs_row_guard(b) && continue
            col_view = view(X2, diags._col_off[j]+1:diags._col_off[j+1], :)
            if first_dense
                fill!(row_bufs[i], 0.0)
                JLUST.execute(SpMMOp, b, col_view, row_bufs[i]; beta=0.0, skip_empty_rows=true)
                first_dense = false
            else
                JLUST.execute(SpMMOp, b, col_view, row_bufs[i]; beta=1.0, skip_empty_rows=true)
            end
        end
        first_dense && fill!(row_bufs[i], 0.0)
    end
end
t_spmm = @belapsed(begin
    $_run_bbm_spmm!($X2, $row_bufs, $diags, $nb_r, $nb_c)
    CUDA.synchronize()
end, samples=N, evals=1) * 1e6
@printf("    diag spmm (%d SpMMs, 2-pass guard): %7.1f μs\n", count(!isnothing, diags.blocks), t_spmm)

# Per-block breakdown: dense pass first (beta=0/1), sparse pass (guarded beta=1)
# Dense blocks in pass 1
println("    -- pass 1 (dense, beta=0 then beta=1):")
first_dense = true
for j in 1:nb_c, i in 1:nb_r
    b = diags.blocks[i, j]; b === nothing && continue
    JLUST.needs_row_guard(b) && continue
    col_view = view(X2, diags._col_off[j]+1:diags._col_off[j+1], :)
    β = first_dense ? 0.0 : 1.0
    t_block = @belapsed(begin
        JLUST.execute(SpMMOp, $b, $col_view, $row_bufs[$i]; beta=$β)
        CUDA.synchronize()
    end, samples=N, evals=1) * 1e6
    sz = size(b)
    @printf("      block[%d,%d] CSR (%dx%d, beta=%.0f): %6.1f μs\n", i, j, sz[1], sz[2], β, t_block)
    JLUST.execute(SpMMOp, b, col_view, row_bufs[i]; beta=β)
    global first_dense = false
end; sync()
println("    -- pass 2 (sparse, guarded):")
for j in 1:nb_c, i in 1:nb_r
    b = diags.blocks[i, j]; b === nothing && continue
    !JLUST.needs_row_guard(b) && continue
    col_view = view(X2, diags._col_off[j]+1:diags._col_off[j+1], :)
    t_block = @belapsed(begin
        JLUST.execute(SpMMOp, $b, $col_view, $row_bufs[$i]; beta=1.0, skip_empty_rows=true)
        CUDA.synchronize()
    end, samples=N, evals=1) * 1e6
    sz = size(b)
    @printf("      block[%d,%d] CSR (%dx%d, guarded beta=1): %6.1f μs\n", i, j, sz[1], sz[2], t_block)
    JLUST.execute(SpMMOp, b, col_view, row_bufs[i]; beta=1.0, skip_empty_rows=true)
end; sync()

# Fused diag scatter (new: 1 kernel) vs old (T*nb_r copyto!)
_run_bbm_spmm!(X2, row_bufs, diags, nb_r, nb_c); sync()
mb_scatter = sum(diags.row_sizes) * T_ms * 8 / 1e6
t_scatter_new = @belapsed(begin
    JLUST._bbm_scatter_diag!($y_ms, $diag_out, $d_period, $BM_ms.n_diag_rows, $T_ms)
    CUDA.synchronize()
end, samples=N, evals=1) * 1e6
t_scatter_old = @belapsed(begin
    for t in 1:$T_ms, i in 1:$nb_r
        y_start = $d_starts[t] + $diags._row_off[i]
        copyto!(view($y_ms, y_start:y_start+$diags.row_sizes[i]-1), view($diag_out, $diags._row_off[i]+1:$diags._row_off[i]+$diags.row_sizes[i], t))
    end; CUDA.synchronize()
end, samples=N, evals=1) * 1e6
@printf("    diag scatter (new 1 kern): %7.1f μs  [%.1f MB, %.0f GB/s]\n",
        t_scatter_new, mb_scatter, mb_scatter/t_scatter_new*1e3)
@printf("    diag scatter (old %d×cpy): %7.1f μs  [%.0f GB/s]  (%.1f× slower)\n",
        T_ms*nb_r, t_scatter_old, mb_scatter/t_scatter_old*1e3, t_scatter_old/t_scatter_new)

# Ramp: batched SpMM + fused scatter (new) vs sequential SpMV (old)
x_lo_mat = reshape(view(x_ms, 1:(T_ms-1)*n_cols), n_cols, T_ms-1)
x_hi_mat = reshape(view(x_ms, n_cols+1:T_ms*n_cols), n_cols, T_ms-1)

t_off_spmm = @belapsed(begin
    LinearAlgebra.mul!($off_buf, $neg_R, $x_lo_mat)
    LinearAlgebra.mul!($off_buf, $pos_R, $x_hi_mat, true, true)
    JLUST._bbm_scatter_off!($y_ms, $off_buf, $d_period,
                             $BM_ms.n_diag_rows + $cum_off[1], $BM_ms.n_off_rows[1], $T_ms-1)
    CUDA.synchronize()
end, samples=N, evals=1) * 1e6
@printf("    off-diag (2 SpMMs+scatter): %7.1f μs\n", t_off_spmm)

# Old off-diagonal path for comparison (sequential SpMVs)
t_off_spv = @belapsed(begin
    for t in 1:$T_ms - 1
        off_row = $d_starts[t] + $BM_ms.n_diag_rows + $cum_off[1]
        y_off   = view($y_ms, off_row:off_row+$BM_ms.n_off_rows[1]-1)
        x_lo    = view($x_ms, (t-1)*$n_cols+1:t*$n_cols)
        x_hi    = view($x_ms, t*$n_cols+1:(t+1)*$n_cols)
        LinearAlgebra.mul!(y_off, $neg_R, x_lo)
        LinearAlgebra.mul!(y_off, $pos_R, x_hi, true, true)
    end; CUDA.synchronize()
end, samples=N, evals=1) * 1e6
@printf("    off-diag OLD (%d SpMVs)  : %7.1f μs  (%.2f× slower)\n",
        2*(T_ms-1), t_off_spv, t_off_spv / t_off_spmm)

println("─"^62)
t_sum = t_spmm + t_scatter_new + t_off_spmm
@printf("  sum(active components)    : %7.1f μs  (spmm %.1f + scatter %.1f + off %.1f)\n",
        t_sum, t_spmm, t_scatter_new, t_off_spmm)
@printf("  overhead/overlap          : %7.1f μs\n", t_full - t_sum)

println("\n" * "═"^62)
println("Optimization summary vs original (fill!+SpMV+scatter path):")

t_fill = @belapsed(begin
    fill!($diag_out, 0.0); CUDA.synchronize()
end, samples=N, evals=1) * 1e6
@printf("  fill! cost (eliminated)   : %7.1f μs  [%.1f MB]\n",
        t_fill, sum(diags.row_sizes) * T_ms * 8 / 1e6)
@printf("  gather: %d×cpy (eliminated): saves %.1f μs\n", nb_c, t_gather)
@printf("  scatter: %d×cpy → 1 kern  :   saves %.1f μs\n",
        T_ms*nb_r, t_scatter_old - t_scatter_new)
@printf("  off-diag: %d SpMV → 2 SpMM:   saves %.1f μs\n", 2*(T_ms-1), t_off_spv - t_off_spmm)
@printf("  total estimated savings   : %.1f μs\n",
        t_fill + t_gather + (t_scatter_old - t_scatter_new) + (t_off_spv - t_off_spmm))
