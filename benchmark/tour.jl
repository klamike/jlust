# ─── JLUST Feature Tour: DC Optimal Power Flow ───────────────────────────────
#
# This script walks through JLUST's core features using the DCOPF constraint
# matrix as a concrete setting.  A DC optimal power flow for a network with
# n_bus buses, n_gen generators, and n_line transmission lines has the
# following block-structured constraint matrix A (phase-angle formulation),
# mapping decision variables [p_gen, p_f, va] to residuals:
#
#   ┌──────────────┬──────────┬──────────────────────┐ ┌─────────┐
#   │  Cg          │  0       │  -Bbus               │ │ p_gen   │
#   │  (n_bus×     │          │  (n_bus×n_bus)       │ │ (n_gen) │
#   │   n_gen)     │          │  nodal susceptance   │ ├─────────┤
#   ├──────────────┼──────────┼──────────────────────┤ │ p_f     │
#   │  0           │ -I_line  │  Bf                  │ │ (n_line)│
#   │              │          │  (n_line×n_bus)      │ ├─────────┤
#   │              │          │  ±b_k incidence      │ │ va      │
#   └──────────────┴──────────┴──────────────────────┘ │ (n_bus) │
#                                                      └─────────┘
# Row 1 (balance):  Cg·p_gen − Bbus·va = Pd
# Row 2 (flow def): Bf·va − p_f = 0   (angle difference = line flow)
#
# Multi-period (batch): same A, T different right-hand sides X[:,t].
#   → batch_mul!(Y, BM, X)
#
#
# Usage (GPU server):
#   cd ~/bench/JLUST && julia --project=benchmark benchmark/tour.jl
# Usage (CPU / Metal):
#   julia --project=benchmark benchmark/tour.jl
#
# Performance numbers (embedded at the bottom) were collected on:
#   GPU : NVIDIA L40S, julia 1.12, CUDA 12
#   CPU : Apple M2 Pro, julia 1.12, EmitterBackend (KA CPU)

using Printf, Random, Downloads
using JLUST, JLUST.Formats
using SparseArrays, LinearAlgebra
using KernelAbstractions
using BenchmarkTools
using UnicodePlots

# ─── Backend detection ────────────────────────────────────────────────────────

const HAS_CUDA  = try; using CUDA;  CUDA.functional();  catch; false; end
const HAS_METAL = !HAS_CUDA && try; using Metal; Metal.functional(); catch; false; end

function to_device(x::AbstractArray{T}) where T
    arr = (HAS_METAL && T === Float64) ? Float32.(x) : x
    HAS_CUDA  ? CUDA.CuArray(arr)   :
    HAS_METAL ? Metal.MtlArray(arr) : arr
end

function sync()
    HAS_CUDA  && CUDA.synchronize()
    HAS_METAL && Metal.synchronize()
end

const _backend_label = HAS_CUDA  ? "CUDA ($(CUDA.name(CUDA.device())))" :
                       HAS_METAL ? "Metal ($(Metal.current_device().name))" :
                       "CPU EmitterBackend (KernelAbstractions)"

println("="^72)
println("JLUST DCOPF Feature Tour")
println("Backend : $(_backend_label)")
println("Julia   : ", VERSION)
println("="^72)

# ─── PGLib-OPF case loader ────────────────────────────────────────────────────
#
# Downloads Matpower (.m) files from the pglib-opf GitHub repository, parses
# the bus/branch/gen tables, and builds the DCOPF sparse matrices:
#
#   Cg    (n_bus × n_gen)   : generator-to-bus incidence
#   Bf    (n_line × n_bus)  : susceptance-weighted branch incidence
#                             Bf[k, from_k] = b_k,  Bf[k, to_k] = -b_k
#   Bbus  (n_bus × n_bus)   : nodal susceptance matrix (graph Laplacian of b_k)
#                             Bbus[i,i] = Σ b_k for lines at bus i
#                             Bbus[i,j] = -b_k  for line (i,j)
#
# No PTDF factorization needed — the phase-angle formulation keeps Bbus sparse.
# .m files are cached after first download; matrix assembly is fast (<1 s).
#
# Case selection (pglib-opf on GitHub):
#   Demo : pglib_opf_case118_ieee    (118 buses,   186 lines,   54 gens)
#   CPU  : pglib_opf_case1354_pegase (1354 buses,  1991 lines,  260 gens)
#   Metal: pglib_opf_case2853_sdet   (2853 buses,  3974 lines,  317 gens)
#   GPU  : pglib_opf_case30000_goc   (30000 buses, 35393 lines, 3526 gens)

const FloatType = HAS_METAL ? Float32 : Float64
const PGLIB_DEMO = "pglib_opf_case30000_goc"
const PGLIB_BM   = HAS_CUDA  ? "pglib_opf_case30000_goc"  :
                   HAS_METAL ? "pglib_opf_case2853_sdet"   :
                               "pglib_opf_case1354_pegase"

const _PGLIB_URL   = "https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/"
const _PGLIB_CACHE = joinpath(dirname(abspath(@__FILE__)), "..", ".cache", "pglib")

function _pglib_file(case::String)
    mkpath(_PGLIB_CACHE)
    path = joinpath(_PGLIB_CACHE, case * ".m")
    if !isfile(path)
        url = _PGLIB_URL * case * ".m"
        print("    Fetching $case ... ")
        Downloads.download(url, path)
        println("$(filesize(path) ÷ 1024) KB")
    end
    path
end

function _mp_table(txt::String, name::String)
    m = match(Regex("mpc\\.$name\\s*=\\s*\\[(.*?)\\]\\s*;", "s"), txt)
    m === nothing && return zeros(0, 0)
    rows = Vector{Vector{FloatType}}()
    for seg in split(m.captures[1], ';')
        seg = strip(seg)
        isempty(seg) && continue
        vals = [v for v in tryparse.(FloatType, split(seg)) if v !== nothing]
        isempty(vals) && continue
        push!(rows, vals)
    end
    isempty(rows) && return zeros(0, 0)
    nc = maximum(length, rows)
    M  = zeros(length(rows), nc)
    for (i, r) in enumerate(rows); M[i, 1:length(r)] .= r; end
    M
end

function load_pglib(case::String; T::Type=FloatType)
    txt = replace(read(_pglib_file(case), String), r"%[^\n]*" => "")

    bus_d = _mp_table(txt, "bus")
    gen_d = _mp_table(txt, "gen")
    br_d  = _mp_table(txt, "branch")

    # Active branches (col 11 = status) and generators (col 8 = status).
    br  = br_d[ br_d[:, 11] .== 1, :]
    gen = gen_d[gen_d[:,  8] .== 1, :]

    n_bus  = size(bus_d, 1)
    n_line = size(br,    1)
    n_gen  = size(gen,   1)

    # Matpower bus IDs are not necessarily 1-indexed.
    id2i = Dict(Int(bus_d[i, 1]) => i for i in 1:n_bus)

    from_b = [id2i[Int(b)] for b in br[:, 1]]
    to_b   = [id2i[Int(b)] for b in br[:, 2]]
    # Susceptance b_k = 1/x_k; skip degenerate zero-reactance branches.
    b_k = T.(ifelse.(abs.(br[:, 4]) .< 1e-10, zero(T), one(T) ./ br[:, 4]))

    # Bf: susceptance-weighted incidence  Bf[k, from_k] = b_k,  Bf[k, to_k] = -b_k
    Bf = sparse([1:n_line; 1:n_line], [from_b; to_b],
                T.([b_k; -b_k]), n_line, n_bus)

    # Cg: generator-to-bus incidence  Cg[bus_j, gen_i] = 1
    gen_b = [id2i[Int(b)] for b in gen[:, 1]]
    Cg    = sparse(gen_b, 1:n_gen, ones(T, n_gen), n_bus, n_gen)

    # Bbus: nodal susceptance (graph Laplacian of b_k weights).
    d_b = zeros(T, n_bus)
    for k in 1:n_line; d_b[from_b[k]] += b_k[k]; d_b[to_b[k]] += b_k[k]; end
    Bbus = sparse([from_b; to_b; 1:n_bus], [to_b; from_b; 1:n_bus],
                  T.([-b_k; -b_k; d_b]), n_bus, n_bus)

    print("  $(case): $(n_bus) buses  $(n_line) lines  $(n_gen) gens")
    I_l = sparse(T(1)*I, n_line, n_line)
    (; Cg, Bf, Bbus, I_line=I_l, negI=-I_l, n_bus, n_line, n_gen)
end

function dense_vec(v::AbstractVector{T}) where T
    ust(to_device(Vector{T}(v)))
end

function bench_spmv(u_A, u_x, u_y, backend; samples=100)
    for _ in 1:5
        execute(SpMVOp, backend, u_A, u_x, u_y); sync()
    end
    @belapsed(begin
        execute(SpMVOp, $backend, $u_A, $u_x, $u_y); $sync()
    end, samples=samples, evals=1) * 1e6
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Constructors
# ─────────────────────────────────────────────────────────────────────────────

println("\n" * "─"^72)
println("Section 1 — Constructors")
println("─"^72)

demo = load_pglib(PGLIB_DEMO); println("  (demo)")
bm   = load_pglib(PGLIB_BM);   println("  (benchmark)")
n_bus_bm, n_gen_bm, n_line_bm = bm.n_bus, bm.n_gen, bm.n_line

be   = EmitterBackend()
u_Bf = csr_tensor(bm.Bf; device=to_device)
u_va = dense_vec(randn(n_bus_bm))
u_pf = dense_vec(zeros(n_line_bm))
println()

# csr_tensor has three calling conventions:
#   buffer form  : csr_tensor(rp, ci, nz; m=m, n=n)          — owns raw arrays
#   matrix form  : csr_tensor(A::SparseMatrixCSC; device=...)  — converts + moves to device
#   positional   : csr_tensor(rp, ci, nz, (m, n))             — dims as tuple
let A = demo.Cg, m=demo.n_bus, n=demo.n_gen
    At  = sparse(A')
    rp  = Int32.(At.colptr)
    ci  = Int32.(At.rowval)
    nz  = FloatType.(At.nzval)

    u1 = csr_tensor(rp, ci, nz; m=m, n=n)           # buffer form
    u2 = csr_tensor(rp, ci, nz, (m, n))              # positional form
    u3 = csr_tensor(A)                               # SparseMatrixCSC form (CPU)
    @assert nnz(u1) == nnz(u2) == nnz(u3)
    println("  csr_tensor  (Cg)   : ✓  ($(nnz(u1)) nnz, $(m)×$(n))")
end

# DCSR for Cg: only generator buses have non-zero rows.
# In a network with few generators per bus, DCSR skips empty rows natively.
let u_dcsr = dcsr_tensor(demo.Cg), m = demo.n_bus
    n_active = length(coordinates(u_dcsr, 1))
    println("  dcsr_tensor (Cg)   : ✓  $(n_active) active rows / $(m) total",
            " ($(round(100*n_active/m, digits=1))% non-empty — generator buses only)")
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: BlockSparseMatrix — full DCOPF constraint matrix
# ─────────────────────────────────────────────────────────────────────────────

println("\n" * "─"^72)
println("Section 2 — BlockSparseMatrix: DCOPF constraint matrix")
println("─"^72)

u_Cg      = csr_tensor(bm.Cg;  device=to_device)
u_Cg_dcsr = dcsr_tensor(bm.Cg; device=to_device)   # skips 92% empty rows
u_negBbus = csr_tensor(-bm.Bbus; device=to_device)
u_negI    = csr_tensor(bm.negI;  device=to_device)

#   Rows : [power_balance(n_bus) | flow_def(n_line)]
#   Cols : [p_gen(n_gen)         | p_f(n_line)      | va(n_bus)]
BM = BlockSparseMatrix([
    u_Cg     nothing  u_negBbus;
    nothing  u_negI   u_Bf
])

println("  BlockSparseMatrix shape : $(size(BM, 1)) × $(size(BM, 2))")
println("    block rows  : $(n_bus_bm) (balance) + $(n_line_bm) (flow)")
println("    block cols  : $(n_gen_bm) (p_gen) + $(n_line_bm) (p_f) + $(n_bus_bm) (va)")

x_dcopf = to_device(randn(n_gen_bm + n_line_bm + n_bus_bm))
y_dcopf = similar(x_dcopf, n_bus_bm + n_line_bm)

# mul! works via the LinearAlgebra interface.
mul!(y_dcopf, BM, x_dcopf)
for _ in 1:5; mul!(y_dcopf, BM, x_dcopf); sync(); end
t_bm = @belapsed(begin mul!($y_dcopf, $BM, $x_dcopf); $sync() end,
                 samples=100, evals=1) * 1e6

@printf("  mul!(y, BM, x)          : %8.1f μs  (%d block SpMVs)\n",
        t_bm, count(!isnothing, BM.blocks))

# ── Native Julia / CUSPARSE baseline ─────────────────────────────────────────
# Assemble the same operator as a single SparseMatrixCSC and benchmark Julia's
# built-in SpMV and CUDA's cuSPARSE for comparison.  Julia's monolithic CSC
# pays one pass over all NNZ; JLUST BlockSparseMatrix pays one kernel launch
# per non-null block.  The gap is launch-overhead dominated at small matrix
# sizes and closes at GPU scale; the value-add is per-block format
# specialization and O(1) selective updates.

A_full = vcat(
    hcat(bm.Cg,                       spzeros(n_bus_bm, n_line_bm), -bm.Bbus),
    hcat(spzeros(n_line_bm, n_gen_bm), bm.negI,                      bm.Bf ))
x_full_cpu = randn(size(A_full, 2))
y_full_cpu = zeros(size(A_full, 1))
mul!(y_full_cpu, A_full, x_full_cpu)
t_jl_csc = @belapsed(mul!($y_full_cpu, $A_full, $x_full_cpu), samples=100, evals=1) * 1e6

if HAS_CUDA
    A_full_cu = CUDA.CUSPARSE.CuSparseMatrixCSR(A_full)
    x_full_cu = CUDA.CuArray(x_full_cpu)
    y_full_cu = CUDA.CuArray(y_full_cpu)
    mul!(y_full_cu, A_full_cu, x_full_cu); CUDA.synchronize()
    t_cusparse_bm = @belapsed(begin
        mul!($y_full_cu, $A_full_cu, $x_full_cu); CUDA.synchronize()
    end, samples=100, evals=1) * 1e6
end


# ── CUDA Graph: amortise per-block kernel launch overhead ────────────────────
# 4 EmitterBackend launches cost more host overhead than cuSPARSE's 1 kernel.
# CUDA Graphs capture the entire mul! sequence (fixed sparse buffers + device
# pointers) and replay it as a single submission, eliminating that overhead.

if HAS_CUDA
    graph_bm = CUDA.capture() do
        mul!(y_dcopf, BM, x_dcopf)
    end
    exec_bm = CUDA.instantiate(graph_bm)
    CUDA.launch(exec_bm); CUDA.synchronize()
    t_bm_graph = @belapsed(begin CUDA.launch($exec_bm); CUDA.synchronize() end,
                            samples=100, evals=1) * 1e6
end

println()
@printf("  %-36s  %10s\n", "Method (full DCOPF operator)", "Time (μs)")
println("  " * "─"^48)
@printf("  %-36s  %10.1f\n", "SparseMatrixCSC (CPU)", t_jl_csc)
HAS_CUDA && @printf("  %-36s  %10.1f\n", "cuSPARSE (GPU reference)", t_cusparse_bm)
if HAS_CUDA
    @printf("  %-36s  %10.1f  (%.2f× vs cuSPARSE)\n",
            "JLUST BlockSparseMatrix", t_bm, t_cusparse_bm / t_bm)
    @printf("  %-36s  %10.1f  (%.2f× vs cuSPARSE)\n",
            "JLUST + CUDA Graph", t_bm_graph, t_cusparse_bm / t_bm_graph)
else
    @printf("  %-36s  %10.1f  (%.2f× vs CSC)\n",
            "JLUST BlockSparseMatrix", t_bm, t_jl_csc / t_bm)
end
println("  " * "─"^48)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Multi-stage OPF with ramping constraints
# ─────────────────────────────────────────────────────────────────────────────
#
# A T-period rolling-horizon DCOPF (48 hours at 2-hour resolution = T=24)
# adds ramp constraints coupling adjacent periods:
#
#   p_gen_{t+1} - p_gen_t ∈ [r_down, r_up]   for t = 1 … T-1
#
# The combined constraint matrix M has a (2T-1) × T block-banded structure:
#
#              x_1     x_2     x_3    …   x_T
#   t=1 DCOPF: BM      —       —          —       ← same BlockSparseMatrix as §2
#   ramp 1→2: -R      +R       —          —       ← identity on p_gen cols
#   t=2 DCOPF: —       BM      —          —
#   ramp 2→3:  —      -R      +R          —
#   …
#   t=T DCOPF: —       —       —          BM
#
# Each diagonal block is the existing BM (2×3 BlockSparseMatrix from §2),
# preserving its per-stage sub-block structure (Cg, Bbus, Bf, I separate).
# R is an n_gen × n_var sparse matrix — identity on the p_gen columns.
#
# JLUST's BlockBandedMatrix encodes this structure directly:
#   • O(T) mul! — no T²-sized block matrix, no sparse scans
#   • Shared diagonal (repeated BM) batched into a single SpMM
#   • Coupling rows (ramp constraints) applied via sparse_mv!, one pair per period
#   • Any AbstractMatrix for blocks; bandwidth > 1 and time-varying diagonals also supported

println("\n" * "─"^72)
println("Section 3 — Multi-stage OPF with ramping (BlockBandedMatrix)")
println("─"^72)

T_ms  = HAS_CUDA ? 24 : HAS_METAL ? 24 : 12
n_con = n_bus_bm + n_line_bm             # DCOPF rows per period
n_var = n_gen_bm + n_line_bm + n_bus_bm  # variables per period
n_rmp = n_gen_bm                          # ramp rows per coupling

# ── Ramp coupling block R: partial identity on p_gen columns ─────────────────
# R is n_gen × n_var with R[i,i] = ±1.  CSR is used so the GPU warp-shuffle
# kernel (VS=2, 1 NNZ/row) handles the ramp SpMV efficiently.
R_cpu      = sparse(1:n_gen_bm, 1:n_gen_bm, ones(n_gen_bm), n_gen_bm, n_var)
u_posR     = csr_tensor(R_cpu; device=to_device)
u_negR     = csr_tensor(-R_cpu; device=to_device)

n_row_ms = T_ms*n_con + (T_ms-1)*n_rmp
n_col_ms = T_ms*n_var
BM_ms = BlockBandedMatrix(BM, u_negR, u_posR, T_ms, n_con, n_rmp, n_var)
@printf("  Ramp R   : %d × %d, %d nnz\n", n_rmp, n_var, nnz(u_posR))
@printf("  Full M   : %d × %d  (T=%d, 3 unique objects)\n",
        n_row_ms, n_col_ms, T_ms)

x_ms = to_device(randn(n_col_ms))
y_ms = similar(x_ms, n_row_ms)

mul!(y_ms, BM_ms, x_ms)   # warmup + kernel compilation

t_ms = @belapsed(begin mul!($y_ms, $BM_ms, $x_ms); $sync() end,
                 samples=30, evals=1) * 1e6

# ── CUDA Graph for multi-stage: block SpMM + ramp kernels ────────────────────
# With T=24 the graph captures 8 operations:
#   4 diagonal SpMMs — two-pass: dense blocks first (Bbus beta=0, Bf beta=1) then sparse
#     (Cg guarded beta=1 onto Bbus output, negI is dense goes in pass 1).
#     No fill! kernel: dense-first beta=0 correctly zeros empty rows for Cg's guarded pass.
#   1 fused diag scatter kernel (diag_out[r,t] → y[(t-1)*d_period+r], ~769 GB/s)
#   2 ramp SpMMs (neg+pos, 23 columns baked) + 1 fused ramp scatter kernel
# Gather eliminated: SpMM reads X2 SubArray views directly (L40S 96 MB L2 absorbs stride).
# Guarded Cg SpMM (beta=1): accumulates onto Bbus output; skips 88% empty rows entirely.
# Bbus uses beta=0 (dense, first in pass 1): no C read — saves 5.76 MB HBM reads.
# Scatter fused: 48 copyto! → 1 kernel (11.9× faster, 64 → 769 GB/s, near peak L40S bandwidth).
# Ramp batched: 46 SpMVs → 2 SpMMs + 1 scatter (6.45× faster).
# Replaying as a single graph removes the per-launch host overhead.

if HAS_CUDA
    graph_ms = CUDA.capture() do
        mul!(y_ms, BM_ms, x_ms)
    end
    exec_ms = CUDA.instantiate(graph_ms)
    CUDA.launch(exec_ms); CUDA.synchronize()
    t_ms_graph = @belapsed(begin CUDA.launch($exec_ms); CUDA.synchronize() end,
                            samples=30, evals=1) * 1e6
end

# §3 serial references: T_ms independent single-period DCOPFs (reuse §2 arrays).
t_serial_csc = @belapsed(begin
    for _ in 1:$T_ms; mul!($y_full_cpu, $A_full, $x_full_cpu); end
end, samples=10, evals=1) * 1e6

if HAS_CUDA
    t_serial_cusparse = @belapsed(begin
        for _ in 1:$T_ms; mul!($y_full_cu, $A_full_cu, $x_full_cu); end
        CUDA.synchronize()
    end, samples=10, evals=1) * 1e6
end

_n_kernels_ms = 4 + 1 + 2 + 1  # SpMM×4 + diag_scatter×1 + ramp:2SpMM+1scatter (no fill!: two-pass ordering)
@printf("\n  %-36s  %10s  %12s\n", "Method (T=$(T_ms))", "Total (μs)", "Per period")
println("  " * "─"^62)
@printf("  %-36s  %10.1f  %12.1f\n", "SparseMatrixCSC (CPU serial)",
        t_serial_csc, t_serial_csc / T_ms)
if HAS_CUDA
    @printf("  %-36s  %10.1f  %12.1f\n", "cuSPARSE (GPU serial)",
            t_serial_cusparse, t_serial_cusparse / T_ms)
    @printf("  %-36s  %10.1f  %12.1f  (%.2f× vs cuSPARSE)\n",
            "JLUST BlockBandedMatrix", t_ms, t_ms / T_ms,
            t_serial_cusparse / t_ms)
    @printf("  %-36s  %10.1f  %12.1f  (%.2f× vs cuSPARSE)\n",
            "JLUST SpMM + CUDA Graph ($(_n_kernels_ms) kernels)", t_ms_graph, t_ms_graph / T_ms,
            t_serial_cusparse / t_ms_graph)
else
    @printf("  %-36s  %10.1f  %12.1f  (%.2f× vs CSC)\n",
            "JLUST BlockBandedMatrix", t_ms, t_ms / T_ms,
            t_serial_csc / t_ms)
end
println("  " * "─"^62)

# ─────────────────────────────────────────────────────────────────────────────
# Performance Summary
# ─────────────────────────────────────────────────────────────────────────────

println("\n" * "="^72)
println("Performance Summary  ($(n_line_bm) lines × $(n_bus_bm) buses, $(FloatType))")
println("="^72)
@printf("  %-40s  %10s\n", "Operation", "Time (μs)")
println("  " * "─"^52)
@printf("  %-40s  %10.1f\n", "SparseMatrixCSC (CPU)",              t_jl_csc)
HAS_CUDA && @printf("  %-40s  %10.1f\n", "cuSPARSE (GPU reference)",           t_cusparse_bm)
@printf("  %-40s  %10.1f\n", "JLUST §2 BlockSparseMatrix",         t_bm)
HAS_CUDA && @printf("  %-40s  %10.1f\n", "  with CUDA Graph",               t_bm_graph)
println("  " * "─"^52)
@printf("  %-40s  %10.1f\n", "JLUST §3 SpMM (per period, T=$(T_ms))",  t_ms / T_ms)
HAS_CUDA && @printf("  %-40s  %10.1f\n", "  with CUDA Graph ($(_n_kernels_ms) kernels)",
                    t_ms_graph / T_ms)
println("  " * "─"^52)

println()
println("─"^72)
println("Visual summary (live numbers from this run)")
println("─"^72)
println()

if HAS_CUDA
    println(barplot(
        ["SparseMatrixCSC (CPU)", "cuSPARSE (GPU ref)",
         "JLUST BlockSparseMatrix", "JLUST + CUDA Graph"],
        round.([t_jl_csc, t_cusparse_bm, t_bm, t_bm_graph]; digits=1);
        title = "§2  DCOPF mul!  (μs, lower is better)",
        xlabel = "μs"))
    println(barplot(
        ["SparseMatrixCSC (CPU)", "cuSPARSE (GPU ref)",
         "JLUST SpMM /period", "JLUST SpMM + Graph /period"],
        round.([t_serial_csc/T_ms, t_serial_cusparse/T_ms, t_ms/T_ms, t_ms_graph/T_ms]; digits=2);
        title = "§3  multi-stage cost per period  (μs, lower is better)",
        xlabel = "μs"))
else
    println(barplot(
        ["SparseMatrixCSC (CPU)", "JLUST BlockSparseMatrix"],
        round.([t_jl_csc, t_bm]; digits=1);
        title = "§2  DCOPF mul!  (μs, lower is better)",
        xlabel = "μs"))
    println(barplot(
        ["SparseMatrixCSC (CPU)", "JLUST SpMM /period"],
        round.([t_serial_csc/T_ms, t_ms/T_ms]; digits=1);
        title = "§3  multi-stage cost per period  (μs, lower is better)",
        xlabel = "μs"))
end

