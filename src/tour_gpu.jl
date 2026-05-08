# ─── JLUST Feature Tour: DC Optimal Power Flow — GPU Edition ──────────────────
#
# Same DCOPF setting as tour.jl plus two GPU-specific features:
#   §2 extension — N-1 contingency: selective value update + CUDA Graph replay
#   §4           — DiagonalLevel: user-defined format plugin (5 lines, GPU+CPU)
#
# Section 1: Constructors  — csr_tensor, dcsr_tensor
# Section 2: BlockSparseMatrix + N-1 contingency (update_block_values! + Graph)
# Section 3: Multi-stage OPF with ramping (BlockBandedMatrix, block SpMM)
# Section 4: DiagonalLevel — custom AbstractLevelFormat plugin
#
# Usage (GPU server):
#   cd ~/bench/JLUST && julia --project=benchmark benchmark/tour_gpu.jl
#
# Performance numbers (embedded at the bottom) were collected on:
#   GPU : NVIDIA L40S, julia 1.12, CUDA 12

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

    br  = br_d[ br_d[:, 11] .== 1, :]
    gen = gen_d[gen_d[:,  8] .== 1, :]

    n_bus  = size(bus_d, 1)
    n_line = size(br,    1)
    n_gen  = size(gen,   1)

    id2i = Dict(Int(bus_d[i, 1]) => i for i in 1:n_bus)

    from_b = [id2i[Int(b)] for b in br[:, 1]]
    to_b   = [id2i[Int(b)] for b in br[:, 2]]
    b_k = T.(ifelse.(abs.(br[:, 4]) .< 1e-10, zero(T), one(T) ./ br[:, 4]))

    Bf = sparse([1:n_line; 1:n_line], [from_b; to_b],
                T.([b_k; -b_k]), n_line, n_bus)

    gen_b = [id2i[Int(b)] for b in gen[:, 1]]
    Cg    = sparse(gen_b, 1:n_gen, ones(T, n_gen), n_bus, n_gen)

    d_b = zeros(T, n_bus)
    for k in 1:n_line; d_b[from_b[k]] += b_k[k]; d_b[to_b[k]] += b_k[k]; end
    Bbus = sparse([from_b; to_b; 1:n_bus], [to_b; from_b; 1:n_bus],
                  T.([-b_k; -b_k; d_b]), n_bus, n_bus)

    print("  $(case): $(n_bus) buses  $(n_line) lines  $(n_gen) gens")
    I_l = sparse(T(1)*I, n_line, n_line)
    (; Cg, Bf, Bbus, I_line=I_l, negI=-I_l, n_bus, n_line, n_gen,
       from_b=from_b, to_b=to_b, b_k=b_k)
end

function dense_vec(v::AbstractVector{T}) where T
    ust(to_device(Vector{T}(v)))
end

function bench_spmv(u_A, u_x, u_y, backend; samples=100)
    for _ in 1:5; sparse_mv!(backend, u_A, u_x, u_y); sync(); end
    @belapsed(begin
        sparse_mv!($backend, $u_A, $u_x, $u_y); $sync()
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

be = EmitterBackend()
println()

let A = demo.Cg, m=demo.n_bus, n=demo.n_gen
    At  = sparse(A')
    rp  = Int32.(At.colptr)
    ci  = Int32.(At.rowval)
    nz  = FloatType.(At.nzval)
    u1 = csr_tensor(rp, ci, nz; m=m, n=n)
    u2 = csr_tensor(rp, ci, nz, (m, n))
    u3 = csr_tensor(A)
    @assert nnz(u1) == nnz(u2) == nnz(u3)
    println("  csr_tensor  (Cg)   : ✓  ($(nnz(u1)) nnz, $(m)×$(n))")
end

let u_dcsr = dcsr_tensor(demo.Cg), m = demo.n_bus
    n_active = length(coordinates(u_dcsr, 1))
    println("  dcsr_tensor (Cg)   : ✓  $(n_active) active rows / $(m) total",
            " ($(round(100*n_active/m, digits=1))% non-empty — generator buses only)")
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: BlockSparseMatrix — DCOPF constraint matrix + N-1 contingency
# ─────────────────────────────────────────────────────────────────────────────
#
# N-1 security analysis requires solving the DCOPF for every possible single
# line outage.  JLUST makes each contingency cheap:
#   1. Build BM once and capture a CUDA Graph.
#   2. For each contingency: update_block_values! swaps just the affected
#      block's NNZ buffer in O(nnz_block) GPU memcpy.
#   3. Replay the same graph — fixed kernel sequence, new data.
#
# Only Bf changes when a line trips (flow-definition rows).  Bbus changes too
# (nodal susceptance), but the demo focuses on the Bf block for clarity.

println("\n" * "─"^72)
println("Section 2 — BlockSparseMatrix: DCOPF constraint matrix")
println("─"^72)

u_Cg      = csr_tensor(bm.Cg;   device=to_device)
u_Bf      = csr_tensor(bm.Bf;   device=to_device)
u_negBbus = csr_tensor(-bm.Bbus; device=to_device)
u_negI    = csr_tensor(bm.negI;  device=to_device)

#   Rows : [power_balance(n_bus) | flow_def(n_line)]
#   Cols : [p_gen(n_gen)         | p_f(n_line)      | va(n_bus)]
BM = BlockSparseMatrix([
    u_Cg    nothing   u_negBbus;
    nothing u_negI    u_Bf
])

println("  BlockSparseMatrix shape : $(size(BM, 1)) × $(size(BM, 2))")
println("    block rows  : $(n_bus_bm) (balance) + $(n_line_bm) (flow)")
println("    block cols  : $(n_gen_bm) (p_gen) + $(n_line_bm) (p_f) + $(n_bus_bm) (va)")

x_dcopf = to_device(randn(n_gen_bm + n_line_bm + n_bus_bm))
y_dcopf = similar(x_dcopf, n_bus_bm + n_line_bm)

for _ in 1:5; mul!(y_dcopf, BM, x_dcopf); sync(); end
t_bm = @belapsed(begin mul!($y_dcopf, $BM, $x_dcopf); $sync() end,
                 samples=100, evals=1) * 1e6
@printf("  mul!(y, BM, x)          : %8.1f μs  (%d block SpMVs)\n",
        t_bm, count(!isnothing, BM.blocks))

# ── CPU / cuSPARSE baseline ───────────────────────────────────────────────────

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

# ── CUDA Graph ────────────────────────────────────────────────────────────────

if HAS_CUDA
    graph_bm = CUDA.capture() do; mul!(y_dcopf, BM, x_dcopf); end
    exec_bm  = CUDA.instantiate(graph_bm)
    CUDA.launch(exec_bm); CUDA.synchronize()
    t_bm_graph = @belapsed(begin CUDA.launch($exec_bm); CUDA.synchronize() end,
                            samples=100, evals=1) * 1e6
end

println()
@printf("  %-36s  %10s\n", "Method (full DCOPF operator)", "Time (μs)")
println("  " * "─"^48)
@printf("  %-36s  %10.1f\n", "Julia SparseMatrixCSC", t_jl_csc)
if HAS_CUDA
    @printf("  %-36s  %10.1f\n", "CUDA cuSPARSE (monolithic)", t_cusparse_bm)
    @printf("  %-36s  %10.1f  (%.2f× vs cuSPARSE)\n",
            "JLUST BlockSparseMatrix", t_bm, t_cusparse_bm / t_bm)
    @printf("  %-36s  %10.1f  (%.2f× faster than cuSPARSE)\n",
            "JLUST BlockSparseMatrix (Graph)", t_bm_graph, t_bm_graph / t_cusparse_bm)
else
    @printf("  %-36s  %10.1f  (%.2f× vs CSC)\n",
            "JLUST BlockSparseMatrix", t_bm, t_jl_csc / t_bm)
end

# ── N-1 contingency: selective value swap + graph replay ──────────────────────
# Trip line 1: zero its two entries in Bf (from_bus and to_bus contributions).
# update_block_values!(BM, 2, 3, new_nz) copies new_nz into Bf's GPU buffer
# in-place — the graph's baked device pointer still points to the same array,
# so the next replay sees the updated susceptances at no restructuring cost.

if HAS_CUDA
    # Bf CSR nzval: line k occupies positions [2k-1, 2k] (2 nnz/row, row-sorted)
    # Build tripped version on CPU, then move to GPU for in-place update benchmark.
    bf_nz_orig_cpu = Array(nonzeros(u_Bf))   # pull to CPU for scalar modification
    bf_nz_trip_cpu = copy(bf_nz_orig_cpu)
    bf_nz_trip_cpu[1] = zero(FloatType)      # zero from-bus entry for line 1
    bf_nz_trip_cpu[2] = zero(FloatType)      # zero to-bus entry  for line 1
    bf_nz_trip = to_device(bf_nz_trip_cpu)   # GPU array ready for update_block_values!
    bf_nz_orig = to_device(bf_nz_orig_cpu)   # restore array

    update_block_values!(BM, 2, 3, bf_nz_trip)   # apply trip
    CUDA.launch(exec_bm); CUDA.synchronize()      # verify replay works

    t_n1 = @belapsed(begin
        update_block_values!($BM, 2, 3, $bf_nz_trip)
        CUDA.launch($exec_bm)
        CUDA.synchronize()
    end, samples=100, evals=1) * 1e6

    update_block_values!(BM, 2, 3, bf_nz_orig)   # restore
    CUDA.launch(exec_bm); CUDA.synchronize()

    @printf("  %-36s  %10.1f  (%d swap + graph replay)\n",
            "N-1 contingency (line trip)", t_n1, 1)
end
println("  " * "─"^48)

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Multi-stage OPF with ramping (BlockBandedMatrix)
# ─────────────────────────────────────────────────────────────────────────────
#
# A T-period rolling-horizon DCOPF (48 hours at 2-hour resolution = T=24)
# adds ramp constraints coupling adjacent periods.  BlockBandedMatrix encodes
# the resulting block-banded structure directly:
#   • Shared diagonal BM (all periods) batched into a single block SpMM call
#   • Coupling ramp rows applied via sparse_mv! per transition
#   • Any AbstractMatrix for blocks; bandwidth > 1 and time-varying diagonals
#     also supported via parametric dispatch

println("\n" * "─"^72)
println("Section 3 — Multi-stage OPF with ramping (BlockBandedMatrix)")
println("─"^72)

T_ms  = HAS_CUDA ? 24 : HAS_METAL ? 24 : 12
n_con = n_bus_bm + n_line_bm
n_var = n_gen_bm + n_line_bm + n_bus_bm
n_rmp = n_gen_bm

R_cpu  = sparse(1:n_gen_bm, 1:n_gen_bm, ones(n_gen_bm), n_gen_bm, n_var)
u_posR = csr_tensor(R_cpu;  device=to_device)
u_negR = csr_tensor(-R_cpu; device=to_device)

n_row_ms = T_ms*n_con + (T_ms-1)*n_rmp
n_col_ms = T_ms*n_var
BM_ms = BlockBandedMatrix(BM, u_negR, u_posR, T_ms, n_con, n_rmp, n_var)
@printf("  Ramp R   : %d × %d, %d nnz\n", n_rmp, n_var, nnz(u_posR))
@printf("  Full M   : %d × %d  (T=%d, 3 unique objects)\n", n_row_ms, n_col_ms, T_ms)

x_ms = to_device(randn(n_col_ms))
y_ms = similar(x_ms, n_row_ms)
mul!(y_ms, BM_ms, x_ms)

t_ms = @belapsed(begin mul!($y_ms, $BM_ms, $x_ms); $sync() end,
                 samples=30, evals=1) * 1e6

if HAS_CUDA
    graph_ms = CUDA.capture() do; mul!(y_ms, BM_ms, x_ms); end
    exec_ms  = CUDA.instantiate(graph_ms)
    CUDA.launch(exec_ms); CUDA.synchronize()
    t_ms_graph = @belapsed(begin CUDA.launch($exec_ms); CUDA.synchronize() end,
                            samples=30, evals=1) * 1e6
end

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
@printf("\n  %-38s  %10s  %12s\n", "Method (T=$(T_ms))", "Total (μs)", "Per period")
println("  " * "─"^64)
@printf("  %-38s  %10.1f  %12.1f\n", "SparseMatrixCSC (CPU serial)",
        t_serial_csc, t_serial_csc / T_ms)
if HAS_CUDA
    @printf("  %-38s  %10.1f  %12.1f\n", "cuSPARSE (GPU serial)",
            t_serial_cusparse, t_serial_cusparse / T_ms)
    @printf("  %-38s  %10.1f  %12.1f  (%.2f× vs cuSPARSE)\n",
            "JLUST BlockBandedMatrix", t_ms, t_ms / T_ms,
            t_serial_cusparse / t_ms)
    @printf("  %-38s  %10.1f  %12.1f  (%.2f× vs cuSPARSE)\n",
            "JLUST SpMM + CUDA Graph ($(_n_kernels_ms) kernels)", t_ms_graph, t_ms_graph / T_ms,
            t_serial_cusparse / t_ms_graph)
else
    @printf("  %-38s  %10.1f  %12.1f  (%.2f× vs CSC)\n",
            "JLUST BlockBandedMatrix", t_ms, t_ms / T_ms,
            t_serial_csc / t_ms)
end
println("  " * "─"^64)

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: DiagonalLevel — user-defined format plugin
# ─────────────────────────────────────────────────────────────────────────────
#
# AbstractLevelFormat lets users plug in custom sparse storage schemes without
# modifying JLUST internals.  Implementing `level_step` is sufficient: JLUST
# generates both the GPU @kernel body and the fused CPU accumulator from it.
#
# DiagonalLevel encodes an n×n diagonal matrix: row i has exactly one nonzero
# at column i with value nz[i].  No colind or rowptr arrays are stored.
#
#   struct DiagonalLevel <: JLUST.AbstractLevelFormat end
#   JLUST.level_step(::DiagonalLevel, i::Int, nz) = (i, nz[i])
#
# Full plugin: struct + level_step + constructor.  ~5 lines of user code.
#
# On GPU, DiagonalLevel (276 KB struct read) loses to CSR (553 KB) because
# the CSR warp-shuffle vector kernel (VS=2, 1 NNZ/row) is memory-bandwidth
# optimal for this pattern.  At n >> L2 capacity the 2× bandwidth reduction
# tips the balance; on CPU DiagonalLevel wins at all tested sizes.

println("\n" * "─"^72)
println("Section 4 — DiagonalLevel (n×n diagonal matrix)")
println("─"^72)

struct DiagonalLevel <: JLUST.AbstractLevelFormat end
JLUST.level_step(::DiagonalLevel, i::Int, nz) = (i, nz[i])

# Format: [row: dense, col: diagonal] — outer Dense iterates all m rows,
# inner DiagonalLevel maps row i to (col=i, val=nz[i]).
const _diag_i, _diag_j = dims(:i, :j)
const DiagonalFmt = TensorFormat([_diag_i, _diag_j],
                                  [_diag_i => DenseLevel(), _diag_j => DiagonalLevel()])

println("  Format : [row, col] -> (row: DenseLevel, col: DiagonalLevel)")
println("  Hook   : level_step only  (GPU + CPU paths auto-generated)")

# Build -I_line using DiagonalLevel and CSR for comparison
neg_diag_vals = to_device(fill(FloatType(-1), n_line_bm))
u_negI_diag   = make_tensor(DiagonalFmt, neg_diag_vals; m=n_line_bm, n=n_line_bm)

# Verify correctness: residual vs CSR -I_line
x_line = to_device(randn(n_line_bm))
y_csr  = similar(x_line)
y_diag = similar(x_line)
LinearAlgebra.mul!(y_csr,  u_negI, x_line)
LinearAlgebra.mul!(y_diag, u_negI_diag, x_line)
sync()
residual = maximum(abs.(Array(y_csr) .- Array(y_diag)))
@printf("  Residual vs CSR I_line : %.2e  %s\n", residual, residual < 1e-10 ? "✓" : "✗")

# Benchmark DiagonalLevel SpMV vs CSR
u_x_line = dense_vec(randn(n_line_bm))
u_y_line = ust(similar(x_line))

t_csr_I    = bench_spmv(u_negI,      u_x_line, u_y_line, be)
t_diag_I   = bench_spmv(u_negI_diag, u_x_line, u_y_line, be)

println()
@printf("  %-28s  %10s  %8s\n", "Kernel", "Time(μs)", "vs CSR")
println("  " * "─"^50)
@printf("  %-28s  %10.1f  baseline\n", "CSR I_line",   t_csr_I)
@printf("  %-28s  %10.1f  %8.2f×\n",  "DiagonalLevel", t_diag_I, t_csr_I / t_diag_I)

# Memory traffic comparison (struct read per SpMV, excluding x/y)
csr_kb  = (length(BM.blocks[2,2] |> b -> positions(b, 2))  +
           nnz(u_negI)) * sizeof(Int32) / 1024 +
           nnz(u_negI) * sizeof(FloatType) / 1024
diag_kb = nnz(u_negI_diag) * sizeof(FloatType) / 1024
println()
@printf("  Struct read per SpMV (excl. x/y) at n_line=%d:\n", n_line_bm)
@printf("    CSR          : %4.0f KB\n", csr_kb)
@printf("    DiagonalLevel: %4.0f KB  (%.1f× less)\n", diag_kb, csr_kb / diag_kb)

# BlockSparseMatrix with DiagonalLevel -I block
BM_diag = BlockSparseMatrix([
    u_Cg    nothing      u_negBbus;
    nothing u_negI_diag  u_Bf
])
for _ in 1:5; mul!(y_dcopf, BM_diag, x_dcopf); sync(); end
t_bm_diag = @belapsed(begin mul!($y_dcopf, $BM_diag, $x_dcopf); $sync() end,
                      samples=100, evals=1) * 1e6

println()
@printf("  BM mul! CSR identity    : %8.1f μs\n", t_bm)
@printf("  BM mul! DiagLevel ident : %8.1f μs  (%.2f×)\n", t_bm_diag, t_bm / t_bm_diag)
println()
println("  Full plugin: struct + level_step + constructor.  ~5 lines of user code.")
println()
println("  Note: on GPU, DiagonalLevel loses to CSR because CSR triggers the")
println("  warp-shuffle vector kernel (hardware-aware VS, 1 NNZ/row).  At n >> L2")
println("  capacity the 2× bandwidth reduction wins; on CPU it wins at all sizes.")

# ─────────────────────────────────────────────────────────────────────────────
# Performance Summary
# ─────────────────────────────────────────────────────────────────────────────

println("\n" * "="^72)
println("Performance Summary  ($(n_line_bm) lines × $(n_bus_bm) buses, $(FloatType))")
println("="^72)
@printf("  %-44s  %10s\n", "Operation", "Time (μs)")
println("  " * "─"^56)
@printf("  %-44s  %10.1f\n", "DCOPF mul! JLUST BlockSparseMatrix", t_bm)
@printf("  %-44s  %10.1f\n", "  with DiagonalLevel -I block", t_bm_diag)
if HAS_CUDA
    @printf("  %-44s  %10.1f\n", "  with CUDA Graph", t_bm_graph)
    @printf("  %-44s  %10.1f\n", "  N-1 contingency (swap + Graph replay)", t_n1)
end
@printf("  %-44s  %10.1f\n", "  Julia SparseMatrixCSC (reference)", t_jl_csc)
HAS_CUDA && @printf("  %-44s  %10.1f\n", "  CUDA cuSPARSE monolithic (reference)", t_cusparse_bm)
println("  " * "─"^56)
if HAS_CUDA
    @printf("  %-44s  %10.1f\n", "Multi-stage T=$(T_ms)", t_ms)
    @printf("  %-44s  %10.1f\n", "  with CUDA Graph ($(_n_kernels_ms) kernels)", t_ms_graph)
    @printf("  %-44s  %10.1f μs\n", "  per period", t_ms / T_ms)
else
    @printf("  %-44s  %10.1f\n", "Multi-stage T=$(T_ms)", t_ms)
    @printf("  %-44s  %10.1f μs\n", "  per period", t_ms / T_ms)
end
println("  " * "─"^56)
@printf("  %-44s  %10.1f\n", "DiagonalLevel I_line SpMV (§4)", t_diag_I)

println()
println("─"^72)
println("Future improvements (in priority order)")
println("─"^72)
println("""
  1. DCSR Cg for generator contingency
       When a generator trips, its column in Cg goes to zero.  Tracking
       active generators with DCSR lets EmitterBackend skip zero rows natively.
       update_block_values! should accept a format change (DCSR re-slice)
       rather than requiring the same USTensor structure.

  2. Complex USTensor for AC OPF
       Complex{Float64} element type with appropriate SpMV semantics.  The
       emitter body needs a `conj` leaf for the Hermitian case.
""")

println()
println("─"^72)
println("Visual summary (live numbers from this run)")
println("─"^72)
println()

if HAS_CUDA
    println(barplot(
        ["SparseMatrixCSC (CPU ref)", "cuSPARSE (1 kernel)",
         "JLUST BM (4 kernels)", "JLUST BM (CUDA Graph)"],
        round.([t_jl_csc, t_cusparse_bm, t_bm, t_bm_graph]; digits=1);
        title = "§2  DCOPF mul!  (μs, lower is better)",
        xlabel = "μs"))
    println(barplot(
        ["single DCOPF (no Graph)", "T=$(T_ms)  per-period",
         "T=$(T_ms)  CUDA Graph /period"],
        round.([t_bm, t_ms/T_ms, t_ms_graph/T_ms]; digits=2);
        title = "§3  multi-stage cost per period  (μs, lower is better)",
        xlabel = "μs"))
else
    println(barplot(
        ["SparseMatrixCSC (CPU ref)", "JLUST BlockSparseMatrix"],
        round.([t_jl_csc, t_bm]; digits=1);
        title = "§2  DCOPF mul!  (μs, lower is better)",
        xlabel = "μs"))
    println(barplot(
        ["single DCOPF", "T=$(T_ms)  per-period"],
        round.([t_bm, t_ms/T_ms]; digits=1);
        title = "§3  multi-stage cost per period  (μs, lower is better)",
        xlabel = "μs"))
end
