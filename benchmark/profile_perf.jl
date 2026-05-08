# Standalone perf profiler — isolates per-launch overhead vs kernel work.
# Run on the L40S: julia --project=benchmark benchmark/profile_perf.jl

using Printf, Random, Downloads, LinearAlgebra
using JLUST, JLUST.Formats, KernelAbstractions, BenchmarkTools, SparseArrays, CUDA

# Reuse the pglib loader from the sweep.
include(joinpath(@__DIR__, "sweep_cases.jl"))

# Override the CASES list to only the largest case for focused profiling.
const CASE = "pglib_opf_case30000_goc"

function profile_case(case::String)
    d = load_pglib(case)
    (; Cg, Bf, Bbus, negI, n_bus, n_line, n_gen) = d

    n_var = n_gen + n_line + n_bus
    n_con = n_bus + n_line

    # CPU sparse + GPU CSR for each block
    A_cpu = vcat(
        hcat(Cg, spzeros(n_bus, n_line), -Bbus),
        hcat(spzeros(n_line, n_gen), negI, Bf))

    u_Cg   = csr_tensor(Cg;    device=CuArray)
    u_Bbus = csr_tensor(-Bbus; device=CuArray)
    u_negI = csr_tensor(negI;  device=CuArray)
    u_Bf   = csr_tensor(Bf;    device=CuArray)
    u_Afull = ust(CUDA.CUSPARSE.CuSparseMatrixCSR(Float64.(A_cpu)))

    BM = BlockSparseMatrix([
        u_Cg    nothing  u_Bbus;
        nothing u_negI   u_Bf
    ])

    x = CUDA.randn(Float64, n_var)
    y = CUDA.zeros(Float64, n_con)

    # Warmup
    for _ in 1:5; mul!(y, BM, x); CUDA.synchronize(); end

    println("="^72)
    println("Profile: $case   ($(n_bus) buses, $(n_var) vars, $(n_con) cons)")
    println("="^72)

    # --- 1) Full BM mul! ---
    t = @belapsed(begin mul!($y, $BM, $x); CUDA.synchronize() end, samples=200, evals=1) * 1e6
    @printf("BM mul! (4 blocks)            : %8.2f μs\n", t)

    # --- 2) cuSPARSE single SpMV on the assembled matrix ---
    h = prepare(CUSPARSEBackend(), SpMVOp, u_Afull)
    u_x = ust(x); u_y = ust(y)
    execute(h, u_x, u_y); CUDA.synchronize()
    t = @belapsed(begin execute($h, $u_x, $u_y); CUDA.synchronize() end, samples=200, evals=1) * 1e6
    @printf("cuSPARSE SpMV (assembled)     : %8.2f μs\n", t)

    # --- 3) Each individual block (emitter) — sums tell us launch overhead ---
    function bench_one_block(b)
        x_sl = ust(view(x, 1:size(b, 2)))
        y_sl = ust(view(y, 1:size(b, 1)))
        execute(SpMVOp, b, x_sl, y_sl); CUDA.synchronize()
        @belapsed(begin execute(SpMVOp, $b, $x_sl, $y_sl); CUDA.synchronize() end, samples=200, evals=1) * 1e6
    end

    t_cg   = bench_one_block(u_Cg)
    t_bbus = bench_one_block(u_Bbus)
    t_negI = bench_one_block(u_negI)
    t_bf   = bench_one_block(u_Bf)
    @printf("emitter SpMV per block        : %.2f + %.2f + %.2f + %.2f = %.2f μs\n",
            t_cg, t_bbus, t_negI, t_bf, t_cg+t_bbus+t_negI+t_bf)

    # --- 4) Empty kernel launch overhead ---
    @kernel function _noop_kern!(); end
    ka = KernelAbstractions.get_backend(x)
    _noop_kern!(ka, 32)(; ndrange=32); CUDA.synchronize()
    t = @belapsed(begin _noop_kern!($ka, 32)(; ndrange=32); CUDA.synchronize() end, samples=500, evals=1) * 1e6
    @printf("KA empty kernel launch        : %8.2f μs\n", t)

    # --- 5) Bare CUDA.@cuda overhead (no KA wrapper) ---
    function _bare_kern() end
    CUDA.@cuda threads=32 blocks=1 _bare_kern(); CUDA.synchronize()
    t = @belapsed(begin CUDA.@cuda threads=32 blocks=1 _bare_kern(); CUDA.synchronize() end, samples=500, evals=1) * 1e6
    @printf("bare CUDA.@cuda launch        : %8.2f μs\n", t)

    # --- 6) Single emitter SpMV on the assembled matrix ---
    u_Afull_emit = csr_tensor(SparseMatrixCSC(Float64.(A_cpu)); device=CuArray)
    mul!(y, u_Afull_emit, x); CUDA.synchronize()
    t = @belapsed(begin mul!($y, $u_Afull_emit, $x); CUDA.synchronize() end, samples=200, evals=1) * 1e6
    @printf("emitter SpMV (assembled)      : %8.2f μs\n", t)

    println()
end

profile_case(CASE)
