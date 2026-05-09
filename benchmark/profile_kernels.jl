# Compare cuSPARSE handle vs JLUST emitter on the SAME assembled CSR.
# Goal: figure out which kernel has the lowest per-call cost on the L40S.
using Printf, LinearAlgebra
using JLUST, JLUST.Formats, KernelAbstractions, BenchmarkTools, SparseArrays, CUDA

let
    src = read(joinpath(@__DIR__, "sweep_cases.jl"), String)
    cut = findfirst("# ── main ─", src)
    include_string(@__MODULE__, src[1:first(cut)-1], "sweep_cases_helpers.jl")
end

ext = Base.get_extension(JLUST, :CUDAExt)

function bench_case_kernels(case::String)
    d = load_pglib(case)
    (; Cg, Bf, Bbus, negI, n_bus, n_line, n_gen) = d
    n_var = n_gen + n_line + n_bus
    n_con = n_bus + n_line

    A_cpu = vcat(
        hcat(Cg, spzeros(n_bus, n_line), -Bbus),
        hcat(spzeros(n_line, n_gen), negI, Bf))

    u_A = ust(CUDA.CUSPARSE.CuSparseMatrixCSR(A_cpu))
    x = CUDA.randn(Float64, n_var)
    y = CUDA.zeros(Float64, n_con)

    # Warmup
    h = prepare(CUSPARSEBackend(), SpMVOp, u_A)
    JLUST.execute(h, x, y); CUDA.synchronize()
    JLUST.execute(JLUST.SpMVOp, u_A, JLUST.ust(x), JLUST.ust(y); backend=JLUST.EmitterBackend()); CUDA.synchronize()

    # Capture each into its own graph
    g_cu  = CUDA.capture() do; JLUST.execute(h, x, y); end
    e_cu  = CUDA.instantiate(g_cu)
    CUDA.launch(e_cu); CUDA.synchronize()

    g_em  = CUDA.capture() do
        JLUST.execute(JLUST.SpMVOp, u_A, JLUST.ust(x), JLUST.ust(y); backend=JLUST.EmitterBackend())
    end
    e_em  = CUDA.instantiate(g_em)
    CUDA.launch(e_em); CUDA.synchronize()

    @printf("%-22s rows=%6d cols=%6d nnz=%7d:\n", case, n_con, n_var, length(nonzeros(u_A)))

    t = @belapsed(begin CUDA.launch($e_cu); CUDA.synchronize() end, samples=300, evals=1) * 1e6
    @printf("  cuSPARSE graph + sync   : %7.3f μs\n", t)

    t = @belapsed(begin CUDA.launch($e_em); CUDA.synchronize() end, samples=300, evals=1) * 1e6
    @printf("  emitter  graph + sync   : %7.3f μs\n", t)

    # Amortise the sync — measure pure kernel time per launch
    t = @belapsed(begin
        CUDA.launch($e_cu); CUDA.launch($e_cu); CUDA.launch($e_cu); CUDA.launch($e_cu); CUDA.launch($e_cu)
        CUDA.launch($e_cu); CUDA.launch($e_cu); CUDA.launch($e_cu); CUDA.launch($e_cu); CUDA.launch($e_cu)
        CUDA.synchronize()
    end, samples=200, evals=1) * 1e6
    @printf("  cuSPARSE 10× / 1 sync   : %7.3f μs  (= %.2f μs/launch)\n", t, t/10)

    t = @belapsed(begin
        CUDA.launch($e_em); CUDA.launch($e_em); CUDA.launch($e_em); CUDA.launch($e_em); CUDA.launch($e_em)
        CUDA.launch($e_em); CUDA.launch($e_em); CUDA.launch($e_em); CUDA.launch($e_em); CUDA.launch($e_em)
        CUDA.synchronize()
    end, samples=200, evals=1) * 1e6
    @printf("  emitter  10× / 1 sync   : %7.3f μs  (= %.2f μs/launch)\n", t, t/10)
    println()
end

for case in ["pglib_opf_case1354_pegase",
             "pglib_opf_case6470_rte",
             "pglib_opf_case30000_goc"]
    bench_case_kernels(case)
end
