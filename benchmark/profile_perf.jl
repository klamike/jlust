# Per-block backend bake-off across sizes.
using Printf, Random, Downloads, LinearAlgebra
using JLUST, JLUST.Formats, KernelAbstractions, BenchmarkTools, SparseArrays, CUDA

# Load pglib helpers from sweep_cases without running the sweep main.
const _ORIG_ARGS = ARGS
include(joinpath(@__DIR__, "sweep_cases.jl"))

const CASES = ["pglib_opf_case1354_pegase",
               "pglib_opf_case6470_rte",
               "pglib_opf_case30000_goc"]

function _bench(label, f; samples=200)
    f(); CUDA.synchronize()
    t = @belapsed(begin $f(); CUDA.synchronize() end, samples=samples, evals=1) * 1e6
    @printf("    %-32s %8.2f μs\n", label, t)
    t
end

function profile_block(name, u_A)
    println("  block: $name  ($(size(u_A, 1)) × $(size(u_A, 2)),  nnz=$(JLUST.nnz(u_A)))")
    x = CUDA.randn(Float64, size(u_A, 2))
    y = CUDA.zeros(Float64, size(u_A, 1))
    u_x = JLUST.ust(x); u_y = JLUST.ust(y)

    _bench("emitter (default)",
           () -> JLUST.execute(JLUST.SpMVOp, u_A, u_x, u_y; backend=JLUST.EmitterBackend()))
    _bench("cuSPARSE direct (CUSPARSE.mv!)",
           () -> JLUST.execute(JLUST.SpMVOp, u_A, u_x, u_y; backend=JLUST.CUSPARSEBackend()))
    h = JLUST.prepare(JLUST.CUSPARSEBackend(), JLUST.SpMVOp, u_A)
    _bench("cuSPARSE handle (prepared)",
           () -> JLUST.execute(h, u_x, u_y))
end

for case in CASES
    println("\n", "="^70)
    println("Case: $case")
    println("="^70)
    d = load_pglib(case)
    (; Cg, Bf, Bbus, negI) = d
    profile_block("Cg",   csr_tensor(Cg;    device=CuArray))
    profile_block("Bbus", csr_tensor(-Bbus; device=CuArray))
    profile_block("negI", csr_tensor(negI;  device=CuArray))
    profile_block("Bf",   csr_tensor(Bf;    device=CuArray))
end
