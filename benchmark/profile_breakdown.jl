# Microbenchmark breakdown: where does the 10 μs floor go?
using Printf, LinearAlgebra
using JLUST, JLUST.Formats, KernelAbstractions, BenchmarkTools, SparseArrays, CUDA

let
    src = read(joinpath(@__DIR__, "sweep_cases.jl"), String)
    cut = findfirst("# ── main ─", src)
    include_string(@__MODULE__, src[1:first(cut)-1], "sweep_cases_helpers.jl")
end

const CASE = "pglib_opf_case1354_pegase"

d = load_pglib(CASE)
(; Cg, Bf, Bbus, negI, n_bus, n_line, n_gen) = d
n_var = n_gen + n_line + n_bus
n_con = n_bus + n_line

u_Cg   = csr_tensor(Cg;    device=CuArray)
u_Bbus = csr_tensor(-Bbus; device=CuArray)
u_negI = csr_tensor(negI;  device=CuArray)
u_Bf   = csr_tensor(Bf;    device=CuArray)
BM = BlockSparseMatrix([
    u_Cg    nothing  u_Bbus;
    nothing u_negI   u_Bf
])

x = CUDA.randn(Float64, n_var)
y = CUDA.zeros(Float64, n_con)
for _ in 1:10; mul!(y, BM, x); CUDA.synchronize(); end

println("="^72)
println("Breakdown — pglib_opf_case1354_pegase, §2 BSM mul!")
println("="^72)

# Pull the compiled BSM out
import JLUST: BlockSparseMatrix
import CUDA
ext = Base.get_extension(JLUST, :CUDAExt)
c = ext._ensure_compiled_bsm(BM)

# 1) noop on host (sanity)
t = @belapsed(nothing) * 1e6
@printf("  noop                                  : %8.3f μs\n", t)

# 2) sync alone (after a no-op stream)
CUDA.synchronize()
t = @belapsed(CUDA.synchronize()) * 1e6
@printf("  CUDA.synchronize() alone              : %8.3f μs\n", t)

# 3) mul! alone, no sync (host return time)
t = @belapsed(mul!($y, $BM, $x)) * 1e6
@printf("  mul! return (no sync)                 : %8.3f μs\n", t)

# 4) mul! + sync (matches benchmark)
t = @belapsed(begin mul!($y, $BM, $x); CUDA.synchronize() end) * 1e6
@printf("  mul! + sync                           : %8.3f μs\n", t)

# 5) Direct CUDA.launch of cached graph + sync
key = (UInt(pointer(y)), UInt(pointer(x)))
exec = c.graph[key]
CUDA.launch(exec); CUDA.synchronize()
t = @belapsed(begin CUDA.launch($exec); CUDA.synchronize() end) * 1e6
@printf("  CUDA.launch(exec) + sync              : %8.3f μs\n", t)

# 6) Bare cusparseSpMV via JLUST.execute(handle) + sync (no graph)
t = @belapsed(begin JLUST.execute($c.handle, $x, $y); CUDA.synchronize() end) * 1e6
@printf("  cuSPARSE handle + sync (no graph)     : %8.3f μs\n", t)

# 7) Two cuLaunch + one sync
t = @belapsed(begin CUDA.launch($exec); CUDA.launch($exec); CUDA.synchronize() end) * 1e6
@printf("  2× CUDA.launch + 1× sync              : %8.3f μs\n", t)

# 8) Five cuLaunch + one sync (amortise sync)
t = @belapsed(begin
    CUDA.launch($exec); CUDA.launch($exec); CUDA.launch($exec); CUDA.launch($exec); CUDA.launch($exec)
    CUDA.synchronize()
end) * 1e6
@printf("  5× CUDA.launch + 1× sync              : %8.3f μs  (= %.2f μs/launch)\n", t, t/5)

# 9) Same but with cuSPARSE handle directly (5× exec + sync)
t = @belapsed(begin
    JLUST.execute($c.handle, $x, $y); JLUST.execute($c.handle, $x, $y); JLUST.execute($c.handle, $x, $y)
    JLUST.execute($c.handle, $x, $y); JLUST.execute($c.handle, $x, $y)
    CUDA.synchronize()
end) * 1e6
@printf("  5× cuSPARSE exec + 1× sync            : %8.3f μs  (= %.2f μs/exec)\n", t, t/5)
