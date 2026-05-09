# Focused profile for pglib_opf_case1354_pegase.
# Goal: identify any remaining gap between JLUST mul! and cuSPARSE on the
# equivalent assembled CSR, for both §2 (single-DCOPF) and §3 (BBM).

using Printf, Random, LinearAlgebra
using JLUST, JLUST.Formats, KernelAbstractions, BenchmarkTools, SparseArrays, CUDA

# Pull just the helpers (load_pglib, etc.) without running the sweep main.
let
    src = read(joinpath(@__DIR__, "sweep_cases.jl"), String)
    cut = findfirst("# ── main ─", src)
    cut === nothing && error("could not locate main marker in sweep_cases.jl")
    include_string(@__MODULE__, src[1:first(cut)-1], "sweep_cases_helpers.jl")
end

const CASE = "pglib_opf_case1354_pegase"
const T_PERIODS = 24

println("="^72)
println("Focused profile: $CASE")
println("="^72)

d = load_pglib(CASE)
(; Cg, Bf, Bbus, negI, n_bus, n_line, n_gen) = d
@printf("  n_bus=%d  n_line=%d  n_gen=%d\n", n_bus, n_line, n_gen)
n_var = n_gen + n_line + n_bus
n_con = n_bus + n_line
T = T_PERIODS

# ─── §2 setup ────────────────────────────────────────────────────────────────
u_Cg   = csr_tensor(Cg;    device=CuArray)
u_Bbus = csr_tensor(-Bbus; device=CuArray)
u_negI = csr_tensor(negI;  device=CuArray)
u_Bf   = csr_tensor(Bf;    device=CuArray)
BM = BlockSparseMatrix([
    u_Cg    nothing  u_Bbus;
    nothing u_negI   u_Bf
])

x_bm = CUDA.randn(Float64, n_var)
y_bm = CUDA.zeros(Float64, n_con)

A_cpu = vcat(
    hcat(Cg, spzeros(n_bus, n_line), -Bbus),
    hcat(spzeros(n_line, n_gen), negI, Bf))

println("\n── §2 single DCOPF ──")
# Warmup
for _ in 1:5; mul!(y_bm, BM, x_bm); CUDA.synchronize(); end

t = @belapsed(begin mul!($y_bm, $BM, $x_bm); CUDA.synchronize() end, samples=500, evals=1) * 1e6
@printf("  JLUST mul!(y, BM, x)             : %8.2f μs\n", t)

# cuSPARSE handle on the exact same assembled matrix
u_A_full = ust(CUDA.CUSPARSE.CuSparseMatrixCSR(A_cpu))
h_sp2    = prepare(CUSPARSEBackend(), SpMVOp, u_A_full)
JLUST.execute(h_sp2, x_bm, y_bm); CUDA.synchronize()
t = @belapsed(begin JLUST.execute($h_sp2, $x_bm, $y_bm); CUDA.synchronize() end, samples=500, evals=1) * 1e6
@printf("  cuSPARSE handle (raw CuVector)   : %8.2f μs\n", t)

# Bare cusparseSpMV via CUSPARSE.mv! (CUDA.jl high-level wrapper)
cusA = CUDA.CUSPARSE.CuSparseMatrixCSR(A_cpu)
CUDA.CUSPARSE.mv!('N', 1.0, cusA, x_bm, 0.0, y_bm, 'O'); CUDA.synchronize()
t = @belapsed(begin CUDA.CUSPARSE.mv!('N', 1.0, $cusA, $x_bm, 0.0, $y_bm, 'O'); CUDA.synchronize() end, samples=500, evals=1) * 1e6
@printf("  bare CUSPARSE.mv!                 : %8.2f μs\n", t)

# Within a CUDA Graph
g = CUDA.capture() do; mul!(y_bm, BM, x_bm); end
e = CUDA.instantiate(g)
CUDA.launch(e); CUDA.synchronize()
t = @belapsed(begin CUDA.launch($e); CUDA.synchronize() end, samples=500, evals=1) * 1e6
@printf("  JLUST mul! captured into Graph    : %8.2f μs\n", t)

# ─── §3 setup ────────────────────────────────────────────────────────────────
R_cpu  = sparse(1:n_gen, 1:n_gen, ones(Float64, n_gen), n_gen, n_var)
u_posR = csr_tensor(R_cpu;  device=CuArray)
u_negR = csr_tensor(-R_cpu; device=CuArray)

n_rmp    = n_gen
n_row_ms = T*n_con + (T-1)*n_rmp
n_col_ms = T*n_var
BM_ms    = BlockBandedMatrix(BM, u_negR, u_posR, T, n_con, n_rmp, n_var)

x_ms = CUDA.randn(Float64, n_col_ms)
y_ms = CUDA.zeros(Float64, n_row_ms)

println("\n── §3 T=$T-period block-banded ──")
print("  Compiling BBM (host assembly + GPU upload) ... "); flush(stdout)
t_compile = @elapsed begin
    mul!(y_ms, BM_ms, x_ms)   # triggers lazy compile
    CUDA.synchronize()
end
@printf("done in %.2f s\n", t_compile)

# Warmup the cached compiled BBM
for _ in 1:5; mul!(y_ms, BM_ms, x_ms); CUDA.synchronize(); end

t = @belapsed(begin mul!($y_ms, $BM_ms, $x_ms); CUDA.synchronize() end, samples=200, evals=1) * 1e6
@printf("  JLUST mul!(y_ms, BM_ms, x_ms)    : %8.2f μs  (= %.2f μs/period)\n", t, t/T)

# Build the same assembled BBM on the host for cuSPARSE comparison
chunks = SparseMatrixCSC{Float64,Int}[]
for t_i in 1:T
    row = spzeros(Float64, n_con, n_col_ms)
    row[:, (t_i-1)*n_var+1 : t_i*n_var] = sparse(A_cpu)
    push!(chunks, row)
    if t_i < T
        off = spzeros(Float64, n_rmp, n_col_ms)
        off[:, (t_i-1)*n_var+1 : t_i*n_var]     = -R_cpu
        off[:, t_i*n_var+1     : (t_i+1)*n_var] =  R_cpu
        push!(chunks, off)
    end
end
BBM_full = vcat(chunks...)

u_A_bbm = ust(CUDA.CUSPARSE.CuSparseMatrixCSR(BBM_full))
h_sp3   = prepare(CUSPARSEBackend(), SpMVOp, u_A_bbm)
JLUST.execute(h_sp3, x_ms, y_ms); CUDA.synchronize()
t = @belapsed(begin JLUST.execute($h_sp3, $x_ms, $y_ms); CUDA.synchronize() end, samples=200, evals=1) * 1e6
@printf("  cuSPARSE handle on BBM_full      : %8.2f μs  (= %.2f μs/period)\n", t, t/T)

# Verify the JLUST and cuSPARSE outputs match on the SAME inputs.
y_jlust = copy(Array(y_ms))
JLUST.execute(h_sp3, x_ms, y_ms); CUDA.synchronize()
y_cu = Array(y_ms)
mul!(y_ms, BM_ms, x_ms); CUDA.synchronize()
y_jlust = Array(y_ms)
diff = norm(y_jlust .- y_cu) / (norm(y_cu) + 1e-30)
@printf("  output match (rel error)         : %.2e\n", diff)

# Captured graph
g3 = CUDA.capture() do; mul!(y_ms, BM_ms, x_ms); end
e3 = CUDA.instantiate(g3)
CUDA.launch(e3); CUDA.synchronize()
t = @belapsed(begin CUDA.launch($e3); CUDA.synchronize() end, samples=200, evals=1) * 1e6
@printf("  JLUST mul! captured into Graph    : %8.2f μs  (= %.2f μs/period)\n", t, t/T)
