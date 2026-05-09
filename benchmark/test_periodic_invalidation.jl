# Correctness test: update_block_values! must invalidate BBM caches built
# from a BSM, so the next mul! sees the new values.  Was a real bug before
# the cache-invalidation hook landed.
using JLUST, JLUST.Formats, KernelAbstractions, SparseArrays, CUDA, LinearAlgebra

let
    src = read(joinpath(@__DIR__, "sweep_cases.jl"), String)
    cut = findfirst("# ── main ─", src)
    include_string(@__MODULE__, src[1:first(cut)-1], "sweep_cases_helpers.jl")
end

d = load_pglib("pglib_opf_case1354_pegase")
(; Cg, Bf, Bbus, negI, n_bus, n_line, n_gen) = d
n_var = n_gen + n_line + n_bus
n_con = n_bus + n_line
T_per = 24

u_Cg   = csr_tensor(Cg;    device=CuArray)
u_Bbus = csr_tensor(-Bbus; device=CuArray)
u_negI = csr_tensor(negI;  device=CuArray)   # negI block — gets the patch!
u_Bf   = csr_tensor(Bf;    device=CuArray)
BM = BlockSparseMatrix([
    u_Cg    nothing  u_Bbus;
    nothing u_negI   u_Bf
])

R_cpu  = sparse(1:n_gen, 1:n_gen, ones(Float64, n_gen), n_gen, n_var)
u_posR = csr_tensor(R_cpu;  device=CuArray)
u_negR = csr_tensor(-R_cpu; device=CuArray)
n_rmp  = n_gen
BM_ms  = BlockBandedMatrix(BM, u_negR, u_posR, T_per, n_con, n_rmp, n_var)

n_row_ms = T_per*n_con + (T_per-1)*n_rmp
n_col_ms = T_per*n_var
x = CUDA.randn(Float64, n_col_ms)
y = CUDA.zeros(Float64, n_row_ms)

# First call: triggers periodic compile w/ BSM-diag patch on negI.
mul!(y, BM_ms, x); CUDA.synchronize()
y_before = Array(y)

# Now mutate negI via update_block_values!: change the constant -1.0 to -2.0.
# After this, the BSM compiled CSR's nzval for negI's slots should be -2.
# More importantly, the BBM PERIODIC cache must be invalidated, otherwise the
# stale leaner CSR + dp_val=-1 will return wrong results.
new_negI_vals = CUDA.fill(-2.0, length(nonzeros(u_negI)))
JLUST.update_block_values!(BM, 2, 2, new_negI_vals)

mul!(y, BM_ms, x); CUDA.synchronize()
y_after = Array(y)

# Diagnostic: inspect a few values from y_after vs y_before.
println("  diff at first 5 cells: ", y_after[1:5] .- y_before[1:5])
println("  diff at row n_bus+1 (negI row 1, period 1): ", y_after[n_bus + 1] - y_before[n_bus + 1])

# Recompute reference on host with the updated negI.  The user set u_negI's
# nzval to -2.0 directly, so the matrix at block (2,2) is -2 * I (not
# -2 * (-I) which would be +2 * I — the original `negI` is already -I).
A_per_cpu = vcat(
    hcat(Cg, spzeros(n_bus, n_line), -Bbus),
    hcat(spzeros(n_line, n_gen), -2 * sparse(I, n_line, n_line), Bf))
chunks = SparseMatrixCSC{Float64,Int}[]
for t in 1:T_per
    row = spzeros(Float64, n_con, n_col_ms)
    row[:, (t-1)*n_var+1 : t*n_var] = sparse(A_per_cpu)
    push!(chunks, row)
    if t < T_per
        off = spzeros(Float64, n_rmp, n_col_ms)
        off[:, (t-1)*n_var+1 : t*n_var]     = -R_cpu
        off[:, t*n_var+1     : (t+1)*n_var] =  R_cpu
        push!(chunks, off)
    end
end
BBM_full = vcat(chunks...)
y_ref = BBM_full * Vector(x)

err = norm(y_after .- y_ref) / (norm(y_ref) + 1e-30)
println("Update test:")
println("  y_before == y_after:  ", y_before == y_after, "   (must be false)")
println("  rel error vs ref:     ", err, "   (must be < 1e-12)")
@assert !(y_before == y_after) "y_after did not change after update — cache not invalidated"
@assert err < 1e-12 "y_after differs from updated reference"
println("  PASS — cache invalidation correct ✓")
