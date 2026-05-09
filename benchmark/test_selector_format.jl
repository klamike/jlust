# Verify the new `selector_tensor` builder + emitter walker produce a correct
# and fast 1-nnz/row SpMV.  Compare against the same matrix stored as CSR.
using Printf, LinearAlgebra
using JLUST, JLUST.Formats, KernelAbstractions, BenchmarkTools, SparseArrays, CUDA

n = 200_000
cols_h = Int32.(rand(1:n, n))   # 1 nnz per row at random col
vals_h = randn(Float64, n)

cu_cols = CuArray(cols_h)
cu_vals = CuArray(vals_h)

# Selector format USTensor — emitter walker generates the optimal kernel
u_sel = selector_tensor(cu_cols, cu_vals; m=n, n=n)
@printf("selector_tensor format: %s\n", format(u_sel))

# Reference: same matrix stored as a SparseMatrixCSC, then CSR USTensor
A_cpu = sparse(1:n, cols_h, vals_h, n, n)
u_csr = csr_tensor(A_cpu; device=CuArray)

# Correctness: compare outputs on the same input
x  = CUDA.randn(Float64, n)
y_sel = CUDA.zeros(Float64, n)
y_csr = CUDA.zeros(Float64, n)

mul!(y_sel, u_sel, x); CUDA.synchronize()
mul!(y_csr, u_csr, x); CUDA.synchronize()

err = norm(Vector(y_sel) - Vector(y_csr)) / (norm(Vector(y_csr)) + 1e-30)
@printf("rel error vs CSR: %.2e   (must be < 1e-12)\n", err)
@assert err < 1e-12

# Bench
for _ in 1:5; mul!(y_sel, u_sel, x); CUDA.synchronize(); end
for _ in 1:5; mul!(y_csr, u_csr, x); CUDA.synchronize(); end

t_sel = @belapsed(begin mul!($y_sel, $u_sel, $x); CUDA.synchronize() end, samples=300, evals=1) * 1e6
t_csr = @belapsed(begin mul!($y_csr, $u_csr, $x); CUDA.synchronize() end, samples=300, evals=1) * 1e6

@printf("\nSpMV on 1-nnz/row matrix (n=%d):\n", n)
@printf("  selector format (Dense, Singleton) : %7.2f μs\n", t_sel)
@printf("  CSR     format (Dense, Compressed) : %7.2f μs\n", t_csr)
@printf("  selector / CSR speedup            : %.2f×\n", t_csr / t_sel)
