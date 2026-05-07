# ─── JLUST format benchmark: EmitterBackend vs CUSPARSEBackend ───────────────
#
# Compares SpMV and SpMM on CSR, COO, and DCSR formats.
# Key story: for DCSR, CUSPARSEBackend must convert to CSR first (costly CPU
# transfer), while EmitterBackend runs device kernels directly.
#
# Usage:  julia --project=. benchmark/bench_formats.jl

using Printf, Random
using CUDA, CUDA.CUSPARSE, JLUST, JLUST.Formats, SparseArrays, BenchmarkTools

Random.seed!(42)

const _kaext   = Base.get_extension(JLUST, :KernelAbstractionsExt)
const _cudaext = Base.get_extension(JLUST, :CUDAExt)
const EmitterBackend  = _kaext.EmitterBackend
const CUSPARSEBackend = _cudaext.CUSPARSEBackend

# ─── Matrix generators ────────────────────────────────────────────────────────

function random_sparse(n::Int, nnz_target::Int; empty_row_frac::Float64=0.0, T=Float64)
    n_active = max(1, round(Int, n * (1 - empty_row_frac)))
    active_rows = sort(randperm(n)[1:n_active])
    nnz_per_row = max(1, div(nnz_target, n_active))
    Is = Int32[]; Js = Int32[]; Vs = T[]
    for i in active_rows
        ncols = min(nnz_per_row, n)
        cols  = sort(randperm(n)[1:ncols])
        for j in cols
            push!(Is, i); push!(Js, j); push!(Vs, randn(T))
        end
    end
    sparse(Is, Js, Vs, n, n)
end

function csr_gpu(A::SparseMatrixCSC{T}) where T
    ust(CuSparseMatrixCSR(A))
end

function coo_gpu(A::SparseMatrixCSC{T}) where T
    ust(CuSparseMatrixCOO(CuSparseMatrixCSR(A)))
end

function dcsr_gpu(A::SparseMatrixCSC{T}) where T
    m, n   = size(A)
    At     = sparse(A')
    rowptr = Int32.(At.colptr)
    colind = Int32.(At.rowval)
    nzval  = T.(At.nzval)
    u_csr  = csr_tensor(rowptr, colind, nzval, (m, n); origin=OneBased())
    u_dcsr = convert_format(u_csr, Formats.DCSR)
    materialize(u_dcsr; device=CUDADevice(0))
end

function dense_vec_gpu(v::Vector{T}) where T
    n   = length(v)
    USTensor{T,Int32,1,CuVector{T},CuVector{Int32},OneBased}(
        (n,), Formats.DensedRight(1),
        Dict{Int,CuVector{Int32}}(), Dict{Int,CuVector{Int32}}(),
        CuArray(v), nothing,
    )
end

# ─── Benchmark helpers ────────────────────────────────────────────────────────

function bench_us(u_A, u_x, u_y, backend; samples=50)
    for _ in 1:3
        sparse_mv!(u_A, u_x, u_y; backend=backend); CUDA.synchronize()
    end
    @belapsed(begin
        sparse_mv!($u_A, $u_x, $u_y; backend=$backend); CUDA.synchronize()
    end, samples=samples, evals=1) * 1e6
end

function bench_us_handle(h, u_x, u_y; samples=50)
    for _ in 1:3
        sparse_mv!(h, u_x, u_y); CUDA.synchronize()
    end
    @belapsed(begin
        sparse_mv!($h, $u_x, $u_y); CUDA.synchronize()
    end, samples=samples, evals=1) * 1e6
end

function bench_mm_us(u_A, u_B, u_C, backend; samples=50)
    for _ in 1:3
        sparse_mm!(u_A, u_B, u_C; backend=backend); CUDA.synchronize()
    end
    @belapsed(begin
        sparse_mm!($u_A, $u_B, $u_C; backend=$backend); CUDA.synchronize()
    end, samples=samples, evals=1) * 1e6
end

function bench_mm_us_handle(h, u_B, u_C; samples=50)
    for _ in 1:3
        sparse_mm!(h, u_B, u_C); CUDA.synchronize()
    end
    @belapsed(begin
        sparse_mm!($h, $u_B, $u_C); CUDA.synchronize()
    end, samples=samples, evals=1) * 1e6
end

function bench_dcsr_via_cusparse(A::SparseMatrixCSC{T}, u_x, u_y) where T
    At    = sparse(A')
    u_cpu = csr_tensor(Int32.(At.colptr), Int32.(At.rowval), T.(At.nzval),
                       (size(A,1), size(A,2)); origin=OneBased())
    for _ in 1:3
        u_csr = materialize(u_cpu; device=CUDADevice(0))
        sparse_mv!(u_csr, u_x, u_y; backend=CUSPARSEBackend()); CUDA.synchronize()
    end
    t_total = @belapsed(begin
        u_csr = materialize($u_cpu; device=CUDADevice(0))
        sparse_mv!(u_csr, $u_x, $u_y; backend=CUSPARSEBackend()); CUDA.synchronize()
    end, samples=30, evals=1) * 1e6
    t_total
end

# ─── Metrics ─────────────────────────────────────────────────────────────────

gflops(nz, t_s) = 2.0 * nz / t_s / 1e9

function gbps_spmv(nz, n, t_s; fmt=:csr, sz_T=8, sz_I=4)
    # CSR: rowptr + colind + nzval + x + y
    # COO: row_crd + col_crd + nzval + x + y  (no rowptr; double crd)
    # DCSR: outer_crd + inner_pos + inner_crd + nzval + x + y
    bytes = if fmt == :coo
        2 * nz * sz_I + nz * sz_T + 2 * n * sz_T
    elseif fmt == :dcsr
        nz * sz_I + nz * sz_I + nz * sz_I + nz * sz_T + 2 * n * sz_T  # approx
    else  # csr
        (n+1) * sz_I + nz * sz_I + nz * sz_T + 2 * n * sz_T
    end
    bytes / t_s / 1e9
end

# ─── Print helpers ────────────────────────────────────────────────────────────

function print_row(fmt_name, bname, t_us, base_us, nz, n; fmt_sym=:csr, sz_T=8)
    spd = base_us > 0 ? @sprintf("%.2f×", base_us / t_us) : "  base"
    gf  = gflops(nz, t_us * 1e-6)
    gb  = gbps_spmv(nz, n, t_us * 1e-6; fmt=fmt_sym, sz_T=sz_T)
    @printf("  %-18s  %-10s  %10.1f  %10s  %10.2f  %10.2f\n",
            fmt_name, bname, t_us, spd, gf, gb)
end

# ─── SpMV Suite ──────────────────────────────────────────────────────────────

function run_spmv_suite(n, nnz_target, empty_frac; T=Float64)
    println("\n── SpMV  n=$n  nnz≈$nnz_target  $(round(Int,empty_frac*100))% empty rows  T=$T ──")
    A  = random_sparse(n, nnz_target; empty_row_frac=empty_frac, T=T)
    nz = nnz(A)

    u_x = dense_vec_gpu(randn(T, n))
    u_y = dense_vec_gpu(zeros(T, n))
    sz_T = sizeof(T)

    @printf("  %-18s  %-10s  %10s  %10s  %10s  %10s\n",
            "Format", "Backend", "Time(μs)", "vs CSR/CU", "GFLOP/s", "GB/s")
    println("  ", "-"^72)

    u_csr = csr_gpu(A)
    t_base = bench_us(u_csr, u_x, u_y, CUSPARSEBackend())
    print_row("CSR", "CUSPARSE", t_base, 0.0, nz, n; sz_T)

    h_spmv = prepare(CUSPARSEBackend(), SpMVOp, u_csr)
    t = bench_us_handle(h_spmv, u_x, u_y)
    t_base_h = t
    print_row("CSR", "CU(handle)", t, t_base, nz, n; sz_T)

    t = bench_us(u_csr, u_x, u_y, EmitterBackend())
    print_row("CSR", "Emitter", t, t_base_h, nz, n; sz_T)

    u_coo = coo_gpu(A)
    t = bench_us(u_coo, u_x, u_y, CUSPARSEBackend())
    print_row("COO", "CUSPARSE", t, t_base_h, nz, n; fmt_sym=:coo, sz_T)

    t = bench_us(u_coo, u_x, u_y, EmitterBackend())
    print_row("COO(warp)", "Emitter", t, t_base_h, nz, n; fmt_sym=:coo, sz_T)

    if empty_frac > 0
        u_dcsr = dcsr_gpu(A)
        t = bench_us(u_dcsr, u_x, u_y, EmitterBackend())
        print_row("DCSR", "Emitter", t, t_base_h, nz, n; fmt_sym=:dcsr, sz_T)

        t_tot = bench_dcsr_via_cusparse(A, u_x, u_y)
        print_row("DCSR→CSR(total)", "CUSPARSE", t_tot, t_base_h, nz, n; sz_T)
    end
end

# ─── SpMM Suite ──────────────────────────────────────────────────────────────

function bench_dcsr_mm_via_cusparse(A::SparseMatrixCSC{T}, u_B, u_C) where T
    At    = sparse(A')
    u_cpu = csr_tensor(Int32.(At.colptr), Int32.(At.rowval), T.(At.nzval),
                       (size(A,1), size(A,2)); origin=OneBased())
    for _ in 1:3
        u_csr = materialize(u_cpu; device=CUDADevice(0))
        sparse_mm!(u_csr, u_B, u_C; backend=CUSPARSEBackend()); CUDA.synchronize()
    end
    @belapsed(begin
        u_csr = materialize($u_cpu; device=CUDADevice(0))
        sparse_mm!(u_csr, $u_B, $u_C; backend=CUSPARSEBackend()); CUDA.synchronize()
    end, samples=20, evals=1) * 1e6
end

function run_spmm_suite(n, nnz_target, k, empty_frac=0.0; T=Float64)
    println("\n── SpMM  n=$n  nnz≈$nnz_target  k=$k  $(round(Int,empty_frac*100))% empty rows  T=$T ──")
    A  = random_sparse(n, nnz_target; empty_row_frac=empty_frac, T=T)
    nz = nnz(A)

    B_d  = CUDA.randn(T, n, k)
    C_d  = CUDA.zeros(T, n, k)

    make_u_dense(arr) = begin
        fmt = Formats.DensedRight(2)
        USTensor{T,Int32,2,typeof(arr),CuVector{Int32},OneBased}(
            size(arr), fmt,
            Dict{Int,CuVector{Int32}}(), Dict{Int,CuVector{Int32}}(),
            arr, nothing)
    end

    u_B = make_u_dense(B_d)
    u_C = make_u_dense(C_d)

    @printf("  %-18s  %-10s  %10s  %10s  %10s\n",
            "Format", "Backend", "Time(μs)", "vs CSR/CU", "GFLOP/s")
    println("  ", "-"^56)

    u_csr = csr_gpu(A)
    t_direct = bench_mm_us(u_csr, u_B, u_C, CUSPARSEBackend())
    @printf("  %-18s  %-10s  %10.1f  %10s  %10.2f\n",
            "CSR", "CUSPARSE", t_direct, "base", 2.0*nz*k/(t_direct*1e-6)/1e9)

    h_spmm = prepare(CUSPARSEBackend(), SpMMOp, u_csr; n_cols=k)
    fill!(C_d, zero(T)); t_base = bench_mm_us_handle(h_spmm, u_B, u_C)
    @printf("  %-18s  %-10s  %10.1f  %10s  %10.2f\n",
            "CSR", "CU(handle)", t_base, @sprintf("%.2f×", t_direct/t_base), 2.0*nz*k/(t_base*1e-6)/1e9)

    fill!(C_d, zero(T)); t = bench_mm_us(u_csr, u_B, u_C, EmitterBackend())
    @printf("  %-18s  %-10s  %10.1f  %10s  %10.2f\n",
            "CSR", "Emitter", t, @sprintf("%.2f×", t_base/t), 2.0*nz*k/(t*1e-6)/1e9)

    if empty_frac > 0
        u_dcsr = dcsr_gpu(A)
        fill!(C_d, zero(T)); t = bench_mm_us(u_dcsr, u_B, u_C, EmitterBackend())
        @printf("  %-18s  %-10s  %10.1f  %10s  %10.2f\n",
                "DCSR", "Emitter", t, @sprintf("%.2f×", t_base/t), 2.0*nz*k/(t*1e-6)/1e9)

        fill!(C_d, zero(T)); t_tot = bench_dcsr_mm_via_cusparse(A, u_B, u_C)
        @printf("  %-18s  %-10s  %10.1f  %10s  %10.2f\n",
                "DCSR→CSR(total)", "CUSPARSE", t_tot, @sprintf("%.2f×", t_base/t_tot), 2.0*nz*k/(t_tot*1e-6)/1e9)
    end
end

# ─── SDDMM Suite ─────────────────────────────────────────────────────────────
#
# C (sparse m×m mask) ← alpha * (A (m×k) * B (k×m)) ∘ C + beta * C
# Compares: Direct CUSPARSE (preprocess + compute every call),
#           Handle CUSPARSE (preprocess cached at prepare() time),
#           EmitterBackend (JIT kernel).

function bench_sddmm_direct(u_A, u_B, u_C; samples=50)
    for _ in 1:3; sparse_sddmm!(CUSPARSEBackend(), u_A, u_B, u_C); CUDA.synchronize(); end
    @belapsed(begin
        sparse_sddmm!(CUSPARSEBackend(), $u_A, $u_B, $u_C); CUDA.synchronize()
    end, samples=samples, evals=1) * 1e6
end

function bench_sddmm_handle(h, u_A, u_B, u_C; samples=50)
    for _ in 1:3; sparse_sddmm!(h, u_A, u_B, u_C); CUDA.synchronize(); end
    @belapsed(begin
        sparse_sddmm!($h, $u_A, $u_B, $u_C); CUDA.synchronize()
    end, samples=samples, evals=1) * 1e6
end

function bench_sddmm_emit(u_A, u_B, u_C; samples=50)
    for _ in 1:3; sparse_sddmm!(EmitterBackend(), u_A, u_B, u_C); CUDA.synchronize(); end
    @belapsed(begin
        sparse_sddmm!(EmitterBackend(), $u_A, $u_B, $u_C); CUDA.synchronize()
    end, samples=samples, evals=1) * 1e6
end

function run_sddmm_suite(n, nnz_target, k; T=Float64)
    A_cpu = random_sparse(n, nnz_target; T=T)
    nz    = nnz(A_cpu)

    # Dense A (n×k) and B (k×n); sparse C (n×n CSR mask)
    A_d = CUDA.randn(T, n, k)
    B_d = CUDA.randn(T, k, n)

    make_u_dense(arr) = begin
        fmt = Formats.DensedRight(2)
        USTensor{T,Int32,2,typeof(arr),CuVector{Int32},OneBased}(
            size(arr), fmt,
            Dict{Int,CuVector{Int32}}(), Dict{Int,CuVector{Int32}}(),
            arr, nothing)
    end

    u_A = make_u_dense(A_d)
    u_B = make_u_dense(B_d)
    u_C = csr_gpu(A_cpu)   # sparse mask/result with same pattern as test matrix

    nflops = 2 * nz * k   # one k-dot per nonzero of C

    println("\n── SDDMM  n=$n  nnz≈$nz  k=$k  T=$T ──")
    @printf("  %-18s  %-10s  %10s  %10s  %10s\n",
            "Variant", "Backend", "Time(μs)", "vs Direct", "GFLOP/s")
    println("  ", "-"^60)

    gf(t_us) = nflops / (t_us * 1e3)

    t_direct = bench_sddmm_direct(u_A, u_B, u_C)
    @printf("  %-18s  %-10s  %10.1f  %10s  %10.2f\n",
            "Direct", "CUSPARSE", t_direct, "base", gf(t_direct))

    h_sddmm = prepare(CUSPARSEBackend(), SDDMMOp, u_A, u_B, u_C)
    t_handle = bench_sddmm_handle(h_sddmm, u_A, u_B, u_C)
    @printf("  %-18s  %-10s  %10.1f  %10s  %10.2f\n",
            "Handle", "CUSPARSE", t_handle, @sprintf("%.2f×", t_direct/t_handle), gf(t_handle))

    t_emit = bench_sddmm_emit(u_A, u_B, u_C)
    @printf("  %-18s  %-10s  %10.1f  %10s  %10.2f\n",
            "Emitter", "Emitter", t_emit, @sprintf("%.2f×", t_direct/t_emit), gf(t_emit))
end

# ─── SpGEMM Suite ─────────────────────────────────────────────────────────────
#
# C (CSR) ← alpha * A (CSR) * B (CSR)
# Compares: Direct CUSPARSE (symbolic + numeric each call),
#           Handle CUSPARSE (SpGEMMreuse: symbolic cached, numeric only per call),
#           EmitterBackend (scatter-sort-reduce per call).

function bench_gemm_direct(u_A, u_B, u_C_templ; samples=30)
    for _ in 1:3; sparse_gemm!(CUSPARSEBackend(), u_A, u_B, u_C_templ); CUDA.synchronize(); end
    @belapsed(begin
        sparse_gemm!(CUSPARSEBackend(), $u_A, $u_B, $u_C_templ); CUDA.synchronize()
    end, samples=samples, evals=1) * 1e6
end

function bench_gemm_handle(h; samples=30)
    for _ in 1:3; sparse_gemm!(h); CUDA.synchronize(); end
    @belapsed(begin; sparse_gemm!($h); CUDA.synchronize(); end,
              samples=samples, evals=1) * 1e6
end

function bench_gemm_emit(u_A, u_B; samples=30)
    for _ in 1:3; sparse_gemm!(EmitterBackend(), u_A, u_B); CUDA.synchronize(); end
    @belapsed(begin
        sparse_gemm!(EmitterBackend(), $u_A, $u_B); CUDA.synchronize()
    end, samples=samples, evals=1) * 1e6
end

function run_spgemm_suite(n, nnz_target; T=Float64)
    A_cpu = random_sparse(n, nnz_target; T=T)
    B_cpu = random_sparse(n, nnz_target; T=T)
    nnz_C = nnz(A_cpu * B_cpu)
    nz_A  = nnz(A_cpu)

    u_A = csr_gpu(A_cpu)
    u_B = csr_gpu(B_cpu)
    # C template: correct sparsity structure for direct CUSPARSE (values overwritten)
    u_C_templ = csr_gpu(A_cpu * B_cpu)

    println("\n── SpGEMM  n=$n  nnz_A≈$nz_A  nnz_C≈$nnz_C  T=$T ──")
    @printf("  %-20s  %-10s  %10s  %10s\n",
            "Variant", "Backend", "Time(μs)", "vs Direct")
    println("  ", "-"^44)

    t_direct = bench_gemm_direct(u_A, u_B, u_C_templ)
    @printf("  %-20s  %-10s  %10.1f  %10s\n",
            "Direct", "CUSPARSE", t_direct, "base")

    h_gemm = prepare(CUSPARSEBackend(), SpGEMMOp, u_A, u_B)
    t_handle = bench_gemm_handle(h_gemm)
    @printf("  %-20s  %-10s  %10.1f  %10s\n",
            "Handle(reuse)", "CUSPARSE", t_handle, @sprintf("%.2f×", t_direct/t_handle))

    t_emit = bench_gemm_emit(u_A, u_B)
    @printf("  %-20s  %-10s  %10.1f  %10s\n",
            "Emitter", "Emitter", t_emit, @sprintf("%.2f×", t_direct/t_emit))
end

# ─── Run ──────────────────────────────────────────────────────────────────────

println("="^80)
println("JLUST Benchmark  —  $(CUDA.name(CUDA.device()))  —  Julia $(VERSION)")
println("="^80)

spmv_configs = [
    (8192,   80_000,   0.0),
    (32768,  200_000,  0.5),
    (131072, 800_000,  0.8),
    (262144, 1_000_000, 0.9),
]

for (n, nnz_t, ef) in spmv_configs
    try; run_spmv_suite(n, nnz_t, ef; T=Float64)
    catch e; @printf("\nSpMV n=%d FAILED: %s\n", n, sprint(showerror, e)); end
end

spmm_configs = [
    (8192,   80_000,  16,  0.0),
    (32768,  200_000, 32,  0.5),
    (131072, 800_000, 64,  0.8),
]
for (n, nnz_t, k, ef) in spmm_configs
    try; run_spmm_suite(n, nnz_t, k, ef; T=Float64)
    catch e; @printf("\nSpMM n=%d k=%d FAILED: %s\n", n, k, sprint(showerror, e)); end
end

sddmm_configs = [
    (8192,   80_000,  32),
    (32768,  200_000, 64),
    (131072, 800_000, 16),
]
for (n, nnz_t, k) in sddmm_configs
    try; run_sddmm_suite(n, nnz_t, k; T=Float64)
    catch e; @printf("\nSDDMM n=%d k=%d FAILED: %s\n", n, k, sprint(showerror, e)); end
end

spgemm_configs = [
    (4_096,  16_000),
    (16_384, 100_000),
    (65_536, 400_000),
]
for (n, nnz_t) in spgemm_configs
    try; run_spgemm_suite(n, nnz_t; T=Float64)
    catch e; @printf("\nSpGEMM n=%d FAILED: %s\n", n, sprint(showerror, e)); end
end

println("\n", "="^80)
println("Benchmark complete.")
