# bench_suitesparse.jl — JLUST performance across diverse sparse instances
#
# Pulls a curated set of matrices from the SuiteSparse Matrix Collection
# (https://sparse.tamu.edu) and benchmarks SpMV and SpMM (k=32) through:
#
#   • SparseMatrixCSC  (Julia stdlib, CPU reference)
#   • JLUST CSR        (csr_tensor)
#   • JLUST CSC        (csc_tensor)
#   • JLUST DCSR       (dcsr_tensor — wins when many rows are empty)
#   • cuSPARSE handle  (when CUDA is available — apples-to-apples GPU ref)
#
# Goal: a benchmark harness we can iterate on for matrices that are *not*
# DCOPF-shaped (irregular row-NNZ, power-law graphs, banded structural,
# rectangular LP, …).  Each row of the output tables is one matrix; columns
# are formats.  Any regression in CSR/DCSR vs. cuSPARSE should jump out.
#
# Caching:  $(JLUST)/benchmark/.cache/suitesparse/<group>__<name>.mtx
# First run downloads ~30 MB total; subsequent runs are local-only.
#
# Usage:
#   julia --project=benchmark benchmark/bench_suitesparse.jl
#
# Single-matrix iteration mode:
#   JLUST_BENCH_ONLY=Hamm/scircuit julia --project=benchmark benchmark/bench_suitesparse.jl

using Printf, Random, Downloads
using JLUST, JLUST.Formats
using SparseArrays, LinearAlgebra
using KernelAbstractions
using BenchmarkTools

const HAS_CUDA  = try; using CUDA;  CUDA.functional();  catch; false; end
const HAS_METAL = !HAS_CUDA && try; using Metal; Metal.functional(); catch; false; end

const FloatType = HAS_METAL ? Float32 : Float64

function to_device(x::AbstractArray{T}) where T
    arr = (HAS_METAL && T === Float64) ? Float32.(x) : x
    HAS_CUDA  ? CUDA.CuArray(arr)   :
    HAS_METAL ? Metal.MtlArray(arr) : arr
end

function sync()
    HAS_CUDA  && CUDA.synchronize()
    HAS_METAL && Metal.synchronize()
end

# ── SuiteSparse fetch + cache ─────────────────────────────────────────────────

# Canonical SuiteSparse Matrix Collection mirror (Tim Davis's UFL host).
# Override with the JLUST_SS_MIRROR env var if you want a faster regional mirror.
const _SS_URL   = get(ENV, "JLUST_SS_MIRROR",
                      "https://www.cise.ufl.edu/research/sparse/MM/")
const _SS_CACHE = joinpath(dirname(abspath(@__FILE__)), "..", ".cache", "suitesparse")

# Fetches <group>/<name>.tar.gz, extracts <name>/<name>.mtx, and stores it
# at .cache/suitesparse/<group>__<name>.mtx.  Skips the download on cache hit.
function ss_fetch(group::String, name::String)
    mkpath(_SS_CACHE)
    cached = joinpath(_SS_CACHE, "$(group)__$(name).mtx")
    isfile(cached) && return cached

    url = _SS_URL * "$group/$name.tar.gz"
    print("    Fetching $group/$name … "); flush(stdout)
    tmpdir = mktempdir()
    tarball = joinpath(tmpdir, "$name.tar.gz")
    Downloads.download(url, tarball)
    # tar lives on macOS and Linux; -xzf is portable enough for our use.
    run(pipeline(`tar -xzf $tarball -C $tmpdir`; stdout=devnull))
    src = joinpath(tmpdir, name, "$name.mtx")
    isfile(src) || error("expected $name.mtx inside $tarball; got: " *
                         join(readdir(joinpath(tmpdir, name)), ", "))
    cp(src, cached; force=true)
    println("$(filesize(cached) ÷ 1024) KB")
    cached
end

# ── Minimal MatrixMarket reader ───────────────────────────────────────────────
#
# Supports `coordinate real {general,symmetric,skew-symmetric}` — covers all
# matrices in the curated list below.  Pattern matrices (no values) get value 1.
# We skip complex/Hermitian for now; trip an error if encountered.

function read_mm(path::String)
    open(path, "r") do io
        header = readline(io)
        startswith(header, "%%MatrixMarket") ||
            error("not a MatrixMarket file: $path")
        toks = split(lowercase(header))
        # %%MatrixMarket matrix coordinate <field> <symmetry>
        length(toks) >= 5 || error("unexpected MM header: $header")
        toks[3] == "coordinate" || error("only coordinate format supported, got $(toks[3])")
        field    = toks[4]   # real | integer | pattern | complex
        symmetry = toks[5]   # general | symmetric | skew-symmetric | hermitian
        symmetry == "hermitian"  && error("hermitian symmetry not supported")
        field    == "complex"    && error("complex field not supported")
        is_pattern = field == "pattern"

        # Skip comment lines.
        line = readline(io)
        while startswith(line, "%"); line = readline(io); end
        m, n, nnz_decl = parse.(Int, split(line))

        rows = Vector{Int32}(undef, 0); sizehint!(rows, nnz_decl)
        cols = Vector{Int32}(undef, 0); sizehint!(cols, nnz_decl)
        vals = Vector{FloatType}(undef, 0); sizehint!(vals, nnz_decl)

        for _ in 1:nnz_decl
            parts = split(readline(io))
            i = parse(Int32, parts[1]); j = parse(Int32, parts[2])
            v = is_pattern ? one(FloatType) : parse(FloatType, parts[3])
            push!(rows, i); push!(cols, j); push!(vals, v)
            if symmetry != "general" && i != j
                push!(rows, j); push!(cols, i)
                push!(vals, symmetry == "skew-symmetric" ? -v : v)
            end
        end
        return sparse(rows, cols, vals, m, n)
    end
end

# ── Curated matrix list ───────────────────────────────────────────────────────
#
# Hand-picked across categories that exercise different SpMV regimes.
# Sized to keep first-run downloads under ~60 MB and runtime under a minute.
#
#   group / name         category    notes
#   HB/bcsstk17          structural  SPD, banded, ~28 nnz/row — predictable
#   Norris/heart3        biology     dense rows, small (~290 nnz/row)
#   SNAP/wiki-Vote       graph       directed social, irregular, sym-augmented
#   Hamm/scircuit        circuit     unsym, very irregular row-NNZ
#   Williams/mac_econ_fwd500  econ   unsym, hyperlink-style
#   GHS_indef/cont-300   opt-indef   symmetric indefinite, sparse
#   Boeing/bcsstk39      structural  larger SPD, ~45 nnz/row
#   Williams/cop20k_A    QC/DFT      symmetric, FEM-like
#   Williams/cant        FEM         symmetric, very dense per row (~64)
#   LPnetlib/lp_cre_b    LP          rectangular 9.6k × 77k
#   SNAP/email-EuAll     graph       has many empty rows — DCSR signal
#   ND/nd6k              3D PDE      symmetric, dense rows (~388 nnz/row)

struct Case
    group::String
    name::String
    category::String
end

const CURATED = Case[
    Case("HB",         "bcsstk17",          "structural"),
    Case("Norris",     "heart3",            "biology"),
    Case("SNAP",       "wiki-Vote",         "graph"),
    Case("Hamm",       "scircuit",          "circuit"),
    Case("Williams",   "mac_econ_fwd500",   "economics"),
    Case("GHS_indef",  "cont-300",          "optimization"),
    Case("Boeing",     "bcsstk39",          "structural"),
    Case("Williams",   "cop20k_A",          "QC/DFT"),
    Case("Williams",   "cant",              "FEM"),
    Case("LPnetlib",   "lp_cre_b",          "LP-rect"),
    Case("SNAP",       "email-EuAll",       "graph"),
    Case("ND",         "nd6k",              "3D-PDE"),
]

# ── Per-matrix benchmark ──────────────────────────────────────────────────────

const SPMM_K   = 32
const N_S_VEC  = 50      # samples for SpMV
const N_S_MAT  = 20      # samples for SpMM (heavier; fewer samples)

# Bytes touched by an SpMV with Int32 indices: nzval + colind + (rowptr ≈ m+1)
# + x_read (n*8) + y_write (m*8).  Useful for "effective bandwidth" reporting.
function spmv_bytes(m::Int, n::Int, nnz::Int, T::Type)
    sizeof(T) * nnz + sizeof(Int32) * (nnz + m + 1) + sizeof(T) * (n + m)
end

function _time_us(f, samples)
    f(); sync()
    @belapsed(begin $f(); $sync() end, samples=samples, evals=1) * 1e6
end

# Run one matrix through every (format × baseline × op) combination we want.
# Returns a NamedTuple keyed by table column.
function bench_matrix(case::Case)
    path = ss_fetch(case.group, case.name)
    A = read_mm(path)
    m, n = size(A)
    nz = SparseArrays.nnz(A)
    avg = nz / m

    Random.seed!(0xBEEF)
    x_cpu = randn(FloatType, n)
    y_cpu = zeros(FloatType, m)
    X_cpu = randn(FloatType, n, SPMM_K)
    Y_cpu = zeros(FloatType, m, SPMM_K)

    # Reference output for correctness.
    y_ref = A * x_cpu
    Y_ref = A * X_cpu

    # ── Build USTensors (host or device) ─────────────────────────────────────
    A_csr  = csr_tensor(A;  device=to_device)
    A_dcsr = dcsr_tensor(A; device=to_device)
    # CSC — JLUST ust(A) is a zero-copy CSC view; for GPU we need a device copy.
    # Build CSC USTensor from buffers.
    A_dev  = HAS_CUDA  ? CUDA.CUSPARSE.CuSparseMatrixCSC(A) :
             HAS_METAL ? sparse(FloatType.(A))                : A
    A_csc  = csc_tensor(to_device(Int32.(A.colptr)),
                        to_device(Int32.(A.rowval)),
                        to_device(FloatType.(A.nzval)); m=m, n=n)

    x_dev = to_device(x_cpu)
    y_dev = to_device(zeros(FloatType, m))
    X_dev = to_device(X_cpu)
    Y_dev = to_device(zeros(FloatType, m, SPMM_K))

    # ── SpMV timings ─────────────────────────────────────────────────────────
    # Force EmitterBackend on the JLUST rows: on GPU, the default backend for
    # CSR/CSC is CUSPARSEBackend, which would make JLUST-CSR ≡ cuSPARSE.  We
    # want JLUST's emitter path explicitly so the table compares vendor
    # vs. emitter (and DCSR, which cuSPARSE can't ingest, works at all).
    A_csc_cpu = sparse(FloatType.(A))   # local CSC copy (FloatType-typed)
    t_sm_csc = _time_us(() -> mul!(y_cpu, A_csc_cpu, x_cpu), N_S_VEC)

    u_x = ust(x_dev); u_y = ust(y_dev)
    spmv_em(A) = execute(EmitterBackend(), SpMVOp(format(A), format(u_x), format(u_y)),
                         A, u_x, u_y)

    fill!(y_dev, 0)
    t_jl_csr  = _time_us(() -> spmv_em(A_csr),  N_S_VEC)
    y_csr_out = Array(y_dev)

    fill!(y_dev, 0)
    t_jl_csc  = _time_us(() -> spmv_em(A_csc),  N_S_VEC)
    y_csc_out = Array(y_dev)

    fill!(y_dev, 0)
    t_jl_dcsr = _time_us(() -> spmv_em(A_dcsr), N_S_VEC)
    y_dcsr_out = Array(y_dev)

    # Adversarial cuSPARSE: try all reasonable SpMV algos and report the best.
    # cuSPARSE_SPMV_ALG_DEFAULT picks based on internal heuristics, but the
    # other algos sometimes beat it — measuring the best avoids penalizing
    # cuSPARSE on workloads where its heuristic is suboptimal.
    t_cusp = NaN
    cusp_algo = ""
    if HAS_CUDA
        u_A_cu = ust(CUDA.CUSPARSE.CuSparseMatrixCSR(FloatType.(A)))
        u_x_cu = ust(CUDA.CuArray(FloatType.(x_cpu)))
        u_y_cu = ust(CUDA.zeros(FloatType, m))
        for (label, alg) in (("DEF", CUDA.CUSPARSE.CUSPARSE_SPMV_ALG_DEFAULT),
                             ("A1",  CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG1),
                             ("A2",  CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2))
            try
                h = prepare(CUSPARSEBackend(), SpMVOp, u_A_cu; algo=alg)
                execute(h, u_x_cu, u_y_cu); CUDA.synchronize()
                t = _time_us(() -> execute(h, u_x_cu, u_y_cu), N_S_VEC)
                if isnan(t_cusp) || t < t_cusp
                    t_cusp = t
                    cusp_algo = label
                end
            catch; end   # some algos are invalid for some matrices
        end
    end

    relerr(y) = norm(Vector{Float64}(y) .- Vector{Float64}(y_ref)) /
                (norm(Vector{Float64}(y_ref)) + 1e-30)
    e_csr  = relerr(y_csr_out)
    e_csc  = relerr(y_csc_out)
    e_dcsr = relerr(y_dcsr_out)
    ok_csr  = e_csr  < 1e-3
    ok_csc  = e_csc  < 1e-3
    ok_dcsr = e_dcsr < 1e-3

    # ── SpMM timings ─────────────────────────────────────────────────────────
    t_mm_sm_csc = _time_us(() -> mul!(Y_cpu, A_csc_cpu, X_cpu), N_S_MAT)

    u_B = ust(X_dev); u_C = ust(Y_dev)
    spmm_em(A) = execute(EmitterBackend(), SpMMOp(format(A), format(u_B), format(u_C)),
                         A, u_B, u_C)

    fill!(Y_dev, 0)
    t_mm_csr  = _time_us(() -> spmm_em(A_csr),  N_S_MAT)
    Y_csr_out = Array(Y_dev)

    fill!(Y_dev, 0)
    t_mm_csc  = _time_us(() -> spmm_em(A_csc),  N_S_MAT)
    Y_csc_out = Array(Y_dev)

    fill!(Y_dev, 0)
    t_mm_dcsr = _time_us(() -> spmm_em(A_dcsr), N_S_MAT)
    Y_dcsr_out = Array(Y_dev)

    # Adversarial cuSPARSE SpMM: sweep DEFAULT, CSR_ALG1/2/3.  ALG3 is tuned
    # for k=1, ALG2 for moderate k, but for k=32 the best can be any of them
    # depending on density and pattern; report whichever wins.
    t_mm_cusp = NaN
    cusp_mm_algo = ""
    if HAS_CUDA
        u_A_cu = ust(CUDA.CUSPARSE.CuSparseMatrixCSR(FloatType.(A)))
        u_B_cu = ust(CUDA.CuArray(FloatType.(X_cpu)))
        u_C_cu = ust(CUDA.zeros(FloatType, m, SPMM_K))
        for (label, alg) in (("DEF", CUDA.CUSPARSE.CUSPARSE_SPMM_ALG_DEFAULT),
                             ("A1",  CUDA.CUSPARSE.CUSPARSE_SPMM_CSR_ALG1),
                             ("A2",  CUDA.CUSPARSE.CUSPARSE_SPMM_CSR_ALG2),
                             ("A3",  CUDA.CUSPARSE.CUSPARSE_SPMM_CSR_ALG3))
            try
                h = prepare(CUSPARSEBackend(), SpMMOp, u_A_cu; n_cols=SPMM_K, algo=alg)
                execute(h, u_B_cu, u_C_cu); CUDA.synchronize()
                t = _time_us(() -> execute(h, u_B_cu, u_C_cu), N_S_MAT)
                if isnan(t_mm_cusp) || t < t_mm_cusp
                    t_mm_cusp = t
                    cusp_mm_algo = label
                end
            catch; end
        end
    end

    Mrel(Y) = norm(Matrix{Float64}(Y) .- Matrix{Float64}(Y_ref)) /
              (norm(Matrix{Float64}(Y_ref)) + 1e-30)
    e_mm_csr  = Mrel(Y_csr_out)
    e_mm_csc  = Mrel(Y_csc_out)
    e_mm_dcsr = Mrel(Y_dcsr_out)
    ok_mm = (e_mm_csr < 1e-3) && (e_mm_csc < 1e-3) && (e_mm_dcsr < 1e-3)

    (; m, n, nnz=nz, avg,
       t_sm_csc, t_jl_csr, t_jl_csc, t_jl_dcsr, t_cusp, cusp_algo,
       t_mm_sm_csc, t_mm_csr, t_mm_csc, t_mm_dcsr, t_mm_cusp, cusp_mm_algo,
       ok_spmv = ok_csr && ok_csc && ok_dcsr, ok_mm,
       e_csr, e_csc, e_dcsr, e_mm_csr, e_mm_csc, e_mm_dcsr)
end

# ── Driver ────────────────────────────────────────────────────────────────────

const _backend_label =
    HAS_CUDA  ? "CUDA ($(CUDA.name(CUDA.device())))" :
    HAS_METAL ? "Metal ($(Metal.current_device().name))" : "CPU EmitterBackend"

println("="^96)
println("JLUST SuiteSparse benchmark   (FloatType=$(FloatType), backend=$(_backend_label))")
println("Cache: $(_SS_CACHE)")
println("="^96)

_only = get(ENV, "JLUST_BENCH_ONLY", "")
const cases = isempty(_only) ? CURATED :
    [c for c in CURATED if "$(c.group)/$(c.name)" == _only ||
                              c.name == _only ||
                              c.group == _only]
isempty(cases) && error("JLUST_BENCH_ONLY=$_only matched no curated case")

results = Tuple{Case, NamedTuple}[]
for c in cases
    @printf("  %-12s / %-22s  [%s]\n", c.group, c.name, c.category)
    flush(stdout)
    r = bench_matrix(c)
    push!(results, (c, r))
end

# ── Output ────────────────────────────────────────────────────────────────────

_fmt(x) = isnan(x) ? "    — " : @sprintf("%6.1f", x)
function _bw_gbs(t_us, m, n, nnz, T::Type)
    isnan(t_us) && return "  —  "
    bytes = spmv_bytes(m, n, nnz, T)
    @sprintf("%5.1f", bytes / (t_us * 1e3))   # µs → ns;  bytes/ns = GB/s
end
_speed(num, den) = (isnan(num) || isnan(den) || den == 0) ? "  —  " :
                    @sprintf("%5.2f×", num / den)

println()
println("─── SpMV (μs) ───────────────────────────────────────────────────────────────────────────────")
@printf("  %-30s %7s %9s %5s │ %6s %6s %6s %6s %6s %4s │ %6s %6s\n",
        "matrix", "rows", "nnz", "nz/r", "spCSC", "JL CSR", "JL CSC", "JL DCSR", "cuSP", "alg",
        "CSR↑", "DCSR↑")
println("  " * "─"^110)
for (c, r) in results
    label = "$(c.group)/$(c.name)"
    @printf("  %-30s %7s %9s %5s │ %6s %6s %6s %6s %6s %4s │ %6s %6s\n",
            length(label) > 30 ? label[1:27]*"…" : label,
            (@sprintf "%d" r.m),
            (@sprintf "%d" r.nnz),
            (@sprintf "%4.1f" r.avg),
            _fmt(r.t_sm_csc), _fmt(r.t_jl_csr), _fmt(r.t_jl_csc),
            _fmt(r.t_jl_dcsr), _fmt(r.t_cusp), r.cusp_algo,
            _speed(HAS_CUDA ? r.t_cusp : r.t_sm_csc, r.t_jl_csr),
            _speed(HAS_CUDA ? r.t_cusp : r.t_sm_csc, r.t_jl_dcsr))
end

println()
println("─── SpMV effective bandwidth (GB/s, JLUST CSR) ──────────────────────────────────────────────")
@printf("  %-30s %7s %9s │ %6s %6s %6s %6s %6s\n",
        "matrix", "rows", "nnz", "spCSC", "JL CSR", "JL CSC", "JL DCSR", "cuSP")
println("  " * "─"^88)
for (c, r) in results
    label = "$(c.group)/$(c.name)"
    @printf("  %-30s %7s %9s │ %6s %6s %6s %6s %6s\n",
            length(label) > 30 ? label[1:27]*"…" : label,
            (@sprintf "%d" r.m), (@sprintf "%d" r.nnz),
            _bw_gbs(r.t_sm_csc, r.m, r.n, r.nnz, FloatType),
            _bw_gbs(r.t_jl_csr,  r.m, r.n, r.nnz, FloatType),
            _bw_gbs(r.t_jl_csc,  r.m, r.n, r.nnz, FloatType),
            _bw_gbs(r.t_jl_dcsr, r.m, r.n, r.nnz, FloatType),
            _bw_gbs(r.t_cusp,    r.m, r.n, r.nnz, FloatType))
end

println()
println("─── SpMM k=$(SPMM_K) (μs) ─────────────────────────────────────────────────────────────────────────")
@printf("  %-30s %7s %9s │ %7s %7s %7s %7s %7s %4s │ %6s\n",
        "matrix", "rows", "nnz", "spCSC", "JL CSR", "JL CSC", "JL DCSR", "cuSP", "alg", "CSR↑")
println("  " * "─"^110)
for (c, r) in results
    label = "$(c.group)/$(c.name)"
    @printf("  %-30s %7s %9s │ %7s %7s %7s %7s %7s %4s │ %6s\n",
            length(label) > 30 ? label[1:27]*"…" : label,
            (@sprintf "%d" r.m), (@sprintf "%d" r.nnz),
            _fmt(r.t_mm_sm_csc), _fmt(r.t_mm_csr), _fmt(r.t_mm_csc),
            _fmt(r.t_mm_dcsr),   _fmt(r.t_mm_cusp), r.cusp_mm_algo,
            _speed(HAS_CUDA ? r.t_mm_cusp : r.t_mm_sm_csc, r.t_mm_csr))
end

# Correctness with per-format relerr.  ✗ at 1e-3 is a real concern; in
# (1e-4, 1e-3] is usually FP-accumulation-order noise on dense rows.
println()
println("─── Correctness (relerr vs SparseMatrixCSC ref) ─────────────────────────────────────────────")
@printf("  %-30s │ %10s %10s %10s │ %10s %10s %10s\n",
        "matrix", "SpMV CSR", "SpMV CSC", "SpMV DCSR",
        "SpMM CSR", "SpMM CSC", "SpMM DCSR")
println("  " * "─"^104)
for (c, r) in results
    label = "$(c.group)/$(c.name)"
    @printf("  %-30s │ %10.2e %10.2e %10.2e │ %10.2e %10.2e %10.2e\n",
            length(label) > 30 ? label[1:27]*"…" : label,
            r.e_csr, r.e_csc, r.e_dcsr,
            r.e_mm_csr, r.e_mm_csc, r.e_mm_dcsr)
end

println()
println("  CSR↑   = baseline ÷ JLUST-CSR.  Baseline is best-of-{DEF,A1,A2(,A3)} cuSPARSE on GPU,")
println("           SparseMatrixCSC on CPU.")
println("  alg    = which cuSPARSE algo won — DEF=ALG_DEFAULT, A1/A2/A3=CSR_ALG{1,2,3}.")
println("  DCSR↑  = same, for JLUST-DCSR.  Wins when many rows are empty.")
println("  GB/s   = nnz·($(sizeof(FloatType))+4) + (m+1)·4 + (m+n)·$(sizeof(FloatType))  divided by SpMV time.")
println("  relerr = ‖y − y_ref‖ / ‖y_ref‖.  >1e-4 typically means FP-accumulation order;")
println("           >1e-3 is suspect (real numerical or correctness issue).")
