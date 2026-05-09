# sweep_cases.jl — JLUST performance across network sizes
#
# Benchmarks the DCOPF BlockSparseMatrix and multi-stage BlockBandedMatrix
# across six pglib-opf cases ranging from 1354 to 30000 buses.  Cases are
# downloaded on first run and cached in .cache/pglib/.
#
# Usage:
#   cd ~/bench/JLUST && julia --project=benchmark benchmark/sweep_cases.jl

using Printf, Random, Downloads
using JLUST, JLUST.Formats
using SparseArrays, LinearAlgebra
using KernelAbstractions
using BenchmarkTools

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

const FloatType = HAS_METAL ? Float32 : Float64

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

function load_pglib(case::String)
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
    b_k = FloatType.(ifelse.(abs.(br[:, 4]) .< 1e-10, zero(FloatType), one(FloatType) ./ br[:, 4]))

    Bf = sparse([1:n_line; 1:n_line], [from_b; to_b],
                FloatType.([b_k; -b_k]), n_line, n_bus)

    gen_b = [id2i[Int(b)] for b in gen[:, 1]]
    Cg    = sparse(gen_b, 1:n_gen, ones(FloatType, n_gen), n_bus, n_gen)

    d_b = zeros(FloatType, n_bus)
    for k in 1:n_line; d_b[from_b[k]] += b_k[k]; d_b[to_b[k]] += b_k[k]; end
    Bbus = sparse([from_b; to_b; 1:n_bus], [to_b; from_b; 1:n_bus],
                  FloatType.([-b_k; -b_k; d_b]), n_bus, n_bus)

    I_l = sparse(FloatType(1)*I, n_line, n_line)
    (; Cg, Bf, Bbus, negI=-I_l, n_bus, n_line, n_gen)
end

# ── benchmark one case ────────────────────────────────────────────────────────

function bench_case(case::String; T::Int=24, n_s2::Int=100, n_s3::Int=20)
    d = load_pglib(case)
    (; Cg, Bf, Bbus, negI, n_bus, n_line, n_gen) = d

    # §2: BlockSparseMatrix
    u_Cg   = csr_tensor(Cg;    device=to_device)
    u_Bbus = csr_tensor(-Bbus; device=to_device)
    u_negI = csr_tensor(negI;  device=to_device)
    u_Bf   = csr_tensor(Bf;    device=to_device)
    BM = BlockSparseMatrix([
        u_Cg   nothing  u_Bbus;
        nothing u_negI  u_Bf
    ])

    n_var = n_gen + n_line + n_bus
    n_con = n_bus + n_line
    x_bm  = to_device(randn(n_var))
    y_bm  = similar(x_bm, n_con)

    for _ in 1:3; mul!(y_bm, BM, x_bm); sync(); end
    t_jlust_s2 = @belapsed(begin mul!($y_bm, $BM, $x_bm); $sync() end,
                            samples=n_s2, evals=1) * 1e6

    # §2: CUDA Graph (best-effort; some graphs fail to relaunch on certain CUDA stacks)
    t_graph_s2 = if HAS_CUDA
        try
            g = CUDA.capture() do; mul!(y_bm, BM, x_bm); end
            e = CUDA.instantiate(g)
            CUDA.launch(e); CUDA.synchronize()
            @belapsed(begin CUDA.launch($e); CUDA.synchronize() end,
                      samples=n_s2, evals=1) * 1e6
        catch err
            @warn "§2 CUDA Graph skipped" exception=(err, catch_backtrace())
            NaN
        end
    else
        NaN
    end

    # §2: cuSPARSE handle reference (prepared, pre-analyzed)
    A_cpu = vcat(
        hcat(Cg,                        spzeros(n_bus, n_line), -Bbus),
        hcat(spzeros(n_line, n_gen),    negI,                    Bf))
    t_cusparse_s2 = if HAS_CUDA
        u_A_full = ust(CUDA.CUSPARSE.CuSparseMatrixCSR(FloatType.(A_cpu)))
        u_x_h    = ust(CUDA.CuArray(Vector{FloatType}(x_bm)))
        u_y_h    = ust(CUDA.zeros(FloatType, n_con))
        h_sp2    = prepare(CUSPARSEBackend(), SpMVOp, u_A_full)
        execute(h_sp2, u_x_h, u_y_h); CUDA.synchronize()
        @belapsed(begin execute($h_sp2, $u_x_h, $u_y_h); CUDA.synchronize() end,
                  samples=n_s2, evals=1) * 1e6
    else
        NaN
    end

    # §3: BlockBandedMatrix
    R_cpu  = sparse(1:n_gen, 1:n_gen, ones(n_gen), n_gen, n_var)
    u_posR = csr_tensor(R_cpu;  device=to_device)
    u_negR = csr_tensor(-R_cpu; device=to_device)

    n_rmp    = n_gen
    n_row_ms = T*n_con + (T-1)*n_rmp
    n_col_ms = T*n_var
    BM_ms    = BlockBandedMatrix(BM, u_negR, u_posR, T, n_con, n_rmp, n_var)

    x_ms = to_device(randn(n_col_ms))
    y_ms = similar(x_ms, n_row_ms)

    mul!(y_ms, BM_ms, x_ms)
    for _ in 1:3; mul!(y_ms, BM_ms, x_ms); sync(); end
    t_jlust_s3 = @belapsed(begin mul!($y_ms, $BM_ms, $x_ms); $sync() end,
                            samples=n_s3, evals=1) * 1e6

    # §3: CUDA Graph (best-effort)
    t_graph_s3 = if HAS_CUDA
        try
            g = CUDA.capture() do; mul!(y_ms, BM_ms, x_ms); end
            e = CUDA.instantiate(g)
            CUDA.launch(e); CUDA.synchronize()
            @belapsed(begin CUDA.launch($e); CUDA.synchronize() end,
                      samples=n_s3, evals=1) * 1e6
        catch err
            @warn "§3 CUDA Graph skipped" exception=(err, catch_backtrace())
            NaN
        end
    else
        NaN
    end

    # §3: cuSPARSE handle on the full T-period block-banded CSR — apples-to-apples
    # vs the BBM mul!.  Builds the exact banded matrix BBM_ms represents:
    #   [D₁; (-R | +R); D₂; (-R | +R); …; D_T]   (D_t = A_cpu, off-diag = ±R coupling)
    t_cusparse_s3 = if HAS_CUDA
        chunks = SparseMatrixCSC{FloatType,Int}[]
        for t in 1:T
            row = spzeros(FloatType, n_con, n_col_ms)
            row[:, (t-1)*n_var+1 : t*n_var] = sparse(A_cpu)
            push!(chunks, row)
            if t < T
                off = spzeros(FloatType, n_rmp, n_col_ms)
                off[:, (t-1)*n_var+1 : t*n_var]     = -R_cpu
                off[:, t*n_var+1     : (t+1)*n_var] =  R_cpu
                push!(chunks, off)
            end
        end
        BBM_full = vcat(chunks...)

        u_A_bbm = ust(CUDA.CUSPARSE.CuSparseMatrixCSR(BBM_full))
        u_xb    = ust(CUDA.randn(FloatType, n_col_ms))
        u_yb    = ust(CUDA.zeros(FloatType, n_row_ms))
        h_sp3   = prepare(CUSPARSEBackend(), SpMVOp, u_A_bbm)
        execute(h_sp3, u_xb, u_yb); CUDA.synchronize()
        @belapsed(begin execute($h_sp3, $u_xb, $u_yb); CUDA.synchronize() end,
                  samples=n_s3, evals=1) * 1e6
    else
        NaN
    end

    # correctness check: JLUST §2 output vs sparse CPU reference
    ok = let x_ref = Vector{Float64}(x_bm),
             y_ref  = Vector{Float64}(A_cpu * x_ref),
             y_got  = Vector{Float64}(y_bm)
        norm(y_got .- y_ref) / (norm(y_ref) + 1e-30) < 1e-6
    end

    (; n_bus, n_line, n_gen,
       t_cusparse_s2, t_graph_s2, t_jlust_s2,
       t_cusparse_s3, t_graph_s3, t_jlust_s3,
       ok)
end

# ── main ──────────────────────────────────────────────────────────────────────

const CASES = [
    "pglib_opf_case1354_pegase",
    "pglib_opf_case2869_pegase",
    "pglib_opf_case6470_rte",
    "pglib_opf_case9241_pegase",
    "pglib_opf_case13659_pegase",
    "pglib_opf_case30000_goc",
]

const _backend = HAS_CUDA  ? "CUDA ($(CUDA.name(CUDA.device())))" :
                 HAS_METAL ? "Metal ($(Metal.current_device().name))" :
                 "CPU"

println("="^72)
println("JLUST multi-case sweep   (T=24, $(FloatType), $_backend)")
println("="^72)
println()
@printf("  %-18s  %6s  │  §2 single DCOPF (μs)                │  §3 per period (μs)\n",
        "Case", "buses")
@printf("  %-18s  %6s  │  %8s  %8s  %8s  %7s  │  %8s  %8s  %8s  %7s  %s\n",
        "", "", "cuSP-h", "JLUST", "Graph", "speedup",
                "cuSP-h", "JLUST", "Graph", "speedup", "ok")
println("  " * "─"^88)

_fmt_or_dash(t) = isnan(t) ? "      — " : @sprintf("%8.2f", t)
function _speedup(num, den)
    (isnan(num) || isnan(den) || den == 0) && return "    —  "
    @sprintf("%5.2f×", num / den)
end

results = Pair{String,Any}[]
for case in CASES
    short = replace(case, "pglib_opf_case" => "")
    print("  $(rpad(short, 18))  ")
    flush(stdout)
    r = bench_case(case)
    push!(results, case => r)

    sp2_jl    = _speedup(r.t_cusparse_s2, r.t_graph_s2)
    sp3_jl    = _speedup(r.t_cusparse_s3, r.t_graph_s3)
    ok        = r.ok ? "✓" : "✗"

    @printf("%6d  │  %s  %s  %s  %7s  │  %s  %s  %s  %7s  %s\n",
            r.n_bus,
            _fmt_or_dash(r.t_cusparse_s2),
            _fmt_or_dash(r.t_jlust_s2),
            _fmt_or_dash(r.t_graph_s2),
            sp2_jl,
            _fmt_or_dash(r.t_cusparse_s3 / 24),
            _fmt_or_dash(r.t_jlust_s3   / 24),
            _fmt_or_dash(r.t_graph_s3   / 24),
            sp3_jl,
            ok)
    flush(stdout)
end

println("  " * "─"^88)
println()
println("  cuSP-h  = cuSPARSE handle on the full T-period block-banded CSR.")
println("  JLUST   = LinearAlgebra.mul! through JLUST (direct dispatch, no graph).")
println("  Graph   = JLUST captured into a CUDA Graph (NaN if the stack rejects relaunch).")
println("  speedup = cuSP-h / Graph (NaN when graph capture is unavailable).")
