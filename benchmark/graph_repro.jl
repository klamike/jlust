# Minimal repro: does cuSPARSE inside CUDA Graph fail to relaunch?
using JLUST, JLUST.Formats, KernelAbstractions, SparseArrays, CUDA, LinearAlgebra

A = sprand(Float64, 5000, 5000, 0.001)
B = randn(Float64, 5000, 8)
u_A = csr_tensor(A; device=CuArray)
u_B = JLUST.ust(CUDA.CuArray(B))
C   = CUDA.zeros(Float64, 5000, 8)
u_C = JLUST.ust(C)

# 1) Pure emitter SpMM in a graph — does relaunch work?
println("--- Test 1: emitter SpMM in graph ---")
mul!(C, u_A, JLUST.nonzeros(u_B), 1.0, 0.0)  # warm
g = CUDA.capture() do
    JLUST.execute(JLUST.SpMMOp, u_A, u_B, u_C; backend=JLUST.EmitterBackend(), beta=0.0)
end
e = CUDA.instantiate(g)
CUDA.launch(e); CUDA.synchronize()
println("  warm launch ok")
try
    for i in 1:5
        CUDA.launch(e); CUDA.synchronize()
    end
    println("  5 relaunches ok ✓")
catch err
    println("  ✗ relaunch failed: ", err)
end

# 2) cuSPARSE SpMM in a graph — does relaunch work?
println("--- Test 2: cuSPARSE SpMM in graph ---")
JLUST.execute(JLUST.SpMMOp, u_A, u_B, u_C; backend=JLUST.CUSPARSEBackend(), beta=0.0); CUDA.synchronize()
g2 = CUDA.capture() do
    JLUST.execute(JLUST.SpMMOp, u_A, u_B, u_C; backend=JLUST.CUSPARSEBackend(), beta=0.0)
end
e2 = CUDA.instantiate(g2)
CUDA.launch(e2); CUDA.synchronize()
println("  warm launch ok")
try
    for i in 1:5
        CUDA.launch(e2); CUDA.synchronize()
    end
    println("  5 relaunches ok ✓")
catch err
    println("  ✗ relaunch failed: ", err)
end

# 3) cuSPARSE handle path — same descriptor reused across calls, NO prepare in graph
println("--- Test 3: cuSPARSE handle (preprepared, only execute in graph) ---")
h = JLUST.prepare(JLUST.CUSPARSEBackend(), JLUST.SpMMOp, u_A; transa='N', transb='N', n_cols=size(JLUST.nonzeros(u_B), 2))
JLUST.execute(h, u_B, u_C; beta=0.0); CUDA.synchronize()
g3 = CUDA.capture() do
    JLUST.execute(h, u_B, u_C; beta=0.0)
end
e3 = CUDA.instantiate(g3)
CUDA.launch(e3); CUDA.synchronize()
println("  warm launch ok")
try
    for i in 1:5
        CUDA.launch(e3); CUDA.synchronize()
    end
    println("  5 relaunches ok ✓")
catch err
    println("  ✗ relaunch failed: ", err)
end

# 4) High-level CUSPARSE.mm! directly — no JLUST layer
println("--- Test 4: bare CUSPARSE.mm! in graph ---")
cusA = JLUST._to_cuspmat(u_A)  # extract once
B_dense = JLUST.nonzeros(u_B)
CUDA.CUSPARSE.mm!('N', 'N', 1.0, cusA, B_dense, 0.0, C, 'O'); CUDA.synchronize()
g4 = CUDA.capture() do
    CUDA.CUSPARSE.mm!('N', 'N', 1.0, cusA, B_dense, 0.0, C, 'O')
end
e4 = CUDA.instantiate(g4)
CUDA.launch(e4); CUDA.synchronize()
println("  warm launch ok")
try
    for i in 1:5
        CUDA.launch(e4); CUDA.synchronize()
    end
    println("  5 relaunches ok ✓")
catch err
    println("  ✗ relaunch failed: ", err)
end
