using Test, JLUST, JLUST.Formats, SparseArrays

# All testsets in this file are gated behind gpu_available().
# @testset skip= was removed in Julia 1.12; use if guards instead.

if gpu_available()

using CUDA, CUDA.CUSPARSE

# CUSPARSEBackend is defined in the CUDAExt package extension; bring it into
# the test namespace the same way EmitterBackend is brought in test_emitter.jl.
const _cudaext = Base.get_extension(JLUST, :CUDAExt)
const CUSPARSEBackend = _cudaext.CUSPARSEBackend

@testset "materialize CPU→GPU" begin
    u_cpu = csr_tensor(Int32[0,2,3,5], Int32[0,2,1,0,2], Float32[1,2,3,4,5], (3,4);
                       origin=ZeroBased())
    u_gpu = materialize(u_cpu; device=CUDADevice(0))
    @test memory_space(u_gpu) == GPUMemory()
    @test index_origin(u_gpu) == ZeroBased()
    @test format(u_gpu) == Formats.CSR
    @test Array(positions(u_gpu, 2))    == positions(u_cpu, 2)
    @test Array(coordinates(u_gpu, 2)) == coordinates(u_cpu, 2)
    @test Array(nonzeros(u_gpu))       ≈  nonzeros(u_cpu)
end

@testset "materialize GPU→CPU round-trip" begin
    u_cpu  = csr_tensor(Int32[0,2,3,5], Int32[0,2,1,0,2], Float32[1,2,3,4,5], (3,4);
                        origin=ZeroBased())
    u_gpu  = materialize(u_cpu; device=CUDADevice(0))
    u_back = materialize(u_gpu; device=CPUDevice())
    @test memory_space(u_back) == CPUMemory()
    @test index_origin(u_back) == ZeroBased()
    @test positions(u_back, 2)    == positions(u_cpu, 2)
    @test coordinates(u_back, 2) == coordinates(u_cpu, 2)
    @test nonzeros(u_back)       ≈  nonzeros(u_cpu)
end

@testset "CSC GPU round-trip preserves format" begin
    u_csc = convert_format(
        csr_tensor(Int32[0,2,3,5], Int32[0,2,1,0,2], Float32[1,2,3,4,5], (3,4);
                   origin=ZeroBased()),
        Formats.CSC)
    u_gpu  = materialize(u_csc; device=CUDADevice(0))
    u_back = materialize(u_gpu; device=CPUDevice())
    @test format(u_back) == Formats.CSC          # must not silently become CSR
    @test positions(u_back, 2)    == positions(u_csc, 2)
    @test coordinates(u_back, 2) == coordinates(u_csc, 2)
    @test nonzeros(u_back)       ≈  nonzeros(u_csc)
end

@testset "materialize preserves origin by default" begin
    A     = sparse([1,1,2,3,3], [1,3,2,1,3], Float32[1,2,3,4,5], 3, 4)
    u_one = ust(A)                              # OneBased CSC view
    u_gpu = materialize(u_one; device=CUDADevice(0))
    @test index_origin(u_gpu) == OneBased()     # unchanged by default
    u_zero = materialize(u_one; device=CUDADevice(0), origin=ZeroBased())
    @test index_origin(u_zero) == ZeroBased()   # explicit change honored
    @test Array(positions(u_zero, 2)) == positions(u_one, 2) .- Int64(1)
    @test Array(coordinates(u_zero, 2)) == coordinates(u_one, 2) .- Int64(1)
end

@testset "CuSparseMatrixCSR adapter" begin
    A    = sparse([1,1,2,3,3], [1,3,2,1,3], Float32[1,2,3,4,5], 3, 4)
    Agpu = CuSparseMatrixCSR(A)
    u    = ust(Agpu)
    # CUDA.jl uses OneBased (verified from source: conversions.jl:353, array.jl:411)
    @test index_origin(u) == OneBased()
    @test format(u) == Formats.CSR
    @test size(u) == (3, 4)
    @test nnz(u) == nnz(A)
    @test memory_space(u) == GPUMemory()
end

@testset "CuSparseMatrixCSC adapter" begin
    A    = sparse([1,1,2,3,3], [1,3,2,1,3], Float32[1,2,3,4,5], 3, 4)
    Agpu = CuSparseMatrixCSC(CuSparseMatrixCSR(A))
    u    = ust(Agpu)
    @test index_origin(u) == OneBased()
    @test format(u) == Formats.CSC
    @test size(u) == (3, 4)
    @test nnz(u) == nnz(A)
end

@testset "CuSparseMatrixBSR adapter" begin
    A    = sparse([1,1,2,3,3], [1,3,2,1,3], Float32[1,2,3,4,5], 4, 4)
    Agpu = CuSparseMatrixBSR(CuSparseMatrixCSR(A), Int32(2))
    u    = ust(Agpu)
    @test index_origin(u) == OneBased()
    bsz  = (Int(Agpu.blockDim), Int(Agpu.blockDim))
    expected_fmt = Agpu.dir == 'R' ? Formats.BSRRight(bsz) : Formats.BSRLeft(bsz)
    @test format(u) == expected_fmt
    @test size(u) == (4, 4)
    # BSR stores full blocks: nnz(u) == nnzb * blockDim^2, not structural nnz(A)
    @test nnz(u) == nnz(Agpu)
    @test memory_space(u) == GPUMemory()
end

@testset "CuSparseMatrixCOO adapter" begin
    A    = sparse([1,1,2,3,3], [1,3,2,1,3], Float32[1,2,3,4,5], 3, 4)
    Agpu = CuSparseMatrixCOO(CuSparseMatrixCSR(A))
    u    = ust(Agpu)
    @test index_origin(u) == OneBased()
    @test format(u) == Formats.COO
    @test size(u) == (3, 4)
    @test nnz(u) == nnz(A)
end

@testset "ust(CuArray) dense adapter" begin
    A = CUDA.rand(Float32, 4, 4)
    u = ust(A)
    @test memory_space(u) == GPUMemory()
    @test nnz(u) == 16
    @test ndims(u) == 2
    @test size(u) == (4, 4)
end

@testset "validate_storage base" begin
    A = sparse([1,1,2,3,3], [1,3,2,1,3], Float32[1,2,3,4,5], 3, 4)
    u = ust(CuSparseMatrixCSR(A))
    op = SpMMOp(Formats.CSR, Formats.DensedRight(2), Formats.DensedRight(2))
    @test (validate_storage(u, CUSPARSEBackend(); op); true)
end

@testset "supports_backend BSR" begin
    bsr2 = Formats.BSRRight((2, 2))
    d1   = Formats.DensedRight(1)
    d2   = Formats.DensedRight(2)
    # SpMV: BSR via legacy bsrmv
    @test  supports_backend(CUSPARSEBackend(), SpMVOp(bsr2, d1, d1))
    # SpMM: BSR via generic ≥12.5.1
    @test  supports_backend(CUSPARSEBackend(), SpMMOp(bsr2, d2, d2))
    # SpSV: BSR via legacy bsrsv2
    @test  supports_backend(CUSPARSEBackend(), SpSVOp(bsr2, d1, d1))
    # SparseToDense: BSR supported
    @test  supports_backend(CUSPARSEBackend(), SparseToDenseOp(bsr2))
    # SDDMM: BSR as C (result/mask) supported ≥12.1.0
    @test  supports_backend(CUSPARSEBackend(), SDDMMOp(d2, d2, bsr2))
    # DenseToSparse: BSR not supported
    @test !supports_backend(CUSPARSEBackend(), DenseToSparseOp(bsr2))
    # SpSM: BSR not supported by generic SpSM
    @test !supports_backend(CUSPARSEBackend(), SpSMOp(bsr2, d2, d2))
end

@testset "supports_convert BSR" begin
    bsr2 = Formats.BSRRight((2, 2))
    @test  supports_convert(CUSPARSEBackend(), Formats.CSR, bsr2)   # csr2bsr
    @test  supports_convert(CUSPARSEBackend(), bsr2, Formats.CSR)   # bsr2csr
    @test !supports_convert(CUSPARSEBackend(), Formats.COO, bsr2)
    @test !supports_convert(CUSPARSEBackend(), bsr2, Formats.CSC)
end

@testset "supports_backend SELL" begin
    sell2 = Formats.SELL(2)
    d1    = Formats.DensedRight(1)
    d2    = Formats.DensedRight(2)
    # SpMV: SELL via SPMV_SELL_ALG1
    @test  supports_backend(CUSPARSEBackend(), SpMVOp(sell2, d1, d1))
    # SpMM: SELL not supported
    @test !supports_backend(CUSPARSEBackend(), SpMMOp(sell2, d2, d2))
end

@testset "supports_backend BlockedELL" begin
    bell4 = Formats.BlockedELL(4)
    d1    = Formats.DensedRight(1)
    d2    = Formats.DensedRight(2)
    # SpMM: BlockedELL via SPMM_BLOCKED_ELL_ALG1
    @test  supports_backend(CUSPARSEBackend(), SpMMOp(bell4, d2, d2))
    # SpMV: BlockedELL not supported
    @test !supports_backend(CUSPARSEBackend(), SpMVOp(bell4, d1, d1))
end

@testset "supports_backend SpVV" begin
    d1 = Formats.DensedRight(1)
    @test  supports_backend(CUSPARSEBackend(), SpVVOp(Formats.CSR, d1))
    @test  supports_backend(CUSPARSEBackend(), SpVVOp(Formats.COO, d1))
    @test !supports_backend(CUSPARSEBackend(), SpVVOp(Formats.CSR, Formats.CSR))
    @test !supports_backend(CUSPARSEBackend(), SpVVOp(Formats.DCSR, d1))
end

@testset "supports_backend SpMV" begin
    d1 = Formats.DensedRight(1)
    @test  supports_backend(CUSPARSEBackend(), SpMVOp(Formats.CSR, d1, d1))
    @test  supports_backend(CUSPARSEBackend(), SpMVOp(Formats.CSC, d1, d1))
    @test  supports_backend(CUSPARSEBackend(), SpMVOp(Formats.COO, d1, d1))
    @test !supports_backend(CUSPARSEBackend(), SpMVOp(Formats.DCSR, d1, d1))
    # wrong vector format
    @test !supports_backend(CUSPARSEBackend(), SpMVOp(Formats.CSR, Formats.CSR, d1))
end

@testset "supports_backend SpMM" begin
    d2 = Formats.DensedRight(2)
    @test  supports_backend(CUSPARSEBackend(), SpMMOp(Formats.CSR, d2, d2))
    @test  supports_backend(CUSPARSEBackend(), SpMMOp(Formats.CSC, d2, d2))
    @test  supports_backend(CUSPARSEBackend(), SpMMOp(Formats.COO, d2, d2))
    @test !supports_backend(CUSPARSEBackend(), SpMMOp(Formats.DCSR, d2, d2))
    @test !supports_backend(CUSPARSEBackend(), SpMMOp(Formats.DCSC, d2, d2))
    @test !supports_backend(CUSPARSEBackend(), SpMMOp(Formats.CSR, d2, Formats.CSR))
end

@testset "supports_backend BatchedSpMM" begin
    d2 = Formats.DensedRight(2)
    @test  supports_backend(CUSPARSEBackend(), BatchedSpMMOp(Formats.CSR, d2, d2))
    @test !supports_backend(CUSPARSEBackend(), BatchedSpMMOp(Formats.CSC, d2, d2))
end

@testset "supports_backend SpGEMM" begin
    @test  supports_backend(CUSPARSEBackend(), SpGEMMOp(Formats.CSR, Formats.CSR, Formats.CSR))
    @test !supports_backend(CUSPARSEBackend(), SpGEMMOp(Formats.CSC, Formats.CSC, Formats.CSC))
    @test !supports_backend(CUSPARSEBackend(), SpGEMMOp(Formats.CSR, Formats.CSC, Formats.CSR))
end

@testset "supports_backend SpSV" begin
    d1 = Formats.DensedRight(1)
    @test  supports_backend(CUSPARSEBackend(), SpSVOp(Formats.CSR, d1, d1))
    @test  supports_backend(CUSPARSEBackend(), SpSVOp(Formats.CSC, d1, d1))
    @test !supports_backend(CUSPARSEBackend(), SpSVOp(Formats.COO, d1, d1))
    @test !supports_backend(CUSPARSEBackend(), SpSVOp(Formats.CSR, Formats.CSR, d1))
end

@testset "supports_backend SpSM" begin
    d2 = Formats.DensedRight(2)
    @test  supports_backend(CUSPARSEBackend(), SpSMOp(Formats.CSR, d2, d2))
    @test  supports_backend(CUSPARSEBackend(), SpSMOp(Formats.CSC, d2, d2))
    @test !supports_backend(CUSPARSEBackend(), SpSMOp(Formats.COO, d2, d2))
    @test !supports_backend(CUSPARSEBackend(), SpSMOp(Formats.CSR, d2, Formats.CSR))
end

@testset "supports_backend SDDMM" begin
    d2 = Formats.DensedRight(2)
    @test  supports_backend(CUSPARSEBackend(), SDDMMOp(d2, d2, Formats.CSR))
    @test  supports_backend(CUSPARSEBackend(), SDDMMOp(d2, d2, Formats.COO))
    @test !supports_backend(CUSPARSEBackend(), SDDMMOp(d2, d2, Formats.CSC))
    @test !supports_backend(CUSPARSEBackend(), SDDMMOp(Formats.CSR, d2, Formats.CSR))
end

@testset "supports_backend SparseToDense / DenseToSparse" begin
    @test  supports_backend(CUSPARSEBackend(), SparseToDenseOp(Formats.CSR))
    @test  supports_backend(CUSPARSEBackend(), SparseToDenseOp(Formats.CSC))
    @test  supports_backend(CUSPARSEBackend(), SparseToDenseOp(Formats.COO))
    @test !supports_backend(CUSPARSEBackend(), SparseToDenseOp(Formats.DCSR))
    @test  supports_backend(CUSPARSEBackend(), DenseToSparseOp(Formats.CSR))
    @test  supports_backend(CUSPARSEBackend(), DenseToSparseOp(Formats.CSC))
    @test  supports_backend(CUSPARSEBackend(), DenseToSparseOp(Formats.COO))
    @test !supports_backend(CUSPARSEBackend(), DenseToSparseOp(Formats.DCSR))
end

@testset "supports_backend vector ops" begin
    d1 = Formats.DensedRight(1)
    @test  supports_backend(CUSPARSEBackend(), GatherOp(Formats.COO))
    @test !supports_backend(CUSPARSEBackend(), GatherOp(Formats.CSR))
    @test  supports_backend(CUSPARSEBackend(), ScatterOp(Formats.COO))
    @test !supports_backend(CUSPARSEBackend(), ScatterOp(Formats.CSR))
    @test  supports_backend(CUSPARSEBackend(), AxpbyOp(Formats.COO, d1))
    @test !supports_backend(CUSPARSEBackend(), AxpbyOp(Formats.CSR, d1))
    @test  supports_backend(CUSPARSEBackend(), RotOp(Formats.COO, d1))
    @test !supports_backend(CUSPARSEBackend(), RotOp(Formats.CSR, d1))
end

@testset "supports_convert" begin
    @test  supports_convert(CUSPARSEBackend(), Formats.CSR, Formats.CSC)
    @test  supports_convert(CUSPARSEBackend(), Formats.CSC, Formats.CSR)
    @test  supports_convert(CUSPARSEBackend(), Formats.CSR, Formats.COO)
    @test  supports_convert(CUSPARSEBackend(), Formats.COO, Formats.CSR)
    @test !supports_convert(CUSPARSEBackend(), Formats.DCSR, Formats.CSR)
    @test !supports_convert(CUSPARSEBackend(), Formats.CSR, Formats.DCSR)
end


# ─── SpMV execution (CUSPARSEBackend direct path) ─────────────────────────────

@testset "sparse_mv! CUSPARSEBackend CSR" begin
    # A = [1 0 2; 0 3 0; 4 0 5],  x = [1, 2, 3]  →  y = [7, 6, 19]
    A    = sparse([1,1,2,3,3], [1,3,2,1,3], Float32[1,2,3,4,5], 3, 3)
    Agpu = CuSparseMatrixCSR(A)
    u_A  = ust(Agpu)
    u_x  = ust(CuArray(Float32[1, 2, 3]))
    u_y  = ust(CUDA.zeros(Float32, 3))

    execute(SpMVOp, u_A, u_x, u_y; backend=CUSPARSEBackend())
    @test Array(nonzeros(u_y)) ≈ Float32[7, 6, 19]
end

@testset "sparse_mv! CUSPARSEBackend CSR handle" begin
    A    = sparse([1,1,2,3,3], [1,3,2,1,3], Float32[1,2,3,4,5], 3, 3)
    Agpu = CuSparseMatrixCSR(A)
    u_A  = ust(Agpu)
    u_x  = ust(CuArray(Float32[1, 2, 3]))
    u_y  = ust(CUDA.zeros(Float32, 3))

    h = prepare(CUSPARSEBackend(), SpMVOp, u_A)
    execute(SpMVOp, h, u_x, u_y)
    @test Array(nonzeros(u_y)) ≈ Float32[7, 6, 19]

    # Second call reuses handle — update values and rerun
    fill!(nonzeros(u_y), 0f0)
    update_values!(h, u_A)
    execute(SpMVOp, h, u_x, u_y)
    @test Array(nonzeros(u_y)) ≈ Float32[7, 6, 19]
end

# ─── SpMM execution (CUSPARSEBackend direct path) ─────────────────────────────

@testset "sparse_mm! CUSPARSEBackend CSR" begin
    # A = [1 0 2; 0 3 0; 4 0 5], B = [1 0; 2 1; 0 3]  →  C = [1 6; 6 3; 4 15]
    A    = sparse([1,1,2,3,3], [1,3,2,1,3], Float32[1,2,3,4,5], 3, 3)
    Agpu = CuSparseMatrixCSR(A)
    u_A  = ust(Agpu)
    u_B  = ust(CuArray(Float32[1 0; 2 1; 0 3]))
    u_C  = ust(CUDA.zeros(Float32, 3, 2))

    execute(SpMMOp, u_A, u_B, u_C; backend=CUSPARSEBackend())
    @test Array(nonzeros(u_C)) ≈ Float32[1 6; 6 3; 4 15]
end

@testset "sparse_mm! CUSPARSEBackend CSR handle" begin
    A    = sparse([1,1,2,3,3], [1,3,2,1,3], Float32[1,2,3,4,5], 3, 3)
    Agpu = CuSparseMatrixCSR(A)
    u_A  = ust(Agpu)
    u_B  = ust(CuArray(Float32[1 0; 2 1; 0 3]))
    u_C  = ust(CUDA.zeros(Float32, 3, 2))

    h = prepare(CUSPARSEBackend(), SpMMOp, u_A; n_cols=2)
    execute(SpMMOp, h, u_B, u_C)
    @test Array(nonzeros(u_C)) ≈ Float32[1 6; 6 3; 4 15]

    fill!(nonzeros(u_C), 0f0)
    update_values!(h, u_A)
    execute(SpMMOp, h, u_B, u_C)
    @test Array(nonzeros(u_C)) ≈ Float32[1 6; 6 3; 4 15]
end

# ─── SpSV / SpSM ──────────────────────────────────────────────────────────────

@testset "sparse_sv! CUSPARSEBackend CSR lower triangular" begin
    # L = [2 0 0; 1 3 0; 0 2 4],  b = [4, 7, 16]  →  x = L\b = [2, 5/3, 17/6]
    # Actually let's use an exact example: L = [2 0; 0 3], b = [6, 9] → x = [3, 3]
    A    = sparse([1,2], [1,2], Float32[2, 3], 2, 2)
    Agpu = CuSparseMatrixCSR(A)
    u_A  = ust(Agpu)
    u_b  = ust(CuArray(Float32[6, 9]))
    u_x  = ust(CUDA.zeros(Float32, 2))

    execute(SpSVOp, u_A, u_b, u_x; backend=CUSPARSEBackend(), uplo='L', diag='N')
    @test Array(nonzeros(u_x)) ≈ Float32[3, 3]
end

@testset "sparse_sm! CUSPARSEBackend CSR lower triangular" begin
    # L = diag([2, 3]),  B = [6 10; 9 12]  →  X = L\B = [3 5; 3 4]
    A    = sparse([1,2], [1,2], Float32[2, 3], 2, 2)
    Agpu = CuSparseMatrixCSR(A)
    u_A  = ust(Agpu)
    u_B  = ust(CuArray(Float32[6 10; 9 12]))
    u_C  = ust(CUDA.zeros(Float32, 2, 2))

    execute(SpSMOp, u_A, u_B, u_C; backend=CUSPARSEBackend(), uplo='L', diag='N')
    @test Array(nonzeros(u_C)) ≈ Float32[3 5; 3 4]
end

@testset "sparse_sv! CUSPARSEBackend CSR handle" begin
    A    = sparse([1,2], [1,2], Float32[2, 3], 2, 2)
    Agpu = CuSparseMatrixCSR(A)
    u_A  = ust(Agpu)
    u_b  = ust(CuArray(Float32[6, 9]))
    u_x  = ust(CUDA.zeros(Float32, 2))

    h = prepare(CUSPARSEBackend(), SpSVOp, u_A; uplo='L', diag='N')
    execute(SpSVOp, h, u_b, u_x)
    @test Array(nonzeros(u_x)) ≈ Float32[3, 3]

    fill!(nonzeros(u_x), 0f0)
    update_values!(h, u_A)
    execute(SpSVOp, h, u_b, u_x)
    @test Array(nonzeros(u_x)) ≈ Float32[3, 3]
end

@testset "sparse_sm! CUSPARSEBackend CSR handle" begin
    A    = sparse([1,2], [1,2], Float32[2, 3], 2, 2)
    Agpu = CuSparseMatrixCSR(A)
    u_A  = ust(Agpu)
    u_B  = ust(CuArray(Float32[6 10; 9 12]))
    u_C  = ust(CUDA.zeros(Float32, 2, 2))

    h = prepare(CUSPARSEBackend(), SpSMOp, u_A; uplo='L', diag='N', n_cols=2)
    execute(SpSMOp, h, u_B, u_C)
    @test Array(nonzeros(u_C)) ≈ Float32[3 5; 3 4]

    fill!(nonzeros(u_C), 0f0)
    update_values!(h, u_A)
    execute(SpSMOp, h, u_B, u_C)
    @test Array(nonzeros(u_C)) ≈ Float32[3 5; 3 4]
end

# ─── sparse_to_dense / dense_to_sparse ────────────────────────────────────────

@testset "sparse_to_dense CUSPARSEBackend" begin
    A    = sparse([1,1,2,3,3], [1,3,2,1,3], Float32[1,2,3,4,5], 3, 3)
    Agpu = CuSparseMatrixCSR(A)
    u_A  = ust(Agpu)
    u_D  = execute(SparseToDenseOp, CUSPARSEBackend(), u_A)
    expected = Matrix(A)
    @test Array(nonzeros(u_D)) ≈ expected
end

@testset "dense_to_sparse CUSPARSEBackend" begin
    D    = Float32[1 0 2; 0 3 0; 4 0 5]
    u_D  = ust(CuArray(D))
    u_S  = dense_to_sparse(CUSPARSEBackend(), u_D, Formats.CSR)
    @test format(u_S) == Formats.CSR
    @test nnz(u_S) == 5
    A_ref = sparse([1,1,2,3,3], [1,3,2,1,3], Float32[1,2,3,4,5], 3, 3)
    @test Array(nonzeros(u_S)) ≈ A_ref.nzval
end

# ─── SDDMM ────────────────────────────────────────────────────────────────────

@testset "sparse_sddmm! CUSPARSEBackend CSR" begin
    # A (3×2 dense), B (2×3 dense), C sparse mask CSR (3×3)
    # SDDMM: C_ij ← alpha * (A*B)_ij * C_ij + beta * C_ij  for stored entries of C
    # C mask = [*,0,*; 0,*,0; *,0,*]  (stored at (1,1),(1,3),(2,2),(3,1),(3,3))
    A_d  = CuArray(Float32[1 0; 0 1; 1 1])   # 3×2
    B_d  = CuArray(Float32[1 0 1; 0 1 1])    # 2×3; A*B = [1 0 1; 0 1 1; 1 1 2]
    C_sp = sparse([1,1,2,3,3], [1,3,2,1,3], ones(Float32, 5), 3, 3)
    Cgpu = CuSparseMatrixCSR(C_sp)
    u_A  = ust(A_d)
    u_B  = ust(B_d)
    u_C  = ust(Cgpu)

    execute(SDDMMOp, u_A, u_B, u_C; backend=CUSPARSEBackend())
    # expected nzval: (A*B) sampled at (1,1)=1,(1,3)=1,(2,2)=1,(3,1)=1,(3,3)=2
    @test Array(nonzeros(u_C)) ≈ Float32[1, 1, 1, 1, 2]
end

# ─── sparse_gemm! CUSPARSEBackend (CSR × CSR → CSR) ─────────────────────────

@testset "sparse_gemm! CUSPARSEBackend CSR" begin
    # A = [1 0 2; 0 3 0; 4 0 5]   (3×3 CSR, same as SpMV tests)
    # B = A  (same matrix)
    # C = A*A = [9 0 12; 0 9 0; 24 0 33]
    A_sp = sparse([1,1,2,3,3], [1,3,2,1,3], Float32[1,2,3,4,5], 3, 3)
    Agpu = CuSparseMatrixCSR(A_sp)
    u_A  = ust(Agpu)
    # Build an empty C for sizing (beta=0)
    C0_sp = spzeros(Float32, 3, 3)
    u_C0  = ust(CuSparseMatrixCSR(C0_sp))

    u_C  = execute(SpGEMMOp, u_A, u_A, u_C0; backend=CUSPARSEBackend())
    C_dense = execute(SparseToDenseOp, u_C; backend=CUSPARSEBackend())
    @test Array(nonzeros(C_dense)) ≈ Float32[9 0 12; 0 9 0; 24 0 33]
end

end # if gpu_available()
