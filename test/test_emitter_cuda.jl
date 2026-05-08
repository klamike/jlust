using Test, JLUST, JLUST.Formats

# CUDA emitter tests — gated behind gpu_available().
# Tests that the EmitterBackend works with CuArrays via the KernelAbstractions
# CUDA backend, producing correct SpMV results on device.

if gpu_available()

using CUDA, KernelAbstractions

const _kaext_cuda = Base.get_extension(JLUST, :KernelAbstractionsExt)
const EmitterBackend_cuda = _kaext_cuda.EmitterBackend

# Dense 1-D USTensor backed by CuVector.
function dense_cuvec(v::Vector{T}) where T
    n  = length(v)
    cv = CuArray(v)
    fmt = Formats.DensedRight(1)
    USTensor{T,Int32,1,CuVector{T},CuVector{Int32},OneBased}(
        (n,), fmt,
        Dict{Int,CuVector{Int32}}(), Dict{Int,CuVector{Int32}}(),
        cv, nothing,
    )
end

# ─── apply_values! on GPU ─────────────────────────────────────────────────────

@testset "apply_values! GPU CSR" begin
    rowptr = CuArray(Int32[1, 3, 4, 6])
    colval = CuArray(Int32[1, 3, 2, 1, 3])
    nzval  = CuArray(Float64[1.0, 2.0, 3.0, 4.0, 5.0])

    u = USTensor{Float64,Int32,2,CuVector{Float64},CuVector{Int32},OneBased}(
        (3, 3), Formats.CSR,
        Dict{Int,CuVector{Int32}}(2 => rowptr),
        Dict{Int,CuVector{Int32}}(2 => colval),
        nzval, nothing,
    )

    apply_values!(x -> x * 2.0, u; backend=EmitterBackend_cuda())
    @test Array(nonzeros(u)) ≈ [2.0, 4.0, 6.0, 8.0, 10.0]
end

# ─── sparse_mv! GPU CSR (OneBased) ────────────────────────────────────────────

@testset "sparse_mv! GPU CSR OneBased" begin
    rowptr = CuArray(Int32[1, 3, 4, 6])
    colval = CuArray(Int32[1, 3, 2, 1, 3])
    nzval  = CuArray(Float64[1.0, 2.0, 3.0, 4.0, 5.0])

    u_A = USTensor{Float64,Int32,2,CuVector{Float64},CuVector{Int32},OneBased}(
        (3, 3), Formats.CSR,
        Dict{Int,CuVector{Int32}}(2 => rowptr),
        Dict{Int,CuVector{Int32}}(2 => colval),
        nzval, nothing,
    )
    u_x = dense_cuvec([1.0, 2.0, 3.0])
    u_y = dense_cuvec(zeros(Float64, 3))

    sparse_mv!(u_A, u_x, u_y; backend=EmitterBackend_cuda())
    @test Array(nonzeros(u_y)) ≈ [7.0, 6.0, 19.0]
end

# ─── sparse_mv! GPU DCSR (OneBased) ──────────────────────────────────────────

@testset "sparse_mv! GPU DCSR OneBased" begin
    outer_crd = CuArray(Int32[1, 3])
    inner_pos = CuArray(Int32[1, 3, 5])
    inner_crd = CuArray(Int32[1, 3, 1, 3])
    nzval     = CuArray(Float64[1.0, 2.0, 4.0, 5.0])

    u_A = USTensor{Float64,Int32,2,CuVector{Float64},CuVector{Int32},OneBased}(
        (3, 3), Formats.DCSR,
        Dict{Int,CuVector{Int32}}(2 => inner_pos),
        Dict{Int,CuVector{Int32}}(1 => outer_crd, 2 => inner_crd),
        nzval, nothing,
    )
    u_x = dense_cuvec([1.0, 2.0, 3.0])
    u_y = dense_cuvec(zeros(Float64, 3))

    sparse_mv!(u_A, u_x, u_y; backend=EmitterBackend_cuda())
    @test Array(nonzeros(u_y)) ≈ [7.0, 0.0, 19.0]
end

# ─── sparse_mv! GPU COO (OneBased, atomic float) ─────────────────────────────

@testset "sparse_mv! GPU COO OneBased (atomic)" begin
    row_crd = CuArray(Int32[1, 1, 2, 3, 3])
    col_crd = CuArray(Int32[1, 3, 2, 1, 3])
    nzval   = CuArray(Float64[1.0, 2.0, 3.0, 4.0, 5.0])

    u_A = USTensor{Float64,Int32,2,CuVector{Float64},CuVector{Int32},OneBased}(
        (3, 3), Formats.COO,
        Dict{Int,CuVector{Int32}}(),
        Dict{Int,CuVector{Int32}}(1 => row_crd, 2 => col_crd),
        nzval, nothing,
    )
    u_x = dense_cuvec([1.0, 2.0, 3.0])
    u_y = dense_cuvec(zeros(Float64, 3))

    sparse_mv!(u_A, u_x, u_y; backend=EmitterBackend_cuda())
    @test Array(nonzeros(u_y)) ≈ [7.0, 6.0, 19.0]
end

# ─── sparse_mm! GPU CSR (OneBased) ───────────────────────────────────────────

@testset "sparse_mm! GPU CSR OneBased" begin
    rowptr = CuArray(Int32[1, 3, 4, 6])
    colval = CuArray(Int32[1, 3, 2, 1, 3])
    nzval  = CuArray(Float64[1.0, 2.0, 3.0, 4.0, 5.0])

    # A = [1 0 2; 0 3 0; 4 0 5], B = [1 0; 2 1; 0 3]
    # C = A*B = [1 6; 6 3; 4 15]
    B_cu = CuArray([1.0 0.0; 2.0 1.0; 0.0 3.0])
    C_cu = CUDA.zeros(Float64, 3, 2)

    u_A = USTensor{Float64,Int32,2,CuVector{Float64},CuVector{Int32},OneBased}(
        (3, 3), Formats.CSR,
        Dict{Int,CuVector{Int32}}(2 => rowptr),
        Dict{Int,CuVector{Int32}}(2 => colval),
        nzval, nothing,
    )
    u_B = USTensor{Float64,Int32,2,CuMatrix{Float64},CuVector{Int32},OneBased}(
        (3, 2), Formats.DensedRight(2),
        Dict{Int,CuVector{Int32}}(), Dict{Int,CuVector{Int32}}(),
        B_cu, nothing,
    )
    u_C = USTensor{Float64,Int32,2,CuMatrix{Float64},CuVector{Int32},OneBased}(
        (3, 2), Formats.DensedRight(2),
        Dict{Int,CuVector{Int32}}(), Dict{Int,CuVector{Int32}}(),
        C_cu, nothing,
    )

    sparse_mm!(u_A, u_B, u_C; backend=EmitterBackend_cuda())
    @test Array(nonzeros(u_C)) ≈ [1.0 6.0; 6.0 3.0; 4.0 15.0]
end


# ─── sparse_sddmm! GPU CSR (OneBased) ────────────────────────────────────────

@testset "sparse_sddmm! GPU CSR OneBased" begin
    A_cu = CuArray([1.0 0.0; 0.0 1.0; 1.0 1.0])   # 3×2
    B_cu = CuArray([1.0 0.0 1.0; 0.0 1.0 1.0])    # 2×3; A*B = [1 0 1; 0 1 1; 1 1 2]
    rowptr = CuArray(Int32[1, 3, 4, 6])
    colval = CuArray(Int32[1, 3, 2, 1, 3])
    nzval  = CUDA.ones(Float64, 5)

    u_A = USTensor{Float64,Int32,2,CuMatrix{Float64},CuVector{Int32},OneBased}(
        (3, 2), Formats.DensedRight(2),
        Dict{Int,CuVector{Int32}}(), Dict{Int,CuVector{Int32}}(),
        A_cu, nothing,
    )
    u_B = USTensor{Float64,Int32,2,CuMatrix{Float64},CuVector{Int32},OneBased}(
        (2, 3), Formats.DensedRight(2),
        Dict{Int,CuVector{Int32}}(), Dict{Int,CuVector{Int32}}(),
        B_cu, nothing,
    )
    u_C = USTensor{Float64,Int32,2,CuVector{Float64},CuVector{Int32},OneBased}(
        (3, 3), Formats.CSR,
        Dict{Int,CuVector{Int32}}(2 => rowptr),
        Dict{Int,CuVector{Int32}}(2 => colval),
        nzval, nothing,
    )

    sparse_sddmm!(u_A, u_B, u_C; backend=EmitterBackend_cuda(), beta=0.0)
    @test Array(nonzeros(u_C)) ≈ [1.0, 1.0, 1.0, 1.0, 2.0]
end

# ─── sparse_to_dense GPU CSR (OneBased) ──────────────────────────────────────

@testset "sparse_to_dense GPU CSR OneBased" begin
    rowptr = CuArray(Int32[1, 3, 4, 6])
    colval = CuArray(Int32[1, 3, 2, 1, 3])
    nzval  = CuArray(Float64[1.0, 2.0, 3.0, 4.0, 5.0])

    u_A = USTensor{Float64,Int32,2,CuVector{Float64},CuVector{Int32},OneBased}(
        (3, 3), Formats.CSR,
        Dict{Int,CuVector{Int32}}(2 => rowptr),
        Dict{Int,CuVector{Int32}}(2 => colval),
        nzval, nothing,
    )

    u_D = sparse_to_dense(u_A; backend=EmitterBackend_cuda())
    @test format(u_D) == Formats.DensedRight(2)
    @test Array(nonzeros(u_D)) ≈ [1.0 0.0 2.0; 0.0 3.0 0.0; 4.0 0.0 5.0]
end

# ─── Operator fusion: input_fn / output_fn ───────────────────────────────────
#
# Named functions required — GPU kernels cannot use closures with captures.
# Zero-capture lambdas (e.g. x -> 2x) are also fine on CUDA.jl >= 5.x.

_fuse_double(x::Float64) = 2.0 * x
_fuse_addone(x::Float64) = x + 1.0

@testset "sparse_mv! GPU CSR input_fn" begin
    # A = [1 0 2; 0 3 0]  (2×3), x = [1,2,3]
    # A * (2*x) = [1*2 + 2*6, 3*4] = [14, 12]
    rowptr = CuArray(Int32[1, 3, 4])
    colval = CuArray(Int32[1, 3, 2])
    nzval  = CuArray(Float64[1.0, 2.0, 3.0])
    u_A = USTensor{Float64,Int32,2,CuVector{Float64},CuVector{Int32},OneBased}(
        (2, 3), Formats.CSR,
        Dict{Int,CuVector{Int32}}(2 => rowptr),
        Dict{Int,CuVector{Int32}}(2 => colval),
        nzval, nothing,
    )
    u_x = dense_cuvec([1.0, 2.0, 3.0])
    u_y = dense_cuvec(zeros(Float64, 2))
    sparse_mv!(u_A, u_x, u_y; backend=EmitterBackend_cuda(), input_fn=_fuse_double)
    @test Array(nonzeros(u_y)) ≈ [14.0, 12.0]
end

@testset "sparse_mv! GPU CSR output_fn" begin
    # A = [1 0 2; 0 3 0], x = [1,2,3]
    # (A*x) .+ 1 = [7+1, 6+1] = [8, 7]
    rowptr = CuArray(Int32[1, 3, 4])
    colval = CuArray(Int32[1, 3, 2])
    nzval  = CuArray(Float64[1.0, 2.0, 3.0])
    u_A = USTensor{Float64,Int32,2,CuVector{Float64},CuVector{Int32},OneBased}(
        (2, 3), Formats.CSR,
        Dict{Int,CuVector{Int32}}(2 => rowptr),
        Dict{Int,CuVector{Int32}}(2 => colval),
        nzval, nothing,
    )
    u_x = dense_cuvec([1.0, 2.0, 3.0])
    u_y = dense_cuvec(zeros(Float64, 2))
    sparse_mv!(u_A, u_x, u_y; backend=EmitterBackend_cuda(), output_fn=_fuse_addone)
    @test Array(nonzeros(u_y)) ≈ [8.0, 7.0]
end

@testset "EmitterSpMVHandle prepare + sparse_mv! GPU CSR" begin
    # A = [1 0 2; 0 3 0], x = [1,2,3]
    # A*(2*x) .+ 1 = [14+1, 12+1] = [15, 13]
    rowptr = CuArray(Int32[1, 3, 4])
    colval = CuArray(Int32[1, 3, 2])
    nzval  = CuArray(Float64[1.0, 2.0, 3.0])
    u_A = USTensor{Float64,Int32,2,CuVector{Float64},CuVector{Int32},OneBased}(
        (2, 3), Formats.CSR,
        Dict{Int,CuVector{Int32}}(2 => rowptr),
        Dict{Int,CuVector{Int32}}(2 => colval),
        nzval, nothing,
    )
    u_x = dense_cuvec([1.0, 2.0, 3.0])
    u_y = dense_cuvec(zeros(Float64, 2))

    h = prepare(EmitterBackend_cuda(), SpMVOp, u_A;
                 input_fn=_fuse_double, output_fn=_fuse_addone)
    @test h isa _kaext_cuda.EmitterSpMVHandle

    sparse_mv!(h, u_A, u_x, u_y)
    @test Array(nonzeros(u_y)) ≈ [15.0, 13.0]

    # Second call reuses compiled kernel — result must be stable
    fill!(nonzeros(u_y), 0.0)
    sparse_mv!(h, u_A, u_x, u_y)
    @test Array(nonzeros(u_y)) ≈ [15.0, 13.0]
end

# ─── UX improvements: constructors, raw vectors, mul!, BlockSparseMatrix ──────

@testset "csr_tensor keyword args" begin
    rowptr = CuArray(Int32[1, 3, 4]); colval = CuArray(Int32[1, 3, 2])
    nzval  = CuArray(Float64[1.0, 2.0, 3.0])
    u = csr_tensor(rowptr, colval, nzval; m=2, n=3)
    @test size(u) == (2, 3)
    @test format(u) == Formats.CSR
end

@testset "dcsr_tensor" begin
    oc = CuArray(Int32[1, 3]); ip = CuArray(Int32[1, 3, 5])
    ic = CuArray(Int32[1, 3, 1, 3]); nz = CuArray(Float64[1.0, 2.0, 4.0, 5.0])
    u = dcsr_tensor(oc, ip, ic, nz; m=3, n=3)
    @test size(u) == (3, 3)
    @test format(u) == Formats.DCSR
end

@testset "sparse_mv! raw CuVector operands" begin
    rowptr = CuArray(Int32[1, 3, 4]); colval = CuArray(Int32[1, 3, 2])
    nzval  = CuArray(Float64[1.0, 2.0, 3.0])
    u_A = csr_tensor(rowptr, colval, nzval; m=2, n=3)
    x = CuArray([1.0, 2.0, 3.0]); y = CUDA.zeros(Float64, 2)
    sparse_mv!(EmitterBackend_cuda(), u_A, x, y)
    @test Array(y) ≈ [7.0, 6.0]
end

@testset "mul! and * for USTensor" begin
    rowptr = CuArray(Int32[1, 3, 4]); colval = CuArray(Int32[1, 3, 2])
    nzval  = CuArray(Float64[1.0, 2.0, 3.0])
    u_A = csr_tensor(rowptr, colval, nzval; m=2, n=3)
    x = CuArray([1.0, 2.0, 3.0]); y = CUDA.zeros(Float64, 2)
    mul!(y, u_A, x)
    @test Array(y) ≈ [7.0, 6.0]
    @test Array(u_A * x) ≈ [7.0, 6.0]
end

@testset "BlockSparseMatrix diagonal" begin
    rp1 = CuArray(Int32[1,3,4]); cv1 = CuArray(Int32[1,3,2]); nz1 = CuArray(Float64[1,2,3])
    rp2 = CuArray(Int32[1,2,4]); cv2 = CuArray(Int32[2,1,3]); nz2 = CuArray(Float64[1,1,1])
    A1 = csr_tensor(rp1, cv1, nz1; m=2, n=3)   # [1 0 2; 0 3 0]
    A2 = csr_tensor(rp2, cv2, nz2; m=2, n=3)   # [0 1 0; 1 0 1]

    BM = BlockSparseMatrix([A1 nothing; nothing A2])
    @test size(BM) == (4, 6)
    x = CuArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # A1*[1,2,3]=[7,6], A2*[4,5,6]=[5,10]
    @test Array(BM * x) ≈ [7.0, 6.0, 5.0, 10.0]
    y = CUDA.zeros(Float64, 4); mul!(y, BM, x)
    @test Array(y) ≈ [7.0, 6.0, 5.0, 10.0]
end

@testset "BlockSparseMatrix accumulate" begin
    rp1 = CuArray(Int32[1,3,4]); cv1 = CuArray(Int32[1,3,2]); nz1 = CuArray(Float64[1,2,3])
    rp2 = CuArray(Int32[1,2,4]); cv2 = CuArray(Int32[2,1,3]); nz2 = CuArray(Float64[1,1,1])
    A1 = csr_tensor(rp1, cv1, nz1; m=2, n=3)   # [1 0 2; 0 3 0]
    A2 = csr_tensor(rp2, cv2, nz2; m=2, n=3)   # [0 1 0; 1 0 1]

    BM = BlockSparseMatrix([A1 A2])             # single block row, two block cols
    x = CuArray([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    # A1*[1,2,3]=[7,6], A2*[1,2,3]=[2,4], sum=[9,10]
    @test Array(BM * x) ≈ [9.0, 10.0]
end

end  # if gpu_available()
