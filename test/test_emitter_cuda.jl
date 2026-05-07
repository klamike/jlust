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

end  # if gpu_available()
