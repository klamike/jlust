using Test, JLUST, KernelAbstractions
using JLUST.Formats

# Resolve EmitterBackend via the extension module directly — extension exports
# are not accessible via `using JLUST: EmitterBackend` until the extension is
# in a loaded state visible to this world age.
const _kaext = Base.get_extension(JLUST, :KernelAbstractionsExt)
const EmitterBackend = _kaext.EmitterBackend

# ─── Helper ───────────────────────────────────────────────────────────────────

# Dense 1-D USTensor wrapping a plain Vector (for x and y in SpMV tests).
function dense_vec(v::Vector{T}) where T
    n   = length(v)
    USTensor{T,Int32,1,Vector{T},Vector{Int32},OneBased}(
        (n,), Formats.DensedRight(1),
        (nothing,), (nothing,),
        v, nothing,
    )
end

# Dense 2-D USTensor wrapping a plain Matrix (for B, C in SpMM / SDDMM tests).
function dense_mat(m::Matrix{T}) where T
    USTensor{T,Int32,2,Matrix{T},Vector{Int32},OneBased}(
        size(m), Formats.DensedRight(2),
        (nothing, nothing), (nothing, nothing),
        m, nothing,
    )
end

# ─── supports_backend ─────────────────────────────────────────────────────────

@testset "EmitterBackend supports_backend" begin
    dense1 = Formats.DensedRight(1)
    dense2 = Formats.DensedRight(2)
    be = EmitterBackend()

    # SpMV
    @test  supports_backend(be, SpMVOp(Formats.CSR, dense1, dense1))
    @test  supports_backend(be, SpMVOp(Formats.DCSR, dense1, dense1))
    @test  supports_backend(be, SpMVOp(Formats.COO, dense1, dense1))
    @test !supports_backend(be, SpMVOp(dense2, dense1, dense1))
    @test !supports_backend(be, SpMVOp(Formats.CSR, dense1, Formats.CSR))

    # SpMM
    @test  supports_backend(be, SpMMOp(Formats.CSR, dense2, dense2))
    @test !supports_backend(be, SpMMOp(dense2, dense2, dense2))

    # SpSV / SpSM — not supported (sequential dependency)
    @test !supports_backend(be, SpSVOp(Formats.CSR, dense1, dense1))
    @test !supports_backend(be, SpSMOp(Formats.CSR, dense2, dense2))

    # SDDMM — supported when C is sparse
    @test  supports_backend(be, SDDMMOp(dense2, dense2, Formats.CSR))
    @test  supports_backend(be, SDDMMOp(dense2, dense2, Formats.COO))
    @test !supports_backend(be, SDDMMOp(Formats.CSR, dense2, Formats.CSR))

    # SparseToDense — supported when src is sparse
    @test  supports_backend(be, SparseToDenseOp(Formats.CSR))
    @test  supports_backend(be, SparseToDenseOp(Formats.COO))
    @test !supports_backend(be, SparseToDenseOp(dense2))
end

# ─── apply_values! ────────────────────────────────────────────────────────────

@testset "apply_values! (CSR, OneBased)" begin
    # Build a 3×3 CSR: A = [1 0 2; 0 3 0; 4 0 5]
    rowptr = Int32[1, 3, 4, 6]
    colval = Int32[1, 3, 2, 1, 3]
    nzval  = Float64[1.0, 2.0, 3.0, 4.0, 5.0]

    u = USTensor{Float64,Int32,2,Vector{Float64},Vector{Int32},OneBased}(
        (3, 3), Formats.CSR,
        (nothing, rowptr),
        (nothing, colval),
        nzval, nothing,
    )

    apply_values!(x -> x * 2.0, u; backend=EmitterBackend())
    @test nonzeros(u) ≈ [2.0, 4.0, 6.0, 8.0, 10.0]
end

# ─── sparse_mv! (CSR, OneBased) ───────────────────────────────────────────────

@testset "sparse_mv! CSR OneBased" begin
    # A = [1 0 2; 0 3 0; 4 0 5],  x = [1, 2, 3]  →  y = [7, 6, 19]
    rowptr = Int32[1, 3, 4, 6]
    colval = Int32[1, 3, 2, 1, 3]
    nzval  = Float64[1.0, 2.0, 3.0, 4.0, 5.0]

    u_A = USTensor{Float64,Int32,2,Vector{Float64},Vector{Int32},OneBased}(
        (3, 3), Formats.CSR,
        (nothing, rowptr),
        (nothing, colval),
        nzval, nothing,
    )
    u_x = dense_vec([1.0, 2.0, 3.0])
    u_y = dense_vec(zeros(Float64, 3))

    sparse_mv!(u_A, u_x, u_y; backend=EmitterBackend())
    @test nonzeros(u_y) ≈ [7.0, 6.0, 19.0]
end

# ─── sparse_mv! (CSR, ZeroBased) ──────────────────────────────────────────────

@testset "sparse_mv! CSR ZeroBased" begin
    # Same matrix, zero-based indices: rowptr = [0,2,3,5], colval = [0,2,1,0,2]
    rowptr = Int32[0, 2, 3, 5]
    colval = Int32[0, 2, 1, 0, 2]
    nzval  = Float64[1.0, 2.0, 3.0, 4.0, 5.0]

    u_A = USTensor{Float64,Int32,2,Vector{Float64},Vector{Int32},ZeroBased}(
        (3, 3), Formats.CSR,
        (nothing, rowptr),
        (nothing, colval),
        nzval, nothing,
    )
    u_x = dense_vec([1.0, 2.0, 3.0])
    u_y = dense_vec(zeros(Float64, 3))

    sparse_mv!(u_A, u_x, u_y; backend=EmitterBackend())
    @test nonzeros(u_y) ≈ [7.0, 6.0, 19.0]
end

# ─── sparse_mv! (DCSR, OneBased) ─────────────────────────────────────────────

@testset "sparse_mv! DCSR OneBased" begin
    # DCSR of the same 3×3 matrix: only rows 1 and 3 are non-empty.
    # fiber_ptr = [1, 3, 5]  (outer pos, 1-based)
    # fiber_crd = [1, 3]      (which rows are stored)
    # col_ptr   = [1, 3, 4, 6] — but in DCSR the inner level is Compressed
    #   inner pos: pos[_fiber] .. pos[_fiber+1]-1 index into colval
    # We flatten inner pos as: for fiber 1 → cols at positions 1..2;
    #                           for fiber 2 → cols at positions 3..4
    # So inner_pos = [1, 3, 5], inner_crd = [1, 3, 1, 3]
    # nzval = [1, 2, 4, 5]
    #
    # A[1,1]=1  A[1,3]=2  A[3,1]=4  A[3,3]=5  (row 2 absent)
    # x = [1, 2, 3]  →  y[1] = 1+6=7,  y[2]=0,  y[3] = 4+15=19

    outer_crd = Int32[1, 3]          # level 1 crd: which rows exist
    inner_pos = Int32[1, 3, 5]       # level 2 pos: extents per fiber
    inner_crd = Int32[1, 3, 1, 3]   # level 2 crd: column indices
    nzval     = Float64[1.0, 2.0, 4.0, 5.0]

    u_A = USTensor{Float64,Int32,2,Vector{Float64},Vector{Int32},OneBased}(
        (3, 3), Formats.DCSR,
        (nothing, inner_pos),
        (outer_crd, inner_crd),
        nzval, nothing,
    )
    u_x = dense_vec([1.0, 2.0, 3.0])
    u_y = dense_vec(zeros(Float64, 3))

    sparse_mv!(u_A, u_x, u_y; backend=EmitterBackend())
    @test nonzeros(u_y) ≈ [7.0, 0.0, 19.0]
end

# ─── sparse_mv! (COO, OneBased, atomic) ───────────────────────────────────────

@testset "sparse_mv! COO OneBased" begin
    # COO of [1 0 2; 0 3 0; 4 0 5]:
    # row_crd = [1,1,2,3,3], col_crd = [1,3,2,1,3], nzval = [1,2,3,4,5]
    # x = [1, 2, 3]  →  y = [7, 6, 19]
    row_crd = Int32[1, 1, 2, 3, 3]
    col_crd = Int32[1, 3, 2, 1, 3]
    nzval   = Float64[1.0, 2.0, 3.0, 4.0, 5.0]

    u_A = USTensor{Float64,Int32,2,Vector{Float64},Vector{Int32},OneBased}(
        (3, 3), Formats.COO,
        (nothing, nothing),
        (row_crd, col_crd),
        nzval, nothing,
    )
    u_x = dense_vec([1.0, 2.0, 3.0])
    u_y = dense_vec(zeros(Float64, 3))

    # COO uses @atomic on float values.  POCL's SPIR-V target may not support
    # SPV_EXT_shader_atomic_float_add; skip rather than fail on those devices.
    try
        sparse_mv!(u_A, u_x, u_y; backend=EmitterBackend())
        @test nonzeros(u_y) ≈ [7.0, 6.0, 19.0]
    catch e
        msg = sprint(showerror, e)
        if contains(msg, "shader_atomic_float") || contains(msg, "SPV_EXT") ||
                contains(msg, "translate LLVM")
            @test_skip "float atomics not supported on this OpenCL/SPIR-V device"
        else
            rethrow(e)
        end
    end
end

# ─── sparse_mm! (CSR, OneBased) ───────────────────────────────────────────────

@testset "sparse_mm! CSR OneBased" begin
    # A = [1 0 2; 0 3 0; 4 0 5] (3×3 CSR),  B = [1 0; 2 1; 0 3] (3×2 dense)
    # C = A*B:
    #   row 1: [1*1+2*0, 1*0+2*3] = [1, 6]
    #   row 2: [3*2, 3*1]         = [6, 3]
    #   row 3: [4*1+5*0, 4*0+5*3] = [4, 15]
    rowptr = Int32[1, 3, 4, 6]
    colval = Int32[1, 3, 2, 1, 3]
    nzval  = Float64[1.0, 2.0, 3.0, 4.0, 5.0]

    B_mat = [1.0 0.0; 2.0 1.0; 0.0 3.0]
    C_mat = zeros(Float64, 3, 2)

    u_A = USTensor{Float64,Int32,2,Vector{Float64},Vector{Int32},OneBased}(
        (3, 3), Formats.CSR,
        (nothing, rowptr),
        (nothing, colval),
        nzval, nothing,
    )
    u_B = USTensor{Float64,Int32,2,Matrix{Float64},Vector{Int32},OneBased}(
        (3, 2), Formats.DensedRight(2),
        (nothing, nothing), (nothing, nothing),
        B_mat, nothing,
    )
    u_C = USTensor{Float64,Int32,2,Matrix{Float64},Vector{Int32},OneBased}(
        (3, 2), Formats.DensedRight(2),
        (nothing, nothing), (nothing, nothing),
        C_mat, nothing,
    )

    sparse_mm!(u_A, u_B, u_C; backend=EmitterBackend())
    @test nonzeros(u_C) ≈ [1.0 6.0; 6.0 3.0; 4.0 15.0]
end

@testset "sparse_mm! DCSR OneBased" begin
    # Same A matrix in DCSR (rows 1 and 3 only), same B
    outer_crd = Int32[1, 3]
    inner_pos = Int32[1, 3, 5]
    inner_crd = Int32[1, 3, 1, 3]
    nzval     = Float64[1.0, 2.0, 4.0, 5.0]

    B_mat = [1.0 0.0; 2.0 1.0; 0.0 3.0]
    C_mat = zeros(Float64, 3, 2)

    u_A = USTensor{Float64,Int32,2,Vector{Float64},Vector{Int32},OneBased}(
        (3, 3), Formats.DCSR,
        (nothing, inner_pos),
        (outer_crd, inner_crd),
        nzval, nothing,
    )
    u_B = USTensor{Float64,Int32,2,Matrix{Float64},Vector{Int32},OneBased}(
        (3, 2), Formats.DensedRight(2),
        (nothing, nothing), (nothing, nothing),
        B_mat, nothing,
    )
    u_C = USTensor{Float64,Int32,2,Matrix{Float64},Vector{Int32},OneBased}(
        (3, 2), Formats.DensedRight(2),
        (nothing, nothing), (nothing, nothing),
        C_mat, nothing,
    )

    sparse_mm!(u_A, u_B, u_C; backend=EmitterBackend())
    # Row 2 absent → C[2,:] stays zero
    @test nonzeros(u_C) ≈ [1.0 6.0; 0.0 0.0; 4.0 15.0]
end

# ─── sparse_sddmm! (CSR, OneBased) ───────────────────────────────────────────

@testset "sparse_sddmm! CSR OneBased" begin
    # A (3×2), B (2×3), C sparse CSR mask = [*,0,*; 0,*,0; *,0,*]
    # A*B = [1 0; 0 1; 1 1] * [1 0 1; 0 1 1] = [1 0 1; 0 1 1; 1 1 2]
    # SDDMM with alpha=1, beta=0: C_ij = (A*B)_ij at stored positions
    A_mat = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    B_mat = [1.0 0.0 1.0; 0.0 1.0 1.0]
    # CSR for C mask: rows=[1,1,2,3,3], cols=[1,3,2,1,3], vals=ones (mask)
    rowptr = Int32[1, 3, 4, 6]
    colval = Int32[1, 3, 2, 1, 3]
    nzval  = ones(Float64, 5)   # mask values (will be overwritten)

    u_A = dense_mat(A_mat)
    u_B = dense_mat(B_mat)
    u_C = USTensor{Float64,Int32,2,Vector{Float64},Vector{Int32},OneBased}(
        (3, 3), Formats.CSR,
        (nothing, rowptr),
        (nothing, colval),
        nzval, nothing,
    )

    sparse_sddmm!(u_A, u_B, u_C; backend=EmitterBackend(), beta=0.0)
    # (A*B) sampled at (1,1)=1,(1,3)=1,(2,2)=1,(3,1)=1,(3,3)=2
    @test nonzeros(u_C) ≈ [1.0, 1.0, 1.0, 1.0, 2.0]
end

@testset "sparse_sddmm! COO OneBased" begin
    A_mat = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    B_mat = [1.0 0.0 1.0; 0.0 1.0 1.0]
    row_crd = Int32[1, 1, 2, 3, 3]
    col_crd = Int32[1, 3, 2, 1, 3]
    nzval   = ones(Float64, 5)

    u_A = dense_mat(A_mat)
    u_B = dense_mat(B_mat)
    u_C = USTensor{Float64,Int32,2,Vector{Float64},Vector{Int32},OneBased}(
        (3, 3), Formats.COO,
        (nothing, nothing),
        (row_crd, col_crd),
        nzval, nothing,
    )

    sparse_sddmm!(u_A, u_B, u_C; backend=EmitterBackend(), beta=0.0)
    @test nonzeros(u_C) ≈ [1.0, 1.0, 1.0, 1.0, 2.0]
end

# ─── sparse_to_dense (CSR, OneBased) ─────────────────────────────────────────

@testset "sparse_to_dense CSR OneBased" begin
    # A = [1 0 2; 0 3 0; 4 0 5]
    rowptr = Int32[1, 3, 4, 6]
    colval = Int32[1, 3, 2, 1, 3]
    nzval  = Float64[1.0, 2.0, 3.0, 4.0, 5.0]

    u_A = USTensor{Float64,Int32,2,Vector{Float64},Vector{Int32},OneBased}(
        (3, 3), Formats.CSR,
        (nothing, rowptr),
        (nothing, colval),
        nzval, nothing,
    )

    u_D = sparse_to_dense(u_A; backend=EmitterBackend())
    @test format(u_D) == Formats.DensedRight(2)
    @test nonzeros(u_D) ≈ [1.0 0.0 2.0; 0.0 3.0 0.0; 4.0 0.0 5.0]
end

@testset "sparse_to_dense COO OneBased" begin
    row_crd = Int32[1, 1, 2, 3, 3]
    col_crd = Int32[1, 3, 2, 1, 3]
    nzval   = Float64[1.0, 2.0, 3.0, 4.0, 5.0]

    u_A = USTensor{Float64,Int32,2,Vector{Float64},Vector{Int32},OneBased}(
        (3, 3), Formats.COO,
        (nothing, nothing),
        (row_crd, col_crd),
        nzval, nothing,
    )

    u_D = sparse_to_dense(u_A; backend=EmitterBackend())
    @test nonzeros(u_D) ≈ [1.0 0.0 2.0; 0.0 3.0 0.0; 4.0 0.0 5.0]
end

# ─── Kernel reuse ─────────────────────────────────────────────────────────────

@testset "Kernel reuse" begin
    # The old _emitter_cache global is gone — Julia's method specialization
    # caches the @generated kernel per (FMT, T) automatically.  This test
    # verifies that repeated calls with structurally-identical tensors give
    # consistent results (proxy for "kernel is reused, not re-emitted").
    rowptr = Int32[1, 3, 4, 6]
    colval = Int32[1, 3, 2, 1, 3]
    nzval  = Float64[1.0, 2.0, 3.0, 4.0, 5.0]
    make_A() = USTensor{Float64,Int32,2,Vector{Float64},Vector{Int32},OneBased}(
        (3, 3), Formats.CSR,
        (nothing, copy(rowptr)),
        (nothing, copy(colval)),
        copy(nzval), nothing,
    )

    u_x = dense_vec([1.0, 2.0, 3.0])

    u_A1 = make_A(); u_y1 = dense_vec(zeros(3))
    sparse_mv!(u_A1, u_x, u_y1; backend=EmitterBackend())
    @test nonzeros(u_y1) ≈ [7.0, 6.0, 19.0]

    u_A2 = make_A(); u_y2 = dense_vec(zeros(3))
    sparse_mv!(u_A2, u_x, u_y2; backend=EmitterBackend())
    @test nonzeros(u_y2) ≈ [7.0, 6.0, 19.0]

    # Both u_A1 and u_A2 have identical Julia types (same FMT type param) →
    # the @generated _ust_spmv_kern method specialization is shared.
    @test typeof(u_A1) === typeof(u_A2)
end
