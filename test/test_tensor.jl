using Test, JLUST, JLUST.Formats, SparseArrays

# ─── Helpers ─────────────────────────────────────────────────────────────────

# 3×4 sparse matrix:
#   row 1: (1,1)=1  (1,3)=2
#   row 2: (2,2)=3
#   row 3: (3,1)=4  (3,3)=5
function make_sparse_3x4()
    sparse([1,1,2,3,3], [1,3,2,1,3], Float32[1,2,3,4,5], 3, 4)
end

# ─── SparseMatrixCSC zero-copy view ──────────────────────────────────────────

@testset "ust(SparseMatrixCSC)" begin
    A = make_sparse_3x4()
    u = ust(A)

    @test index_origin(u) == OneBased()
    @test format(u) == Formats.CSC
    @test size(u) == (3, 4)
    @test ndims(u) == 2
    @test eltype(u) == Float32
    @test nnz(u) == 5

    # CSC: pos[2] = colptr, crd[2] = rowval
    @test has_positions(u, 2)
    @test has_coordinates(u, 2)
    @test !has_positions(u, 1)
    @test !has_coordinates(u, 1)

    @test positions(u, 2) === A.colptr    # zero-copy
    @test coordinates(u, 2) === A.rowval  # zero-copy
    @test nonzeros(u) === A.nzval         # zero-copy

    # Throws on missing level
    @test_throws InvalidLevelAccess positions(u, 1)
    @test_throws InvalidLevelAccess coordinates(u, 1)
end

# ─── Manual csr_tensor ───────────────────────────────────────────────────────

@testset "csr_tensor (ZeroBased)" begin
    # 0-based CSR for the same 3×4 matrix
    rowptr = Int32[0, 2, 3, 5]
    colind = Int32[0, 2, 1, 0, 2]
    vals   = Float32[1, 2, 3, 4, 5]

    u = csr_tensor(rowptr, colind, vals, (3, 4); origin=ZeroBased())

    @test index_origin(u) == ZeroBased()
    @test format(u) == Formats.CSR
    @test size(u) == (3, 4)
    @test nnz(u) == 5
    @test positions(u, 2) === rowptr
    @test coordinates(u, 2) === colind
    @test nonzeros(u) === vals
end

@testset "csr_tensor (OneBased)" begin
    rowptr = Int32[1, 3, 4, 6]
    colind = Int32[1, 3, 2, 1, 3]
    vals   = Float32[1, 2, 3, 4, 5]

    u = csr_tensor(rowptr, colind, vals, (3, 4); origin=OneBased())
    @test index_origin(u) == OneBased()
end

# ─── csc_tensor ──────────────────────────────────────────────────────────────

@testset "csc_tensor (OneBased)" begin
    A = make_sparse_3x4()
    u = csc_tensor(A.colptr, A.rowval, A.nzval, (3, 4); origin=OneBased())
    @test format(u) == Formats.CSC
    @test size(u) == (3, 4)
    @test positions(u, 2) === A.colptr
end

# ─── coo_tensor ──────────────────────────────────────────────────────────────

@testset "coo_tensor (ZeroBased)" begin
    rows = Int32[0, 0, 1, 2, 2]
    cols = Int32[0, 2, 1, 0, 2]
    vals = Float32[1, 2, 3, 4, 5]

    u = coo_tensor(rows, cols, vals, (3, 4); origin=ZeroBased())
    @test format(u) == Formats.COO
    @test size(u) == (3, 4)
    @test nnz(u) == 5
    @test !has_positions(u, 1)
    @test has_coordinates(u, 1)
    @test has_coordinates(u, 2)
    @test coordinates(u, 1) === rows
    @test coordinates(u, 2) === cols
end

# ─── getindex ────────────────────────────────────────────────────────────────

@testset "getindex on ZeroBased CSR" begin
    rowptr = Int32[0, 2, 3, 5]
    colind = Int32[0, 2, 1, 0, 2]
    vals   = Float32[1, 2, 3, 4, 5]
    u = csr_tensor(rowptr, colind, vals, (3, 4); origin=ZeroBased())

    # Stored values (1-based Julia indexing)
    @test u[1, 1] === 1.0f0
    @test u[1, 3] === 2.0f0
    @test u[2, 2] === 3.0f0
    @test u[3, 1] === 4.0f0
    @test u[3, 3] === 5.0f0

    # Structural zeros
    @test u[1, 2] === 0.0f0
    @test u[1, 4] === 0.0f0
    @test u[2, 1] === 0.0f0
    @test u[3, 2] === 0.0f0
end

@testset "getindex on OneBased CSC" begin
    A = make_sparse_3x4()
    u = ust(A)

    @test u[1, 1] === 1.0f0
    @test u[1, 3] === 2.0f0
    @test u[2, 2] === 3.0f0
    @test u[3, 1] === 4.0f0
    @test u[3, 3] === 5.0f0
    @test u[1, 2] === 0.0f0
    @test u[2, 4] === 0.0f0
end

# ─── copy ────────────────────────────────────────────────────────────────────

@testset "copy" begin
    rowptr = Int32[0, 2, 3, 5]
    colind = Int32[0, 2, 1, 0, 2]
    vals   = Float32[1, 2, 3, 4, 5]
    u  = csr_tensor(rowptr, colind, vals, (3, 4); origin=ZeroBased())
    u2 = copy(u)

    @test u2[1, 1] === 1.0f0
    @test positions(u2, 2) == positions(u, 2)
    @test positions(u2, 2) !== positions(u, 2)  # independent copy
    @test nonzeros(u2) == nonzeros(u)
    @test nonzeros(u2) !== nonzeros(u)
end

# ─── Base traits ─────────────────────────────────────────────────────────────

@testset "Base traits" begin
    u = csr_tensor(Int32[0,2,3,5], Int32[0,2,1,0,2], Float32[1,2,3,4,5], (3,4); origin=ZeroBased())
    @test size(u)     == (3, 4)
    @test size(u, 1)  == 3
    @test size(u, 2)  == 4
    @test ndims(u)    == 2
    @test eltype(u)   == Float32
    @test length(u)   == 12
    @test nnz(u)      == 5
end

# ─── show (smoke test) ───────────────────────────────────────────────────────

@testset "show" begin
    u = csr_tensor(Int32[0,2,3,5], Int32[0,2,1,0,2], Float32[1,2,3,4,5], (3,4); origin=ZeroBased())
    s = sprint(show, u)
    @test occursin("CSR", s)
    @test occursin("ZeroBased", s)
    @test occursin("nnz      : 5", s)
end
