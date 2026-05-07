using Test, JLUST, JLUST.Formats, SparseArrays

# 3×4 matrix used throughout:
#   (1,1)=1  (1,3)=2  (2,2)=3  (3,1)=4  (3,3)=5  (0-based rows/cols)
const ROWPTR3x4 = Int32[0, 2, 3, 5]
const COLIND3x4 = Int32[0, 2, 1, 0, 2]
const VALS3x4   = Float32[1, 2, 3, 4, 5]

csr_fixture() = csr_tensor(copy(ROWPTR3x4), copy(COLIND3x4), copy(VALS3x4), (3, 4); origin=ZeroBased())

# Helper: sort (rows, cols, vals) together by (row, col) for comparison
function sorted_coo(rows, cols, vals)
    perm = sortperm(collect(zip(rows, cols)))
    rows[perm], cols[perm], vals[perm]
end

# ─── Decomposer: visits correct elements ─────────────────────────────────────

@testset "TensorDecomposer visits all elements" begin
    u = csr_fixture()
    visited = Tuple{Vector{Int},Float64}[]
    run!(TensorDecomposer(u, (dims, val) -> push!(visited, (copy(dims), Float64(val)))))
    @test length(visited) == 5
    sort!(visited, by = x -> (x[1][1], x[1][2]))
    @test visited[1] == ([0, 0], 1.0)
    @test visited[2] == ([0, 2], 2.0)
    @test visited[3] == ([1, 1], 3.0)
    @test visited[4] == ([2, 0], 4.0)
    @test visited[5] == ([2, 2], 5.0)
end

# ─── CSR → CSR round-trip ────────────────────────────────────────────────────

@testset "convert_format: CSR → CSR (identity)" begin
    u  = csr_fixture()
    u2 = convert_format(u, Formats.CSR)
    @test positions(u2, 2)    == ROWPTR3x4
    @test coordinates(u2, 2)  == COLIND3x4
    @test nonzeros(u2)        ≈  VALS3x4
end

# ─── CSR → CSC ───────────────────────────────────────────────────────────────

@testset "convert_format: CSR → CSC" begin
    u   = csr_fixture()
    u2  = convert_format(u, Formats.CSC)
    @test format(u2) == Formats.CSC
    @test size(u2) == (3, 4)
    @test nnz(u2) == 5
    # col 0: rows [0,2], col 1: row [1], col 2: rows [0,2], col 3: empty
    colptr = Int32[0, 2, 3, 5, 5]
    rowind = Int32[0, 2, 1, 0, 2]
    nzval  = Float32[1, 4, 3, 2, 5]
    @test positions(u2, 2)    == colptr
    @test coordinates(u2, 2)  == rowind
    @test nonzeros(u2)        ≈  nzval
    @test index_origin(u2) == ZeroBased()
end

# ─── CSR → COO ───────────────────────────────────────────────────────────────

@testset "convert_format: CSR → COO" begin
    u  = csr_fixture()
    u2 = convert_format(u, Formats.COO)
    @test format(u2) == Formats.COO
    @test nnz(u2) == 5
    @test has_positions(u2, 1)
    @test positions(u2, 1) == Int32[0, 5]   # pos = [0, nnz], 0-based
    @test has_coordinates(u2, 1)
    @test has_coordinates(u2, 2)
    rows  = Int32[0, 0, 1, 2, 2]
    cols  = Int32[0, 2, 1, 0, 2]
    nzval = Float32[1, 2, 3, 4, 5]
    r, c, v   = sorted_coo(coordinates(u2, 1), coordinates(u2, 2), nonzeros(u2))
    rr, cc, vv = sorted_coo(rows, cols, nzval)
    @test r  == rr
    @test c  == cc
    @test v  ≈  vv
end

# ─── CSR → DCSR ──────────────────────────────────────────────────────────────

@testset "convert_format: CSR → DCSR" begin
    u  = csr_fixture()
    u2 = convert_format(u, Formats.DCSR)
    @test format(u2) == Formats.DCSR
    @test nnz(u2) == 5
    # All 3 rows are non-empty in DCSR
    @test has_positions(u2, 1)
    @test has_coordinates(u2, 1)
    @test has_positions(u2, 2)
    @test has_coordinates(u2, 2)
end

# ─── CSR → CSC → CSR round-trip ──────────────────────────────────────────────

@testset "round-trip CSR → CSC → CSR" begin
    u1 = csr_fixture()
    u2 = convert_format(u1, Formats.CSC)
    u3 = convert_format(u2, Formats.CSR)
    @test positions(u3, 2)   == ROWPTR3x4
    @test coordinates(u3, 2) == COLIND3x4
    @test nonzeros(u3)       ≈  VALS3x4
end

# ─── Round-trip through several formats ──────────────────────────────────────

@testset "round-trip through various formats" begin
    u = csr_fixture()
    for fmt in [Formats.CSC, Formats.COO, Formats.DCSR, Formats.DCSC]
        u2 = convert_format(convert_format(u, fmt), Formats.CSR)
        @test positions(u2, 2)   == ROWPTR3x4
        @test coordinates(u2, 2) == COLIND3x4
        @test nonzeros(u2)       ≈  VALS3x4
    end
end

# ─── convert_index_type / convert_value_type ─────────────────────────────────

@testset "convert_index_type" begin
    u  = csr_fixture()
    u2 = convert_index_type(u, Int64)
    @test eltype(positions(u2, 2))    == Int64
    @test eltype(coordinates(u2, 2))  == Int64
    @test positions(u2, 2)   == Int64.(ROWPTR3x4)
    @test coordinates(u2, 2) == Int64.(COLIND3x4)
    @test nonzeros(u2) ≈ VALS3x4
end

@testset "convert_value_type" begin
    u  = csr_fixture()
    u2 = convert_value_type(u, Float64)
    @test eltype(nonzeros(u2)) == Float64
    @test nonzeros(u2) ≈ Float64.(VALS3x4)
end

# ─── Base.convert → SparseMatrixCSC ──────────────────────────────────────────

@testset "convert to SparseMatrixCSC" begin
    u = csr_fixture()
    A = convert(SparseMatrixCSC{Float32,Int32}, u)
    @test size(A)    == (3, 4)
    @test nnz(A)     == 5
    @test A[1, 1]    == 1.0f0
    @test A[1, 3]    == 2.0f0
    @test A[2, 2]    == 3.0f0
    @test A[3, 1]    == 4.0f0
    @test A[3, 3]    == 5.0f0
    @test A[1, 2]    == 0.0f0
end

@testset "convert SparseMatrixCSC → ust → SparseMatrixCSC" begin
    A  = sparse([1,1,2,3,3], [1,3,2,1,3], Float32[1,2,3,4,5], 3, 4)
    u  = ust(A)
    A2 = convert(SparseMatrixCSC{Float32,Int32}, u)
    @test A == A2
end

# ─── Preserves index origin ───────────────────────────────────────────────────

@testset "convert_format preserves index origin" begin
    rowptr = Int32[1, 3, 4, 6]
    colind = Int32[1, 3, 2, 1, 3]
    vals   = Float32[1, 2, 3, 4, 5]
    u   = csr_tensor(rowptr, colind, vals, (3, 4); origin=OneBased())
    u2  = convert_format(u, Formats.CSC)
    @test index_origin(u2) == OneBased()
end
