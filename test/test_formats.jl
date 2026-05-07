using Test, JLUST, JLUST.Formats

@testset "format properties" begin
    # From Python test_format_properties
    @test Formats.COO.name == :COO
    @test length(Formats.COO.dimensions) == 2
    @test length(Formats.COO.levels) == 2
    @test Formats.COO.is_identity
    @test Formats.COO.is_ordered
    @test Formats.COO.is_unique  # nonunique compressed + singleton → unique overall

    @test Formats.CSR.name == :CSR
    @test length(Formats.CSR.dimensions) == 2
    @test length(Formats.CSR.levels) == 2
    @test Formats.CSR.is_identity
    @test Formats.CSR.is_ordered
    @test Formats.CSR.is_unique

    @test Formats.CSC.name == :CSC
    @test length(Formats.CSC.dimensions) == 2
    @test length(Formats.CSC.levels) == 2
    @test !Formats.CSC.is_identity
    @test Formats.CSC.is_ordered
    @test Formats.CSC.is_unique

    bsr = Formats.BSRRight((4, 8))
    @test bsr.name == :BSRRight4x8
    @test length(bsr.dimensions) == 2
    @test length(bsr.levels) == 4
    @test !bsr.is_identity
    @test bsr.is_ordered
    @test bsr.is_unique
end

@testset "level format predicates" begin
    @test is_ordered(DenseLevel())
    @test is_ordered(CompressedLevel())
    @test !is_ordered(CompressedLevel(ordered=false, unique=false))
    @test is_unique(DenseLevel())
    @test is_unique(CompressedLevel())
    @test !is_unique(CompressedLevel(unique=false))
    @test is_unique(SingletonLevel())
    @test is_unique(DeltaLevel(3))
end

@testset "dim2lvl 2D" begin
    # Expected values from Python test_dim2lvl_2d
    @test dim2lvl(Formats.COO,   [2, 1]) == [2, 1]
    @test dim2lvl(Formats.CSR,   [2, 1]) == [2, 1]
    @test dim2lvl(Formats.CSC,   [2, 1]) == [1, 2]
    @test dim2lvl(Formats.DCSR,  [2, 1]) == [2, 1]
    @test dim2lvl(Formats.DCSC,  [2, 1]) == [1, 2]
    @test dim2lvl(Formats.CROW,  [2, 1]) == [2, 1]
    @test dim2lvl(Formats.CCOL,  [2, 1]) == [1, 2]
    @test dim2lvl(Formats.DIAI,  [2, 1]) == [-1, 2]
    @test dim2lvl(Formats.DIAJ,  [2, 1]) == [-1, 1]
    @test dim2lvl(Formats.SkewDIAI, [2, 1]) == [3, 2]
    @test dim2lvl(Formats.SkewDIAJ, [2, 1]) == [3, 1]
    @test dim2lvl(Formats.BSRRight((2, 2)), [2, 1]) == [1, 0, 0, 1]
    @test dim2lvl(Formats.BSRLeft((2, 2)),  [2, 1]) == [1, 0, 1, 0]
    @test dim2lvl(Formats.BSCRight((2, 2)), [2, 1]) == [0, 1, 0, 1]
    @test dim2lvl(Formats.BSCLeft((2, 2)),  [2, 1]) == [0, 1, 1, 0]

    # as_size=true
    @test dim2lvl(Formats.COO,  [8, 4]; as_size=true) == [8, 4]
    @test dim2lvl(Formats.CSR,  [8, 4]; as_size=true) == [8, 4]
    @test dim2lvl(Formats.CSC,  [8, 4]; as_size=true) == [4, 8]
    @test dim2lvl(Formats.DIAI, [8, 4]; as_size=true) == [11, 8]
    @test dim2lvl(Formats.DIAJ, [8, 4]; as_size=true) == [11, 4]
    @test dim2lvl(Formats.BSRRight((2, 2)), [8, 4]; as_size=true) == [4, 2, 2, 2]
    @test dim2lvl(Formats.BSCRight((2, 2)), [8, 4]; as_size=true) == [2, 4, 2, 2]
end

@testset "lvl2dim 2D" begin
    # Expected values from Python test_lvl2dim_2d
    @test lvl2dim(Formats.COO,   [2, 1])  == [2, 1]
    @test lvl2dim(Formats.CSR,   [2, 1])  == [2, 1]
    @test lvl2dim(Formats.CSC,   [1, 2])  == [2, 1]
    @test lvl2dim(Formats.DCSR,  [2, 1])  == [2, 1]
    @test lvl2dim(Formats.DCSC,  [1, 2])  == [2, 1]
    @test lvl2dim(Formats.CROW,  [2, 1])  == [2, 1]
    @test lvl2dim(Formats.CCOL,  [1, 2])  == [2, 1]
    @test lvl2dim(Formats.DIAI,  [-1, 2]) == [2, 1]
    @test lvl2dim(Formats.DIAJ,  [-1, 1]) == [2, 1]
    @test lvl2dim(Formats.SkewDIAI, [3, 2]) == [2, 1]
    @test lvl2dim(Formats.SkewDIAJ, [3, 1]) == [2, 1]
    @test lvl2dim(Formats.BSRRight((2, 2)), [1, 0, 0, 1]) == [2, 1]
    @test lvl2dim(Formats.BSRLeft((2, 2)),  [1, 0, 1, 0]) == [2, 1]
    @test lvl2dim(Formats.BSCRight((2, 2)), [0, 1, 0, 1]) == [2, 1]
    @test lvl2dim(Formats.BSCLeft((2, 2)),  [0, 1, 1, 0]) == [2, 1]
end

@testset "round-trip lvl2dim(dim2lvl(crd)) 2D" begin
    # From Python test_dim2lvl2dim_2d: all (i,j) in 0:16 × 0:110
    for i in 0:16, j in 0:110
        crd = [i, j]
        for fmt in (Formats.CSR, Formats.CSC, Formats.DIAI, Formats.DIAJ,
                    Formats.SkewDIAI, Formats.SkewDIAJ, Formats.DELTA(2))
            @test lvl2dim(fmt, dim2lvl(fmt, crd)) == crd
        end
    end
end

@testset "round-trip lvl2dim(dim2lvl(crd)) 3D" begin
    # From Python test_dim2lvl2dim_3d
    for i in 0:16, j in 0:10, k in 0:20
        crd = [i, j, k]
        @test lvl2dim(Formats.COOd(3), dim2lvl(Formats.COOd(3), crd)) == crd
        @test lvl2dim(Formats.CSFd(3), dim2lvl(Formats.CSFd(3), crd)) == crd
    end
end

@testset "validation errors" begin
    i, j, k = dims(:i, :j, :k)

    # Repeated dimensions
    @test_throws InvalidTensorFormat TensorFormat([i, i], [i => DenseLevel()])

    # Dimension not in spec
    @test_throws InvalidTensorFormat TensorFormat([i], [j => DenseLevel()])

    # RHS of add is not a dimension
    @test_throws InvalidTensorFormat TensorFormat([i], [i+1 => DenseLevel()])

    # RHS of div is not an integer
    @test_throws InvalidTensorFormat TensorFormat([i, j], [i÷j => DenseLevel()])

    # Divisor is zero
    @test_throws InvalidTensorFormat TensorFormat([i], [i÷0 => DenseLevel()])

    # Unused dimension
    @test_throws InvalidTensorFormat TensorFormat([i, j, k], [i => DenseLevel(), j => CompressedLevel()])

    # Modulo without prior division
    @test_throws InvalidTensorFormat TensorFormat([i], [i%4 => DenseLevel()])

    # Modulo block size doesn't match division block size
    @test_throws InvalidTensorFormat TensorFormat([i], [i÷4 => DenseLevel(), i%8 => DenseLevel()])

    # Division reuses dimension
    @test_throws InvalidTensorFormat TensorFormat([i], [i÷4 => DenseLevel(), i÷8 => DenseLevel()])

    # Division without matching modulo
    @test_throws InvalidTensorFormat TensorFormat([i], [i÷4 => DenseLevel()])

    # Add/sub dimension reuse in range computation
    @test_throws InvalidTensorFormat TensorFormat([i, j], [i+j => DenseLevel(), j-i => DenseLevel()])

    # Range on compound expression
    @test_throws InvalidTensorFormat TensorFormat([i], [i÷4 => RangeLevel()])

    # Range on dimension not defined via add/sub
    @test_throws InvalidTensorFormat TensorFormat([i], [i => RangeLevel()])

    # Add/sub without matching range
    @test_throws InvalidTensorFormat TensorFormat([i, j], [i+j => DenseLevel()])

    # DeltaLevel bits must be positive
    @test_throws InvalidTensorFormat DeltaLevel(0)
    @test_throws InvalidTensorFormat DeltaLevel(-1)
end

@testset "@tensor_format macro" begin
    # Define a format with the macro and check it matches the constructor
    @tensor_format MacroCSR  (i, j) -> (i : dense, j : compressed)
    @tensor_format MacroDIAI (i, j) -> ((j - i) : compressed, i : range)
    @tensor_format MacroBSR  (i, j) -> ((i÷4) : dense, (j÷4) : compressed, (i%4) : dense, (j%4) : dense)
    @tensor_format MacroCOO  (i, j) -> (i : compressed(nonunique), j : singleton)

    @test MacroCSR  == Formats.CSR
    @test MacroDIAI == Formats.DIAI
    @test MacroBSR  == Formats.BSRRight((4, 4))
    @test MacroCOO  == Formats.COO

    # Level properties
    @tensor_format MacroCOL (i, j) -> (i : compressed(nonunique, unordered), j : singleton)
    col = Formats.COOd(2)  # same structure but named COO — just check the level
    @test MacroCOL.levels[1].second == CompressedLevel(unique=false, ordered=false)

    # Delta
    @tensor_format MacroDelta (i, j) -> (i : dense, j : delta(8))
    @test MacroDelta.levels[2].second == DeltaLevel(8)
end

@testset "named format builders" begin
    @test Formats.BSRRight((2, 4)).name == :BSRRight2x4
    @test length(Formats.BSRRight((2, 4)).levels) == 4

    @test Formats.DELTA(3).name == :Delta3
    @test Formats.DELTA(3).levels[2].second == DeltaLevel(3)

    @test Formats.CSFd(3).name == :CSF3
    @test length(Formats.CSFd(3).levels) == 3

    @test Formats.COOd(3).name == :COO3
    @test length(Formats.COOd(3).levels) == 3

    @test Formats.CSRd(0, 0) == Formats.CSR
    @test Formats.DensedRight(3).name == :Dense3Right
    @test length(Formats.DensedRight(3).levels) == 3
end

@testset "Scalar format" begin
    @test Formats.Scalar.name == :Scalar
    @test isempty(Formats.Scalar.dimensions)
    @test isempty(Formats.Scalar.levels)
    @test Formats.Scalar.is_identity
end
