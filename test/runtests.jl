using Test

gpu_available() = get(ENV, "JLUST_GPU_TESTS", "0") == "1"

@testset "JLUST" begin
    include("test_formats.jl")
    include("test_tensor.jl")
    include("test_convert.jl")
    include("test_cuda.jl")
    include("test_emitter.jl")
    include("test_emitter_cuda.jl")
end
