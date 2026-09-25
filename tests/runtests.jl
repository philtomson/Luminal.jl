# Runs every tests/test_*.jl file, each in its own testset and module.
#   julia --project=. tests/runtests.jl            # all tests
#   julia --project=. tests/runtests.jl llama rope # files whose name contains a pattern
using Test

files = sort(filter(f -> startswith(f, "test_") && endswith(f, ".jl"), readdir(@__DIR__)))
isempty(ARGS) || filter!(f -> any(p -> occursin(p, f), ARGS), files)

@testset "Luminal.jl" begin
    for f in files
        @testset "$f" begin
            Base.include(Module(Symbol(splitext(f)[1])), joinpath(@__DIR__, f))
        end
    end
end
