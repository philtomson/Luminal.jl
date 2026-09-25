using Luminal
using Test

# Building a graph records operations; nothing is computed until the graph is
# compiled and executed.
@testset "Lazy Graph Evaluation" begin
    @testset "Deferred computation — no eager evaluation" begin
        g = Luminal.Graph()
        a = Luminal.tensor(g, [2, 2])
        b = Luminal.tensor(g, [2, 2])
        c = a + b
        @test length(g.nodes) == 3                 # a, b, a+b
        @test c isa Luminal.GraphTensor            # a handle, not a value
        @test isempty(g.tensors)                   # no data anywhere yet

        av, bv = rand(Float32, 2, 2), rand(Float32, 2, 2)
        ex = compile(g; device=CPUDevice(), retain=[c.id])
        @test ex(Dict{Int,Any}(a.id => av, b.id => bv); device=CPUDevice())[c.id] ≈ av .+ bv
    end

    @testset "Chained operations stay deferred" begin
        g = Luminal.Graph()
        a = Luminal.tensor(g, [4]); b = Luminal.tensor(g, [4]); c = Luminal.tensor(g, [4])
        result = Luminal.relu((a + b) * c)
        @test isempty(g.tensors)
        @test result.id == length(g.nodes)         # the last node recorded

        av, bv, cv = randn(Float32, 4), randn(Float32, 4), randn(Float32, 4)
        ex = compile(g; device=CPUDevice(), retain=[result.id])
        out = ex(Dict{Int,Any}(a.id => av, b.id => bv, c.id => cv); device=CPUDevice())[result.id]
        @test out ≈ max.((av .+ bv) .* cv, 0f0)
    end
end
