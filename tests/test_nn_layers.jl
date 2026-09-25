using Test
using Luminal
using Luminal.NN

# Layers use the (Hidden, Seq..., Batch) layout: features along dim 1.
@testset "NN Layers" begin

    @testset "Linear Layer" begin
        g = Luminal.Graph()
        # x: (in=3, seq=2); weight stored (out=4, in=3); y = W * x .+ b
        input_data = Float32[1 4; 2 5; 3 6]
        weights = Float32[1 5 9; 2 6 10; 3 7 11; 4 8 12]
        bias = Float32[0.1, 0.2, 0.3, 0.4]
        a = Luminal.tensor(g, input_data)
        model = Linear(3, 4, g; bias=true)
        out = model(a)
        res = Luminal.execute(g, out.id, Dict(a.id => input_data, model.weight.id => weights,
                                              model.bias.id => bias))
        @test res ≈ weights * input_data .+ bias
    end

    @testset "Embedding Layer" begin
        g = Luminal.Graph()
        # table (vocab=3, dim=4); tokens [2, 0] (0-indexed) -> (dim=4, seq=2)
        matrix_data = Float32[1 2 3 4; 5 6 7 8; 9 10 11 12]
        indices = Float32[2, 0]
        a = Luminal.tensor(g, indices)
        model = Embedding(3, 4, g)
        out = model(a)
        res = Luminal.execute(g, out.id, Dict(a.id => indices, model.weight.id => matrix_data))
        @test res ≈ Float32[9 1; 10 2; 11 3; 12 4]
    end

    @testset "LayerNorm" begin
        g = Luminal.Graph()
        # x: (features=3, seq=2); each column normalized to mean 0, RMS 1
        data = Float32[1 4; 2 5; 3 6]
        a = Luminal.tensor(g, data)
        model = LayerNorm(3, g; weight=false, bias=false)
        out = model(a)
        res = Luminal.execute(g, out.id, Dict(a.id => data))
        for j in 1:2
            col = res[:, j]
            @test isapprox(sum(col) / 3, 0.0, atol=1e-5)
            @test isapprox(sqrt(sum(col .^ 2) / 3), 1.0, atol=1e-3)
        end
    end
end
