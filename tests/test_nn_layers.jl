using Test
using Luminal
using Luminal.NN
using CUDA

@testset "NN Layers" begin
    
    @testset "Linear Layer" begin
        g = Luminal.Graph()
        # (batch=2, in=3) * (in=3, out=4) -> (batch=2, out=4)
        input_data = Float32[1 2 3; 4 5 6] # 2x3
        # Weight matrix for in=3, out=4 should be 4x3 in standard storage.
        # But our test expectations use (input * weights).
        # weights here is 3x4.
        weights = Float32[1 2 3 4; 5 6 7 8; 9 10 11 12]
        bias = Float32[0.1, 0.2, 0.3, 0.4] # 4
        
        a = Luminal.tensor(g, input_data)
        model = Linear(3, 4, g; bias=true)
        
        out = model(a)
        
        expected = (input_data * weights) .+ bias'
        
        res = Luminal.execute(g, out.id, Dict(
            a.id => input_data,
            model.weight.id => weights',
            model.bias.id => bias
        ))
        
        @test res ≈ expected
    end

    @testset "Embedding Layer" begin
        g = Luminal.Graph()
        # matrix (vocab=3, dim=4)
        # indexes (batch=2) -> [2, 0] (0-indexed in Luminal's Gather op)
        matrix_data = Float32[1 2 3 4; 5 6 7 8; 9 10 11 12]
        indices = Float32[2, 0]
        
        a = Luminal.tensor(g, indices)
        model = Embedding(3, 4, g)
        out = model(a)
        
        res = Luminal.execute(g, out.id, Dict(
            a.id => indices,
            model.weight.id => matrix_data
        ))
        
        # vocab[2] is [9, 10, 11, 12], vocab[0] is [1, 2, 3, 4]
        @test res ≈ [9 10 11 12; 1 2 3 4]
    end

    @testset "LayerNorm" begin
        g = Luminal.Graph()
        # x: (2, 3)
        data = Float32[1 2 3; 4 5 6]
        a = Luminal.tensor(g, data)
        model = LayerNorm(3, g; weight=false, bias=false)
        out = model(a)
        
        res = Luminal.execute(g, out.id, Dict(a.id => data))
        
        for i in 1:2
            row = res[i, :]
            @test isapprox(sum(row)/3, 0.0, atol=1e-5)
            @test isapprox(sqrt(sum(row.^2)/3), 1.0, atol=1e-5)
        end
    end
end
