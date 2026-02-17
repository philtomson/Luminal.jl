using Luminal
using Luminal.NN
using Test
using Statistics

# Helper to execute a graph by providing random data for all required inputs
function execute_with_random_inputs(graph::Graph, output_id::Int, input_overrides::Dict = Dict())
    inputs = Dict{Int, Any}()
    for (id, node) in enumerate(graph.nodes)
        if node.op isa Luminal.Function && node.op.name == "InputTensor"
            if haskey(input_overrides, id)
                inputs[id] = input_overrides[id]
            else
                # Generate random data based on shape
                dims = Luminal.realized_dims(graph.shapes[id])
                inputs[id] = randn(Float32, dims...)
            end
        end
    end
    return execute(graph, output_id, inputs)
end

@testset "Llama Components" begin
    @testset "RMSNorm" begin
        graph = Graph()
        dim = 128
        ln = NN.RMSNorm(dim, graph; epsilon=1e-5)
        
        # input: batch=2, dim=128
        input_data = randn(Float32, 2, dim)
        x = tensor(graph, [2, dim]) # Note: 2rd arg is shape
        out = ln(x)
        
        result = execute_with_random_inputs(graph, out.id, Dict(x.id => input_data))
        
        @test all(!isnan, result)
        @test size(result) == (2, dim)
    end

    @testset "Mlp" begin
        graph = Graph()
        hidden = 128
        inter = 256
        mlp = NN.Mlp(hidden, inter, graph)
        
        input_data = randn(Float32, 2, hidden)
        x = tensor(graph, [2, hidden])
        out = mlp(x)
        
        result = execute_with_random_inputs(graph, out.id, Dict(x.id => input_data))
        
        @test size(result) == (2, hidden)
        @test all(!isnan, result)
    end

    @testset "RoPE" begin
        graph = Graph()
        batch, n_heads, seq, head_dim = 1, 4, 10, 32
        input_data = randn(Float32, batch, n_heads, seq, head_dim)
        x = tensor(graph, [batch, n_heads, seq, head_dim])
        
        out = NN.apply_rotary_embeddings(x, 0)
        
        result = execute_with_random_inputs(graph, out.id, Dict(x.id => input_data))
        
        @test size(result) == (1, 4, 10, 32)
        @test all(!isnan, result)
        
        # RoPE should preserve norm of the complex pairs
        norm_in = sqrt(input_data[1, 1, 1, 1]^2 + input_data[1, 1, 1, 2]^2)
        norm_out = sqrt(result[1, 1, 1, 1]^2 + result[1, 1, 1, 2]^2)
        @test isapprox(norm_in, norm_out, atol=1e-5)
    end

    @testset "Attention" begin
        graph = Graph()
        hidden = 128
        n_heads = 4
        sa = NN.SelfAttention(hidden, n_heads, n_heads, graph)
        
        input_data = randn(Float32, 1, 8, hidden) # batch=1, seq=8
        x = tensor(graph, [1, 8, hidden])
        
        out = sa(x, 0)
        
        result = execute_with_random_inputs(graph, out.id, Dict(x.id => input_data))
        
        @test size(result) == (1, 8, hidden)
        @test all(!isnan, result)
    end
    
    @testset "Top-level Llama" begin
        graph = Graph()
        # Small Llama for testing
        llama = NN.Llama(graph; 
                         vocab_size=1000, 
                         hidden=64, 
                         n_layers=2, 
                         n_heads=4, 
                         n_kv_heads=4, 
                         intermediate=128)
                         
        input_ids = Float32.(rand(0:999, 1, 8)) # batch=1, seq=8
        x = tensor(graph, [1, 8])
        
        out = llama(x, 0)
        
        result = execute_with_random_inputs(graph, out.id, Dict(x.id => input_ids))
        
        @test size(result) == (1, 8, 1000)
        @test all(!isnan, result)
    end
end
