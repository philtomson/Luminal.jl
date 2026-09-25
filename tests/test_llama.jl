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

# Llama components in the (Hidden, Seq, Batch) layout, run through the interpreter.
@testset "Llama Components" begin
    @testset "RMSNorm" begin
        graph = Graph()
        dim = 128
        ln = NN.RMSNorm(dim, graph; epsilon=1f-5)
        input_data = randn(Float32, dim, 2)          # (hidden, seq)
        x = tensor(graph, [dim, 2])
        out = ln(x)
        w = randn(Float32, dim)
        result = execute_with_random_inputs(graph, out.id, Dict(x.id => input_data, ln.weight.id => w))
        @test result ≈ input_data ./ sqrt.(sum(abs2, input_data; dims=1) ./ dim .+ 1f-5) .* w rtol = 1e-4
    end

    @testset "Mlp" begin
        graph = Graph()
        hidden, inter = 128, 256
        mlp = NN.Mlp(hidden, inter, graph)
        x = tensor(graph, [hidden, 2])               # (hidden, seq)
        out = mlp(x)
        result = execute_with_random_inputs(graph, out.id, Dict(x.id => randn(Float32, hidden, 2)))
        @test size(result) == (hidden, 2)
        @test all(!isnan, result)
    end

    @testset "RoPE" begin
        graph = Graph()
        head_dim, seq, n_heads, batch = 32, 10, 4, 1
        input_data = randn(Float32, head_dim, seq, n_heads, batch)   # (D, S, H, B)
        x = tensor(graph, [head_dim, seq, n_heads, batch])
        out = NN.apply_rotary_embeddings(x, 0)
        result = execute_with_random_inputs(graph, out.id, Dict(x.id => input_data))
        @test size(result) == (head_dim, seq, n_heads, batch)
        # rotate-half RoPE rotates each (i, i + D/2) pair, preserving its norm
        half = head_dim ÷ 2
        for t in 1:seq, i in (1, 7, half)
            nin = hypot(input_data[i, t, 1, 1], input_data[i + half, t, 1, 1])
            nout = hypot(result[i, t, 1, 1], result[i + half, t, 1, 1])
            @test isapprox(nin, nout, atol=1e-4)
        end
        @test result[:, 1, :, :] ≈ input_data[:, 1, :, :]   # position 0: no rotation
    end

    @testset "Attention" begin
        graph = Graph()
        hidden, n_heads = 128, 4
        sa = NN.SelfAttention(hidden, n_heads, n_heads, graph)
        x = tensor(graph, [hidden, 8, 1])            # (hidden, seq=8, batch=1)
        out = sa(x, 0)
        result = execute_with_random_inputs(graph, out.id, Dict(x.id => randn(Float32, hidden, 8, 1)))
        @test size(result) == (hidden, 8, 1)
        @test all(!isnan, result)
    end

    @testset "Top-level Llama" begin
        graph = Graph()
        llama = NN.Llama(graph; vocab_size=1000, hidden=64, n_layers=2, n_heads=4, n_kv_heads=4,
                         intermediate=128)
        input_ids = Float32.(rand(0:999, 8, 1))     # (seq=8, batch=1) token ids
        x = tensor(graph, [8, 1])
        out = llama(x, 0)
        result = execute_with_random_inputs(graph, out.id, Dict(x.id => input_ids))
        @test size(result) == (1000, 8, 1)           # (vocab, seq, batch)
        @test all(!isnan, result)
    end
end
