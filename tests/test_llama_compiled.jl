
using Luminal
using Luminal.NN
using Test
using Statistics
using CUDA

# Helper to execute a graph using compilation
function execute_compiled_with_random_inputs(graph::Graph, output_id::Int, input_overrides::Dict = Dict())
    # 1. Compile the graph
    println("Compiling graph with $(length(graph.nodes)) nodes...")
    exec_fn = compile(graph)
    println("Graph compiled.")
    
    # 2. Prepare inputs
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
    
    # 3. Execute
    device = get_device()
    println("Executing on device: $device")
    
    # Note: compiled function returns results[end], which matches output_id usually
    # But if output_id is not the last node, this might be wrong.
    # Llama tests usually take the result of the last operation added.
    # Let's hope output_id is the last ID.
    if output_id != length(graph.nodes)
        @warn "Output ID $output_id is not the last node $(length(graph.nodes)). Compiled execution returned all results."
    end
    
    # exec_fn returns Vector{Any} indexed by NodeID
    all_results = exec_fn(inputs; device=device)
    res = all_results[output_id]
    
    return from_device(res)
end

@testset "Llama Components (Compiled)" begin
    @testset "RMSNorm" begin
        graph = Graph()
        dim = 128
        ln = NN.RMSNorm(dim, graph; epsilon=1e-5)
        
        # Layout: (Hidden, Seq)
        input_data = randn(Float32, dim, 2)
        x = tensor(graph, [dim, 2])
        out = ln(x)
        
        result = execute_compiled_with_random_inputs(graph, out.id, Dict(x.id => input_data))
        
        @test all(!isnan, result)
        @test size(result) == (dim, 2)
    end

    @testset "Mlp" begin
        graph = Graph()
        hidden = 128
        inter = 256
        mlp = NN.Mlp(hidden, inter, graph)
        
        x = tensor(graph, [hidden, 2])  # (Hidden, Seq)
        out = mlp(x)
        
        result = execute_compiled_with_random_inputs(graph, out.id)
        
        @test size(result) == (hidden, 2)
        @test all(!isnan, result)
    end

    @testset "RoPE" begin
        graph = Graph()
        batch, n_heads, seq, head_dim = 1, 4, 10, 32
        x = tensor(graph, [head_dim, seq, n_heads, batch])  # (D, S, H, B)
        
        out = NN.apply_rotary_embeddings(x, 0)
        
        result = execute_compiled_with_random_inputs(graph, out.id)
        
        @test size(result) == (head_dim, seq, n_heads, batch)
        @test all(!isnan, result)
    end

    @testset "Attention" begin
        graph = Graph()
        hidden = 128
        n_heads = 4
        sa = NN.SelfAttention(hidden, n_heads, n_heads, graph)
        
        x = tensor(graph, [hidden, 8, 1])  # (Hidden, Seq, Batch)
        out = sa(x, 0)
        
        result = execute_compiled_with_random_inputs(graph, out.id)
        
        @test size(result) == (hidden, 8, 1)
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
                         
        x = tensor(graph, [8, 1])  # token ids, (Seq, Batch)
        out = llama(x, 0)
        
        ids = Float32.(rand(0:999, 8, 1))
        result = execute_compiled_with_random_inputs(graph, out.id, Dict(x.id => ids))
        
        @test size(result) == (1000, 8, 1)  # (Vocab, Seq, Batch)
        @test all(!isnan, result)
    end
end
