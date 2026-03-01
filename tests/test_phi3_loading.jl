using Luminal
using Luminal.NN
using Test

@testset "Phi3 Weight Loading" begin
    graph = Graph()
    reg = WeightRegistry()
    
    # Scale down Phi-3 for faster testing
    vocab_size = 100
    hidden = 64
    n_layers = 1
    n_heads = 4
    n_kv_heads = 1
    intermediate = 128
    
    phi3 = NN.Phi3(graph, reg; 
                   vocab_size=vocab_size, 
                   hidden=hidden, 
                   n_layers=n_layers, 
                   n_heads=n_heads, 
                   n_kv_heads=n_kv_heads, 
                   intermediate=intermediate)
    
    # Verify mapping
    @test haskey(reg.mapping, "model.embed_tokens.weight")
    @test haskey(reg.mapping, "model.layers.0.self_attn.q_proj.weight")
    @test haskey(reg.mapping, "model.layers.0.mlp.gate_proj.weight")
    @test haskey(reg.mapping, "model.norm.weight")
    @test haskey(reg.mapping, "lm_head.weight")
    
    # Prepare dummy data for all registered keys
    tensors = Dict{String, Array{Float32}}()
    for key in keys(reg.mapping)
        # Find the node and its shape
        node_id = reg.mapping[key]
        shape = Luminal.realized_dims(graph.shapes[node_id])
        tensors[key] = randn(Float32, shape...)
    end
    
    # Load weights using the dictionary-based overload
    load_weights!(graph, reg, tensors)
    
    # Verify that graph.tensors is populated
    for (key, node_id) in reg.mapping
        @test haskey(graph.tensors, (node_id, 1))
        # graph.tensors stores the data (Array or CuArray)
        data = graph.tensors[(node_id, 1)]
        @test !any(isnan, data)
    end
    
    println("Phi3 weight loading verification successful (via dict).")
end
