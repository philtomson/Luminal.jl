using Luminal
using Luminal.NN
using BenchmarkTools
using CUDA
using Random

function benchmark_llama()
    @info "Initializing Llama Model..."
    graph = Graph()
    # Use a small configuration for quick benchmarking, adjust for larger tests
    vocab_size = 32000
    hidden_dim = 1024 # 1K for decent size but fast
    n_layers = 2
    n_heads = 16
    n_kv_heads = 16 # Temporarily avoid GQA path
    intermediate_dim = 2048
    
    # Check for GPU
    device = get_device()
    @info "Using device: $(typeof(device))"

    # Define model
    llama = NN.Llama(graph; 
                     vocab_size=vocab_size, 
                     hidden=hidden_dim, 
                     n_layers=n_layers, 
                     n_heads=n_heads, 
                     n_kv_heads=n_kv_heads, 
                     intermediate=intermediate_dim)

    # Input data
    batch_size = 1
    seq_len = 128
    input_ids = zeros(Int, batch_size, seq_len)
    
    # Create input tensor
    x = tensor(graph, [batch_size, seq_len])
    
    # Forward pass symbolic
    out = llama(x, 0)
    display(out.shape)
    
    @info "Compiling and Executing..."
    
    # Create input dictionary
    # Llama expects Float inputs for embedding currently? 
    # Wait, Embedding layer takes indices. 
    # Let's check NN.Embedding implementation.
    # It uses `gather(weight, input)`. 
    # `gather` expects `indexes` to be the second argument.
    # In `Execution.jl`: indices = Int.(input_values[2]) .+ 1
    # So input should be the indices.
    
    inputs = Dict{Int, Any}()
    
    for (id, node) in enumerate(graph.nodes)
        if node.op isa Luminal.Function && node.op.name == "InputTensor"
            if id == x.id
                inputs[id] = Float32.(input_ids)
            else
                # Generate random weights/biases
                dims = Luminal.realized_dims(graph.shapes[id])
                # Use smaller range to avoid overflow/NaN issues if any
                inputs[id] = randn(Float32, dims...) * 0.02f0
            end
        end
    end
    
    # Warmup
    @info "Warmup run..."
    execute(graph, out.id, inputs)
    
    # Benchmark
    @info "Benchmarking..."
    # We want to benchmark the execution time, including data transfer if part of the loop,
    # or just the execute call.
    # The `execute` function handles transfer `to_device` inside. 
    # For a fair benchmark, we might want to pre-transfer if possible, 
    # but `execute` interface takes a Dict of raw inputs.
    
    # Let's verify `execute` logic.
    # `results = to_device(initial_inputs, device)` is called at start.
    
    # To benchmark pure execution, we'd need a lower level API or modify execute.
    # For now, let's benchmark the full `execute` call as it represents end-to-end usage.
    
    t = @benchmark execute($graph, $(out.id), $inputs)
    
    display(t)
end

benchmark_llama()
