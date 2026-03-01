# Llama Compilation and Execution Example
# 
# This example demonstrates how to set up the Llama model architecture,
# compile the execution graph ahead-of-time, and simulate a token generation loop
# to measure the inference performance.

using Luminal
using Luminal.NN
using Printf

function main()
    println("=========================================")
    println("   Luminal.jl - Llama Compiled Example   ")
    println("=========================================")
    
    # 1. Initialize the computation Graph
    graph = Graph()
    
    # 2. Setup Model Configuration
    # We use a scaled-down architecture (TinyLlama-like) for demonstration purposes
    vocab_size = 32000
    hidden_dim = 512
    n_layers = 4
    n_heads = 8
    n_kv_heads = 8
    intermediate = 1024
    
    println("\n[1/4] Constructing Llama Architecture...")
    println("Config: Layers=$n_layers, Hidden=$hidden_dim, Heads=$n_heads")
    
    llama = NN.Llama(graph; 
                     vocab_size=vocab_size, 
                     hidden=hidden_dim, 
                     n_layers=n_layers, 
                     n_heads=n_heads, 
                     n_kv_heads=n_kv_heads, 
                     intermediate=intermediate)
    
    # Define input tensor placeholder (Batch=1, SeqLen=1 for decoding simulation)
    batch_size = 1
    seq_len = 1
    input_tensor = tensor(graph, [batch_size, seq_len])
    
    # Build the forward pass through the network
    # Note: 0 is the starting positional offset for RoPE embeddings
    out = llama(input_tensor, 0)
    
    device = get_device()
    println("\n[2/4] Hardware Detected: $device")
    
    # 3. Compile Graph
    # This phase runs the SymbolicUtils.jl optimizations (Operator Fusion, etc.)
    # and pre-allocates all necessary buffers perfectly sized for the execution.
    println("\n[3/4] Compiling Execution Graph ahead-of-time...")
    exec_fn = compile(graph)
    println("Compilation complete. Fused graph has $(length(graph.nodes)) nodes.")
    
    # 4. Simulate generation loop
    gen_tokens = 100
    println("\n[4/4] Simulating generation of $gen_tokens tokens...")
    
    # Prepare initial dummy tokens
    input_ids = Float32.(rand(0:(vocab_size-1), batch_size, seq_len))
    inputs = Dict{Int, Any}(input_tensor.id => input_ids)
    
    # Warmup kernel (compiles CUDA kernels if using GPU)
    print("Running Warmup...")
    _ = exec_fn(inputs, device)
    println(" Done.")
    
    # Benchmarking Loop
    start_time = time()
    for i in 1:gen_tokens
        # In a real generation loop, you would sample the next token by inspecting logits
        # For simulation, we simply inject a randomly generated dummy token IDs.
        inputs[input_tensor.id] = Float32.(rand(0:(vocab_size-1), batch_size, seq_len))
        
        # Execute the compiled function. Return value is the evaluation of all graph nodes.
        results = exec_fn(inputs, device)
        
        # Access the final output logits via the output node ID: `out.id`
        logits = results[out.id]
        
        if i % 25 == 0
            print(".")
        end
    end
    println()
    
    end_time = time()
    
    # 5. Performance Metrics
    total_time = end_time - start_time
    ms_per_token = (total_time / gen_tokens) * 1000
    tok_per_sec = gen_tokens / total_time
    
    println("\n--- Performance Summary ---")
    @printf("Generated %d tokens in %.2fs\n", gen_tokens, total_time)
    @printf("Speed: %.2f ms/token (%.2f tok/s)\n", ms_per_token, tok_per_sec)
    println("=========================================")
end

main()
