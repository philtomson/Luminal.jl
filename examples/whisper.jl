# Whisper Compilation and Execution Example
# 
# This example demonstrates how to set up the Whisper model architecture,
# compile the execution graph ahead-of-time, and simulate a transcription loop
# to measure the inference performance.

using Luminal
using Luminal.NN
using Printf

function main()
    println("=========================================")
    println("  Luminal.jl - Whisper Compiled Example  ")
    println("=========================================")
    
    # 1. Initialize the computation Graph
    graph = Graph()
    
    println("\n[1/4] Constructing Whisper Architecture...")
    
    audio_encoder = NN.AudioEncoder(graph)
    text_decoder = NN.TextDecoder(graph)
    
    # Define input tensor placeholders
    # Audio input (Batch=1, Mels=80, Frames=50)
    batch_size = 1
    mels = 80
    audio_frames = 50
    audio_tensor = tensor(graph, [batch_size, mels, audio_frames])
    
    # Text input tokens (Batch=1, SeqLen=10)
    seq_len = 10
    text_tensor = tensor(graph, [batch_size, seq_len])
    
    # Build the forward pass through the network
    enc_out = audio_encoder(audio_tensor)
    logits = text_decoder(enc_out, text_tensor)
    
    device = get_device()
    println("\n[2/4] Hardware Detected: $device")
    
    # Initialize Random Weights since weight loading isn't implemented
    println("Initializing weights randomly for benchmarking...")
    
    inputs = Dict{Int, Any}()
    
    for (i, node) in enumerate(graph.nodes)
        id = i
        if node.op isa Luminal.Function && node.op.name == "InputTensor" && !haskey(inputs, id)
            dims = Luminal.realized_dims(graph.shapes[id])
            has_syms = !all(x -> x isa Int, dims)
            if !has_syms && id != audio_tensor.id && id != text_tensor.id
                inputs[id] = randn(Float32, Tuple(dims))
            end
        end
    end
    
    # 3. Compile Graph
    # This phase runs the SymbolicUtils.jl optimizations (Operator Fusion, etc.)
    # and pre-allocates all necessary buffers perfectly sized for the execution.
    println("\n[3/4] Compiling Execution Graph ahead-of-time...")
    exec_fn = compile(graph)
    println("Compilation complete. Fused graph has $(length(graph.nodes)) nodes.")
    
    # 4. Simulate transcription loop
    gen_tokens = 50
    println("\n[4/4] Simulating transcription of $gen_tokens iterations...")
    
    # Prepare initial dummy sequence
    inputs[audio_tensor.id] = Float32.(randn(batch_size, mels, audio_frames))
    inputs[text_tensor.id] = Float32.(rand(0:(NN.VOCAB_SIZE-1), batch_size, seq_len))
    
    # Warmup kernel (compiles CUDA kernels if using GPU)
    print("Running Warmup...")
    _ = exec_fn(inputs, device)
    println(" Done.")
    
    # Benchmarking Loop
    start_time = time()
    for i in 1:gen_tokens
        # In a real generation loop, you would append the argmax token to the sequence
        # and re-run. Here we just inject dummy variables.
        inputs[audio_tensor.id] = Float32.(randn(batch_size, mels, audio_frames))
        inputs[text_tensor.id] = Float32.(rand(0:(NN.VOCAB_SIZE-1), batch_size, seq_len))
        
        # Execute the compiled function. Return value is the evaluation of all graph nodes.
        results = exec_fn(inputs, device)
        
        # Access the final output logits via the output node ID: `logits.id`
        out_logits = results[logits.id]
        
        if i % 10 == 0
            print(".")
        end
    end
    println()
    
    end_time = time()
    
    # 5. Performance Metrics
    total_time = end_time - start_time
    ms_per_step = (total_time / gen_tokens) * 1000
    steps_per_sec = gen_tokens / total_time
    
    println("\n--- Performance Summary ---")
    @printf("Ran %d decoding steps in %.2fs\n", gen_tokens, total_time)
    @printf("Speed: %.2f ms/step (%.2f steps/s)\n", ms_per_step, steps_per_sec)
    println("=========================================")
end

main()
