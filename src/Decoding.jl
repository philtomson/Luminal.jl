# Decoding.jl
#
# Inference loops for Whisper models.
# Supports greedy decoding using the incremental graph and KV caching.

module Decoding

using ..Luminal
using ..Luminal.NN

export greedy_decode

"""
    greedy_decode(td, tokenizer, enc_output_array, model_weights_dir; 
                  language="en", task=:transcribe, max_len=448, device=nothing)

Run greedy decoding for the Whisper model.

Args:
- `td`                 : A Whisper TextDecoder instance (template for structure)
- `tokenizer`          : A WhisperTokenizer instance
- `enc_output_array`   : The output of the audio encoder (batch, enc_seq, d_model)
- `model_weights_dir`  : Path to the directory containing weights.
- `language`           : Target language code (e.g., "en")
- `task`               : :transcribe or :translate
- `max_len`            : Maximum number of tokens to decode
- `device`             : Optional device to run on

Returns:
- `String`             : The decoded transcript
- `Vector{Int}`        : The sequence of token IDs (including SOT and EOT)
"""
function greedy_decode(td::TextDecoder, 
                       tokenizer::WhisperTokenizer, 
                       enc_output_array::Array{Float32, 3},
                       model_weights_dir::String;
                       language::String="en",
                       task::Symbol=:transcribe,
                       max_len::Int=448,
                       device=nothing)
    
    batch, enc_seq, d_model = size(enc_output_array)
    @assert batch == 1 "Batch size > 1 not supported in greedy_decode yet"
    target_device = (device === nothing ? get_device() : device)

    # 1. Load weights once into memory
    files = filter(f -> endswith(f, ".safetensors"), readdir(model_weights_dir; join=true))
    weights_dict = Dict{String, Array}()
    for f in files
        merge!(weights_dict, Luminal.SafeTensors.load_safetensors(f))
    end

    # 2. Precompute Cross KV
    # We build a small graph just for this
    cg = Graph()
    creg = WeightRegistry()
    ctd = TextDecoder(cg; reg=creg)
    
    enc_in = Luminal.tensor(cg, [batch, enc_seq, d_model])
    kv_tensors = project_cross_kv(ctd, cg, enc_in)
    
    # Pack K and V IDs for compilation
    kv_ids = Int[]
    for (k, v) in kv_tensors
        push!(kv_ids, k.id)
        push!(kv_ids, v.id)
    end
    
    load_weights!(cg, creg, weights_dict; device=target_device)
    c_exec = compile(cg)
    
    # Run precompute
    kv_results = Luminal.execute(cg, kv_ids, Dict(enc_in.id => enc_output_array))
    
    # Format into layers: [(k,v), (k,v), ...]
    cross_kv_arrays = Vector{Tuple{Array{Float32,4}, Array{Float32,4}}}()
    for i in 1:length(td.layers)
        push!(cross_kv_arrays, (kv_results[kv_ids[2i-1]], kv_results[kv_ids[2i]]))
    end

    # 3. Initialize KV Cache
    cache = KVCacheState(batch, length(td.layers), HEADS, HEAD_DIM, enc_seq; 
                         max_seq=MAX_TARGET_POSITION)

    # 4. Starting sequence
    tokens = sot_sequence(tokenizer; language=language, task=task, notimestamps=true)
    
    # 5. Greedy Decoding Loop
    compiled_graphs = Dict{Int, Any}()

    for _ in 1:max_len
        pos = length(tokens) - 1
        if pos >= MAX_TARGET_POSITION - 1
            break
        end
        
        # Get/Compile graph for this step position
        if !haskey(compiled_graphs, pos)
            g = Graph()
            reg = WeightRegistry()
            td_step = TextDecoder(g; reg=reg)
            idg = build_decode_step!(td_step, g, enc_seq, pos; batch=batch)
            load_weights!(g, reg, weights_dict; device=target_device)
            exec_fn = compile(g)
            compiled_graphs[pos] = (exec_fn, idg)
        end
        
        exec_fn, idg = compiled_graphs[pos]
        
        # Run one step
        current_token = Float32[tokens[end]]
        token_input = Base.reshape(current_token, (1, 1)) # (batch, 1)
        
        exec_fn, idg = compiled_graphs[pos]
        logits = decode_step!(exec_fn, idg, cache, token_input, cross_kv_arrays; device=target_device)
        # logits shape: (batch, 1, vocab_size) -> (vocab_size,)
        
        # Greedy sample
        next_token = argmax(view(logits, 1, 1, :))
        
        # Julia argmax is 1-indexed, but Whisper vocab is 0-indexed?
        # Check: Whisper vocab is 0-indexed in vocab.json. 
        # So next_token - 1.
        push!(tokens, next_token - 1)
        
        # Stop if EOT
        if tokens[end] == tokenizer.eot_id
            break
        end
    end

    # 6. Decode tokens to text
    decoded_text = decode(tokenizer, tokens; skip_special=true)
    
    return decoded_text, tokens
end

end # module Decoding
