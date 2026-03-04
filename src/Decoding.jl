# Decoding.jl
#
# Inference loops for Whisper models.
# Supports greedy decoding using the incremental graph and KV caching.

module Decoding

using ..Luminal
using ..Luminal.NN
using ..Luminal.LlamaTokenization
using AMDGPU

export greedy_decode, llama_generate

"""
    llama_generate(model, tokenizer, prompt, model_dir;
                   max_new_tokens=200, max_seq=2048, rope_base=500000f0,
                   device=nothing) -> String

End-to-end greedy text generation for Llama-style (decoder-only) models.

# Steps
1. Encode `prompt` with `tokenizer` (adds BOS).
2. Prefill: run the full prompt forward (`model(ids, 0)`) to warm up, then
   grab logits for the last token position.
3. Decode: iteratively call `llama_decode_step!` with a KV cache until EOS
   or `max_new_tokens` is reached.
4. Decode the output token IDs back to a string.

# Arguments
- `model`         : A `Luminal.NN.Llama` (or `Phi3`) instance
- `tokenizer`     : A `LlamaTokenizer` loaded from the model directory
- `prompt`        : Input string
- `model_dir`     : Path to the directory containing `.safetensors` weights
- `max_new_tokens`: Maximum tokens to generate (default 200)
- `max_seq`       : KV cache capacity (default 2048)
- `rope_base`     : RoPE base frequency (500000 for Llama-3, 10000 for Llama-2/Phi-3)
- `device`        : Device to run on; defaults to `get_device()`
"""
function llama_generate(model,
                         tokenizer::LlamaTokenizer,
                         prompt::String,
                         model_dir::String;
                         max_new_tokens::Int=200,
                         max_seq::Int=2048,
                         rope_base::Float32=500000f0,
                         device=nothing)

    target_device = (device === nothing ? get_device() : device)

    # Note: We now use the streaming load_weights! to load directly into the graph
    # to avoid OutOfMemory errors on large checkpoints.

    # 2. Encode prompt
    prompt_ids = LlamaTokenization.encode(tokenizer, prompt; bos=true)
    @info "Prompt: $(length(prompt_ids)) tokens"

    # 3. Prefill — full prompt in one shot
    #    Build & compile a prefill graph of the right sequence length
    plen = length(prompt_ids)
    pfx_graph = Graph()
    pfx_reg   = WeightRegistry()
    # Rebuild model structure for the prefill length graph
    pfx_model = _rebuild_model_like(model, pfx_graph, pfx_reg; rope_base=rope_base)
    pfx_input = Luminal.tensor(pfx_graph, [plen, 1])
    pfx_out, pfx_kvs = pfx_model(pfx_input, 0; return_kv=true)
    
    # Mark K/V tensors for retrieval so we can populate the cache
    push!(pfx_graph.to_retrieve, pfx_out.id)
    for (k, v) in pfx_kvs
        push!(pfx_graph.to_retrieve, k.id)
        push!(pfx_graph.to_retrieve, v.id)
    end

    @info "Loading weights for prefill graph..."
    # We load ALL weights into a dictionary on the target device once, 
    # and reuse them across all graphs to save VRAM.
    weights_dict = load_weights_to_dict(model_dir; device=target_device)
    
    load_weights!(pfx_graph, pfx_reg, weights_dict; device=target_device)
    
    # We must explicitly retain nodes we want to retrieve 
    retain_pfx = collect(pfx_graph.to_retrieve)
    pfx_exec = compile(pfx_graph; device=target_device, retain=retain_pfx)

    pfx_inputs = Dict{Int,Any}(pfx_input.id => Float32.(Base.reshape(prompt_ids, plen, 1)))
    pfx_results = pfx_exec(pfx_inputs; device=target_device)
    prefill_logits = Array{Float32}(pfx_results[pfx_out.id])  # (vocab, plen, 1)

    # Greedy-pick the first generated token from the last position of prefill
    first_token = argmax(view(prefill_logits, :, plen, 1)) - 1  # 0-indexed

    eos_id = tokenizer.eos_id
    if first_token == eos_id
        return LlamaTokenization.decode(tokenizer, Int[])
    end

    generated = [first_token]
    
    # Free prefill memory
    pfx_exec = nothing
    pfx_graph = nothing
    prefill_logits = nothing
    GC.gc()
    Luminal.reclaim!(target_device)

    # 4. KV-cached decode loop
    avail = Luminal.available_memory(target_device)
    if avail >= 0
        @info "VRAM before decode loop" available_mb=avail
    end
    
    first_attn = model.layers[1].attention
    cache = LlamaKVCacheState(
        length(model.layers), first_attn.n_kv_heads, first_attn.head_dim;
        batch=1, max_seq=max_seq)

    # Step 0 = position immediately after the prefill sequence
    # (step_pos semantics: position in the *cache* of the current token)
    start_pos = plen  # first decode token lands at position plen in the cache
    cache.step_pos = start_pos

    # Populate cache from prefill results
    for (i, (k, v)) in enumerate(pfx_kvs)
        # Prefill K/V shape: (head_dim, plen, kv_heads, batch)
        # Cache slot shape: (head_dim, max_seq, kv_heads, batch)
        k_val = Array{Float16}(pfx_results[k.id])
        v_val = Array{Float16}(pfx_results[v.id])
        
        # In-place update of the cache slices
        cache.self_cache[i][1][:, 1:plen, :, :] = k_val
        cache.self_cache[i][2][:, 1:plen, :, :] = v_val
    end

    @info "Compiling position-agnostic decode graph..."
    dg = Base.invokelatest(Graph)
    dreg = WeightRegistry()
    dm = _rebuild_model_like(model, dg, dreg; rope_base=rope_base)
    pos_sym = Luminal.Sym{Int}(:pos)
    idg = build_llama_decode_step!(dm, dg, pos_sym;
                                   max_seq=max_seq,
                                   rope_base=rope_base)
    
    load_weights!(dg, dreg, weights_dict; device=target_device)
    retain_nodes = vcat(
        idg.logits_id, idg.new_self_k_ids, idg.new_self_v_ids,
        idg.token_input_id, idg.pos_input_id, idg.self_k_ids, idg.self_v_ids
    )
    exec_fn = compile(dg; device=target_device, retain=retain_nodes)

    for step in 0:(max_new_tokens - 2)
        pos = start_pos + step
        current_token = generated[end]
        
        logits = llama_decode_step!(exec_fn, idg, cache, current_token; 
                                    sym_vals=Dict(:pos => pos),
                                    device=target_device)
        next_token = argmax(view(Array{Float32}(logits), :, 1, 1)) - 1  # 0-indexed

        push!(generated, next_token)
        next_token == eos_id && break
    end

    # 5. Decode token IDs to text (skip prompt tokens)
    return LlamaTokenization.decode(tokenizer, generated)
end

"""
    _rebuild_model_like(model, graph, reg)

Clone the model architecture into a new `graph` with a fresh `reg`,
using the same hyperparameters as the original. Supports `Llama` and `Phi3`.
"""
function _rebuild_model_like(model::Llama, graph::Luminal.Graph, reg::WeightRegistry;
                            rope_base=model.rope_base)
    attn   = model.layers[1].attention
    n_h    = attn.n_heads
    n_kv   = attn.n_kv_heads
    hd     = attn.head_dim
    hidden = n_h * hd
    inter  = Luminal.realized_dims(model.layers[1].feed_forward.gate_proj.weight.shape)[1]
    vsize  = Luminal.realized_dims(model.head.weight.shape)[1]
    return Llama(graph, reg;
                 vocab_size=vsize, hidden=hidden,
                 n_layers=length(model.layers),
                 n_heads=n_h, n_kv_heads=n_kv,
                 intermediate=inter,
                 rope_base=rope_base)
end

function _rebuild_model_like(model::Phi3, graph::Luminal.Graph, reg::WeightRegistry;
                            rope_base=model.rope_base)
    attn   = model.layers[1].attention
    n_h    = attn.n_heads
    n_kv   = attn.n_kv_heads
    hd     = attn.head_dim
    hidden = n_h * hd
    inter  = Luminal.realized_dims(model.layers[1].feed_forward.gate_proj.weight.shape)[1]
    vsize  = Luminal.realized_dims(model.head.weight.shape)[1]
    return Phi3(graph, reg;
                vocab_size=vsize, hidden=hidden,
                n_layers=length(model.layers),
                n_heads=n_h, n_kv_heads=n_kv,
                intermediate=inter,
                rope_base=rope_base)
end

export llama_generate

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
