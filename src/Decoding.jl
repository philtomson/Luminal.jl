# Decoding.jl
#
# Inference loops for Whisper models.
# Supports greedy decoding using the incremental graph and KV caching.

module Decoding

using ..Luminal
using ..Luminal.NN
using ..Luminal.LlamaTokenization
using AMDGPU
using JSON3

export greedy_decode, llama_generate

"""
    llama_generate(model, tokenizer, prompt, model_dir; kwargs...) -> String
    llama_generate(model, tokenizer, prompts::Vector{String}, model_dir; kwargs...) -> Vector{String}

End-to-end greedy text generation for Llama-style (decoder-only) models. A vector
of prompts is generated as one batch: one prefill and one decode step per token
for all of them, which raises throughput because decode is bound by reading the
weights, not by the number of sequences.

# Steps
1. Encode each prompt with `tokenizer` (adds BOS).
2. Prefill: run the prompts forward in one graph, right-padded to the longest
   (attention is causal, so padding cannot affect the real positions), and take
   each prompt's last-position logits.
3. Decode: repeatedly call `llama_decode_step!` with a device-resident KV cache.
   Every sequence runs at its own position; a finished sequence (EOS or
   `max_new_tokens`) stops advancing while the others continue.
4. Decode each sequence's generated token ids back to a string.

# Arguments
- `model`         : A `Luminal.NN.Llama` (or `Phi3`) instance
- `tokenizer`     : A `LlamaTokenizer` loaded from the model directory
- `prompt(s)`     : Input string, or a vector of them
- `model_dir`     : Path to the directory containing `.safetensors` weights
- `max_new_tokens`: Maximum tokens to generate per sequence (default 200)
- `max_seq`       : KV cache capacity (default 2048)
- `rope_base`     : RoPE base frequency (500000 for Llama-3, 10000 for Llama-2/Phi-3)
- `device`        : Device to run on; defaults to `get_device()`
- `search`        : `:none`, `:static` or `:measured`: compile the decode graph
                    through the e-graph rewrite layer (see `compile`). On GPU the
                    search also chooses Float16 weights per matmul. `:measured`
                    results are cached per model and device.
- `decode_weights`: storage for decode's matmul weights (default `Float16` on GPU,
                    `Float32` on CPU). `Int8` (group-wise int8, weight-only) is ~1.6x
                    faster decode on TinyLlama at ~+0.25% perplexity.
"""
function llama_generate(model, tokenizer::LlamaTokenizer, prompt::String, model_dir::String; kwargs...)
    return only(llama_generate(model, tokenizer, [prompt], model_dir; kwargs...))
end

function llama_generate(model,
                         tokenizer::LlamaTokenizer,
                         prompts::Vector{String},
                         model_dir::String;
                         max_new_tokens::Int=200,
                         max_seq::Int=2048,
                         rope_base::Float32=500000f0,
                         device=nothing,
                         search::Symbol=:none,
                         decode_weights::Union{Nothing,Type}=nothing)

    target_device = (device === nothing ? get_device() : device)
    B = length(prompts)
    B >= 1 || error("no prompts")

    # 1. Encode prompts
    prompt_ids = [LlamaTokenization.encode(tokenizer, p; bos=true) for p in prompts]
    plens = length.(prompt_ids)
    @info "Prompts: $(B) × ≤$(maximum(plens)) tokens"
    maximum(plens) < max_seq || error("prompt longer than max_seq=$max_seq")

    # 2. Prefill — all prompts in one graph, right-padded with token 0 to the longest.
    #    With `search`, the length is padded further to a power-of-two bucket so one
    #    searched graph (cached per bucket) serves every prompt up to that length.
    #    Attention is causal, so padding cannot change the first `plen` positions.
    slen = search === :none ? maximum(plens) : max(16, nextpow(2, maximum(plens)))
    pfx_graph = Graph()
    pfx_reg   = WeightRegistry()
    pfx_model = _rebuild_model_like(model, pfx_graph, pfx_reg; rope_base=rope_base)
    pfx_input = Luminal.tensor(pfx_graph, [slen, B])
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

    retain_pfx = collect(pfx_graph.to_retrieve)
    on_gpu = target_device isa Luminal.AbstractGPUDevice
    # Decode (one column per sequence per matmul) is fastest with Float16 weights.
    # Prefill is not: the Float16 kernel is a GEMV looping over the columns, while
    # rocBLAS's Float32 GEMM stays bandwidth-bound. Measured on TinyLlama: f16 is ~1x
    # f32 at 16 columns and ~3x slower at 64+, so prefill uses Float16 only for very
    # few columns.
    wdtype = decode_weights !== nothing ? decode_weights : (on_gpu ? Float16 : Float32)
    pfx_wdtype = (on_gpu && slen * B <= 8) ? Float16 : Float32
    ids = zeros(Float32, slen, B)
    for (b, p) in enumerate(prompt_ids)
        ids[1:plens[b], b] .= p
    end
    pfx_inputs = Dict{Int,Any}(pfx_input.id => ids)
    pfx_exec = if search === :none
        compile(pfx_graph; device=target_device, retain=retain_pfx, weight_dtype=pfx_wdtype)
    else
        @info "Searching equivalent prefill graphs ($search, bucket $slen × $B)..."
        compile(pfx_graph; device=target_device, retain=vcat(retain_pfx, pfx_input.id),
                search=search, precision=on_gpu,
                search_inputs=Dict{Int,Any}(pfx_input.id => Luminal.to_device(ids, target_device)))
    end

    pfx_results = pfx_exec(pfx_inputs; device=target_device)
    prefill_logits = Array{Float32}(pfx_results[pfx_out.id])  # (vocab, slen, B)

    # Greedy-pick each sequence's first generated token from its last prompt position
    eos_id = tokenizer.eos_id
    generated = [[argmax(view(prefill_logits, :, plens[b], b)) - 1] for b in 1:B]  # 0-indexed
    done = [g[1] == eos_id || max_new_tokens <= 1 for g in generated]

    # 3. KV cache: sequence b's next token goes to position plens[b]
    first_attn = model.layers[1].attention
    cache = LlamaKVCacheState(
        length(model.layers), first_attn.n_kv_heads, first_attn.head_dim;
        batch=B, max_seq=max_seq, device=target_device)
    cache.positions .= plens
    for (i, (k, v)) in enumerate(pfx_kvs)
        # Prefill K/V (head_dim, slen, kv_heads, B) -> cache (head_dim, max_seq, kv_heads, B),
        # in place on the device; only each prompt's own positions.
        for b in 1:B
            view(cache.self_cache[i][1], :, 1:plens[b], :, b) .= view(pfx_results[k.id], :, 1:plens[b], :, b)
            view(cache.self_cache[i][2], :, 1:plens[b], :, b) .= view(pfx_results[v.id], :, 1:plens[b], :, b)
        end
    end

    # Free prefill memory
    pfx_exec = nothing
    pfx_graph = nothing
    pfx_results = nothing
    prefill_logits = nothing
    GC.gc()
    Luminal.reclaim!(target_device)

    avail = Luminal.available_memory(target_device)
    if avail >= 0
        @info "VRAM before decode loop" available_mb=avail
    end

    @info "Compiling position-agnostic decode graph (batch $B)..."
    dg = Base.invokelatest(Graph)
    dreg = WeightRegistry()
    dm = _rebuild_model_like(model, dg, dreg; rope_base=rope_base)
    idg = build_llama_decode_step!(dm, dg, 0; max_seq=max_seq, batch=B, rope_base=rope_base)

    load_weights!(dg, dreg, weights_dict; device=target_device)
    retain_nodes = vcat(
        idg.logits_id, idg.new_self_k_ids, idg.new_self_v_ids,
        idg.token_input_id, idg.pos_input_id, idg.self_k_ids, idg.self_v_ids
    )
    # The decode graph is shape-static (positions are data), so on AMD GPUs it is
    # captured once as a HIP graph and replayed each token.
    capture = target_device isa Luminal.AMDDevice
    exec_fn = if search === :none
        compile(dg; device=target_device, retain=retain_nodes, free_intermediates=false,
                weight_dtype=wdtype, capture=capture)
    else
        @info "Searching equivalent decode graphs ($search)..."
        compile(dg; device=target_device, retain=retain_nodes, free_intermediates=false,
                capture=capture, search=search,
                precision=wdtype === Int8 ? (:weights, :int8) : wdtype === Float16 ? :weights : false)
    end

    # 4. Decode. A finished sequence keeps being fed (its last token, at the same
    #    position, so it never runs into max_seq) until every sequence is done.
    while !all(done)
        tokens = [g[end] for g in generated]
        logits = llama_decode_step!(exec_fn, idg, cache, tokens;
                                    advance=.!done, device=target_device)
        host = Array{Float32}(logits)                                     # (vocab, 1, B)
        for b in 1:B
            done[b] && continue
            next = argmax(view(host, :, 1, b)) - 1                        # 0-indexed
            push!(generated[b], next)
            done[b] = next == eos_id || length(generated[b]) >= max_new_tokens ||
                      cache.positions[b] >= max_seq
        end
    end

    # 5. Decode token IDs to text (the generated tokens only)
    return [LlamaTokenization.decode(tokenizer, g) for g in generated]
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
    greedy_decode(td, tokenizer, enc_output, weights; language="en", task=:transcribe,
                  max_len=448, device=nothing, weight_dtype=Float32) -> (text, tokens)

Greedy Whisper decoding with a device-resident KV cache.

- `td`         : a `TextDecoder`, used only as the architecture template
- `tokenizer`  : a `WhisperTokenizer`
- `enc_output` : the audio encoder output, (Hidden, S_enc, 1)
- `weights`    : model directory, or a `Dict` from `load_weights_to_dict`
- `max_len`    : maximum length of the token sequence, prompt included

The cross-attention K/V are projected once; then a single position-independent
decode-step graph (captured as a HIP graph on AMD GPUs) is replayed per token:
first for the start-of-transcript prompt, then for each greedily chosen token
until end-of-text. Returns the decoded text and all token ids, the prompt included.
"""
function greedy_decode(td::TextDecoder,
                       tokenizer::WhisperTokenizer,
                       enc_output::AbstractArray{Float32, 3},
                       weights;
                       language::String="en",
                       task::Symbol=:transcribe,
                       max_len::Int=MAX_TARGET_POSITION,
                       device=nothing,
                       weight_dtype::Type=Float32)
    hidden, enc_seq, batch = size(enc_output)
    @assert batch == 1 "greedy_decode supports batch size 1"
    dev = device === nothing ? get_device() : device
    W = weights isa AbstractString ? load_weights_to_dict(weights; device=dev) : weights
    cfg = decoder_config(td)
    head_dim = cfg.d_model ÷ cfg.n_heads

    # 1. Cross-attention K/V, projected once from the encoder output
    cg = Graph(); creg = WeightRegistry()
    ctd = TextDecoder(cg; reg=creg, cfg...)
    enc_in = Luminal.tensor(cg, [hidden, enc_seq, batch])
    kv = project_cross_kv(ctd, enc_in)
    load_weights!(cg, creg, W; device=dev)
    kv_ids = [id for (k, v) in kv for id in (k.id, v.id)]
    cexec = compile(cg; device=dev, retain=kv_ids)
    kv_res = cexec(Dict{Int,Any}(enc_in.id => enc_output); device=dev)

    cache = KVCacheState(cfg.n_layers, cfg.n_heads, head_dim, enc_seq;
                         batch=batch, max_seq=cfg.max_positions, device=dev)
    for (i, (k, v)) in enumerate(kv)
        cache.cross_cache[i][1] .= kv_res[k.id]
        cache.cross_cache[i][2] .= kv_res[v.id]
    end
    cexec = kv_res = nothing

    # 2. The decode step: one shape-static graph for every position
    g = Graph(); reg = WeightRegistry()
    dtd = TextDecoder(g; reg=reg, cfg...)
    idg = build_decode_step!(dtd, g, enc_seq; max_seq=cfg.max_positions, batch=batch)
    load_weights!(g, reg, W; device=dev)
    retain = vcat(idg.logits_id, idg.new_self_k_ids, idg.new_self_v_ids,
                  idg.token_input_id, idg.pos_input_id, idg.self_k_ids, idg.self_v_ids,
                  idg.cross_k_ids, idg.cross_v_ids)
    exec = compile(g; device=dev, retain=retain, free_intermediates=false,
                   weight_dtype=weight_dtype, capture=dev isa Luminal.AMDDevice)

    # 3. Feed the prompt, then decode greedily
    tokens = sot_sequence(tokenizer; language=language, task=task, notimestamps=true)
    logits = nothing
    for t in tokens
        logits = decode_step!(exec, idg, cache, [t]; device=dev)
    end
    while length(tokens) < max_len
        next = argmax(view(Array{Float32}(logits), :, 1, 1)) - 1   # 0-indexed
        push!(tokens, next)
        (next == tokenizer.eot_id || cache.step_pos >= cache.max_seq) && break
        logits = decode_step!(exec, idg, cache, [next]; device=dev)
    end

    return decode(tokenizer, tokens; skip_special=true), tokens
end

# Whisper dimensions from a Hugging Face config.json (defaults: whisper-tiny).
function _whisper_config(model_dir::String)
    path = joinpath(model_dir, "config.json")
    c = isfile(path) ? JSON3.read(read(path, String)) : Dict{Symbol,Any}()
    get_(k, d) = Int(get(c, k, d))
    enc = (d_model=get_(:d_model, D_MODEL), n_layers=get_(:encoder_layers, NN.ENC_LAYERS),
           n_heads=get_(:encoder_attention_heads, HEADS),
           ffn_dim=get_(:encoder_ffn_dim, NN.ENC_FFN_DIM), n_mels=get_(:num_mel_bins, NN.N_MEL_BINS),
           max_positions=get_(:max_source_positions, NN.MAX_SOURCE_POSITION))
    dec = (d_model=get_(:d_model, D_MODEL), n_layers=get_(:decoder_layers, DEC_LAYERS),
           n_heads=get_(:decoder_attention_heads, HEADS),
           ffn_dim=get_(:decoder_ffn_dim, NN.DEC_FFN_DIM), vocab_size=get_(:vocab_size, VOCAB_SIZE),
           max_positions=get_(:max_target_positions, MAX_TARGET_POSITION))
    return enc, dec
end

"""
    transcribe(model_dir, audio; language="en", task=:transcribe, max_len=448,
               device=nothing, weight_dtype=Float32) -> (text, tokens)

Transcribe (or translate) up to 30 s of audio with a Hugging Face Whisper
checkpoint directory (`config.json`, `model.safetensors` and tokenizer files).
`audio` is a path (decoded with ffmpeg) or a 16 kHz mono `Vector{Float32}`.
"""
function transcribe(model_dir::String, audio;
                    language::String="en", task::Symbol=:transcribe,
                    max_len::Int=MAX_TARGET_POSITION, device=nothing,
                    weight_dtype::Type=Float32)
    dev = device === nothing ? get_device() : device
    samples = audio isa AbstractString ? load_audio_file(audio) : Vector{Float32}(audio)
    enc_cfg, dec_cfg = _whisper_config(model_dir)
    mel = log_mel_spectrogram(pad_or_trim(samples); n_mels=enc_cfg.n_mels)
    W = load_weights_to_dict(model_dir; device=dev)

    g = Graph(); reg = WeightRegistry()
    enc = AudioEncoder(g; reg=reg, enc_cfg...)
    mel_in = Luminal.tensor(g, [size(mel)..., 1])
    enc_out = enc(mel_in)
    load_weights!(g, reg, W; device=dev)
    exec = compile(g; device=dev, retain=[enc_out.id])
    enc_arr = exec(Dict{Int,Any}(mel_in.id => Base.reshape(mel, size(mel)..., 1)); device=dev)[enc_out.id]

    td = TextDecoder(Graph(); dec_cfg...)
    return greedy_decode(td, WhisperTokenizer(model_dir), enc_arr, W;
                         language=language, task=task, max_len=max_len, device=dev,
                         weight_dtype=weight_dtype)
end

export transcribe

end # module Decoding
