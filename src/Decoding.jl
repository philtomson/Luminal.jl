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

export greedy_decode, llama_generate, LlamaSession, generate

"""
    LlamaSession(model, tokenizer, model_dir; max_seq=2048, rope_base=500000f0,
                 device=nothing, search=:none, decode_weights=nothing)

A loaded Llama-style model for repeated generation. The weights are read once;
compiled graphs are built on first use and kept:
- prefill graphs per (padded prompt length, batch size). Prompts are right-padded
  to a bucket (16, 32, 64, 128, 256, then multiples of 256) so similar lengths
  share a graph. Attention is causal, so padding cannot change the real positions.
- decode graphs and their device-resident KV caches per batch size. Reusing the
  cache arrays keeps a captured HIP graph valid from one call to the next.

Generate with `generate(session, prompt_or_prompts; max_new_tokens)`.

# Arguments
- `model`         : a `Luminal.NN.Llama` (or `Phi3`) instance, used as the architecture template
- `tokenizer`     : a `LlamaTokenizer` loaded from the model directory
- `model_dir`     : directory containing the `.safetensors` weights
- `max_seq`       : KV cache capacity per sequence (default 2048)
- `rope_base`     : RoPE base frequency (500000 for Llama-3, 10000 for Llama-2/Phi-3)
- `device`        : device to run on; defaults to `get_device()`
- `search`        : `:none`, `:static` or `:measured`: compile the graphs through the
                    e-graph rewrite layer (see `compile`). On GPU the search also
                    chooses Float16 weights per matmul. `:measured` results are
                    cached per model and device.
- `decode_weights`: storage for decode's matmul weights (default `Float16` on GPU,
                    `Float32` on CPU). `Int8` (group-wise int8, weight-only) is ~1.6x
                    faster decode on TinyLlama at ~+0.25% perplexity.
"""
mutable struct LlamaSession{M, D}
    model::M
    tokenizer::LlamaTokenizer
    weights::Dict{String, Any}       # Float32 weights on `device`, shared by every graph
    device::D
    max_seq::Int
    rope_base::Float32
    search::Symbol
    decode_weights::Type
    prefill::Dict{Tuple{Int,Int}, Any}   # (padded length, batch) => compiled prefill
    decode::Dict{Int, Any}               # batch => compiled decode step and its cache
end

function LlamaSession(model, tokenizer::LlamaTokenizer, model_dir::String;
                      max_seq::Int=2048, rope_base::Float32=500000f0, device=nothing,
                      search::Symbol=:none, decode_weights::Union{Nothing,Type}=nothing)
    dev = device === nothing ? get_device() : device
    on_gpu = dev isa Luminal.AbstractGPUDevice
    wdtype = decode_weights !== nothing ? decode_weights : (on_gpu ? Float16 : Float32)
    @info "Loading weights..." model_dir
    weights = load_weights_to_dict(model_dir; device=dev)
    return LlamaSession(model, tokenizer, weights, dev, max_seq, rope_base, search, wdtype,
                        Dict{Tuple{Int,Int}, Any}(), Dict{Int, Any}())
end

Base.show(io::IO, s::LlamaSession) =
    print(io, "LlamaSession($(nameof(typeof(s.model))), $(s.device), decode weights $(s.decode_weights), ",
          "$(length(s.prefill)) prefill / $(length(s.decode)) decode graphs compiled)")

# Padded prefill length for a prompt of `n` tokens.
_prefill_bucket(n::Int) = n <= 256 ? max(16, nextpow(2, n)) : cld(n, 256) * 256

# The compiled prefill graph for (slen, B), built on first use.
function _prefill_graph(s::LlamaSession, slen::Int, B::Int)
    get!(s.prefill, (slen, B)) do
        g = Graph(); reg = WeightRegistry()
        m = _rebuild_model_like(s.model, g, reg; rope_base=s.rope_base)
        input = Luminal.tensor(g, [slen, B])
        out, kvs = m(input, 0; return_kv=true)
        load_weights!(g, reg, s.weights; device=s.device)
        retain = vcat(out.id, [t.id for kv in kvs for t in kv])
        on_gpu = s.device isa Luminal.AbstractGPUDevice
        # Prefill multiplies each weight by slen*B >= 16 columns, where rocBLAS's
        # Float32 GEMM beats the Float16 GEMV (which suits decode's single column).
        exec = if s.search === :none
            @info "Compiling prefill graph ($slen × $B)..."
            compile(g; device=s.device, retain=retain)
        else
            @info "Searching equivalent prefill graphs ($(s.search), $slen × $B)..."
            ids = Luminal.to_device(zeros(Float32, slen, B), s.device)
            compile(g; device=s.device, retain=vcat(retain, input.id), search=s.search,
                    precision=on_gpu, search_inputs=Dict{Int,Any}(input.id => ids))
        end
        (exec=exec, input_id=input.id, out_id=out.id, kv_ids=[(k.id, v.id) for (k, v) in kvs])
    end
end

# The compiled decode step and KV cache for batch size B, built on first use.
function _decode_graph(s::LlamaSession, B::Int)
    get!(s.decode, B) do
        g = Base.invokelatest(Graph); reg = WeightRegistry()
        m = _rebuild_model_like(s.model, g, reg; rope_base=s.rope_base)
        idg = build_llama_decode_step!(m, g, 0; max_seq=s.max_seq, batch=B, rope_base=s.rope_base)
        load_weights!(g, reg, s.weights; device=s.device)
        retain = vcat(idg.logits_id, idg.new_self_k_ids, idg.new_self_v_ids,
                      idg.token_input_id, idg.pos_input_id, idg.self_k_ids, idg.self_v_ids)
        # The decode graph is shape-static (positions are data), so on AMD GPUs it is
        # captured once as a HIP graph and replayed each token.
        capture = s.device isa Luminal.AMDDevice
        wd = s.decode_weights
        exec = if s.search === :none
            @info "Compiling decode graph (batch $B)..."
            compile(g; device=s.device, retain=retain, free_intermediates=false,
                    weight_dtype=wd, capture=capture)
        else
            @info "Searching equivalent decode graphs ($(s.search), batch $B)..."
            compile(g; device=s.device, retain=retain, free_intermediates=false,
                    capture=capture, search=s.search,
                    precision=wd === Int8 ? (:weights, :int8) : wd === Float16 ? :weights : false)
        end
        attn = s.model.layers[1].attention
        cache = LlamaKVCacheState(length(s.model.layers), attn.n_kv_heads, attn.head_dim;
                                  batch=B, max_seq=s.max_seq, device=s.device)
        (exec=exec, idg=idg, cache=cache)
    end
end

"""
    generate(session, prompt; max_new_tokens=200) -> String
    generate(session, prompts::Vector{String}; max_new_tokens=200) -> Vector{String}

Greedy generation with a `LlamaSession`. A vector of prompts is generated as one
batch: one prefill and one decode step per token for all of them, which raises
throughput because decode is bound by reading the weights, not by the number of
sequences.

1. Encode each prompt (adds BOS).
2. Prefill: the prompts right-padded in one graph; take each prompt's
   last-position logits and copy its K/V into the cache.
3. Decode: `llama_decode_step!` until each sequence has produced EOS or
   `max_new_tokens`. Every sequence runs at its own position, and a finished one
   stops advancing while the others continue.
4. Return each sequence's generated text.
"""
generate(s::LlamaSession, prompt::String; kwargs...) = only(generate(s, [prompt]; kwargs...))

function generate(s::LlamaSession, prompts::Vector{String}; max_new_tokens::Int=200)
    B = length(prompts)
    B >= 1 || error("no prompts")
    prompt_ids = [LlamaTokenization.encode(s.tokenizer, p; bos=true) for p in prompts]
    plens = length.(prompt_ids)
    maximum(plens) < s.max_seq || error("prompt longer than max_seq=$(s.max_seq)")

    # Prefill
    slen = _prefill_bucket(maximum(plens))
    pf = _prefill_graph(s, slen, B)
    ids = zeros(Float32, slen, B)
    for (b, p) in enumerate(prompt_ids)
        ids[1:plens[b], b] .= p
    end
    res = pf.exec(Dict{Int,Any}(pf.input_id => ids); device=s.device)
    logits = Array{Float32}(res[pf.out_id])                                  # (vocab, slen, B)
    eos_id = s.tokenizer.eos_id
    generated = [[argmax(view(logits, :, plens[b], b)) - 1] for b in 1:B]   # 0-indexed
    done = [g[1] == eos_id || max_new_tokens <= 1 for g in generated]

    # KV cache: sequence b's next token goes to position plens[b]. Slots past a
    # sequence's position are never read, so the reused cache needs no clearing.
    dc = _decode_graph(s, B)
    cache = dc.cache
    cache.positions .= plens
    for (i, (k, v)) in enumerate(pf.kv_ids)
        # Prefill K/V (head_dim, slen, kv_heads, B) -> cache (head_dim, max_seq, kv_heads, B),
        # in place on the device; only each prompt's own positions.
        for b in 1:B
            view(cache.self_cache[i][1], :, 1:plens[b], :, b) .= view(res[k], :, 1:plens[b], :, b)
            view(cache.self_cache[i][2], :, 1:plens[b], :, b) .= view(res[v], :, 1:plens[b], :, b)
        end
    end

    # Decode. A finished sequence keeps being fed (its last token, at the same
    # position, so it never runs into max_seq) until every sequence is done.
    while !all(done)
        tokens = [g[end] for g in generated]
        out = llama_decode_step!(dc.exec, dc.idg, cache, tokens; advance=.!done, device=s.device)
        host = Array{Float32}(out)                                          # (vocab, 1, B)
        for b in 1:B
            done[b] && continue
            next = argmax(view(host, :, 1, b)) - 1                          # 0-indexed
            push!(generated[b], next)
            done[b] = next == eos_id || length(generated[b]) >= max_new_tokens ||
                      cache.positions[b] >= s.max_seq
        end
    end

    return [LlamaTokenization.decode(s.tokenizer, g) for g in generated]
end

"""
    llama_generate(model, tokenizer, prompt(s), model_dir; max_new_tokens=200, kwargs...)

One-shot generation: builds a `LlamaSession` (loading the weights and compiling
the graphs) and generates. For more than one call, create a `LlamaSession` once
and use `generate`, which reuses the weights and compiled graphs. The remaining
keyword arguments are `LlamaSession`'s.
"""
function llama_generate(model, tokenizer::LlamaTokenizer, prompts::Union{String, Vector{String}},
                        model_dir::String; max_new_tokens::Int=200, kwargs...)
    return generate(LlamaSession(model, tokenizer, model_dir; kwargs...), prompts;
                    max_new_tokens=max_new_tokens)
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
