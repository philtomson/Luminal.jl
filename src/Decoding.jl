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
    LlamaSession(model, tokenizer, model_dir; max_seq=2048, rope_base=model.rope_base,
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
- `rope_base`     : RoPE base frequency; defaults to the model's (see `llama_config`)
- `device`        : device to run on; defaults to `get_device()`
- `search`        : `:none`, `:static` or `:measured`: compile the graphs through the
                    e-graph rewrite layer (see `compile`). On GPU the search also
                    chooses Float16 weights per matmul. `:measured` results are
                    cached per model and device.
- `decode_weights`: storage for decode's matmul weights (default `Float16` on GPU,
                    `Float32` on CPU). `Int8` (group-wise int8, weight-only) is ~1.6x
                    faster decode on TinyLlama at ~+0.25% perplexity; `Luminal.Int4`
                    (group-wise 4-bit) is smaller and faster again, at a larger cost.
                    A function of the Hugging Face weight name chooses per tensor, e.g.
                    `name -> occursin("down_proj", name) ? Int8 : Luminal.Int4`.
"""
mutable struct LlamaSession{M, D}
    model::M
    tokenizer::LlamaTokenizer
    weights::Dict{String, Any}       # Float32 weights on `device`, shared by every graph
    device::D
    max_seq::Int
    rope_base::Float32
    search::Symbol
    decode_weights::Any              # a Type, or name -> Type (per tensor)
    prefill::Dict{Tuple{Int,Int}, Any}   # (padded length, batch) => compiled prefill
    decode::Dict{Int, Any}               # batch => compiled decode step and its cache
end

function LlamaSession(model, tokenizer::LlamaTokenizer, model_dir::String;
                      max_seq::Int=2048, rope_base::Real=model.rope_base, device=nothing,
                      search::Symbol=:none, decode_weights=nothing)
    dev = device === nothing ? get_device() : device
    on_gpu = dev isa Luminal.AbstractGPUDevice
    wdtype = decode_weights !== nothing ? decode_weights : (on_gpu ? Float16 : Float32)
    @info "Loading weights..." model_dir
    weights = load_weights_to_dict(model_dir; device=dev)
    return LlamaSession(model, tokenizer, weights, dev, max_seq, Float32(rope_base), search, wdtype,
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
        if !(wd isa Type)
            # per-tensor policy by weight name -> by node id (other tensors: Float32)
            by_id = Dict{Int, Type}(id => wd(name) for (name, id) in reg.mapping)
            wd = id -> get(by_id, id, Float32)
        end
        exec = if s.search === :none
            @info "Compiling decode graph (batch $B)..."
            compile(g; device=s.device, retain=retain, free_intermediates=false,
                    weight_dtype=wd, capture=capture)
        else
            # One reduced precision per search: offering Float16 variants to an int8
            # search would hold a Float16 copy of every weight as well (on an 8B model,
            # 16 GB more) for choices int8 already beats in decode.
            @info "Searching equivalent decode graphs ($(s.search), batch $B)..."
            compile(g; device=s.device, retain=retain, free_intermediates=false,
                    capture=capture, search=s.search,
                    precision=wd === Int8 ? :int8 : wd === Float16 ? :weights : false)
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
3. Decode: `llama_decode_step!` until each sequence has produced an
   end-of-sequence token (any of `tokenizer.eos_ids`) or `max_new_tokens`. Every sequence runs at its own position, and a finished one
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
    eos_ids = s.tokenizer.eos_ids
    generated = [[argmax(view(logits, :, plens[b], b)) - 1] for b in 1:B]   # 0-indexed
    done = [g[1] in eos_ids || max_new_tokens <= 1 for g in generated]

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
            done[b] = next in eos_ids || length(generated[b]) >= max_new_tokens ||
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
    WhisperSession(model_dir; device=nothing, weight_dtype=Float32)

A loaded Whisper checkpoint for repeated transcription: weights and tokenizer are
read once, and the compiled graphs (audio encoder, cross-attention projection,
decode step with its device-resident KV cache) are built per batch size on
first use and reused. Transcribe with `transcribe(session, audio_or_clips)`.
"""
mutable struct WhisperSession{D}
    tokenizer::WhisperTokenizer
    weights::Dict{String, Any}
    device::D
    enc_cfg::Any                  # AudioEncoder kwargs (nothing: decoding only)
    dec_cfg::Any                  # TextDecoder kwargs
    weight_dtype::Type            # decode-step matmul weight storage
    encoders::Dict{Int, Any}      # batch => compiled encoder
    cross::Dict{Int, Any}         # batch => compiled cross-attention K/V projection
    decoders::Dict{Int, Any}      # batch => compiled decode step and KV cache
end

function WhisperSession(model_dir::String; device=nothing, weight_dtype::Type=Float32)
    dev = device === nothing ? get_device() : device
    enc_cfg, dec_cfg = _whisper_config(model_dir)
    return WhisperSession(WhisperTokenizer(model_dir), load_weights_to_dict(model_dir; device=dev),
                          dev, enc_cfg, dec_cfg, weight_dtype,
                          Dict{Int,Any}(), Dict{Int,Any}(), Dict{Int,Any}())
end

Base.show(io::IO, s::WhisperSession) =
    print(io, "WhisperSession(d_model $(s.dec_cfg.d_model), $(s.dec_cfg.n_layers) decoder layers, ",
          "$(s.device), batch sizes compiled: $(sort!(collect(keys(s.decoders)))))")

# Encoder output (Hidden, S_enc, B) on the device for log-mel spectrograms (n_mels, frames, B).
function _encode(s::WhisperSession, mels::Array{Float32, 3})
    s.enc_cfg === nothing && error("this session has no encoder configuration")
    B = size(mels, 3)
    e = get!(s.encoders, B) do
        g = Graph(); reg = WeightRegistry()
        enc = AudioEncoder(g; reg=reg, s.enc_cfg...)
        mel_in = Luminal.tensor(g, collect(size(mels)))
        out = enc(mel_in)
        load_weights!(g, reg, s.weights; device=s.device)
        (exec=compile(g; device=s.device, retain=[out.id]), in_id=mel_in.id, out_id=out.id)
    end
    return e.exec(Dict{Int,Any}(e.in_id => mels); device=s.device)[e.out_id]
end

# Greedy decoding for every sequence of the (Hidden, S_enc, B) encoder output.
# Sequences share the prompt, so they stay at one position; a finished sequence
# keeps being fed end-of-text, and its output is ignored, until all are done.
function _greedy(s::WhisperSession, enc_output::AbstractArray{Float32, 3};
                 language::String="en", task::Symbol=:transcribe,
                 max_len::Int=MAX_TARGET_POSITION)
    hidden, enc_seq, B = size(enc_output)
    cfg = s.dec_cfg
    head_dim = cfg.d_model ÷ cfg.n_heads

    # Cross-attention K/V, projected once per call
    c = get!(s.cross, B) do
        g = Graph(); reg = WeightRegistry()
        td = TextDecoder(g; reg=reg, cfg...)
        enc_in = Luminal.tensor(g, [hidden, enc_seq, B])
        kv = project_cross_kv(td, enc_in)
        load_weights!(g, reg, s.weights; device=s.device)
        ids = [(k.id, v.id) for (k, v) in kv]
        (exec=compile(g; device=s.device, retain=[i for p in ids for i in p]), in_id=enc_in.id, kv_ids=ids)
    end
    # The decode step: one shape-static graph for every position, and its cache.
    # Reusing the cache arrays keeps a captured HIP graph valid across calls.
    d = get!(s.decoders, B) do
        g = Graph(); reg = WeightRegistry()
        td = TextDecoder(g; reg=reg, cfg...)
        idg = build_decode_step!(td, g, enc_seq; max_seq=cfg.max_positions, batch=B)
        load_weights!(g, reg, s.weights; device=s.device)
        retain = vcat(idg.logits_id, idg.new_self_k_ids, idg.new_self_v_ids,
                      idg.token_input_id, idg.pos_input_id, idg.self_k_ids, idg.self_v_ids,
                      idg.cross_k_ids, idg.cross_v_ids)
        exec = compile(g; device=s.device, retain=retain, free_intermediates=false,
                       weight_dtype=s.weight_dtype, capture=s.device isa Luminal.AMDDevice)
        cache = KVCacheState(cfg.n_layers, cfg.n_heads, head_dim, enc_seq;
                             batch=B, max_seq=cfg.max_positions, device=s.device)
        (exec=exec, idg=idg, cache=cache)
    end
    kv_res = c.exec(Dict{Int,Any}(c.in_id => enc_output); device=s.device)
    cache = d.cache
    for (i, (k, v)) in enumerate(c.kv_ids)
        cache.cross_cache[i][1] .= kv_res[k]
        cache.cross_cache[i][2] .= kv_res[v]
    end
    cache.step_pos = 0      # self-attention slots are rewritten before they are read

    tok = s.tokenizer
    prompt = sot_sequence(tok; language=language, task=task, notimestamps=true)
    tokens = [copy(prompt) for _ in 1:B]
    logits = nothing
    for t in prompt
        logits = decode_step!(d.exec, d.idg, cache, fill(t, B); device=s.device)
    end
    done = falses(B)
    while !all(done)
        host = Array{Float32}(logits)                                     # (vocab, 1, B)
        for b in 1:B
            done[b] && continue
            next = argmax(view(host, :, 1, b)) - 1                        # 0-indexed
            push!(tokens[b], next)
            done[b] = next == tok.eot_id || length(tokens[b]) >= max_len
        end
        (all(done) || cache.step_pos >= cache.max_seq) && break
        logits = decode_step!(d.exec, d.idg, cache, [done[b] ? tok.eot_id : tokens[b][end] for b in 1:B];
                              device=s.device)
    end
    return [(decode(tok, t; skip_special=true), t) for t in tokens]
end

_samples(audio::AbstractString) = load_audio_file(audio)
_samples(audio::AbstractVector{<:Real}) = Vector{Float32}(audio)

"""
    transcribe(session::WhisperSession, audio; language="en", task=:transcribe, max_len=448) -> (text, tokens)
    transcribe(session::WhisperSession, clips::Vector; kwargs...) -> Vector{(text, tokens)}
    transcribe(model_dir, audio; device=nothing, weight_dtype=Float32, kwargs...)

Transcribe (or, with `task=:translate`, translate to English) up to 30 s of
audio. `audio` is a path (decoded with ffmpeg) or a 16 kHz mono vector of
samples; a vector of them is encoded and decoded as one batch. `tokens`
includes the start-of-transcript prompt. The `model_dir` form is one-shot: it
creates a `WhisperSession` (loading and compiling) for a single call.
"""
# A vector of clips (paths and/or sample vectors); a vector of numbers is one clip
# (the more specific method below).
function transcribe(s::WhisperSession, clips::AbstractVector;
                    language::String="en", task::Symbol=:transcribe, max_len::Int=MAX_TARGET_POSITION)
    isempty(clips) && return Tuple{String, Vector{Int}}[]
    mels = [log_mel_spectrogram(pad_or_trim(_samples(a)); n_mels=s.enc_cfg.n_mels) for a in clips]
    enc = _encode(s, cat(mels...; dims=3))
    return _greedy(s, enc; language=language, task=task, max_len=max_len)
end

transcribe(s::WhisperSession, audio::Union{AbstractString, AbstractVector{<:Real}}; kwargs...) =
    only(transcribe(s, [audio]; kwargs...))

function transcribe(model_dir::String, audio; device=nothing, weight_dtype::Type=Float32, kwargs...)
    return transcribe(WhisperSession(model_dir; device=device, weight_dtype=weight_dtype), audio; kwargs...)
end

"""
    greedy_decode(td, tokenizer, enc_output, weights; language="en", task=:transcribe,
                  max_len=448, device=nothing, weight_dtype=Float32)

Greedy Whisper decoding of an encoder output (Hidden, S_enc, B) with a
device-resident KV cache; `td` is a `TextDecoder` used only as the architecture
template, and `weights` a model directory or a `Dict` from `load_weights_to_dict`.
Returns `(text, tokens)` for batch 1, or a vector of them. (`transcribe` and
`WhisperSession` wrap this with the audio frontend and encoder.)
"""
function greedy_decode(td::TextDecoder, tokenizer::WhisperTokenizer,
                       enc_output::AbstractArray{Float32, 3}, weights;
                       device=nothing, weight_dtype::Type=Float32, kwargs...)
    dev = device === nothing ? get_device() : device
    W = weights isa AbstractString ? load_weights_to_dict(weights; device=dev) : weights
    s = WhisperSession(tokenizer, W, dev, nothing, decoder_config(td), weight_dtype,
                       Dict{Int,Any}(), Dict{Int,Any}(), Dict{Int,Any}())
    out = _greedy(s, enc_output; kwargs...)
    return length(out) == 1 ? only(out) : out
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

export transcribe, WhisperSession

end # module Decoding
