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
using Random

export PrefixCache, release_stale!, Sampler, sample_token, greedy_decode, llama_generate, LlamaSession, generate, generate_ids, weight_preset, MTPDrafter, generate_ids_speculative, generate_speculative

"""
    weight_preset(name) -> (weight name -> Type)

Named per-tensor weight policies for `LlamaSession(...; decode_weights=name)`,
chosen with examples/quant_sensitivity.jl (WikiText-2 perplexity, TinyLlama and
Llama-3-8B, two texts):
- `:int4`: every matmul weight 4-bit (smallest; +5-8% perplexity on these models)
- `:int4_mixed`: int8 for down_proj, lm_head, k_proj and v_proj, 4-bit for the
  rest. down_proj and lm_head are the most sensitive types on Llama-3-8B; k/v are
  tiny under grouped-query attention, and v_proj is TinyLlama's most sensitive.
- `:int4_mixed_plus`: also up_proj and o_proj at int8 (within ~0.1% of int8 on 8B)
"""
function weight_preset(name::Symbol)
    int8_types = name === :int4 ? () :
                 name === :int4_mixed ? ("down_proj", "lm_head", "k_proj", "v_proj") :
                 name === :int4_mixed_plus ? ("down_proj", "lm_head", "k_proj", "v_proj", "up_proj", "o_proj") :
                 error("unknown weight preset $name (use :int4, :int4_mixed, :int4_mixed_plus)")
    return w -> any(t -> occursin(t, w), int8_types) ? Int8 : Luminal.Int4
end

"""
    LlamaSession(model, tokenizer, model_dir; max_seq=2048, rope_base=model.rope_base,
                 device=nothing, search=:none, decode_weights=nothing, kv_dtype=Float32)

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
- `kv_dtype`      : element type of the K/V caches (default Float32; Float16 halves them; Int8 quarters them: per-token,
                    per-head absmax scales, AMD GPU only, see src/DecodeAttn.jl)
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
                    `name -> occursin("down_proj", name) ? Int8 : Luminal.Int4`, and
                    a Symbol names a preset (`weight_preset`: `:int4_mixed`, ...).
- `awq`           : AWQ scales from `awq_scales`, folded into the weights at load
                    (improves 4-bit accuracy; Float32 results are unchanged).
- `weight_storage`: load the matmul weights straight into reduced precision
                    (`Luminal.Int4`, `Int8`, `Float16`, a preset Symbol or a function
                    of the name), never holding the model in Float32; prefill runs
                    on them too (dequantized per matmul into one GEMM). For models
                    too large for Float32 in memory, e.g. Qwen3.6-27B.
- `weights`       : a preloaded weight dict (as from `load_weights_to_dict`).
"""
mutable struct LlamaSession{M, D}
    model::M
    tokenizer::Union{LlamaTokenizer, Nothing}   # nothing: generate_ids only, no end-of-sequence ids
    weights::Dict{String, Any}       # Float32 weights on `device`, shared by every graph
    device::D
    max_seq::Int
    rope_base::Float32
    search::Symbol
    decode_weights::Any              # a Type, or name -> Type (per tensor)
    prefill::Dict{Tuple{Int,Int}, Any}   # (padded length, batch) => compiled prefill
    decode::Dict{Int, Any}               # batch => compiled decode step and its cache
    kv_dtype::DataType                   # element type of the K/V caches
    spec::Dict{Any, Any}                 # speculative decoding: compiled multi-token verify steps, MTP graphs
end

function LlamaSession(model, tokenizer::Union{LlamaTokenizer, Nothing}, model_dir::String;
                      max_seq::Int=2048, rope_base::Real=model.rope_base, device=nothing,
                      search::Symbol=:none, decode_weights=nothing, awq=nothing,
                      weight_storage=nothing, weights=nothing, kv_dtype::Type=Float32)
    dev = device === nothing ? get_device() : device
    on_gpu = dev isa Luminal.AbstractGPUDevice
    kv_dtype === Int8 && !(dev isa Luminal.AMDDevice) && error("kv_dtype=Int8 (the int8 KV cache) needs an AMD GPU")
    # weight_storage: load straight into reduced precision (prefill and decode both
    # use it; decode_weights follows it unless given)
    ws = weight_storage isa Symbol ? weight_preset(weight_storage) :
         weight_storage isa Type ? (T = weight_storage; _ -> T) : weight_storage
    ws !== nothing && awq !== nothing && error("awq needs Float32 weights to fold into; don't combine it with weight_storage")
    decode_weights === nothing && weight_storage !== nothing && (decode_weights = weight_storage)
    wdtype = decode_weights isa Symbol ? weight_preset(decode_weights) :
             decode_weights !== nothing ? decode_weights : (on_gpu ? Float16 : Float32)
    if weights === nothing
        @info "Loading weights..." model_dir
        weights = load_weights_to_dict(model_dir; device=dev, quantize=ws)
    end
    awq === nothing || awq_apply!(weights, awq, model)
    return LlamaSession(model, tokenizer, weights, dev, max_seq, Float32(rope_base), search, wdtype,
                        Dict{Tuple{Int,Int}, Any}(), Dict{Int, Any}(), kv_dtype, Dict{Any, Any}())
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
        # recurrent layers (Qwen3.5) consume each prompt only up to its length
        lens = NN._needs_lens(m) ? Luminal.tensor(g, [B]) : nothing
        last_only = NN._prefill_last_only(m)
        href = Ref{Any}(nothing)     # the final residual stream, kept for the MTP head (models that offer it)
        out, kvs = lens === nothing ? m(input, 0; return_kv=true) :
                   !last_only ? m(input, 0; return_kv=true, lens=lens) :
                   NN._prefill_hidden(m) ? m(input, 0; return_kv=true, lens=lens, last_only=true, hidden_ref=href) :
                   m(input, 0; return_kv=true, lens=lens, last_only=true)
        hidden_id = href[] === nothing ? 0 : href[].id
        load_weights!(g, reg, s.weights; device=s.device)
        retain = vcat(out.id, [t.id for kv in kvs for t in kv], hidden_id == 0 ? Int[] : [hidden_id])
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
        (exec=exec, input_id=input.id, out_id=out.id, kv_ids=[(k.id, v.id) for (k, v) in kvs],
         lens_id=lens === nothing ? 0 : lens.id, last_only=last_only, hidden_id=hidden_id)
    end
end

# The compiled decode step and KV cache for batch size B, built on first use.
function _decode_graph(s::LlamaSession, B::Int)
    get!(s.decode, B) do
        g = Base.invokelatest(Graph); reg = WeightRegistry()
        m = _rebuild_model_like(s.model, g, reg; rope_base=s.rope_base)
        idg = build_llama_decode_step!(m, g, 0; max_seq=s.max_seq, batch=B, rope_base=s.rope_base, kv_dtype=s.kv_dtype)
        load_weights!(g, reg, s.weights; device=s.device)
        retain = vcat(idg.logits_id, idg.new_self_k_ids, idg.new_self_v_ids,
                      idg.token_input_id, idg.pos_input_id, idg.self_k_ids, idg.self_v_ids,
                      idg.lens_input_id == 0 ? Int[] : [idg.lens_input_id])
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
                    precision=wd === Luminal.Int4 ? :int4 : wd === Int8 ? :int8 : wd === Float16 ? :weights : false)
        end
        cache = LlamaKVCacheState(s.model; batch=B, max_seq=s.max_seq, device=s.device, kv_dtype=s.kv_dtype)
        (exec=exec, idg=idg, cache=cache)
    end
end

# The compiled multi-token decode step for speculative decoding: `seq` tokens per sequence in one
# pass (the weights are read once for all of them), the recurrent layers also recording their state
# after the first token (snapshot buffers `snaps`, for rolling a rejected draft back), and the final
# residual stream (Hidden, seq, B) returned for the MTP head. It runs on the same cache arrays as the
# plain decode step of batch B.
function _verify_graph(s::LlamaSession, B::Int, seq::Int)
    get!(s.spec, (:verify, B, seq)) do
        dc = _decode_graph(s, B)
        g = Base.invokelatest(Graph); reg = WeightRegistry()
        m = _rebuild_model_like(s.model, g, reg; rope_base=s.rope_base)
        idg = build_llama_decode_step!(m, g, 0; max_seq=s.max_seq, batch=B, rope_base=s.rope_base, kv_dtype=s.kv_dtype,
                                       seq=seq, snapshots=seq - 1, want_hidden=true, want_argmax=true)
        load_weights!(g, reg, s.weights; device=s.device)
        retain = vcat(idg.logits_id, idg.hidden_id, idg.argmax_id, idg.token_input_id, idg.pos_input_id, idg.self_k_ids, idg.self_v_ids,
                      idg.lens_input_id == 0 ? Int[] : [idg.lens_input_id],
                      [i for ab in idg.snap_ids for i in ab])
        capture = s.device isa Luminal.AMDDevice
        wd = s.decode_weights
        if !(wd isa Type)
            by_id = Dict{Int, Type}(id => wd(name) for (name, id) in reg.mapping)
            wd = id -> get(by_id, id, Float32)
        end
        @info "Compiling verify graph (batch $B, $seq tokens)..."
        exec = compile(g; device=s.device, retain=retain, free_intermediates=false, weight_dtype=wd, capture=capture)
        # (the state after each of the first seq - 1 tokens: seq - 1 times the batch dim)
        sn(a) = (d = collect(size(a)); d[end] *= seq - 1; d)
        snaps = [(Luminal.zero_tensor(s.device, Float32, sn(dc.cache.self_cache[i][1])...),
                  Luminal.zero_tensor(s.device, Float32, sn(dc.cache.self_cache[i][2])...))
                 for (i, l) in enumerate(s.model.layers) if NN._is_state_layer(l)]
        (exec=exec, idg=idg, snaps=snaps, cache=dc.cache)
    end
end

# Run one verify step: tokens (seq, B) at each sequence's cache position, `lens` = seq. Returns
# (logits (vocab, seq, B), hidden (Hidden, seq, B)) as device arrays and the greedy tokens (seq, B)
# (0-based, host); the cache has the new K/V, the recurrent states are at the end of the `seq` tokens,
# and `snaps` hold them after the first token.
function _verify_step!(s::LlamaSession, vg, tokens::AbstractMatrix{<:Integer})
    cache = vg.cache; idg = vg.idg
    inputs = Dict{Int, Any}()
    inputs[idg.token_input_id] = Float32.(tokens)
    inputs[idg.pos_input_id] = Float32.(cache.positions)
    idg.lens_input_id == 0 || (inputs[idg.lens_input_id] = fill(Float32(idg.seq), size(tokens, 2)))
    for (i, (k_id, v_id)) in enumerate(zip(idg.self_k_ids, idg.self_v_ids))
        inputs[k_id] = cache.self_cache[i][1]; inputs[v_id] = cache.self_cache[i][2]
    end
    j = 0
    for (i, l) in enumerate(s.model.layers)
        NN._is_state_layer(l) || continue
        j += 1
        inputs[idg.snap_ids[j][1]] = vg.snaps[j][1]; inputs[idg.snap_ids[j][2]] = vg.snaps[j][2]
    end
    res = vg.exec(inputs; device=s.device)
    return res[idg.logits_id], res[idg.hidden_id], Int.(Array(res[idg.argmax_id]))
end

# Roll the recurrent states back to their snapshots (the state after the first verified token).
function _rollback_states!(s::LlamaSession, vg; slot::Int=1)
    j = 0
    for (i, l) in enumerate(s.model.layers)
        NN._is_state_layer(l) || continue
        j += 1
        for q in 1:2
            dst = vg.cache.self_cache[i][q]; B = size(dst, ndims(dst))
            src = selectdim(vg.snaps[j][q], ndims(dst), (slot - 1) * B + 1:slot * B)
            copyto!(dst, src)
        end
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

function generate(s::LlamaSession, prompts::Vector{String}; max_new_tokens::Int=200, sampler=nothing)
    prompt_ids = [LlamaTokenization.encode(s.tokenizer, p; bos=true) for p in prompts]
    return [LlamaTokenization.decode(s.tokenizer, g) for g in generate_ids(s, prompt_ids; max_new_tokens, sampler)]
end

# --- Chunked prefill -------------------------------------------------------------------------------
# A long prompt goes through the model in chunks of `prefill_chunk()` tokens, each one a multi-token
# decode step (the same graph builder as a speculative verify step, with seq = chunk): the recurrent
# layers continue from their states, the attention layers write the chunk's K/V into the cache and attend
# to all of it (`chunk_attention!`). Memory stays at one chunk's intermediates (~32 MB per token) whatever
# the prompt length, and the cost stays linear in the recurrent layers. Float16 cache, batch 1, Qwen3.5.
# (a young-generation GC is enough and ~10x cheaper than a full one; LUMINAL_CHUNK_GC=full for the full collection)
# With the launch-bounds AMDGPU.jl branch (`DERIVE_SCOPE`) the graph drops every dead intermediate and the
# derived arrays of each step itself, so a chunk leaves ~0.05 GB behind and no collection is needed between
# chunks. Otherwise (stock AMDGPU.jl) a young-generation GC is enough and ~10x cheaper than a full one.
# LUMINAL_CHUNK_GC=none|young|full overrides.
function _chunk_reclaim!(device)
    # (measured on the 27B with the launch-bounds branch too: "none" left ~0.4-0.9 MB per prompt token behind, 28 GB over a 32k prompt, until a GC
    # happened to run; "young" after every chunk keeps the memory flat at the same speed)
    mode = get(ENV, "LUMINAL_CHUNK_GC", "young")
    mode == "full" && return Luminal.reclaim!(device)
    mode == "none" && return nothing
    GC.gc(false); device isa Luminal.AMDDevice && AMDGPU.HIP.reclaim()
    return nothing
end
_chunk_reuse() = get(ENV, "LUMINAL_CHUNK_REUSE", "0") == "1"
prefill_chunk() = parse(Int, get(ENV, "LUMINAL_PREFILL_CHUNK", "512"))

function _chunk_graph(s::LlamaSession, C::Int; extract=())
    get!(s.spec, isempty(extract) ? (:chunk, C) : (:chunkx, C, Tuple(extract))) do
        dc = _decode_graph(s, 1)
        g = Base.invokelatest(Graph); reg = WeightRegistry()
        m = _rebuild_model_like(s.model, g, reg; rope_base=s.rope_base)
        idg = build_llama_decode_step!(m, g, 0; max_seq=s.max_seq, batch=1, rope_base=s.rope_base, kv_dtype=s.kv_dtype,
                                       seq=C, want_hidden=true, head=:last, extract_layers=extract)
        load_weights!(g, reg, s.weights; device=s.device)
        retain = vcat(idg.logits_id, idg.hidden_id, idg.token_input_id, idg.pos_input_id, idg.self_k_ids, idg.self_v_ids,
                      [idg.lens_input_id], idg.extract_ids)
        wd = s.decode_weights
        if !(wd isa Type)
            by_id = Dict{Int, Type}(id => wd(name) for (name, id) in reg.mapping)
            wd = id -> get(by_id, id, Float32)
        end
        @info "Compiling prefill chunk graph ($C tokens)..."
        # LUMINAL_CHUNK_REUSE=1: keep every node's buffer between chunks (more memory, no per-op allocation or per-chunk GC)
        # (HIP graph capture of the chunk was tried and removed: no faster, see TODO.md)
        exec = compile(g; device=s.device, retain=retain, weight_dtype=wd, free_intermediates=!_chunk_reuse())
        (exec=exec, idg=idg, cache=dc.cache)
    end
end

"""
    prefill_hiddens(session, ids, layers; chunk=prefill_chunk()) -> Vector of (Hidden, n, 1) device arrays

The residual stream after each of the (0-based) `layers`, for every token of `ids`, from a chunked prefill of `ids` from empty states (the session's
decode cache is overwritten). The block drafter reads these as its context (see docs/dflash2_port_plan.md).
"""
function prefill_hiddens(s::LlamaSession, ids::Vector{Int}, layers; chunk::Int=prefill_chunk())
    n = length(ids)
    n + chunk <= s.max_seq || error("prefill_hiddens of $n tokens needs max_seq >= $(n + chunk)")
    gr = _chunk_graph(s, chunk; extract=Tuple(layers)); cache = gr.cache; idg = gr.idg
    for (i, l) in enumerate(s.model.layers)
        NN._is_state_layer(l) && (fill!(cache.self_cache[i][1], 0); fill!(cache.self_cache[i][2], 0))
    end
    Hd = s.model.config.hidden
    out = [Luminal.zero_tensor(s.device, Float32, Hd, n, 1) for _ in layers]
    a = 0
    while a < n
        m = min(chunk, n - a)
        toks = zeros(Float32, chunk, 1); toks[1:m, 1] .= ids[a+1:a+m]
        inputs = Dict{Int, Any}(idg.token_input_id => toks, idg.pos_input_id => Float32[a], idg.lens_input_id => Float32[m])
        for (i, (k_id, v_id)) in enumerate(zip(idg.self_k_ids, idg.self_v_ids))
            inputs[k_id] = cache.self_cache[i][1]; inputs[v_id] = cache.self_cache[i][2]
        end
        Luminal.CHUNK_POS_HINT[] = a
        res = try gr.exec(inputs; device=s.device) finally Luminal.CHUNK_POS_HINT[] = -1 end
        for (j, eid) in enumerate(idg.extract_ids)
            view(out[j], :, a+1:a+m, :) .= view(res[eid], :, 1:m, :)
        end
        res = nothing
        a += m
        a < n && _chunk_reclaim!(s.device)
    end
    return out
end

# Can this session prefill in chunks?
_can_chunk(s::LlamaSession) = s.device isa Luminal.AMDDevice && s.kv_dtype in (Float16, Int8) && NN._needs_lens(s.model) &&
                              NN._prefill_hidden(s.model) && get(ENV, "LUMINAL_PREFILL_CHUNK", "") != "0"

"""
    PrefixCache(; max_sys=2, max_turn=2, min_tokens=1024)

Saved prefill states for reusing the start of the next prompt: a chat turn that continues the previous conversation, or a new conversation
that starts with the same system prompt (coding harnesses send ~10k tokens of it). An entry holds the token ids it covers and copies of
everything a prompt that starts with them needs: the recurrent layers' states, the full-attention layers' K/V below that position, the
residual stream at its last token and (for speculative decoding) the MTP head's K/V. Entries are independent of the live caches, which any
other request may overwrite. Pass the cache as `prefix=` to `generate_ids` / `generate_ids_speculative`, and as `prefix_at` the places to save:
`:sys => n` (the end of the system message: shared by every conversation with that system prompt) and `:turn => n` (the start of the last
assistant header: a later prompt of the same conversation shares everything before it however the reply is re-rendered). The longest entry
that prefixes the prompt is resumed; the rest is prefilled and the new entries replace older ones (a turn entry that the new one extends,
the least recently used beyond `max_sys` / `max_turn`). Prompts shorter than `min_tokens` are not saved. Chunked prefill only (Float16 KV,
Qwen3.5), batch 1.
"""
# seconds spent in the phases of the last chunked prefill / MTP priming (restore, chunks, save, mtp_restore, mtp_prime, mtp_save): for the server's log
const PREFILL_TIMES = Dict{Symbol,Float64}()
_tic() = (AMDGPU.synchronize(); time())

mutable struct PrefixEntry
    kind::Symbol
    ids::Vector{Int}
    states::Vector{Any}
    kv::Vector{Any}
    h::Any
    mtp::Any
end
# A recurrent-state checkpoint of the live prompt: the K/V it needs is the live cache itself (valid while `PrefixCache.live` starts with the
# same tokens), so only the recurrent layers' states and the residual stream at the last token are kept (~150 MB on the 27B, whatever the position).
mutable struct StateCkpt
    pos::Int
    states::Vector{Any}
    h::Any
end
mutable struct PrefixCache
    entries::Vector{PrefixEntry}      # least recently used first
    max_sys::Int
    max_turn::Int
    min_tokens::Int
    pending::Vector{PrefixEntry}      # entries saved by the last prefill (the MTP part is added after its priming)
    resumed::Union{Nothing,PrefixEntry}
    stale::Vector{PrefixEntry}        # replaced entries, freed at the next prefill (the resumed one is still read after this one)
    live::Vector{Int}                 # the prompt whose K/V (below its length) the live cache holds: an entry that it starts with needs no K/V copied back
    live_mtp::Int                     # how many MTP pairs (positions) of the live MTP cache are valid for `live`
    mtp_live::Bool                    # the entry resumed by the last prefill needs no MTP K/V copied back either
    ckpts::Vector{StateCkpt}          # state checkpoints along the live prompt, ascending position
    ckpt_every::Int                   # distance between checkpoints (0: none)
    resumed_kind::Symbol              # what the last prefill resumed from: :sys, :turn, :ckpt or :none
end
PrefixCache(; max_sys::Int=2, max_turn::Int=2, min_tokens::Int=1024, ckpt_every::Int=parse(Int, get(ENV, "LUMINAL_CKPT_EVERY", "4096"))) =
    PrefixCache(PrefixEntry[], max_sys, max_turn, min_tokens, PrefixEntry[], nothing, PrefixEntry[], Int[], 0, false, StateCkpt[], ckpt_every, :none)

function _free_ckpt!(k::StateCkpt)
    for (a, b) in k.states; Luminal._free_now!(a); Luminal._free_now!(b); end
    Luminal._free_now!(k.h)
    return nothing
end
function _lcp(a::AbstractVector, b::AbstractVector)
    m = min(length(a), length(b)); i = 0
    while i < m && a[i+1] == b[i+1]; i += 1; end
    return i
end

_prefix_points(x::Integer) = Pair{Symbol,Int}[:turn => Int(x)]
_prefix_points(x::AbstractVector{<:Integer}) = Pair{Symbol,Int}[:turn => Int(v) for v in x]
_prefix_points(x::AbstractVector{<:Pair}) = Pair{Symbol,Int}[Symbol(k) => Int(v) for (k, v) in x]

function _free_entry!(e::PrefixEntry)
    for (a, b) in e.states; Luminal._free_now!(a); Luminal._free_now!(b); end
    for (a, b) in e.kv; Luminal._free_now!(a); Luminal._free_now!(b); end
    e.h === nothing || Luminal._free_now!(e.h)
    e.mtp === nothing || (Luminal._free_now!(e.mtp[1]); Luminal._free_now!(e.mtp[2]))
    return nothing
end

# The replaced entries (kept until now because the resumed one is read after the prefill: its MTP K/V).
release_stale!(pc::Union{Nothing,PrefixCache}) = pc === nothing || (foreach(_free_entry!, pc.stale); empty!(pc.stale); nothing)

function _register!(pc::PrefixCache, new::Vector{PrefixEntry}, used::Union{Nothing,PrefixEntry})
    used === nothing || (i = findfirst(e -> e === used, pc.entries); i === nothing || push!(pc.entries, popat!(pc.entries, i)))
    for e in new
        if e.kind == :turn                                    # the conversation moved on: the entry it extends is stale
            for old in filter(o -> o.kind == :turn && length(o.ids) < length(e.ids) && view(e.ids, 1:length(o.ids)) == o.ids, pc.entries)
                push!(pc.stale, old); filter!(o -> o !== old, pc.entries)
            end
        end
        push!(pc.entries, e)
    end
    for (kind, mx) in ((:sys, pc.max_sys), (:turn, pc.max_turn))
        while count(e -> e.kind == kind, pc.entries) > mx
            i = findfirst(e -> e.kind == kind, pc.entries)
            push!(pc.stale, pc.entries[i]); deleteat!(pc.entries, i)
        end
    end
end

# Prefill `prompt` (0-based ids) chunk by chunk into the decode cache; returns the last token's logits
# (vocab, 1, 1) on the host, the final residual stream (Hidden, n - off, 1) on the device: the columns of positions off .. n-1 (off = 0
# unless a `prefix` entry was resumed: then its last position is the first column), and off.
# prefix / prefix_at: see `PrefixCache`; need_mtp: only resume entries that carry the MTP head's K/V.
function _prefill_chunked!(s::LlamaSession, prompt::Vector{Int}; chunk::Int=prefill_chunk(), prefix::Union{Nothing,PrefixCache}=nothing, prefix_at=Int[],
                           need_mtp::Bool=false, on_progress=nothing)
    n = length(prompt)
    n < s.max_seq || error("prompt longer than max_seq=$(s.max_seq)")
    n + chunk <= s.max_seq || error("chunked prefill of $n tokens needs max_seq >= $(n + chunk) (the last chunk is padded)")
    s.kv_dtype === Float16 || (prefix = nothing)
    empty!(PREFILL_TIMES); t_start = _tic()
    gr0 = _chunk_graph(s, chunk)
    cache = gr0.cache
    tails = get(ENV, "LUMINAL_TAIL_CHUNKS", "1") == "0" ? () : Tuple(c for c in (64, 256) if c < chunk)     # smaller graphs for a short last chunk (a padded 512-token one costs ~2 s at 120k context)
    Hd = s.model.config.hidden
    best = nothing
    if prefix !== nothing
        release_stale!(prefix)
        prefix.pending = PrefixEntry[]; prefix.resumed = nothing
        for e in prefix.entries
            L = length(e.ids)
            L < n && (!need_mtp || e.mtp !== nothing) && (best === nothing || L > length(best.ids)) && view(prompt, 1:L) == e.ids && (best = e)
        end
    end
    ck = nothing                                             # a state checkpoint of the live prompt further along than any entry
    if prefix !== nothing
        c = min(_lcp(prompt, prefix.live), n - 1)            # the live K/V holds this prompt's first c tokens
        for k in filter(k -> k.pos > c, prefix.ckpts); _free_ckpt!(k); end
        filter!(k -> k.pos <= c, prefix.ckpts); sort!(prefix.ckpts, by = k -> k.pos)
        i = findlast(k -> (!need_mtp || prefix.live_mtp >= k.pos - 1) && get(ENV, "LUMINAL_SKIP_RESTORE", "1") != "0", prefix.ckpts)
        i !== nothing && (best === nothing || prefix.ckpts[i].pos > length(best.ids)) && (ck = prefix.ckpts[i])
        prefix.resumed_kind = ck !== nothing ? :ckpt : best !== nothing ? best.kind : :none
    end
    resume = best !== nothing || ck !== nothing
    a = ck !== nothing ? ck.pos : best !== nothing ? length(best.ids) : 0; off = resume ? a - 1 : 0
    skip_kv = ck !== nothing || (best !== nothing && get(ENV, "LUMINAL_SKIP_RESTORE", "1") != "0" && length(prefix.live) >= a && view(prefix.live, 1:a) == best.ids)   # the live K/V already is the entry's
    mtp_live = skip_kv && prefix.live_mtp >= a - 1
    ck !== nothing && (best = nothing)
    prefix === nothing || (prefix.live = Int[]; prefix.live_mtp = 0)           # (until this prefill is done)
    Hall = Luminal.zero_tensor(s.device, Float32, Hd, n - off, 1)
    logits = nothing
    j = 0
    for (i, l) in enumerate(s.model.layers)
        NN._is_state_layer(l) || continue
        j += 1
        if resume                                            # continue from the saved recurrent states
            st = ck !== nothing ? ck.states[j] : best.states[j]
            copyto!(cache.self_cache[i][1], st[1]); copyto!(cache.self_cache[i][2], st[2])
        else                                                 # the recurrent layers start from empty states (the cache may hold a previous prompt's)
            fill!(cache.self_cache[i][1], 0); fill!(cache.self_cache[i][2], 0)
        end
    end
    if resume && !skip_kv
        j = 0
        for (i, l) in enumerate(s.model.layers)
            NN._is_state_layer(l) && continue
            j += 1
            view(cache.self_cache[i][1], :, 1:a, :, :) .= best.kv[j][1]; view(cache.self_cache[i][2], :, 1:a, :, :) .= best.kv[j][2]
        end
    end
    if resume
        view(Hall, :, 1:1, :) .= (ck !== nothing ? ck.h : best.h)
        prefix.resumed = best; prefix.mtp_live = mtp_live
    end
    PREFILL_TIMES[:restore] = _tic() - t_start; t_chunks = time()
    points = prefix === nothing || n < prefix.min_tokens ? Pair{Symbol,Int}[] :
             sort!(Pair{Symbol,Int}[k => p for (k, p) in _prefix_points(prefix_at) if p > a && p < n], by = last)
    if prefix !== nothing && n >= prefix.min_tokens && prefix.ckpt_every > 0
        have = Set(k.pos for k in prefix.ckpts)
        for p in (a ÷ prefix.ckpt_every + 1) * prefix.ckpt_every:prefix.ckpt_every:n - 1
            p in have || push!(points, :ckpt => p)
        end
        sort!(points, by = last)
    end
    while a < n
        m = min(chunk, n - a)
        for (_, p) in points
            p > a && (m = min(m, p - a); break)              # chunks end at the positions to save
        end
        Cc = something(findfirst(c -> m <= c, tails), 0) == 0 ? chunk : tails[findfirst(c -> m <= c, tails)]
        gr = Cc == chunk ? gr0 : _chunk_graph(s, Cc); idg = gr.idg
        toks = zeros(Float32, Cc, 1); toks[1:m, 1] .= prompt[a+1:a+m]
        inputs = Dict{Int, Any}(idg.token_input_id => toks, idg.pos_input_id => Float32[a], idg.lens_input_id => Float32[m])
        for (i, (k_id, v_id)) in enumerate(zip(idg.self_k_ids, idg.self_v_ids))
            inputs[k_id] = cache.self_cache[i][1]; inputs[v_id] = cache.self_cache[i][2]
        end
        Luminal.CHUNK_POS_HINT[] = a                    # (see chunk_attention!)
        res = try gr.exec(inputs; device=s.device) finally Luminal.CHUNK_POS_HINT[] = -1 end
        view(Hall, :, a+1-off:a+m-off, :) .= view(res[idg.hidden_id], :, 1:m, :)
        a + m == n && (logits = Array{Float32}(res[idg.logits_id]))
        a += m
        for (kind, p) in points
            p == a || continue
            t_sv = _tic()
            states = Any[]; kv = Any[]
            for (i, l) in enumerate(s.model.layers)
                if kind == :ckpt
                    NN._is_state_layer(l) && push!(states, (copy(cache.self_cache[i][1]), copy(cache.self_cache[i][2])))
                elseif NN._is_state_layer(l)
                    push!(states, (copy(cache.self_cache[i][1]), copy(cache.self_cache[i][2])))
                else
                    K = cache.self_cache[i][1]
                    kk = similar(K, size(K, 1), a, size(K, 3), size(K, 4)); vv = similar(kk)
                    kk .= view(K, :, 1:a, :, :); vv .= view(cache.self_cache[i][2], :, 1:a, :, :)
                    push!(kv, (kk, vv))
                end
            end
            hh = Luminal.zero_tensor(s.device, Float32, Hd, 1, 1); view(hh, :, 1:1, :) .= view(Hall, :, a-off:a-off, :)
            kind == :ckpt ? push!(prefix.ckpts, StateCkpt(a, states, hh)) : push!(prefix.pending, PrefixEntry(kind, prompt[1:a], states, kv, hh, nothing))
            PREFILL_TIMES[:save] = get(PREFILL_TIMES, :save, 0.0) + (_tic() - t_sv)
        end
        # the chunk's intermediates that the graph did not free itself are only released by the GC: at ~32 MB per
        # token they would pile up (24 GB per 1024 tokens on the 27B) until it ran
        res = nothing
        a < n && (_chunk_reuse() || _chunk_reclaim!(s.device))      # (not after the last chunk: no more garbage to pile up, and the server collects between requests)
        on_progress === nothing || on_progress(a, n)
    end
    cache.positions .= n
    prefix === nothing || (prefix.live = copy(prompt))                       # the live K/V below n is this prompt's (MTP: set once primed)
    PREFILL_TIMES[:chunks] = _tic() - t_chunks - get(PREFILL_TIMES, :save, 0.0)
    t_reg = time()
    prefix === nothing || _register!(prefix, prefix.pending, best)
    PREFILL_TIMES[:register] = time() - t_reg
    return logits, Hall, off
end


"""
    Sampler(; temperature=1.0, top_p=1.0, top_k=0, rng=Random.default_rng())

Token sampling for `generate_ids` / `generate` (`sampler=`; the default `nothing` is greedy): the `top_k` highest logits are kept (0: all),
divided by `temperature`, softmaxed, cut to the smallest set whose probability reaches `top_p`, and a token is drawn from the renormalised
rest. Qwen3.8's recommended settings for coding: `Sampler(temperature=1.0, top_p=0.95, top_k=20)`. `temperature == 0` is greedy.
"""
Base.@kwdef struct Sampler{R<:AbstractRNG}
    temperature::Float64 = 1.0
    top_p::Float64 = 1.0
    top_k::Int = 0
    rng::R = Random.default_rng()
end

"""The candidate tokens (1-based) and their probabilities after top-k, temperature and top-p, most probable first."""
function _sampling_distribution(sm::Sampler, logits::AbstractVector{<:Real})
    n = length(logits)
    k = sm.top_k > 0 ? min(sm.top_k, n) : n
    idx = k < n ? partialsortperm(logits, 1:k; rev = true) : sortperm(logits; rev = true)
    z = Float64.(logits[idx]) ./ sm.temperature
    z .-= maximum(z)
    p = exp.(z); p ./= sum(p)
    if sm.top_p < 1
        c = cumsum(p)
        keep = something(findfirst(>=(sm.top_p), c), length(p))
        idx = idx[1:keep]; p = p[1:keep]; p ./= sum(p)
    end
    return idx, p
end

"""`sample_token(sampler, logits) -> Int`: a 0-indexed token drawn from `logits` (`nothing` or temperature 0: the argmax)."""
sample_token(::Nothing, logits::AbstractVector) = argmax(logits) - 1
function sample_token(sm::Sampler, logits::AbstractVector)
    sm.temperature <= 0 && return argmax(logits) - 1
    idx, p = _sampling_distribution(sm, logits)
    r = rand(sm.rng); c = 0.0
    for (i, pi) in zip(idx, p)
        c += pi
        r < c && return i - 1
    end
    return idx[end] - 1
end

"""
    generate_ids(session, prompt_ids::Vector{Vector{Int}}; max_new_tokens=200,
                 last_logits=nothing, step_logits=nothing, sampler=nothing) -> Vector{Vector{Int}}

`generate` on token ids (0-indexed, BOS included as wanted): each prompt's
generated ids, the end-of-sequence token included if one was produced. With a
`last_logits` vector, each prompt's prefill logits at its last position are
appended to it, and with `step_logits` each decode step's (vocab, B) logits
(for validation). `sampler` (a [`Sampler`](@ref)) draws each token instead of taking the argmax.
"""
# Prefill `prompt_ids` into the session's decode cache for batch size B = length(prompt_ids): the K/V of
# the full layers are copied in, the recurrent states are left at each prompt's end, and the cache
# positions set. Returns (pf, res, dc, plens, logits): the prefill graph and its results, the decode
# graph (and cache), the prompt lengths and the host logits ((vocab, slen, B), or (vocab, 1, B)).
function _prefill_into_cache!(s::LlamaSession, prompt_ids::Vector{Vector{Int}}; prefix::Union{Nothing,PrefixCache}=nothing, prefix_at=Int[], need_mtp::Bool=false, on_progress=nothing)
    B = length(prompt_ids)
    plens = length.(prompt_ids)
    maximum(plens) < s.max_seq || error("prompt longer than max_seq=$(s.max_seq)")
    if B == 1 && (plens[1] > prefill_chunk() || prefix !== nothing) && _can_chunk(s)
        logits, Hall, off = _prefill_chunked!(s, prompt_ids[1]; prefix, prefix_at, need_mtp, on_progress)
        pf = (last_only=true, hidden_id=1, chunked=true, off=off)
        return pf, Dict{Any,Any}(1 => Hall), _decode_graph(s, 1), plens, logits
    end

    # Prefill. Recurrent layers' states are bound straight to the decode cache's
    # buffers, which the prefill leaves at each prompt's end.
    slen = _prefill_bucket(maximum(plens))
    pf = _prefill_graph(s, slen, B)
    dc = _decode_graph(s, B)
    cache = dc.cache
    ids = zeros(Float32, slen, B)
    for (b, p) in enumerate(prompt_ids)
        ids[1:plens[b], b] .= p
    end
    pins = Dict{Int,Any}(pf.input_id => ids)
    pf.lens_id == 0 || (pins[pf.lens_id] = Float32.(plens))
    for (i, (a, b)) in enumerate(pf.kv_ids)
        NN._is_state_layer(s.model.layers[i]) || continue
        pins[a], pins[b] = cache.self_cache[i]
    end
    res = pf.exec(pins; device=s.device)
    logits = Array{Float32}(res[pf.out_id])                                  # (vocab, slen, B), or (vocab, 1, B)

    # KV cache: sequence b's next token goes to position plens[b]. Slots past a
    # sequence's position are never read, so the reused cache needs no clearing.
    cache.positions .= plens
    for (i, (k, v)) in enumerate(pf.kv_ids)
        NN._is_state_layer(s.model.layers[i]) && continue                  # written in place
        # Prefill K/V (head_dim, slen, kv_heads, B) -> cache (head_dim, max_seq, kv_heads, B),
        # in place on the device; only each prompt's own positions.
        for b in 1:B
            _store_kv!(cache.self_cache[i][1], res[k], plens[b], b)
            _store_kv!(cache.self_cache[i][2], res[v], plens[b], b)
        end
    end
    return pf, res, dc, plens, logits
end

# prefill's K/V (head_dim, slen, kv_heads, B) -> the cache's first n slots of sequence b (an int8 cache quantizes)
_store_kv!(dst, src, n::Int, b::Int) =
    eltype(dst) === Int8 ? Luminal.quantize_kv!(dst, src, n, b) : (view(dst, :, 1:n, :, b) .= view(src, :, 1:n, :, b); dst)

function generate_ids(s::LlamaSession, prompt_ids::Vector{Vector{Int}}; max_new_tokens::Int=200,
                      last_logits=nothing, step_logits=nothing, sampler=nothing, on_token=nothing,
                      prefix=nothing, prefix_at=Int[], on_progress=nothing)
    B = length(prompt_ids)
    B >= 1 || error("no prompts")
    pf, res, dc, plens, logits = _prefill_into_cache!(s, prompt_ids; prefix, prefix_at, on_progress)
    cache = dc.cache
    lastpos(b) = pf.last_only ? 1 : plens[b]
    eos_ids = s.tokenizer === nothing ? Int[] : s.tokenizer.eos_ids
    generated = [[sample_token(sampler, view(logits, :, lastpos(b), b))] for b in 1:B]   # 0-indexed
    last_logits === nothing || append!(last_logits, [logits[:, lastpos(b), b] for b in 1:B])
    done = [g[1] in eos_ids || max_new_tokens <= 1 for g in generated]
    on_token !== nothing && B == 1 && on_token([generated[1][1]]) && return generated

    # Decode. A finished sequence keeps being fed (its last token, at the same
    # position, so it never runs into max_seq) until every sequence is done.
    while !all(done)
        tokens = [g[end] for g in generated]
        out = llama_decode_step!(dc.exec, dc.idg, cache, tokens; advance=.!done, device=s.device)
        host = Array{Float32}(out)                                          # (vocab, 1, B)
        step_logits === nothing || push!(step_logits, host[:, 1, :])
        for b in 1:B
            done[b] && continue
            next = sample_token(sampler, view(host, :, 1, b))               # 0-indexed
            push!(generated[b], next)
            on_token !== nothing && B == 1 && on_token([next]) && return generated
            done[b] = next in eos_ids || length(generated[b]) >= max_new_tokens ||
                      cache.positions[b] >= s.max_seq
        end
    end
    return generated
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
_rebuild_model_like(model::Llama, graph::Luminal.Graph, reg::WeightRegistry; rope_base=model.rope_base) =
    Llama(graph, reg; model.config..., rope_base=rope_base)
_rebuild_model_like(model::Gemma3, graph::Luminal.Graph, reg::WeightRegistry; rope_base=model.rope_base) =
    Gemma3(graph, reg; model.config..., rope_base=rope_base)
_rebuild_model_like(model::Qwen35, graph::Luminal.Graph, reg::WeightRegistry; rope_base=model.rope_base) =
    Qwen35(graph, reg; model.config..., rope_base=rope_base)

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

include("AWQ.jl")

include("Speculative.jl")

end # module Decoding
