# Gemma 3 (text): a Llama-family decoder with
#   - embeddings scaled by sqrt(hidden), tied to the output head;
#   - RMSNorm scaled by (1 + w), and four per block: before and after attention,
#     before and after the MLP (the "post" norms apply to the branch output);
#   - QK-norm (per-head (1 + w) RMSNorm of q and k before RoPE);
#   - GELU (tanh approximation) gating in the MLP;
#   - interleaved attention: local layers see a sliding window of recent tokens
#     with RoPE base `rope_local_base`; every `pattern`-th layer is global, with
#     base `rope_base` and the model's rope scaling (linear x8 on 4B);
#   - scores scaled by query_pre_attn_scalar^-1/2.
# Prefill, the cached decode step, LlamaSession and generate all work with it.

struct Gemma3Block
    attention::SelfAttention
    input_norm::LayerNorm
    post_attention_norm::LayerNorm
    pre_feedforward_norm::LayerNorm
    post_feedforward_norm::LayerNorm
    feed_forward::Mlp
end

struct Gemma3
    embedding::Embedding
    embed_scale::Float32
    layers::Vector{Gemma3Block}
    norm::LayerNorm
    head::Linear
    rope_base::Float32
    config::NamedTuple
end

# Gemma's RMSNorm multiplies by (1 + w): the registered tensor is w, the norm
# reads w + 1 (a node compile folds into a constant)
function _gemma_rmsnorm(dim, cx, reg, name; epsilon)
    w = Luminal.tensor(cx, [dim])
    reg !== nothing && register_weight!(reg, "$(name).weight", w)
    return LayerNorm(w + 1f0, nothing, Float32(epsilon), false)
end

"""
    Gemma3(graph, reg; vocab_size, hidden, n_layers, n_heads, n_kv_heads, head_dim,
           intermediate, rope_base, rope_local_base, rope_scaling, sliding_window,
           pattern, query_pre_attn_scalar, rms_eps, prefix)

Gemma 3's text decoder. Build it from a checkpoint with
`Gemma3(graph, reg; gemma3_config(dir)...)`; `prefix` is where the text model's
weights live (`language_model.model` in the multimodal checkpoints).
"""
function Gemma3(graph::Luminal.Graph, reg=nothing;
                vocab_size=262208, hidden=2560, n_layers=34, n_heads=8, n_kv_heads=4, head_dim=256,
                intermediate=10240, rope_base=1f6, rope_local_base=1f4, rope_scaling=(type=:linear, factor=8.0),
                sliding_window=1024, pattern=6, query_pre_attn_scalar=256, rms_eps=1f-6,
                prefix="language_model.model")
    config = (; vocab_size, hidden, n_layers, n_heads, n_kv_heads, head_dim, intermediate,
              rope_base=Float32(rope_base), rope_local_base=Float32(rope_local_base), rope_scaling,
              sliding_window, pattern, query_pre_attn_scalar, rms_eps=Float32(rms_eps), prefix)
    scale = 1 / sqrt(query_pre_attn_scalar)
    layers = map(0:n_layers-1) do i
        p = "$(prefix).layers.$i"
        is_global = (i + 1) % pattern == 0
        qn = _gemma_rmsnorm(head_dim, graph, reg, "$(p).self_attn.q_norm"; epsilon=rms_eps)
        kn = _gemma_rmsnorm(head_dim, graph, reg, "$(p).self_attn.k_norm"; epsilon=rms_eps)
        attn = SelfAttention(
            _linear(hidden, n_heads * head_dim, graph, reg, "$(p).self_attn.q_proj"; bias=false),
            _linear(hidden, n_kv_heads * head_dim, graph, reg, "$(p).self_attn.k_proj"; bias=false),
            _linear(hidden, n_kv_heads * head_dim, graph, reg, "$(p).self_attn.v_proj"; bias=false),
            _linear(n_heads * head_dim, hidden, graph, reg, "$(p).self_attn.o_proj"; bias=false),
            n_heads, n_kv_heads, head_dim, qn, kn,
            is_global ? rope_scaling : nothing,                 # local layers: unscaled
            is_global ? nothing : Float32(rope_local_base),     # global: the model's base
            Float32(scale),
            is_global ? 0 : sliding_window)
        Gemma3Block(attn,
                    _gemma_rmsnorm(hidden, graph, reg, "$(p).input_layernorm"; epsilon=rms_eps),
                    _gemma_rmsnorm(hidden, graph, reg, "$(p).post_attention_layernorm"; epsilon=rms_eps),
                    _gemma_rmsnorm(hidden, graph, reg, "$(p).pre_feedforward_layernorm"; epsilon=rms_eps),
                    _gemma_rmsnorm(hidden, graph, reg, "$(p).post_feedforward_layernorm"; epsilon=rms_eps),
                    Mlp(hidden, intermediate, graph, reg, "$(p).mlp"; act=:gelu_tanh))
    end
    emb = Luminal.tensor(graph, [vocab_size, hidden])
    reg !== nothing && register_weight!(reg, "$(prefix).embed_tokens.weight", emb)
    # tied head: its own node, loaded from the embedding's array (see tie_weight!)
    head = _linear(hidden, vocab_size, graph, reg, "lm_head"; bias=false)
    reg !== nothing && tie_weight!(reg, "lm_head.weight", "$(prefix).embed_tokens.weight")
    return Gemma3(Embedding(emb), Float32(sqrt(hidden)), layers,
                  _gemma_rmsnorm(hidden, graph, reg, "$(prefix).norm"; epsilon=rms_eps),
                  head, Float32(rope_base), config)
end

function (b::Gemma3Block)(x::Luminal.GraphTensor, prev_seq::Int; rope_base=1f6, return_kv::Bool=false)
    a = b.attention(b.input_norm(x), prev_seq; rope_base=rope_base, return_kv=return_kv)
    attn_out, k, v = return_kv ? a : (a, nothing, nothing)
    x = x + b.post_attention_norm(attn_out)
    x = x + b.post_feedforward_norm(b.feed_forward(b.pre_feedforward_norm(x)))
    return return_kv ? (x, k, v) : x
end

_embed(m::Gemma3, tokens) = m.embedding(tokens) * m.embed_scale

function (m::Gemma3)(input::Luminal.GraphTensor, prev_seq::Int; return_kv::Bool=false)
    x = _embed(m, input)
    kvs = Tuple{Luminal.GraphTensor, Luminal.GraphTensor}[]
    for layer in m.layers
        if return_kv
            x, k, v = layer(x, prev_seq; rope_base=m.rope_base, return_kv=true)
            push!(kvs, (k, v))
        else
            x = layer(x, prev_seq; rope_base=m.rope_base)
        end
    end
    logits = m.head(m.norm(x))
    return return_kv ? (logits, kvs) : logits
end

function _decode_block(b::Gemma3Block, x, step_pos, pos_tensor, pk, pv, rope_base)
    attn_out, nk, nv = llama_self_attn_cached(b.attention, b.input_norm(x), step_pos, pos_tensor, pk, pv;
                                              rope_base=rope_base)
    x = x + b.post_attention_norm(attn_out)
    x = x + b.post_feedforward_norm(b.feed_forward(b.pre_feedforward_norm(x)))
    return x, nk, nv
end

"""
    gemma3_config(model_dir) -> NamedTuple

`Gemma3` keyword arguments from a Hugging Face `config.json` (the multimodal
checkpoints' `text_config`, or a text-only one). Errors on features this
implementation does not support.
"""
function gemma3_config(model_dir::String)
    c0 = JSON3.read(read(joinpath(model_dir, "config.json"), String))
    c = haskey(c0, :text_config) ? c0[:text_config] : c0
    get_(k, d) = haskey(c, k) && c[k] !== nothing ? c[k] : d
    unsupported = String[]
    get_(:attn_logit_softcapping, nothing) === nothing || push!(unsupported, "attn_logit_softcapping")
    get_(:final_logit_softcapping, nothing) === nothing || push!(unsupported, "final_logit_softcapping")
    get_(:attention_bias, false) && push!(unsupported, "attention_bias")
    String(get_(:hidden_activation, "gelu_pytorch_tanh")) == "gelu_pytorch_tanh" ||
        push!(unsupported, "hidden_activation=$(c[:hidden_activation])")
    rs = get_(:rope_scaling, nothing)
    scaling = rs === nothing ? nothing : _rope_scaling(rs)
    scaling === :unsupported && push!(unsupported, "rope_scaling=$rs")
    # layer pattern: newer configs list layer_types, older give sliding_window_pattern
    n_layers = Int(c[:num_hidden_layers])
    pattern = Int(get_(:sliding_window_pattern, 6))
    if haskey(c, :layer_types)
        lt = String.(c[:layer_types])
        globals = findall(==("full_attention"), lt)
        globals == collect(pattern:pattern:n_layers) || push!(unsupported, "layer_types=$lt")
    end
    isempty(unsupported) || error("unsupported Gemma3 config in $model_dir: " * join(unsupported, ", "))
    # where the text weights live: "language_model.model." in multimodal checkpoints
    idx = joinpath(model_dir, "model.safetensors.index.json")
    keys_ = isfile(idx) ? String.(collect(keys(JSON3.read(read(idx, String))[:weight_map]))) : String[]
    prefix = any(k -> startswith(k, "language_model.model."), keys_) ? "language_model.model" :
             any(k -> startswith(k, "model.language_model."), keys_) ? "model.language_model" : "model"
    hidden = Int(c[:hidden_size]); n_heads = Int(c[:num_attention_heads])
    return (vocab_size=Int(c[:vocab_size]), hidden=hidden, n_layers=n_layers, n_heads=n_heads,
            n_kv_heads=Int(get_(:num_key_value_heads, n_heads)),
            head_dim=Int(get_(:head_dim, div(hidden, n_heads))),
            intermediate=Int(c[:intermediate_size]),
            rope_base=Float32(get_(:rope_theta, 1f6)), rope_local_base=Float32(get_(:rope_local_base_freq, 1f4)),
            rope_scaling=scaling, sliding_window=Int(get_(:sliding_window, 4096)), pattern=pattern,
            query_pre_attn_scalar=Float64(get_(:query_pre_attn_scalar, get_(:head_dim, div(hidden, n_heads)))),
            rms_eps=Float32(get_(:rms_norm_eps, 1f-6)), prefix=prefix)
end

"""
    model_template(model_dir) -> Llama or Gemma3

The architecture of a checkpoint, from its `config.json`, as an unregistered
template model (for `LlamaSession`, `llama_generate`, `awq_scales`, ...):
`Gemma3` for Gemma 3, otherwise `Llama` (Llama 2/3/3.x, TinyLlama, Qwen3).
"""
function model_template(model_dir::String)
    c = JSON3.read(read(joinpath(model_dir, "config.json"), String))
    mt = String(get(c, :model_type, "llama"))
    startswith(mt, "gemma3") && return Gemma3(Luminal.Graph(), nothing; gemma3_config(model_dir)...)
    return Llama(Luminal.Graph(), nothing; llama_config(model_dir)...)
end
