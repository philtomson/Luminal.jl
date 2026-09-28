# Qwen3.5 / Qwen3.6 (`qwen3_5`) building blocks. So far: the Gated DeltaNet
# linear-attention layer, which three of every four Qwen3.5/3.6 layers use in
# place of softmax attention. Instead of a KV cache it carries two states per
# sequence, updated in place by the layer's ops (see CausalConv and DeltaRule):
#   conv: (C, K, B), the last K pre-convolution inputs (C = 2 nk dk + nv dv)
#   rec:  (dv, dk, nv, B), the transposed delta-rule state of each value head
# The same graph code serves prefill (fresh=true: states start from zero and end
# at the prompt's) and decode (fresh=false: continue from the states, S = 1).

struct GatedDeltaNet
    in_proj_qkv::Linear
    in_proj_z::Linear
    in_proj_b::Linear
    in_proj_a::Linear
    conv_w::Luminal.GraphTensor      # (C, 1, K)
    A_log::Luminal.GraphTensor       # (nv,)
    dt_bias::Luminal.GraphTensor     # (nv,)
    norm::LayerNorm                  # gated RMSNorm over dv (plain weight)
    out_proj::Linear
    nk::Int; nv::Int; dk::Int; dv::Int; K::Int
end

function GatedDeltaNet(hidden::Int, graph::Luminal.Graph, reg=nothing, prefix::String="linear_attn";
                       nk::Int, nv::Int, dk::Int, dv::Int, K::Int=4, epsilon=1f-6)
    C = 2nk * dk + nv * dv
    t(dims, name) = (x = Luminal.tensor(graph, dims); reg !== nothing && register_weight!(reg, "$(prefix).$(name)", x); x)
    return GatedDeltaNet(
        _linear(hidden, C, graph, reg, "$(prefix).in_proj_qkv"; bias=false),
        _linear(hidden, nv * dv, graph, reg, "$(prefix).in_proj_z"; bias=false),
        _linear(hidden, nv, graph, reg, "$(prefix).in_proj_b"; bias=false),
        _linear(hidden, nv, graph, reg, "$(prefix).in_proj_a"; bias=false),
        t([C, 1, K], "conv1d.weight"), t([nv], "A_log"), t([nv], "dt_bias"),
        _rmsnorm(dv, graph, reg, "$(prefix).norm"; epsilon=epsilon),
        _linear(nv * dv, hidden, graph, reg, "$(prefix).out_proj"; bias=false),
        nk, nv, dk, dv, K)
end

"""
    deltanet_state_shapes(dn, batch) -> (conv, rec)

Shapes of a Gated DeltaNet layer's two state buffers.
"""
deltanet_state_shapes(dn::GatedDeltaNet, batch::Int) =
    ([2dn.nk * dn.dk + dn.nv * dn.dv, dn.K, batch], [dn.dv, dn.dk, dn.nv, batch])

# x / ||x|| over dim 1, as FLA's l2norm: x * rsqrt(sum(x^2) + eps)
function _l2norm(x::Luminal.GraphTensor; eps=1f-6)
    n = Luminal.realized_dims(x.shape)[1]
    return x * Luminal.expand(Luminal.reciprocal(sqrt(Luminal.sum(x * x, 1) + eps)), 1, n)
end

"""
    (dn::GatedDeltaNet)(x, conv_state, rec_state, lens; fresh) -> (hidden, S, B)

The layer on `x` (hidden, S, B), reading and updating the state tensors (graph
inputs bound to the state buffers; see `deltanet_state_shapes`). `lens` (B,)
is how many of each sequence's tokens to consume (the rest are padding, or a
sequence holding its position). `fresh=true` for a prefill from empty states,
`false` to continue (decode).
"""
function (dn::GatedDeltaNet)(x::Luminal.GraphTensor, conv_state::Luminal.GraphTensor,
                             rec_state::Luminal.GraphTensor, lens::Luminal.GraphTensor; fresh::Bool)
    g_ = x.graph_ref
    _, S, B = Luminal.realized_dims(x.shape)
    nk, nv, dk, dv = dn.nk, dn.nv, dn.dk, dn.dv
    C = 2nk * dk + nv * dv

    # projections, then the causal convolution over q, k and v together
    qkv = dn.in_proj_qkv(x)                                                   # (C, S, B)
    conv_ins = [(t.id, 0, t.shape) for t in (qkv, conv_state, dn.conv_w, lens)]
    qkv = Luminal.add_op!(g_, Luminal.CausalConv(fresh), conv_ins, Luminal.ShapeTracker([C, S, B]))

    q = Luminal.reshape(Luminal.slice_along(qkv, 1, 0, nk * dk), [dk, nk, S, B])
    k = Luminal.reshape(Luminal.slice_along(qkv, 1, nk * dk, 2nk * dk), [dk, nk, S, B])
    v = Luminal.reshape(Luminal.slice_along(qkv, 1, 2nk * dk, C), [dv, nv, S, B])
    q, k = _l2norm(q), _l2norm(k)

    # gates: beta = sigmoid(b), g = -exp(A_log) softplus(a + dt_bias), per value head
    beta = Luminal.sigmoid(dn.in_proj_b(x))                                   # (nv, S, B)
    a = dn.in_proj_a(x) + Luminal.expand(Luminal.expand(dn.dt_bias, 2, S), 3, B)
    g = Luminal.expand(Luminal.expand(-exp(dn.A_log), 2, S), 3, B) * Luminal.softplus(a)

    ins = [(t.id, 0, t.shape) for t in (q, k, v, g, beta, rec_state, lens)]
    o = Luminal.add_op!(g_, Luminal.DeltaRule(Float32(1 / sqrt(dk)), fresh), ins,
                        Luminal.ShapeTracker([dv, nv, S, B]))

    # gated RMSNorm per value head: norm(o) * w * silu(z)
    z = Luminal.reshape(dn.in_proj_z(x), [dv, nv, S, B])
    o = dn.norm(o) * Luminal.silu(z)
    return dn.out_proj(Luminal.reshape(o, [nv * dv, S, B]))
end

"""
    qwen35_attention(hidden, graph, reg, prefix; n_heads, n_kv_heads, head_dim,
                     rotary_dim, rope_theta, epsilon) -> SelfAttention

Qwen3.5 / Qwen3.6's gated full attention: q_proj yields each head's query and
an output gate (the output is multiplied by sigmoid(gate) before o_proj),
(1 + w) RMSNorms on q and k per head, and RoPE on each head's first
`rotary_dim` dims only. (Its "interleaved mRoPE" differs from plain RoPE only
for image / video positions; for text the three position streams coincide.)
"""
function qwen35_attention(hidden::Int, graph::Luminal.Graph, reg, prefix::String;
                          n_heads::Int, n_kv_heads::Int, head_dim::Int, rotary_dim::Int,
                          rope_theta, epsilon=1f-6)
    return SelfAttention(
        _linear(hidden, 2n_heads * head_dim, graph, reg, "$(prefix).q_proj"; bias=false),
        _linear(hidden, n_kv_heads * head_dim, graph, reg, "$(prefix).k_proj"; bias=false),
        _linear(hidden, n_kv_heads * head_dim, graph, reg, "$(prefix).v_proj"; bias=false),
        _linear(n_heads * head_dim, hidden, graph, reg, "$(prefix).o_proj"; bias=false),
        n_heads, n_kv_heads, head_dim,
        _gemma_rmsnorm(head_dim, graph, reg, "$(prefix).q_norm"; epsilon=epsilon),
        _gemma_rmsnorm(head_dim, graph, reg, "$(prefix).k_norm"; epsilon=epsilon),
        nothing, Float32(rope_theta), Float32(1 / sqrt(head_dim)), 0,
        rotary_dim, true)
end

# ── The model ─────────────────────────────────────────────────────────────────

struct Qwen35Block
    kind::Symbol                                  # :linear (Gated DeltaNet) or :full (gated attention)
    mixer::Union{GatedDeltaNet, SelfAttention}
    input_norm::LayerNorm                         # (1 + w) RMSNorms
    post_attention_norm::LayerNorm
    feed_forward::Mlp
end

struct Qwen35
    embedding::Embedding
    layers::Vector{Qwen35Block}
    norm::LayerNorm
    head::Linear
    rope_base::Float32
    config::NamedTuple
end

"""
    Qwen35(graph, reg; qwen35_config(dir)...)

Qwen3.5 / Qwen3.6 (dense) text decoder: layers of Gated DeltaNet
(`layer_types` "linear_attention") or gated full attention, each followed by a
SwiGLU MLP, with (1 + w) RMSNorms. Prefill (`model(tokens, 0; return_kv, lens)`),
the cached decode step, `LlamaSession` and `generate` work with it; the full
layers keep a KV cache, the linear ones their convolution and recurrent states.
"""
function Qwen35(graph::Luminal.Graph, reg=nothing; vocab_size, hidden, n_layers, n_heads, n_kv_heads, head_dim,
                intermediate, layer_types, linear_nk, linear_nv, linear_dk, linear_dv, conv_kernel=4,
                rope_base=1f7, rotary_dim, rms_eps=1f-6, tie_embeddings=false, prefix="model.language_model")
    config = (; vocab_size, hidden, n_layers, n_heads, n_kv_heads, head_dim, intermediate, layer_types,
              linear_nk, linear_nv, linear_dk, linear_dv, conv_kernel, rope_base=Float32(rope_base), rotary_dim,
              rms_eps=Float32(rms_eps), tie_embeddings, prefix)
    layers = map(0:n_layers-1) do i
        p = "$(prefix).layers.$i"
        kind = layer_types[i+1] == "linear_attention" ? :linear : :full
        mixer = kind === :linear ?
            GatedDeltaNet(hidden, graph, reg, "$(p).linear_attn"; nk=linear_nk, nv=linear_nv, dk=linear_dk,
                          dv=linear_dv, K=conv_kernel, epsilon=rms_eps) :
            qwen35_attention(hidden, graph, reg, "$(p).self_attn"; n_heads=n_heads, n_kv_heads=n_kv_heads,
                             head_dim=head_dim, rotary_dim=rotary_dim, rope_theta=rope_base, epsilon=rms_eps)
        Qwen35Block(kind, mixer,
                    _gemma_rmsnorm(hidden, graph, reg, "$(p).input_layernorm"; epsilon=rms_eps),
                    _gemma_rmsnorm(hidden, graph, reg, "$(p).post_attention_layernorm"; epsilon=rms_eps),
                    Mlp(hidden, intermediate, graph, reg, "$(p).mlp"))
    end
    emb = Luminal.tensor(graph, [vocab_size, hidden])
    reg !== nothing && register_weight!(reg, "$(prefix).embed_tokens.weight", emb)
    head = _linear(hidden, vocab_size, graph, reg, "lm_head"; bias=false)
    tie_embeddings && reg !== nothing && tie_weight!(reg, "lm_head.weight", "$(prefix).embed_tokens.weight")
    return Qwen35(Embedding(emb), layers, _gemma_rmsnorm(hidden, graph, reg, "$(prefix).norm"; epsilon=rms_eps),
                  head, Float32(rope_base), config)
end

"""
    (m::Qwen35)(tokens, 0; return_kv=false, lens) -> logits, or (logits, states)

Prefill from empty states. `lens` (B,) holds each prompt's length (tokens past
it are padding). With `return_kv`, `states[i]` is layer i's (K, V) outputs for
a full layer, or its (conv, rec) state *inputs* for a linear layer: bind those
to the state buffers, which the prefill leaves at the prompts' end.
"""
function (m::Qwen35)(input::Luminal.GraphTensor, prev_seq::Int; return_kv::Bool=false, lens::Luminal.GraphTensor)
    prev_seq == 0 || error("Qwen35 prefill starts from empty states (prev_seq = 0)")
    g = input.graph_ref
    B = Luminal.realized_dims(input.shape)[2]
    x = m.embedding(input)
    states = Tuple{Luminal.GraphTensor, Luminal.GraphTensor}[]
    for layer in m.layers
        h = layer.input_norm(x)
        if layer.kind === :linear
            cs, rs = deltanet_state_shapes(layer.mixer, B)
            conv = Luminal.tensor(g, cs); rec = Luminal.tensor(g, rs)
            a = layer.mixer(h, conv, rec, lens; fresh=true)
            push!(states, (conv, rec))
        else
            a, k, v = layer.mixer(h, 0; rope_base=m.rope_base, return_kv=true)
            push!(states, (k, v))
        end
        x = x + a
        x = x + layer.feed_forward(layer.post_attention_norm(x))
    end
    logits = m.head(m.norm(x))
    return return_kv ? (logits, states) : logits
end

# decode-step hooks (see build_llama_decode_step!)
_needs_lens(::Qwen35) = true
_state_shapes(l::Qwen35Block, max_seq, batch) =
    l.kind === :linear ? deltanet_state_shapes(l.mixer, batch) :
    ([l.mixer.head_dim, max_seq, l.mixer.n_kv_heads, batch], [l.mixer.head_dim, max_seq, l.mixer.n_kv_heads, batch])
_is_state_layer(l::Qwen35Block) = l.kind === :linear

function _decode_block(l::Qwen35Block, x, step_pos, pos_tensor, s1, s2, rope_base; lens=nothing)
    h = l.input_norm(x)
    if l.kind === :linear
        a = l.mixer(h, s1, s2, lens; fresh=false)
        nk, nv = s1, s2
    else
        a, nk, nv = llama_self_attn_cached(l.mixer, h, step_pos, pos_tensor, s1, s2; rope_base=rope_base)
    end
    x = x + a
    x = x + l.feed_forward(l.post_attention_norm(x))
    return x, nk, nv
end

"""
    qwen35_config(model_dir) -> NamedTuple

`Qwen35` keyword arguments from a Qwen3.5 / Qwen3.6 `config.json` (its
`text_config`). Errors on what this implementation doesn't support (the MoE
variants, biases).
"""
function qwen35_config(model_dir::String)
    c0 = JSON3.read(read(joinpath(model_dir, "config.json"), String))
    c = haskey(c0, :text_config) ? c0[:text_config] : c0
    get_(k, d) = haskey(c, k) && c[k] !== nothing ? c[k] : d
    unsupported = String[]
    mt = String(get_(:model_type, "qwen3_5_text"))
    mt in ("qwen3_5_text", "qwen3_5") || push!(unsupported, "model_type=$mt")
    haskey(c, :num_experts) && push!(unsupported, "mixture of experts")
    get_(:attention_bias, false) && push!(unsupported, "attention_bias")
    get_(:attn_output_gate, true) || push!(unsupported, "attn_output_gate=false")
    String(get_(:hidden_act, "silu")) == "silu" || push!(unsupported, "hidden_act=$(c[:hidden_act])")
    rp = get_(:rope_parameters, Dict{Symbol,Any}())
    String(get(rp, :rope_type, "default")) == "default" || push!(unsupported, "rope_type=$(rp[:rope_type])")
    isempty(unsupported) || error("unsupported Qwen3.5 config in $model_dir: " * join(unsupported, ", "))
    n_layers = Int(c[:num_hidden_layers])
    lt = haskey(c, :layer_types) ? String.(c[:layer_types]) :
         [(i + 1) % Int(get_(:full_attention_interval, 4)) == 0 ? "full_attention" : "linear_attention" for i in 0:n_layers-1]
    hidden = Int(c[:hidden_size]); n_heads = Int(c[:num_attention_heads])
    head_dim = Int(get_(:head_dim, div(hidden, n_heads)))
    idx = joinpath(model_dir, "model.safetensors.index.json")
    st = joinpath(model_dir, "model.safetensors")
    keys_ = isfile(idx) ? String.(collect(keys(JSON3.read(read(idx, String))[:weight_map]))) :
            isfile(st) ? String.(filter(!=(:__metadata__), collect(keys(open(io -> Luminal.SafeTensors.load_header(io)[1], st))))) : String[]
    prefix = any(k -> startswith(k, "model.language_model."), keys_) ? "model.language_model" : "model"
    return (vocab_size=Int(c[:vocab_size]), hidden=hidden, n_layers=n_layers, n_heads=n_heads,
            n_kv_heads=Int(get_(:num_key_value_heads, n_heads)), head_dim=head_dim,
            intermediate=Int(c[:intermediate_size]), layer_types=lt,
            linear_nk=Int(c[:linear_num_key_heads]), linear_nv=Int(c[:linear_num_value_heads]),
            linear_dk=Int(c[:linear_key_head_dim]), linear_dv=Int(c[:linear_value_head_dim]),
            conv_kernel=Int(get_(:linear_conv_kernel_dim, 4)),
            rope_base=Float32(get(rp, :rope_theta, get_(:rope_theta, 1f7))),
            rotary_dim=round(Int, head_dim * Float64(get(rp, :partial_rotary_factor, get_(:partial_rotary_factor, 1.0)))),
            rms_eps=Float32(get_(:rms_norm_eps, 1f-6)),
            tie_embeddings=Bool(get(c0, :tie_word_embeddings, get_(:tie_word_embeddings, false))),
            prefix=prefix)
end
