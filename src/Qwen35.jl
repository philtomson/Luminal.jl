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
    return Luminal.add_op!(x.graph_ref, Luminal.L2NormOp(Float32(eps)), [(x.id, 0, x.shape)],
                           Luminal.ShapeTracker(Luminal.realized_dims(x.shape)))
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
                             rec_state::Luminal.GraphTensor, lens::Luminal.GraphTensor; fresh::Bool,
                             snaps=nothing, snap_t::Int=1)
    g_ = x.graph_ref
    _, S, B = Luminal.realized_dims(x.shape)
    nk, nv, dk, dv = dn.nk, dn.nv, dn.dk, dn.dv
    C = 2nk * dk + nv * dv

    # projections, then the causal convolution over q, k and v together
    qkv = dn.in_proj_qkv(x)                                                   # (C, S, B)
    conv_ins = [(t.id, 0, t.shape) for t in (qkv, conv_state, dn.conv_w, lens)]
    snaps === nothing || push!(conv_ins, (snaps[1].id, 0, snaps[1].shape))   # state after `snap_t` tokens lands here
    qkv = Luminal.add_op!(g_, Luminal.CausalConv(fresh, snaps === nothing ? 0 : snap_t), conv_ins, Luminal.ShapeTracker([C, S, B]))

    q = Luminal.reshape(Luminal.slice_along(qkv, 1, 0, nk * dk), [dk, nk, S, B])
    k = Luminal.reshape(Luminal.slice_along(qkv, 1, nk * dk, 2nk * dk), [dk, nk, S, B])
    v = Luminal.reshape(Luminal.slice_along(qkv, 1, 2nk * dk, C), [dv, nv, S, B])
    q, k = _l2norm(q), _l2norm(k)

    # gates: beta = sigmoid(b), g = -exp(A_log) softplus(a + dt_bias), per value head
    beta = Luminal.sigmoid(dn.in_proj_b(x))                                   # (nv, S, B)
    a = dn.in_proj_a(x) + Luminal.expand(Luminal.expand(dn.dt_bias, 2, S), 3, B)
    g = Luminal.expand(Luminal.expand(-exp(dn.A_log), 2, S), 3, B) * Luminal.softplus(a)

    ins = [(t.id, 0, t.shape) for t in (q, k, v, g, beta, rec_state, lens)]
    snaps === nothing || push!(ins, (snaps[2].id, 0, snaps[2].shape))
    o = Luminal.add_op!(g_, Luminal.DeltaRule(Float32(1 / sqrt(dk)), fresh, snaps === nothing ? 0 : snap_t), ins,
                        Luminal.ShapeTracker([dv, nv, S, B]))

    # gated RMSNorm per value head: norm(o) * w * silu(z)
    z = Luminal.reshape(dn.in_proj_z(x), [dv, nv, S, B])
    # (one kernel for rmsnorm(o) * silu(z); the weight and epsilon are dn.norm's)
    o = Luminal.add_op!(g_, Luminal.GatedRMSNormOp(dn.norm.epsilon),
                        [(o.id, 0, o.shape), (dn.norm.weight.id, 0, dn.norm.weight.shape), (z.id, 0, z.shape)],
                        Luminal.ShapeTracker([dv, nv, S, B]))
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
                    Mlp(hidden, intermediate, graph, reg, "$(p).mlp"; swiglu=true))
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
function (m::Qwen35)(input::Luminal.GraphTensor, prev_seq::Int; return_kv::Bool=false, lens::Luminal.GraphTensor,
                     last_only::Bool=false, hidden_ref=nothing)
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
    hidden_ref === nothing || (hidden_ref[] = x)    # the final residual stream, for the MTP head
    if last_only   # logits at each sequence's position lens[b] only: (vocab, 1, B)
        H, S, _ = Luminal.realized_dims(x.shape)
        x = Luminal.add_op!(g, Luminal.LastToken(), [(x.id, 0, x.shape), (lens.id, 0, lens.shape)],
                            Luminal.ShapeTracker([H, 1, B]))
    end
    logits = m.head(m.norm(x))
    return return_kv ? (logits, states) : logits
end

# decode-step hooks (see build_llama_decode_step!)
_needs_lens(::Qwen35) = true
_prefill_last_only(::Qwen35) = true
_prefill_hidden(::Qwen35) = true
_state_shapes(l::Qwen35Block, max_seq, batch) =
    l.kind === :linear ? deltanet_state_shapes(l.mixer, batch) :
    ([l.mixer.head_dim, max_seq, l.mixer.n_kv_heads, batch], [l.mixer.head_dim, max_seq, l.mixer.n_kv_heads, batch])
_is_state_layer(l::Qwen35Block) = l.kind === :linear

function _decode_block(l::Qwen35Block, x, step_pos, pos_tensor, s1, s2, rope_base; lens=nothing, snaps=nothing, nsnap=1)
    h = l.input_norm(x)
    if l.kind === :linear
        a = l.mixer(h, s1, s2, lens; fresh=false, snaps=snaps, snap_t=nsnap)
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

"""
    qwen35_config(f::GGUFFile) -> NamedTuple

`Qwen35` keyword arguments from a llama.cpp GGUF file's metadata (dense
`qwen35` architecture only), for files that come without a `config.json`. The
weights are named as `load_qwen35_gguf` returns them.
"""
function qwen35_config(f::Luminal.GGUFFile)
    md = f.metadata
    arch = String(get(md, "general.architecture", "?"))
    arch == "qwen35" || error("unsupported GGUF architecture \"$arch\" (need the dense qwen35)")
    g(k, d=nothing) = (v = get(md, "qwen35.$k", d); v === nothing ? error("GGUF metadata qwen35.$k is missing") : v)
    n_layers = Int(g("block_count")) - Int(g("nextn_predict_layers", 0))   # MTP layers trail the stack
    hidden = Int(g("embedding_length"))
    n_heads = Int(g("attention.head_count")); head_dim = Int(g("attention.key_length"))
    interval = Int(g("full_attention_interval", 4))
    nv = Int(g("ssm.time_step_rank"))
    return (vocab_size=length(md["tokenizer.ggml.tokens"]), hidden=hidden, n_layers=n_layers,
            n_heads=n_heads, n_kv_heads=Int(g("attention.head_count_kv")), head_dim=head_dim,
            intermediate=Int(g("feed_forward_length")),
            layer_types=[(i + 1) % interval == 0 ? "full_attention" : "linear_attention" for i in 0:n_layers-1],
            linear_nk=Int(g("ssm.group_count")), linear_nv=nv,
            linear_dk=Int(g("ssm.state_size")), linear_dv=Int(g("ssm.inner_size")) ÷ nv,
            conv_kernel=Int(g("ssm.conv_kernel")),
            rope_base=Float32(g("rope.freq_base", 1f7)), rotary_dim=Int(g("rope.dimension_count")),
            rms_eps=Float32(g("attention.layer_norm_rms_epsilon", 1f-6)),
            tie_embeddings=!haskey(f.tensors, "output.weight"), prefix="model")
end

# ── Loading llama.cpp GGUF files ──────────────────────────────────────────────

# Row order of a V-head dimension: llama.cpp stores Qwen3.5's value heads tiled
# ([K0, K1, ...] repeated) where Hugging Face groups them by key head
# ([K0's G heads, K1's, ...]). p[o] is the GGUF index of HF index o (1-based).
function _untile_perm(nk::Int, G::Int, hd::Int)
    p = Vector{Int}(undef, nk * G * hd)
    for k in 0:nk-1, g in 0:G-1, d in 0:hd-1
        p[(k * G + g) * hd + d + 1] = (g * nk + k) * hd + d + 1
    end
    return p
end

"""
    load_qwen35_gguf(path, cfg; device=get_device(), quantize=nothing) -> Dict{String, Any}

The weights of a Qwen3.5 / Qwen3.6 (dense) llama.cpp GGUF file under Hugging
Face names, for `Qwen35(graph, reg; cfg...)` / `LlamaSession(...; weights=...)`
(`cfg` from `qwen35_config` on the model's `config.json`). Each tensor is
dequantized to Float32 (any GGUF type `gguf_tensor` reads: F32/F16/BF16, Q8_0,
Q4_K/Q5_K/Q6_K), transposed to HF's (out, in), un-transformed from llama.cpp's
conversion (norm weights stored as 1 + w, A as -exp(A_log), and the linear
layers' value heads reordered from grouped to tiled when there are fewer key
heads than value heads), moved to `device` and, with `quantize` (as
`load_weights_to_dict`), converted to reduced precision one tensor at a time.
"""
function load_qwen35_gguf(path::String, cfg; device=Luminal.get_device(), quantize=nothing)
    f = Luminal.read_gguf(path)
    p = cfg.prefix
    nk, nv, dk, dv = cfg.linear_nk, cfg.linear_nv, cfg.linear_dk, cfg.linear_dv
    G = nv ÷ nk
    pv = G > 1 ? _untile_perm(nk, G, dv) : nothing         # value-head channels
    p1 = G > 1 ? _untile_perm(nk, G, 1) : nothing          # one value per head
    W = Dict{String, Any}()
    converted = 0
    # Matrices go to the device in GGUF's (in, out) order and are transposed to
    # HF's (out, in) there, and reordered there (as are their row permutations):
    # permuting tens of GB on one CPU thread would dominate the load.
    dperm(perm) = perm === nothing ? nothing : Luminal.to_device(perm, device)
    pvd, p1d = dperm(pv), dperm(p1)
    function mat(name)
        A = Luminal.to_device(Luminal.gguf_tensor(f, name), device)
        B = permutedims(A, (2, 1))
        Luminal._free_now!(A)
        return B                                                     # HF (out, in), on the device
    end
    vec_(name) = vec(Luminal.gguf_tensor(f, name))
    function rows(A, perm)                                           # A[perm, :] on the device
        perm === nothing && return A
        B = A[perm, :]
        Luminal._free_now!(A)
        return B
    end
    function store!(key, A)
        data = A isa Array ? Luminal.to_device(Array{Float32}(A), device) : A
        data, n = Luminal._quantize_loaded(key, data, quantize)
        W[key] = data
        converted += n
        converted >= 2^31 && (Luminal.reclaim!(device); GC.gc(false); converted = 0)
    end
    store!("$p.embed_tokens.weight", mat("token_embd.weight"))
    store!("lm_head.weight", haskey(f.tensors, "output.weight") ? mat("output.weight") : mat("token_embd.weight"))
    store!("$p.norm.weight", vec_("output_norm.weight") .- 1f0)
    for i in 0:cfg.n_layers-1
        b = "blk.$i"; h = "$p.layers.$i"
        store!("$h.input_layernorm.weight", vec_("$b.attn_norm.weight") .- 1f0)
        store!("$h.post_attention_layernorm.weight", vec_("$b.post_attention_norm.weight") .- 1f0)
        for (g, hf) in (("ffn_gate", "gate_proj"), ("ffn_up", "up_proj"), ("ffn_down", "down_proj"))
            store!("$h.mlp.$hf.weight", mat("$b.$g.weight"))
        end
        if cfg.layer_types[i+1] == "linear_attention"
            la = "$h.linear_attn"
            qkv = mat("$b.attn_qkv.weight")
            qk = 2nk * dk
            store!("$la.in_proj_qkv.weight", vcat(qkv[1:qk, :], rows(qkv[qk+1:end, :], pvd)))
            Luminal._free_now!(qkv)
            store!("$la.in_proj_z.weight", rows(mat("$b.attn_gate.weight"), pvd))
            store!("$la.in_proj_a.weight", rows(mat("$b.ssm_alpha.weight"), p1d))
            store!("$la.in_proj_b.weight", rows(mat("$b.ssm_beta.weight"), p1d))
            A = vec_("$b.ssm_a")                                             # -exp(A_log)
            store!("$la.A_log", log.(-(p1 === nothing ? A : A[p1])))
            dt = vec_("$b.ssm_dt.bias")
            store!("$la.dt_bias", p1 === nothing ? dt : dt[p1])
            cw = permutedims(Luminal.gguf_tensor(f, "$b.ssm_conv1d.weight"), (2, 1))   # (C, K)
            cw = vcat(cw[1:qk, :], rows(cw[qk+1:end, :], pv))
            store!("$la.conv1d.weight", Base.reshape(cw, size(cw, 1), 1, size(cw, 2)))
            store!("$la.norm.weight", vec_("$b.ssm_norm.weight"))              # plain weight, not 1 + w
            op = mat("$b.ssm_out.weight")
            store!("$la.out_proj.weight", pvd === nothing ? op : (o2 = op[:, pvd]; Luminal._free_now!(op); o2))
        else
            sa = "$h.self_attn"
            for (g, hf) in (("attn_q", "q_proj"), ("attn_k", "k_proj"), ("attn_v", "v_proj"), ("attn_output", "o_proj"))
                store!("$sa.$hf.weight", mat("$b.$g.weight"))
            end
            store!("$sa.q_norm.weight", vec_("$b.attn_q_norm.weight") .- 1f0)
            store!("$sa.k_norm.weight", vec_("$b.attn_k_norm.weight") .- 1f0)
        end
    end
    Luminal.reclaim!(device)
    return W
end


# ── The MTP (multi-token prediction) head ─────────────────────────────────────

"""
    Qwen35MTP(graph, reg; qwen35_config(...)...)

The one-layer multi-token-prediction head some Qwen3.5 / Qwen3.6 GGUFs carry (`blk.N.nextn.*`, N = the
number of base layers). Given the *next* token and the base model's final residual stream (before its
output norm) at the position before it, it predicts the token after that one:

    x = eh_proj([enorm(embed(token)); hnorm(hidden)])       # (2 hidden) -> hidden
    x = x + attn(norm(x)); x = x + mlp(norm(x))              # a full gated-attention layer, own KV cache
    logits = lm_head(shared_head_norm(x))                    # the base model's embedding and lm_head

Used for speculative decoding (`generate_ids_speculative`): drafts one token that the base model then
verifies two-at-a-time. Weights are registered under `mtp.*`; the embedding and head under the base names.
"""
struct Qwen35MTP
    embedding::Embedding
    enorm::LayerNorm
    hnorm::LayerNorm
    eh_proj::Linear
    block::Qwen35Block
    head_norm::LayerNorm
    head::Linear
    rope_base::Float32
    config::NamedTuple
end

function Qwen35MTP(graph::Luminal.Graph, reg=nothing; vocab_size, hidden, n_heads, n_kv_heads, head_dim,
                   intermediate, rope_base=1f7, rotary_dim, rms_eps=1f-6, prefix="model.language_model",
                   draft_vocab::Int=0, kwargs...)
    emb = Luminal.tensor(graph, [vocab_size, hidden])
    reg !== nothing && register_weight!(reg, "$(prefix).embed_tokens.weight", emb)
    # draft_vocab > 0: the head scores only the first draft_vocab token ids (the BPE's most frequent), which
    # cuts the dominant weight read of a draft step; the base model still verifies every draft.
    head = draft_vocab > 0 ? _linear(hidden, draft_vocab, graph, reg, "mtp.draft_head"; bias=false) :
                             _linear(hidden, vocab_size, graph, reg, "lm_head"; bias=false)
    p = "mtp.block"
    block = Qwen35Block(:full,
                        qwen35_attention(hidden, graph, reg, "$(p).self_attn"; n_heads=n_heads, n_kv_heads=n_kv_heads,
                                         head_dim=head_dim, rotary_dim=rotary_dim, rope_theta=rope_base, epsilon=rms_eps),
                        _gemma_rmsnorm(hidden, graph, reg, "$(p).input_layernorm"; epsilon=rms_eps),
                        _gemma_rmsnorm(hidden, graph, reg, "$(p).post_attention_layernorm"; epsilon=rms_eps),
                        Mlp(hidden, intermediate, graph, reg, "$(p).mlp"; swiglu=true))
    config = (; vocab_size, hidden, n_heads, n_kv_heads, head_dim, intermediate, rotary_dim,
              rope_base=Float32(rope_base), rms_eps=Float32(rms_eps), prefix)
    return Qwen35MTP(Embedding(emb),
                     _gemma_rmsnorm(hidden, graph, reg, "mtp.enorm"; epsilon=rms_eps),
                     _gemma_rmsnorm(hidden, graph, reg, "mtp.hnorm"; epsilon=rms_eps),
                     _linear(2hidden, hidden, graph, reg, "mtp.eh_proj"; bias=false),
                     block,
                     _gemma_rmsnorm(hidden, graph, reg, "mtp.shared_head_norm"; epsilon=rms_eps),
                     head, Float32(rope_base), config)
end

# tokens (S, B) and the base hidden states (Hidden, S, B) -> the layer's input (Hidden, S, B)
function _mtp_input(m::Qwen35MTP, tokens::Luminal.GraphTensor, hidden::Luminal.GraphTensor)
    e = m.enorm(m.embedding(tokens))
    h = m.hnorm(hidden)
    return m.eh_proj(Luminal.concat_along(e, h, 1))
end

# Prefill over S (token, hidden) pairs: returns the layer's K and V, (D, S, KV_H, B) each, for the
# MTP cache. (The pairs are (x[i+1], H[i]) for a prompt x with base hidden states H.)
function mtp_prefill(m::Qwen35MTP, tokens::Luminal.GraphTensor, hidden::Luminal.GraphTensor)
    x = _mtp_input(m, tokens, hidden)
    b = m.block
    a, k, v = b.mixer(b.input_norm(x), 0; rope_base=m.rope_base, return_kv=true)
    return k, v
end

# One draft step: token (1, B), hidden (Hidden, 1, B), the MTP layer's cache tensors. Returns
# (logits (Vocab, 1, B), new K, new V); the attention op writes K/V into the cache at `pos`.
function mtp_decode_step(m::Qwen35MTP, tokens, hidden, step_pos, pos_tensor, pk, pv)
    logits, nk, nv, _ = mtp_decode_step_hidden(m, tokens, hidden, step_pos, pos_tensor, pk, pv)
    return logits, nk, nv
end

# The same, also returning the layer's output before the head norm (Hidden, S, B): the "previous hidden state"
# a chained second draft feeds back in.
function mtp_decode_step_hidden(m::Qwen35MTP, tokens, hidden, step_pos, pos_tensor, pk, pv)
    x = _mtp_input(m, tokens, hidden)
    x, nk, nv = _decode_block(m.block, x, step_pos, pos_tensor, pk, pv, m.rope_base)
    return m.head(m.head_norm(x)), nk, nv, x
end

"""
    load_qwen35_mtp(path, cfg; device, quantize, block=cfg.n_layers) -> Dict{String,Any}

The MTP head's weights (`blk.<block>.*` of a GGUF that has them; `block` is the base model's layer count), under the `mtp.*` names
`Qwen35MTP` registers; merge the result into the base model's weight dict.
"""
function load_qwen35_mtp(path::String, cfg; device=Luminal.get_device(), quantize=nothing, block::Int=cfg.n_layers)
    f = Luminal.read_gguf(path)
    b = "blk.$block"
    haskey(f.tensors, "$b.nextn.eh_proj.weight") || error("$path has no MTP head ($b.nextn.eh_proj.weight)")
    W = Dict{String, Any}()
    function mat(name)
        A = Luminal.to_device(Luminal.gguf_tensor(f, name), device)
        B = permutedims(A, (2, 1))
        Luminal._free_now!(A)
        return B
    end
    vec_(name) = vec(Luminal.gguf_tensor(f, name))
    function store!(key, A)
        data = A isa Array ? Luminal.to_device(Array{Float32}(A), device) : A
        data, _ = Luminal._quantize_loaded(key, data, quantize)
        W[key] = data
    end
    store!("mtp.enorm.weight", vec_("$b.nextn.enorm.weight") .- 1f0)
    store!("mtp.hnorm.weight", vec_("$b.nextn.hnorm.weight") .- 1f0)
    store!("mtp.shared_head_norm.weight", vec_("$b.nextn.shared_head_norm.weight") .- 1f0)
    store!("mtp.eh_proj.weight", mat("$b.nextn.eh_proj.weight"))
    h = "mtp.block"
    store!("$h.input_layernorm.weight", vec_("$b.attn_norm.weight") .- 1f0)
    store!("$h.post_attention_layernorm.weight", vec_("$b.post_attention_norm.weight") .- 1f0)
    for (g, hf) in (("ffn_gate", "gate_proj"), ("ffn_up", "up_proj"), ("ffn_down", "down_proj"))
        store!("$h.mlp.$hf.weight", mat("$b.$g.weight"))
    end
    for (g, hf) in (("attn_q", "q_proj"), ("attn_k", "k_proj"), ("attn_v", "v_proj"), ("attn_output", "o_proj"))
        store!("$h.self_attn.$hf.weight", mat("$b.$g.weight"))
    end
    store!("$h.self_attn.q_norm.weight", vec_("$b.attn_q_norm.weight") .- 1f0)
    store!("$h.self_attn.k_norm.weight", vec_("$b.attn_k_norm.weight") .- 1f0)
    Luminal.reclaim!(device)
    return W
end
