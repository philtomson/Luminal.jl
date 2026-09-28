module NN

using ..Luminal
import JSON3

export Linear, Conv1D, Embedding, LayerNorm, RMSNorm, Mlp, SelfAttention, TransformerBlock, Llama, Phi3, repeat_kv, llama_config

# Layer Designs
# --------------

# Linear Layer: y = xW^T + b
struct Linear
    weight::Luminal.GraphTensor
    bias::Union{Luminal.GraphTensor, Nothing}
end

function Linear(in_features::Int, out_features::Int, graph::Luminal.Graph; bias=true)
    weight = Luminal.tensor(graph, [out_features, in_features])
    b = bias ? Luminal.tensor(graph, [out_features]) : nothing
    return Linear(weight, b)
end

function (l::Linear)(x::Luminal.GraphTensor)
    # x is (In, Seq..., Batch)
    # l.weight is (Out, In)
    # out = W * x -> (Out, Seq..., Batch)
    out = Luminal.matmul(l.weight, x)
    if l.bias !== nothing
        # Expand bias (Out) to match out shape
        dims = Luminal.realized_dims(out.shape)
        b_expanded = l.bias
        for i in 2:length(dims)
            b_expanded = Luminal.expand(b_expanded, i, dims[i])
        end
        out = out + b_expanded
    end
    return out
end

# Conv1D Layer
struct Conv1D
    weight::Luminal.GraphTensor          # (ch_out, ch_in, kernel), the PyTorch layout
    bias::Union{Luminal.GraphTensor, Nothing}
    kernel::Int
    stride::Int
    padding::Int
    dilation::Int
    ch_in::Int
    ch_out::Int
end

function Conv1D(ch_in::Int, ch_out::Int, kernel::Int, graph::Luminal.Graph; stride=1, padding=0, dilation=1, bias=true)
    weight = Luminal.tensor(graph, [ch_out, ch_in, kernel])
    b = bias ? Luminal.tensor(graph, [ch_out]) : nothing
    return Conv1D(weight, b, kernel, stride, padding, dilation, ch_in, ch_out)
end

function (c::Conv1D)(x::Luminal.GraphTensor)
    # x: (ch_in, length, batch...) -> (ch_out, out_length, batch...)
    # One matmul per kernel tap: out = sum_k W[:, :, k] * x[:, t*stride + k*dilation].
    dims = Luminal.realized_dims(x.shape)
    L, rest = dims[2], dims[3:end]
    lp = L + 2 * c.padding
    lout = (lp - c.dilation * (c.kernel - 1) - 1) ÷ c.stride + 1
    # A strided tap is a slice of stride*lout columns reshaped to (C, stride, lout);
    # the last tap's slice may run past the padded input by up to stride-1 columns,
    # which are zero-padded (and never read).
    extra = max(0, c.dilation * (c.kernel - 1) + c.stride * lout - lp)
    padded = (c.padding > 0 || extra > 0) ? Luminal.pad_along(x, 2, c.padding, c.padding + extra) : x

    out = nothing
    for k in 0:(c.kernel - 1)
        off = k * c.dilation
        tap = Luminal.slice_along(padded, 2, off, off + c.stride * lout)
        if c.stride > 1
            tap = Luminal.reshape(Luminal.contiguous(tap), [c.ch_in, c.stride, lout, rest...])
            tap = Luminal.reshape(Luminal.contiguous(Luminal.slice_along(tap, 2, 0, 1)),
                                  [c.ch_in, lout, rest...])
        end
        wk = Luminal.reshape(Luminal.contiguous(Luminal.slice_along(c.weight, 3, k, k + 1)),
                             [c.ch_out, c.ch_in])
        y = Luminal.matmul(wk, tap)
        out = out === nothing ? y : out + y
    end
    return c.bias === nothing ? out : out + c.bias
end

# Embedding Layer: y = weight[indexes]
struct Embedding
    weight::Luminal.GraphTensor
end

function Embedding(vocab_size::Int, embed_dim::Int, graph::Luminal.Graph)
    weight = Luminal.tensor(graph, [vocab_size, embed_dim])
    return Embedding(weight)
end

function (e::Embedding)(x::Luminal.GraphTensor)
    # gather(weight(V, H), x(S, B)) -> (S, B, H)
    out = Luminal.gather(e.weight, x)
    # Align to (H, S, B) 
    rank = length(Luminal.realized_dims(out.shape))
    if rank == 3
        # (S, B, H) -> (H, S, B)
        return Luminal.permute(out, [3, 1, 2])
    elseif rank == 2
        # (S, H) -> (H, S)
        return Luminal.permute(out, [2, 1])
    end
    return out
end

# LayerNorm / RMSNorm
struct LayerNorm
    weight::Union{Luminal.GraphTensor, Nothing}
    bias::Union{Luminal.GraphTensor, Nothing}
    epsilon::Float32
    mean_norm::Bool
end

function LayerNorm(dim::Int, graph::Luminal.Graph; weight=true, bias=true, epsilon=1f-5, mean_norm=true)
    w = weight ? Luminal.tensor(graph, [dim]) : nothing
    b = bias ? Luminal.tensor(graph, [dim]) : nothing
    return LayerNorm(w, b, Float32(epsilon), mean_norm)
end

function RMSNorm(dim::Int, graph::Luminal.Graph; epsilon=1f-5)
    return LayerNorm(dim, graph; weight=true, bias=false, epsilon=epsilon, mean_norm=false)
end

function (ln::LayerNorm)(x::Luminal.GraphTensor)
    # x is (Hidden, Seq..., Batch)
    # normalize over dimension 1 (Hidden)
    # RMS norm with a weight and no bias (Llama, Phi-3): one fused kernel
    if !ln.mean_norm && ln.weight !== nothing && ln.bias === nothing
        return Luminal.add_op!(x.graph_ref, Luminal.RMSNormOp(ln.epsilon),
                               [(x.id, 0, x.shape), (ln.weight.id, 0, ln.weight.shape)],
                               Luminal.ShapeTracker(Luminal.realized_dims(x.shape)))
    end
    # Use layer_norm (with mean subtraction) or std_norm (RMS) based on flag
    out = ln.mean_norm ? Luminal.layer_norm(x, 1, ln.epsilon) : Luminal.std_norm(x, 1, ln.epsilon)
    
    if ln.weight !== nothing
        out = out * ln.weight
    end
    
    if ln.bias !== nothing
        out = out + ln.bias
    end
    
    return out
end

# ──────────────────────────────────────────────────────────────────────────────
# Weight Registration Helpers
# ──────────────────────────────────────────────────────────────────────────────

# Helper: create a named Linear layer with optional registration
function _linear(in_f, out_f, cx, reg, prefix; bias=true)
    l = Linear(in_f, out_f, cx; bias=bias)
    if reg !== nothing
        register_weight!(reg, "$(prefix).weight", l.weight)
        bias && register_weight!(reg, "$(prefix).bias", l.bias)
    end
    return l
end

# Helper: create a named RMSNorm with optional registration
function _rmsnorm(dim, cx, reg, prefix; epsilon=1f-5)
    ln = RMSNorm(dim, cx; epsilon=epsilon)
    if reg !== nothing
        register_weight!(reg, "$(prefix).weight", ln.weight)
    end
    return ln
end

# Helper: create a named Embedding with optional registration
function _embedding(vocab_size, dim, cx, reg, prefix)
    weight = Luminal.tensor(cx, [vocab_size, dim])
    if reg !== nothing
        register_weight!(reg, "$(prefix).weight", weight)
    end
    return Embedding(weight)
end

# Helper: create a named LayerNorm with optional registration
function _layernorm(dim, cx, reg, prefix; epsilon=1f-5)
    ln = LayerNorm(dim, cx; epsilon=epsilon)
    if reg !== nothing
        register_weight!(reg, "$(prefix).weight", ln.weight)
        ln.bias !== nothing && register_weight!(reg, "$(prefix).bias", ln.bias)
    end
    return ln
end

# Helper: create a Conv1D with optional registration
function _conv1d(ch_in, ch_out, kernel, cx, reg, prefix; stride=1, padding=0, bias=true)
    c = Conv1D(ch_in, ch_out, kernel, cx; stride=stride, padding=padding, bias=bias)
    if reg !== nothing
        register_weight!(reg, "$(prefix).weight", c.weight)
        bias && register_weight!(reg, "$(prefix).bias", c.bias)
    end
    return c
end

# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────

# Llama MLP
struct Mlp
    gate_proj::Linear
    down_proj::Linear
    up_proj::Linear
    act::Symbol          # gate activation: :silu (Llama, Qwen) or :gelu_tanh (Gemma)
end
Mlp(gate_proj::Linear, down_proj::Linear, up_proj::Linear) = Mlp(gate_proj, down_proj, up_proj, :silu)

function Mlp(hidden::Int, intermediate::Int, graph::Luminal.Graph, reg=nothing, prefix::String="mlp"; act::Symbol=:silu)
    return Mlp(
        _linear(hidden, intermediate, graph, reg, "$(prefix).gate_proj"; bias=false),
        _linear(intermediate, hidden, graph, reg, "$(prefix).down_proj"; bias=false),
        _linear(hidden, intermediate, graph, reg, "$(prefix).up_proj"; bias=false),
        act
    )
end

function (m::Mlp)(x::Luminal.GraphTensor)
    g = m.gate_proj(x)
    gate = m.act === :gelu_tanh ? Luminal.gelu(g; approximate=true) : Luminal.silu(g)
    up = m.up_proj(x)
    return m.down_proj(gate * up)
end

# RoPE (Rotary Positional Embeddings)
"""
    rope_inv_freqs(head_dim, base, scaling) -> Vector{Float32}

RoPE inverse frequencies `base^(-2i/head_dim)`, i = 0..head_dim/2-1, adjusted by
a Hugging Face `rope_scaling`:
- `(type=:linear, factor)`: all divided by `factor` (position interpolation;
  Gemma3's global layers);
- `(type=:llama3, factor, low_freq_factor, high_freq_factor, original_max_position_embeddings)`:
  wavelengths shorter than `original / high_freq_factor` are kept, longer than
  `original / low_freq_factor` divided by `factor`, and blended linearly in
  between (Llama-3.1 / 3.2).
Computed in Float64, as HF does in Float32 up to rounding.
"""
function rope_inv_freqs(head_dim::Int, base, scaling=nothing)
    inv = [Float64(base)^(-2(i - 1) / head_dim) for i in 1:head_dim ÷ 2]
    scaling === nothing && return Float32.(inv)
    if scaling.type === :linear
        inv ./= scaling.factor
    elseif scaling.type === :llama3
        f, lo, hi, orig = scaling.factor, scaling.low_freq_factor, scaling.high_freq_factor,
                          scaling.original_max_position_embeddings
        lo_wav, hi_wav = orig / lo, orig / hi
        inv = map(inv) do w
            wav = 2π / w
            wav < hi_wav && return w
            wav > lo_wav && return w / f
            smooth = (orig / wav - lo) / (hi - lo)
            return (1 - smooth) * w / f + smooth * w
        end
    else
        error("unsupported rope scaling $(scaling.type)")
    end
    return Float32.(inv)
end

function apply_rotary_embeddings(input::Luminal.GraphTensor, prev_seq; base=10000.0f0, scaling=nothing)
    # input: D, S, H, B
    dims = Luminal.realized_dims(input.shape)
    head_dim, seq, n_heads, batch = dims[1], dims[2], dims[3], dims[4]
    
    graph = input.graph_ref
    
    # Get freqs
    half_dim = div(head_dim, 2)
    inv_freqs = if scaling === nothing
        freqs = Luminal.arange(graph, half_dim) * 2.0f0 / Float32(head_dim)
        # inv_freqs = 1.0 / base^(2i/d)
        # Using exp2(-x) to avoid overflow in intermediate exp2(x) when base=500k
        Luminal.exp2(-freqs * log2(Float32(base)))
    else
        # scaled frequencies: a constant table, computed on the host
        Luminal.add_op!(graph, Luminal.Constant(rope_inv_freqs(head_dim, base, scaling)),
                        Tuple{Int, Int, Luminal.ShapeTracker}[], Luminal.ShapeTracker([half_dim]))
    end
    
    if prev_seq isa Luminal.GraphTensor && batch isa Int && batch > 1 &&
       Luminal.realized_dims(prev_seq.shape) == [batch]
        # One start position per sequence (batched decode): tables (half, seq, batch)
        pos = Luminal.expand(Luminal.arange(graph, seq), 2, batch) + Luminal.expand(prev_seq, 1, seq)
        emb = Luminal.expand(Luminal.expand(inv_freqs, 2, seq), 3, batch) * Luminal.expand(pos, 1, half_dim)
        cos_b, sin_b = Luminal.cos(emb), Luminal.sin(emb)
        return Luminal.add_op!(graph, Luminal.RotaryEmbed(),
                               [(input.id, 0, input.shape), (cos_b.id, 0, cos_b.shape), (sin_b.id, 0, sin_b.shape)],
                               Luminal.ShapeTracker(collect(dims)))
    end

    pos_seq = Luminal.arange(graph, seq) # shape (seq)
    if prev_seq isa Int
        pos = pos_seq + Float32(prev_seq)
    else
        pos = pos_seq + prev_seq
    end
    
    # emb = pos @ inv_freqs
    # pos: (seq), inv_freqs: (half_dim)
    # emb: (seq, half_dim)
    emb = Luminal.matmul(Luminal.expand(pos, 2, 1), Luminal.expand(inv_freqs, 1, 1))
    
    # Tables (half_dim, seq); identical for q and k and across layers, so they are
    # computed once per graph (common-subexpression elimination)
    emb_t = Luminal.permute(emb, [2, 1])
    cos_t = Luminal.cos(emb_t)
    sin_t = Luminal.sin(emb_t)

    # Standard Llama RoPE, rotate-half form, as one kernel (see Luminal.RotaryEmbed):
    #   out_0 = x0 * cos - x1 * sin,  out_1 = x1 * cos + x0 * sin
    return Luminal.add_op!(graph, Luminal.RotaryEmbed(),
                           [(input.id, 0, input.shape), (cos_t.id, 0, cos_t.shape), (sin_t.id, 0, sin_t.shape)],
                           Luminal.ShapeTracker(collect(dims)))
end

function repeat_kv(keys::Luminal.GraphTensor, groups::Int)
    if groups == 1
        return keys
    end
    # keys: (D, S, KV_H, B)
    dims = Luminal.realized_dims(keys.shape)
    head_dim, seq, kv_heads, batch = dims[1], dims[2], dims[3], dims[4]
    
    # expand to (D, S, groups, KV_H, B)
    # This ensures that 'groups' is faster than 'KV_H' (in Julia column-major),
    # so when we reshape to (D, S, H, B), we get (k0, k0, ..., k1, k1, ...)
    expanded = Luminal.expand(keys, 3, groups)
    # reshape to (D, S, KV_H * groups, B)
    return Luminal.reshape(expanded, [head_dim, seq, kv_heads * groups, batch])
end

# SelfAttention
struct SelfAttention
    q_proj::Linear
    k_proj::Linear
    v_proj::Linear
    o_proj::Linear
    n_heads::Int
    n_kv_heads::Int
    head_dim::Int
    q_norm::Union{LayerNorm, Nothing}   # per-head RMSNorm of q and k before RoPE (Qwen3)
    k_norm::Union{LayerNorm, Nothing}
    rope_scaling::Any                   # nothing, or a rope_inv_freqs scaling NamedTuple
    rope_theta::Union{Float32, Nothing} # this layer's RoPE base, if not the model's
    scale::Float32                      # score scale, 1/sqrt(head_dim) unless the model says otherwise
    window::Int                         # sliding-window size (Gemma3 local layers); 0: full causal
    rotary_dim::Int                     # RoPE on the first rotary_dim dims of each head (Qwen3.5: a quarter)
    output_gate::Bool                   # q_proj also yields a gate per head: out * sigmoid(gate) (Qwen3.5)
end

# RoPE on `t` (D, S, H, B): all of each head, or its first rotary_dim dims (the
# rest pass through; frequencies over rotary_dim, as HF's partial rotary)
function _rope(sa::SelfAttention, t, pos, base)
    D = sa.head_dim; rd = sa.rotary_dim
    (rd == 0 || rd == D) && return apply_rotary_embeddings(t, pos; base=base, scaling=sa.rope_scaling)
    rot = apply_rotary_embeddings(Luminal.slice_along(t, 1, 0, rd), pos; base=base, scaling=sa.rope_scaling)
    return Luminal.concat_along(rot, Luminal.slice_along(t, 1, rd, D), 1)
end

# q_proj's output (as (rows, H, S, B)) split into the query and, with an output
# gate, the gate: each head's rows are [query; gate]
function _query_and_gate(sa::SelfAttention, qp, S, B)
    D, H = sa.head_dim, sa.n_heads
    sa.output_gate || return Luminal.reshape(qp, [D, H, S, B]), nothing
    r = Luminal.reshape(qp, [2D, H, S, B])
    return Luminal.slice_along(r, 1, 0, D), Luminal.slice_along(r, 1, D, 2D)
end

# attention output (H * D, S, B), gated if the layer has an output gate
_gate_output(sa::SelfAttention, out, gate, S, B) =
    gate === nothing ? out : out * Luminal.sigmoid(Luminal.reshape(gate, [sa.head_dim * sa.n_heads, S, B]))

_rope_base(sa::SelfAttention, model_base) = sa.rope_theta === nothing ? model_base : sa.rope_theta

"""
    SelfAttention(hidden, n_heads, n_kv_heads, graph, reg, prefix; head_dim, qk_norm, epsilon)

Grouped-query attention with RoPE. `head_dim` defaults to `hidden ÷ n_heads`;
a model may set it independently (Qwen3: 16 heads of 128 on a 1024 hidden), in
which case q and o map between `hidden` and `n_heads * head_dim`. `qk_norm`
adds an RMSNorm over each head of q and k (weights `q_norm`, `k_norm`, of size
`head_dim`) before RoPE.
"""
function SelfAttention(hidden::Int, n_heads::Int, n_kv_heads::Int, graph::Luminal.Graph, reg=nothing, prefix::String="self_attn";
                       head_dim::Int=div(hidden, n_heads), qk_norm::Bool=false, epsilon=1f-5,
                       rope_scaling=nothing, rope_theta=nothing, scale=1 / sqrt(head_dim), window::Int=0,
                       rotary_dim::Int=0, output_gate::Bool=false)
    return SelfAttention(
        _linear(hidden, n_heads * head_dim * (output_gate ? 2 : 1), graph, reg, "$(prefix).q_proj"; bias=false),
        _linear(hidden, n_kv_heads * head_dim, graph, reg, "$(prefix).k_proj"; bias=false),
        _linear(hidden, n_kv_heads * head_dim, graph, reg, "$(prefix).v_proj"; bias=false),
        _linear(n_heads * head_dim, hidden, graph, reg, "$(prefix).o_proj"; bias=false),
        n_heads,
        n_kv_heads,
        head_dim,
        qk_norm ? _rmsnorm(head_dim, graph, reg, "$(prefix).q_norm"; epsilon=epsilon) : nothing,
        qk_norm ? _rmsnorm(head_dim, graph, reg, "$(prefix).k_norm"; epsilon=epsilon) : nothing,
        rope_scaling,
        rope_theta === nothing ? nothing : Float32(rope_theta),
        Float32(scale),
        window,
        rotary_dim,
        output_gate,
    )
end

# q or k as (D, S, H, B), per-head normalized if the model has QK-norm (the fused
# RMSNorm normalizes dim 1, the head dim)
_qk_norm(norm::Nothing, t) = t
_qk_norm(norm::LayerNorm, t) = norm(t)

function (sa::SelfAttention)(x::Luminal.GraphTensor, prev_seq::Int; rope_base=10000.0f0, return_kv::Bool=false)
    # x: (hidden, seq, batch)
    hidden, seq, batch = Luminal.realized_dims(x.shape)
    
    # Project queries, keys, values
    # Llama weights are packed as (Heads * HeadDim, In). 
    # In Julia column-major matrix W(Out, In), HeadDim is the faster dimension.
    # W * x -> (Heads * HeadDim, Seq, Batch)
    queries, gate = _query_and_gate(sa, sa.q_proj(x), seq, batch)
    queries = Luminal.permute(queries, [1, 3, 2, 4]) # (D, S, H, B)
    
    keys = Luminal.reshape(sa.k_proj(x), [sa.head_dim, sa.n_kv_heads, seq, batch])
    keys = Luminal.permute(keys, [1, 3, 2, 4]) # (D, S, KV_H, B)
    queries = _qk_norm(sa.q_norm, queries)
    keys = _qk_norm(sa.k_norm, keys)
    
    values = Luminal.reshape(sa.v_proj(x), [sa.head_dim, sa.n_kv_heads, seq, batch])
    values = Luminal.permute(values, [1, 3, 2, 4]) # (D, S, KV_H, B)
    
    # RoPE
    base = _rope_base(sa, rope_base)
    queries = _rope(sa, queries, prev_seq, base)
    keys = _rope(sa, keys, prev_seq, base)
    
    # Save keys/values before repeatability expansion for GQA
    new_keys = keys
    new_values = values

    # Attention: (Q^T @ K) / sqrt(D)
    # GQA: Repeat KV heads to match Q heads
    if sa.n_kv_heads < sa.n_heads
        groups = div(sa.n_heads, sa.n_kv_heads)
        keys = repeat_kv(keys, groups)
        values = repeat_kv(values, groups)
    end
    
    # (S, D, H, B) @ (D, S, H, B) -> (S, S, H, B)
    # Use permute to make (S, D) the leading dims for matmul
    q_t = Luminal.permute(queries, [2, 1, 3, 4])
    weights = Luminal.matmul(q_t, keys) * sa.scale
    
    # Mask
    if seq > 1
        # causal; with a sliding window, keys more than `window - 1` behind the
        # query are masked too (weights are (S_q, S_k, H, B))
        W = sa.window
        mask = W > 0 && seq > W ?
            Luminal.iota(x.graph_ref, [seq, seq], (i, j) -> (j > i || i - j >= W) ? -9f9 : 0f0) :
            Luminal.triu(x.graph_ref, seq, 1) * -9f9
        # Expand mask (S, S) to (S, S, H, B)
        mask_expanded = Luminal.expand(Luminal.expand(mask, 3, sa.n_heads), 4, batch)
        weights = weights + mask_expanded
    end
    
    probs = Luminal.softmax(weights, 2) # Softmax over dimension 2 (columns S_k)
    
    # (S, S, H, B) @ (S, D, H, B) -> (S, D, H, B) 
    # Wait, (S, S) * (S, D) -> (S, D). Correct!
    v_t = Luminal.permute(values, [2, 1, 3, 4])
    out = Luminal.matmul(probs, v_t)
    
    # (S, D, H, B) -> (D, H, S, B) -> (H * D, S, Batch)
    out = Luminal.permute(out, [2, 3, 1, 4])
    out = Luminal.reshape(out, [sa.n_heads * sa.head_dim, seq, batch])
    out = _gate_output(sa, out, gate, seq, batch)
    
    out = sa.o_proj(out)
    
    if return_kv
        return out, new_keys, new_values
    end
    return out
end

# Transformer Block
struct TransformerBlock
    attention::SelfAttention
    attention_norm::LayerNorm
    feed_forward::Mlp
    feed_forward_norm::LayerNorm
end

function TransformerBlock(hidden::Int, n_heads::Int, n_kv_heads::Int, intermediate::Int, graph::Luminal.Graph, reg=nothing, prefix::String="block";
                          head_dim::Int=div(hidden, n_heads), qk_norm::Bool=false, epsilon=1f-5,
                          rope_scaling=nothing)
    return TransformerBlock(
        SelfAttention(hidden, n_heads, n_kv_heads, graph, reg, "$(prefix).self_attn";
                      head_dim=head_dim, qk_norm=qk_norm, epsilon=epsilon, rope_scaling=rope_scaling),
        _rmsnorm(hidden, graph, reg, "$(prefix).input_layernorm"; epsilon=epsilon),
        Mlp(hidden, intermediate, graph, reg, "$(prefix).mlp"),
        _rmsnorm(hidden, graph, reg, "$(prefix).post_attention_layernorm"; epsilon=epsilon)
    )
end

function (tb::TransformerBlock)(x::Luminal.GraphTensor, prev_seq::Int; rope_base=10000.0f0, return_kv::Bool=false)
    normed_x = tb.attention_norm(x)
    if return_kv
        attn_out, k, v = tb.attention(normed_x, prev_seq; rope_base=rope_base, return_kv=true)
        x = x + attn_out
    else
        attn_out = tb.attention(normed_x, prev_seq; rope_base=rope_base)
        x = x + attn_out
        k, v = nothing, nothing
    end
    
    normed_x = tb.feed_forward_norm(x)
    ff_out = tb.feed_forward(normed_x)
    out = x + ff_out
    return return_kv ? (out, k, v) : out
end

# Top-level Llama Model
struct Llama
    embedding::Embedding
    layers::Vector{TransformerBlock}
    norm::LayerNorm
    head::Linear
    rope_base::Float32
    config::NamedTuple      # the constructor's keywords, to rebuild the architecture
end

"""
    Llama(graph, reg; vocab_size, hidden, n_layers, n_heads, n_kv_heads, intermediate,
          rope_base, head_dim=hidden ÷ n_heads, qk_norm=false, rms_eps=1f-5,
          tie_embeddings=false, rope_scaling=nothing)

A Llama-family decoder: Llama 2/3, TinyLlama, and Qwen3 (`head_dim` set
independently, `qk_norm=true`, `rms_eps=1f-6`, `tie_embeddings=true`: the output
head reuses the embedding matrix, and there is no `lm_head.weight`). Build it
from a checkpoint with `Llama(graph, reg; llama_config(dir)...)`. `rope_scaling`
(Llama-3.1 / 3.2's `llama3`, or `linear`) adjusts the RoPE frequencies, see
`rope_inv_freqs`.
"""
function Llama(graph::Luminal.Graph, reg=nothing; 
               vocab_size=128256, 
               hidden=4096, 
               n_layers=32, 
               n_heads=32, 
               n_kv_heads=8, 
               intermediate=14336,
               rope_base=500000.0f0,
               head_dim=div(hidden, n_heads),
               qk_norm=false,
               rms_eps=1f-5,
               tie_embeddings=false,
               rope_scaling=nothing)
    config = (; vocab_size, hidden, n_layers, n_heads, n_kv_heads, intermediate, rope_base=Float32(rope_base),
              head_dim, qk_norm, rms_eps=Float32(rms_eps), tie_embeddings, rope_scaling)
    pfx = "model"
    layers = [TransformerBlock(hidden, n_heads, n_kv_heads, intermediate, graph, reg, "$(pfx).layers.$(i-1)";
                               head_dim=head_dim, qk_norm=qk_norm, epsilon=rms_eps, rope_scaling=rope_scaling)
              for i in 1:n_layers]
    
    emb_weight = Luminal.tensor(graph, [vocab_size, hidden])
    if reg !== nothing
        register_weight!(reg, "$(pfx).embed_tokens.weight", emb_weight)
    end
    # Tied (Qwen3): the head is the embedding matrix, (vocab, hidden) as a Linear
    # weight. It stays its own node, loaded from the embedding's array: the
    # embedding's Gather needs Float32, while the head can be stored in reduced
    # precision like any other matmul weight.
    head = _linear(hidden, vocab_size, graph, reg, "lm_head"; bias=false)
    tie_embeddings && reg !== nothing && tie_weight!(reg, "lm_head.weight", "$(pfx).embed_tokens.weight")
    
    return Llama(
        Embedding(emb_weight),
        layers,
        _rmsnorm(hidden, graph, reg, "$(pfx).norm"; epsilon=rms_eps),
        head,
        Float32(rope_base),
        config
    )
end

"""
    llama_config(model_dir) -> NamedTuple

`Llama` keyword arguments from a Hugging Face `config.json`, e.g.
`Llama(graph, reg; llama_config(dir)...)`. Errors on features this
implementation does not support (RoPE scaling other than `llama3` / `linear`, attention or MLP biases,
activations other than SiLU), rather than building a model that would silently
compute something else. Handles Llama 2/3, TinyLlama and Qwen3 (`model_type`
"qwen3": QK-norm, `head_dim` from the config, tied embeddings).
"""
# A Hugging Face rope_scaling entry as a rope_inv_freqs scaling; nothing for
# "default" (none), :unsupported otherwise
function _rope_scaling(rs)
    t = String(get(rs, :rope_type, get(rs, :type, "")))
    t == "linear" && return (type=:linear, factor=Float64(rs[:factor]))
    t == "llama3" && return (type=:llama3, factor=Float64(rs[:factor]), low_freq_factor=Float64(rs[:low_freq_factor]),
                             high_freq_factor=Float64(rs[:high_freq_factor]),
                             original_max_position_embeddings=Float64(rs[:original_max_position_embeddings]))
    t == "default" && return nothing
    return :unsupported
end

function llama_config(model_dir::String)
    c = JSON3.read(read(joinpath(model_dir, "config.json"), String))
    get_(k, d) = haskey(c, k) && c[k] !== nothing ? c[k] : d
    unsupported = String[]
    rs = get_(:rope_scaling, nothing)
    rope_scaling = rs === nothing ? nothing : _rope_scaling(rs)
    rope_scaling === :unsupported && push!(unsupported, "rope_scaling=$rs")
    get_(:attention_bias, false) && push!(unsupported, "attention_bias")
    get_(:mlp_bias, false) && push!(unsupported, "mlp_bias")
    get_(:hidden_act, "silu") == "silu" || push!(unsupported, "hidden_act=$(c[:hidden_act])")
    mt = String(get_(:model_type, "llama"))
    mt in ("llama", "qwen3") || push!(unsupported, "model_type=$mt")
    get_(:use_sliding_window, false) && push!(unsupported, "use_sliding_window")
    isempty(unsupported) || error("unsupported Llama config in $model_dir: " * join(unsupported, ", "))
    n_heads = Int(c[:num_attention_heads]); hidden = Int(c[:hidden_size])
    return (vocab_size=Int(c[:vocab_size]), hidden=hidden,
            n_layers=Int(c[:num_hidden_layers]), n_heads=n_heads,
            n_kv_heads=Int(get_(:num_key_value_heads, n_heads)),
            intermediate=Int(c[:intermediate_size]),
            rope_base=Float32(get_(:rope_theta, 10000)),
            head_dim=Int(get_(:head_dim, div(hidden, n_heads))),
            qk_norm=mt == "qwen3",
            rms_eps=Float32(get_(:rms_norm_eps, 1f-5)),
            tie_embeddings=Bool(get_(:tie_word_embeddings, false)),
            rope_scaling=rope_scaling)
end

function (l::Llama)(input::Luminal.GraphTensor, prev_seq::Int; return_kv::Bool=false)
    x = l.embedding(input)
    kvs = Tuple{Luminal.GraphTensor, Luminal.GraphTensor}[]
    for layer in l.layers
        if return_kv
            x, k, v = layer(x, prev_seq; rope_base=l.rope_base, return_kv=true)
            push!(kvs, (k, v))
        else
            x = layer(x, prev_seq; rope_base=l.rope_base)
        end
    end
    x = l.norm(x)
    logits = l.head(x)
    return return_kv ? (logits, kvs) : logits
end

# ──────────────────────────────────────────────────────────────────────────────
# Top-level Phi-3 Model
# ──────────────────────────────────────────────────────────────────────────────

struct Phi3
    embedding::Embedding
    layers::Vector{TransformerBlock}
    norm::LayerNorm
    head::Linear
    rope_base::Float32
end

"""
    Phi3(; vocab_size=32064, hidden=3072, n_layers=32, n_heads=32, n_kv_heads=8, intermediate=8192)

Phi-3-mini-4k-instruct model architecture.
"""
function Phi3(graph::Luminal.Graph, reg=nothing; 
               vocab_size=32064, 
               hidden=3072, 
               n_layers=32, 
               n_heads=32, 
               n_kv_heads=8, 
               intermediate=8192,
               rope_base=10000.0f0)
    
    pfx = "model"
    # Phi-3 uses similar layer naming to Llama but sometimes with slight variations.
    # We'll use Llama-style as default for now which matches most HF Phi-3 mini checkpoints.
    layers = [TransformerBlock(hidden, n_heads, n_kv_heads, intermediate, graph, reg, "$(pfx).layers.$(i-1)") for i in 1:n_layers]
    
    emb_weight = Luminal.tensor(graph, [vocab_size, hidden])
    if reg !== nothing
        register_weight!(reg, "$(pfx).embed_tokens.weight", emb_weight)
    end

    return Phi3(
        Embedding(emb_weight),
        layers,
        _rmsnorm(hidden, graph, reg, "$(pfx).norm"),
        _linear(hidden, vocab_size, graph, reg, "lm_head"; bias=false),
        rope_base
    )
end

function (p::Phi3)(input::Luminal.GraphTensor, prev_seq::Int; return_kv::Bool=false)
    x = p.embedding(input)
    kvs = Tuple{Luminal.GraphTensor, Luminal.GraphTensor}[]
    for layer in p.layers
        if return_kv
            x, k, v = layer(x, prev_seq; rope_base=p.rope_base, return_kv=true)
            push!(kvs, (k, v))
        else
            x = layer(x, prev_seq; rope_base=p.rope_base)
        end
    end
    x = p.norm(x)
    logits = p.head(x)
    return return_kv ? (logits, kvs) : logits
end


# ──────────────────────────────────────────────────────────────────────────────
# Llama KV-Cache Infrastructure
#
# Mirrors the Whisper KV-cache pattern but for decoder-only Llama/Phi-3 models.
# Usage:
#   cache = LlamaKVCacheState(model, max_seq=2048)
#   idg   = build_llama_decode_step!(model, graph, step_pos)
#   logits = llama_decode_step!(exec_fn, idg, cache, token_id)
# ──────────────────────────────────────────────────────────────────────────────

"""
    LlamaKVCacheState

Device-resident storage for past K/V tensors for one decode session.
- `self_cache[i]` = `(K, V)` arrays for layer i, shape (head_dim, max_seq, n_kv_heads, batch),
  allocated once on the decode device and updated in place one slot per step
- `positions[b]`: 0-indexed position of sequence b's next token. Sequences in a
  batch advance independently (prompts of different lengths, finished sequences).
- `step_pos`: the common position, for batch 1 (or when all sequences agree);
  assigning it sets every sequence's position.
"""
mutable struct LlamaKVCacheState{T}
    positions::Vector{Int}
    max_seq::Int
    self_cache::Vector{T}     # per layer: (K, V), or a recurrent layer's (conv, rec) states
end

function Base.getproperty(c::LlamaKVCacheState, name::Symbol)
    if name === :step_pos
        pos = getfield(c, :positions)
        all(==(pos[1]), pos) || error("sequences are at different positions; use `positions`")
        return pos[1]
    end
    return getfield(c, name)
end
function Base.setproperty!(c::LlamaKVCacheState, name::Symbol, v)
    name === :step_pos ? fill!(getfield(c, :positions), v) : setfield!(c, name, v)
    return v
end
Base.propertynames(::LlamaKVCacheState) = (fieldnames(LlamaKVCacheState)..., :step_pos)

"""
    LlamaKVCacheState(n_layers, n_kv_heads, head_dim; batch=1, max_seq=2048, device=CPUDevice())
"""
function LlamaKVCacheState(n_layers::Int, n_kv_heads::Int, head_dim::Int;
                            batch::Int=1, max_seq::Int=2048,
                            device::Luminal.AbstractDevice=Luminal.CPUDevice())
    self = [(Luminal.zero_tensor(device, Float32, head_dim, max_seq, n_kv_heads, batch),
             Luminal.zero_tensor(device, Float32, head_dim, max_seq, n_kv_heads, batch))
            for _ in 1:n_layers]
    return LlamaKVCacheState(zeros(Int, batch), max_seq, self)
end

"""
    LlamaKVCacheState(model; batch=1, max_seq=2048, device=CPUDevice())

The decode-step state buffers of `model`: a K/V cache per attention layer, and
the convolution and recurrent states of a recurrent (Gated DeltaNet) layer.
"""
function LlamaKVCacheState(model; batch::Int=1, max_seq::Int=2048,
                           device::Luminal.AbstractDevice=Luminal.CPUDevice())
    self = [Tuple(Luminal.zero_tensor(device, Float32, s...) for s in _state_shapes(l, max_seq, batch))
            for l in model.layers]
    return LlamaKVCacheState(zeros(Int, batch), max_seq, self)
end



"""
    llama_self_attn_cached(sa, x, step_pos, step_pos_tensor, past_k, past_v; rope_base=500000f0)

Single-token cached self-attention for Llama decoder-only models.
- `x` : (hidden, 1, batch)
- `step_pos_tensor` : (1,) current 0-indexed decode position, as data
- `past_k`, `past_v` : (head_dim, max_seq, n_kv_heads, batch); slots `>= pos` are ignored
Returns `(output, k_new, v_new)`, where `k_new`/`v_new` are this token's
(D, 1, KV_H, B) K/V slot; the attention op also writes them into the cache at
`step_pos` (in place), so running the graph updates the cache.

All shapes are static (independent of the position), so the step can be
captured once and replayed. The cache is scored in place under a mask
instead of being sliced, and grouped-query attention is computed per KV head
without materializing repeated K/V.
"""
function llama_self_attn_cached(sa::SelfAttention,
                                 x::Luminal.GraphTensor,
                                 step_pos::Luminal.DimType,
                                 step_pos_tensor::Luminal.GraphTensor,
                                 past_k::Luminal.GraphTensor,
                                 past_v::Luminal.GraphTensor;
                                 rope_base::Float32=500000f0)
    hidden, _, batch = Luminal.realized_dims(x.shape)
    D, KVH = sa.head_dim, sa.n_kv_heads
    G = div(sa.n_heads, KVH)              # query heads per KV head
    max_seq = Luminal.realized_dims(past_k.shape)[2]

    # Project current token: (Hidden, 1, B) -> (D, 1, H, B), then RoPE
    q, gate = _query_and_gate(sa, sa.q_proj(x), 1, batch)
    q     = Luminal.permute(q, [1, 3, 2, 4])
    k_new = Luminal.permute(Luminal.reshape(sa.k_proj(x), [D, KVH, 1, batch]), [1, 3, 2, 4])
    v_new = Luminal.permute(Luminal.reshape(sa.v_proj(x), [D, KVH, 1, batch]), [1, 3, 2, 4])
    base = _rope_base(sa, rope_base)
    q     = _rope(sa, _qk_norm(sa.q_norm, q),     step_pos_tensor, base)
    k_new = _rope(sa, _qk_norm(sa.k_norm, k_new), step_pos_tensor, base)

    # Attention over cache slots [0, pos) plus the current token, in one kernel
    # (see Luminal.DecodeAttention): scores, softmax and the weighted sum of V,
    # grouped-query heads reading their KV head directly. (D, H, B) is the
    # head-major order the output projection expects.
    ins = [(t.id, 0, t.shape) for t in (q, past_k, past_v, k_new, v_new, step_pos_tensor)]
    # The op also writes this token's K/V into the cache slot (write_cache=true).
    out = Luminal.add_op!(x.graph_ref, Luminal.DecodeAttention(sa.scale, true, sa.window), ins,
                          Luminal.ShapeTracker([D, sa.n_heads, batch]))
    out = Luminal.reshape(out, [D * sa.n_heads, 1, batch])
    out = _gate_output(sa, out, gate, 1, batch)

    return sa.o_proj(out), k_new, v_new
end


"""
    LlamaDecodeGraph

Node IDs for driving one step of the incremental Llama decode graph.
"""
struct LlamaDecodeGraph
    token_input_id::Int
    pos_input_id::Int
    self_k_ids::Vector{Int}
    self_v_ids::Vector{Int}
    logits_id::Int
    new_self_k_ids::Vector{Int}   # this step's K slot per layer, (D, 1, KV_H, B)
    new_self_v_ids::Vector{Int}   # this step's V slot per layer, (D, 1, KV_H, B)
    step_pos::Luminal.DimType
    lens_input_id::Int            # (B,) tokens each sequence consumes (recurrent layers); 0 if none
end


"""
    build_llama_decode_step!(model, graph, step_pos; max_seq, batch, rope_base)

Build the single-step incremental decode graph for a Llama-style model.
`model` must be a `Llama` or `Phi3` instance whose weight tensors are already
registered (or randomly initialized) in `graph`.

Returns a `LlamaDecodeGraph`.
"""
function build_llama_decode_step!(model,
                                   graph::Luminal.Graph,
                                   step_pos::Luminal.DimType;
                                   max_seq::Int=2048,
                                   batch::Int=1,
                                   rope_base::Float32=500000f0)
    # ── Inputs ────────────────────────────────────────────────────────────────
    token_in = Luminal.tensor(graph, [1, batch])
    pos_tensor = Luminal.tensor(graph, [batch])   # each sequence's position
    lens = _needs_lens(model) ? Luminal.tensor(graph, [batch]) : nothing

    # per layer: the K/V cache, or a recurrent layer's two state buffers
    shapes = [_state_shapes(l, max_seq, batch) for l in model.layers]
    self_k_tensors = [Luminal.tensor(graph, s[1]) for s in shapes]
    self_v_tensors = [Luminal.tensor(graph, s[2]) for s in shapes]

    # ── Embedding ─────────────────────────────────────────────────────────────
    x = _embed(model, token_in)  # (hidden, 1, batch)

    # ── Decoder layers ────────────────────────────────────────────────────────
    new_k_tensors = Luminal.GraphTensor[]
    new_v_tensors = Luminal.GraphTensor[]

    for (i, layer) in enumerate(model.layers)
        x, nk, nv = _decode_block(layer, x, step_pos, pos_tensor, self_k_tensors[i], self_v_tensors[i], rope_base;
                                  lens=lens)
        push!(new_k_tensors, nk)
        push!(new_v_tensors, nv)
    end

    # ── Head ─────────────────────────────────────────────────────────────────
    out    = model.norm(x)   # (Hidden, 1, Batch)
    logits = model.head(out) # (Vocab, 1, Batch)

    return LlamaDecodeGraph(
        token_in.id,
        pos_tensor.id,
        [t.id for t in self_k_tensors],
        [t.id for t in self_v_tensors],
        logits.id,
        [t.id for t in new_k_tensors],
        [t.id for t in new_v_tensors],
        step_pos,
        lens === nothing ? 0 : lens.id)
end


# Model / block hooks of the decode step: the embedding, the per-layer state
# buffers (a KV cache, or a recurrent layer's states), whether the step needs a
# `lens` input, and one decoder layer.
_embed(model, tokens) = model.embedding(tokens)
_needs_lens(model) = false
_is_state_layer(layer) = false
_state_shapes(layer, max_seq, batch) =
    (a = layer.attention; ([a.head_dim, max_seq, a.n_kv_heads, batch], [a.head_dim, max_seq, a.n_kv_heads, batch]))
function _decode_block(layer::TransformerBlock, x, step_pos, pos_tensor, pk, pv, rope_base; lens=nothing)
    attn_out, nk, nv = llama_self_attn_cached(layer.attention, layer.attention_norm(x), step_pos, pos_tensor,
                                              pk, pv; rope_base=rope_base)
    x = x + attn_out
    x = x + layer.feed_forward(layer.feed_forward_norm(x))
    return x, nk, nv
end

"""
    llama_decode_step!(exec_fn, idg, cache, tokens; device=get_device())

Execute one cached decode step.
- `exec_fn`: compiled decode graph
- `idg`: its LlamaDecodeGraph
- `cache`: LlamaKVCacheState (mutated in place)
- `tokens`: one 0-indexed token per sequence (a scalar for batch 1)
- `advance`: which sequences move to the next position (default: all). A
  sequence that does not advance is recomputed at the same position next step,
  e.g. once it has finished.

Each sequence's token is processed at its own `cache.positions[b]`. Returns the
(vocab, 1, batch) logits.
"""
function llama_decode_step!(exec_fn,
                             idg::LlamaDecodeGraph,
                             cache::LlamaKVCacheState,
                             tokens::AbstractVector{<:Integer};
                             advance::AbstractVector{Bool}=trues(length(tokens)),
                             sym_vals::Dict{Symbol, Int}=Dict{Symbol, Int}(),
                             device=Luminal.get_device())
    pos = cache.positions
    length(tokens) == length(pos) || error("expected $(length(pos)) tokens, got $(length(tokens))")
    maximum(pos) < cache.max_seq || error("KV cache full ($(cache.max_seq) positions)")
    inputs = Dict{Int, Any}()
    inputs[idg.token_input_id] = Float32.(Base.reshape(tokens, 1, :))   # (1, B)
    inputs[idg.pos_input_id] = Float32.(pos)                            # (B,)
    idg.lens_input_id == 0 || (inputs[idg.lens_input_id] = Float32.(advance))   # recurrent layers: 0 holds

    # The cache arrays already live on `device`; the compiled graph aliases them.
    for (i, (k_id, v_id)) in enumerate(zip(idg.self_k_ids, idg.self_v_ids))
        inputs[k_id] = cache.self_cache[i][1]
        inputs[v_id] = cache.self_cache[i][2]
    end

    results = exec_fn(inputs; sym_vals=sym_vals, device=device)
    logits  = results[idg.logits_id]

    # The step graph's DecodeAttention nodes have already written each sequence's
    # K/V into its cache slot (in place, inside the captured graph).
    pos .+= advance
    return logits
end

llama_decode_step!(exec_fn, idg::LlamaDecodeGraph, cache::LlamaKVCacheState, token_id::Integer; kw...) =
    llama_decode_step!(exec_fn, idg, cache, [token_id]; kw...)

export LlamaKVCacheState, LlamaDecodeGraph, build_llama_decode_step!, llama_decode_step!, llama_self_attn_cached

include("Gemma3.jl")
export Gemma3, Gemma3Block, gemma3_config, model_template

include("Qwen35.jl")
export GatedDeltaNet, deltanet_state_shapes, qwen35_attention, Qwen35, Qwen35Block, qwen35_config

include("Whisper.jl")
export WhisperAttention, WhisperSelfAttention, WhisperCrossAttention, EncoderTransformerBlock, AudioEncoder,
       DecoderTransformerBlock, TextDecoder,
       # Audio preprocessing
       mel_filters, get_mel_filters, log_mel_spectrogram, stft_power,
       pad_or_trim, load_audio_file,
       SAMPLE_RATE, N_FFT, HOP_LENGTH, N_SAMPLES, N_FRAMES,
       # KV Caching
       KVCacheState, IncrementalDecodeGraph, build_decode_step!, decode_step!,
       whisper_self_attn_cached, whisper_cross_attn_cached

include("WhisperTokenizer.jl")
export WhisperTokenizer, encode, decode, sot_sequence, LANGUAGES

end # module NN
