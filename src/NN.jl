module NN

using ..Luminal

export Linear, Conv1D, Embedding, LayerNorm, RMSNorm, Mlp, SelfAttention, TransformerBlock, Llama, Phi3, repeat_kv

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
    # x: (batch, in_features)
    # weight: (out_features, in_features)
    # out: (batch, out_features)
    out = Luminal.matmul(x, Luminal.permute(l.weight, [2, 1]))
    if l.bias !== nothing
        # Expand bias to match batch dimension
        # out.shape: (batch, out_features)
        # l.bias: (out_features)
        b_expanded = Luminal.expand(l.bias, 1, Luminal.realized_dims(out.shape)[1])
        out = out + b_expanded
    end
    return out
end

# Conv1D Layer
struct Conv1D
    weight::Luminal.GraphTensor
    bias::Union{Luminal.GraphTensor, Nothing}
    kernel::Int
    stride::Int
    padding::Int
    dilation::Int
    ch_in::Int
    ch_out::Int
end

function Conv1D(ch_in::Int, ch_out::Int, kernel::Int, graph::Luminal.Graph; stride=1, padding=0, dilation=1, bias=true)
    weight = Luminal.tensor(graph, [ch_out, ch_in * kernel])
    b = bias ? Luminal.tensor(graph, [ch_out]) : nothing
    return Conv1D(weight, b, kernel, stride, padding, dilation, ch_in, ch_out)
end

function (c::Conv1D)(x::Luminal.GraphTensor)
    # x: (batch..., channels, length)
    rank = length(Luminal.realized_dims(x.shape))
    padded = x
    if c.padding > 0
        padded = Luminal.pad_along(x, rank, c.padding, c.padding)
    end
    
    # unfold shape: [batch..., channels, out_length, kernel]
    unfolded = Luminal.unfold(padded, [c.kernel], [c.stride], [c.dilation])
    
    # permute unfolded to [batch..., out_length, channels, kernel]
    axes = collect(1:(rank+1))
    chan_idx = rank - 1
    out_len_idx = rank
    axes[chan_idx] = out_len_idx
    axes[out_len_idx] = chan_idx
    unfolded_permuted = Luminal.permute(unfolded, axes)
    
    # reshape to [batch..., out_length, channels * kernel]
    dims = Luminal.realized_dims(unfolded_permuted.shape)
    new_shape = [dims[1:end-2]..., dims[end-1] * dims[end]]
    reshaped_for_matmul = Luminal.reshape(unfolded_permuted, new_shape)
    
    # matmul with weight^T ([ch_in * kernel, ch_out]) => [batch..., out_length, ch_out]
    out = Luminal.matmul(reshaped_for_matmul, Luminal.permute(c.weight, [2, 1]))
    
    # permute back to [batch..., ch_out, out_length]
    out_axes = collect(1:rank)
    out_chan_idx = rank - 1
    out_len_idx = rank
    out_axes[out_chan_idx] = out_len_idx
    out_axes[out_len_idx] = out_chan_idx
    out_final = Luminal.permute(out, out_axes)
    
    if c.bias !== nothing
        # bias is (ch_out,)
        b_expanded = c.bias
        out_dims = Luminal.realized_dims(out_final.shape)
        # expand over batch dimensions
        for i in 1:(rank-2)
            b_expanded = Luminal.expand(b_expanded, i, out_dims[i])
        end
        # expand over out_length
        b_expanded = Luminal.expand(b_expanded, rank, out_dims[rank])
        out_final = out_final + b_expanded
    end
    
    return out_final
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
    return Luminal.gather(e.weight, x)
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
    # x: (batch, ..., dim)
    dims = Luminal.realized_dims(x.shape)
    axis = length(dims)
    
    out = x
    if ln.mean_norm
        out = Luminal.mean_norm(out, axis)
    end
    out = Luminal.std_norm(out, axis, ln.epsilon)
    
    if ln.weight !== nothing
        # Expand weight to match input shape except the last dimension.
        # If x is (B, ..., D), weight is (D). We need to expand weight to (B, ..., D).
        # This is a bit tricky with my current expand. 
        # For now, let's just do a simple expansion for common cases.
        w_dims = Luminal.realized_dims(ln.weight.shape)
        target_dims = Luminal.realized_dims(out.shape)
        w_expanded = ln.weight
        for i in 1:(length(target_dims)-1)
            w_expanded = Luminal.expand(w_expanded, i, target_dims[i])
        end
        out = out * w_expanded
    end
    
    if ln.bias !== nothing
        b_expanded = ln.bias
        target_dims = Luminal.realized_dims(out.shape)
        for i in 1:(length(target_dims)-1)
            b_expanded = Luminal.expand(b_expanded, i, target_dims[i])
        end
        out = out + b_expanded
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
end

function Mlp(hidden::Int, intermediate::Int, graph::Luminal.Graph, reg=nothing, prefix::String="mlp")
    return Mlp(
        _linear(hidden, intermediate, graph, reg, "$(prefix).gate_proj"; bias=false),
        _linear(intermediate, hidden, graph, reg, "$(prefix).down_proj"; bias=false),
        _linear(hidden, intermediate, graph, reg, "$(prefix).up_proj"; bias=false)
    )
end

function (m::Mlp)(x::Luminal.GraphTensor)
    gate = Luminal.silu(m.gate_proj(x))
    up = m.up_proj(x)
    return m.down_proj(gate * up)
end

# RoPE (Rotary Positional Embeddings)
function apply_rotary_embeddings(input::Luminal.GraphTensor, prev_seq::Int; base=10000.0f0)
    # input: batch, n_heads, seq, head_dim
    dims = Luminal.realized_dims(input.shape)
    batch, n_heads, seq, head_dim = dims[1], dims[2], dims[3], dims[4]
    
    graph = input.graph_ref
    
    # Get freqs
    half_dim = div(head_dim, 2)
    freqs = Luminal.arange(graph, half_dim) * 2.0f0 / Float32(head_dim)
    inv_freqs = Luminal.reciprocal(Luminal.exp2(freqs * log2(Float32(base))))
    
    pos = Luminal.arange(graph, seq) + Float32(prev_seq)
    
    # emb = pos @ inv_freqs
    # pos: (seq), inv_freqs: (half_dim)
    # emb: (seq, half_dim)
    emb = Luminal.matmul(Luminal.expand(pos, 2, 1), Luminal.expand(inv_freqs, 1, 1))
    
    # Split input into evens and odds along last dimension
    # input is (B, H, S, D) -> reshape to (B, H, S, D/2, 2)
    split = Luminal.reshape(input, [batch, n_heads, seq, half_dim, 2])
    x0 = Luminal.slice_along(split, 5, 0, 1) # slice last dim, start 0, stop 1
    x1 = Luminal.slice_along(split, 5, 1, 2) # slice last dim, start 1, stop 2
    
    # Apply sin/cos
    # emb is (seq, half_dim), needs to be expanded to (batch, n_heads, seq, half_dim)
    # or just broadcasted.
    # In Julia, we'll expand it manually for now to be safe.
    emb_expanded = Luminal.expand(Luminal.expand(emb, 1, n_heads), 1, batch)
    
    sin_emb = Luminal.sin(emb_expanded)
    cos_emb = Luminal.cos(emb_expanded)
    
    # Reshape x0, x1 to (B, H, S, D/2)
    # slice returned non-contiguous tensors, so we must make them contiguous
    x0 = Luminal.reshape(Luminal.contiguous(x0), [batch, n_heads, seq, half_dim])
    x1 = Luminal.reshape(Luminal.contiguous(x1), [batch, n_heads, seq, half_dim])
    
    x0_out = x0 * cos_emb - x1 * sin_emb
    x1_out = x0 * sin_emb + x1 * cos_emb
    
    # Combine back: concat along last dimension
    # x0_out, x1_out are (B, H, S, D/2)
    # Result should be (B, H, S, D/2, 2) or (B, H, S, D)
    # Luminal Rust uses concat_along(4) which is the head_dim/2 dimension.
    # Wait, in Rust it's (B, H, S, D/2, 2) and they concat along the '2' dimension.
    
    res = Luminal.concat_along(Luminal.expand(x0_out, 5, 1), Luminal.expand(x1_out, 5, 1), 5)
    return Luminal.reshape(res, [batch, n_heads, seq, head_dim])
end

function repeat_kv(keys::Luminal.GraphTensor, groups::Int)
    if groups == 1
        return keys
    end
    # keys: (B, KV_H, S, D)
    dims = Luminal.realized_dims(keys.shape)
    batch, kv_heads, seq, head_dim = dims[1], dims[2], dims[3], dims[4]
    
    # expand to (B, KV_H, groups, S, D)
    expanded = Luminal.expand(keys, 3, groups)
    # reshape to (B, KV_H * groups, S, D)
    return Luminal.reshape(expanded, [batch, kv_heads * groups, seq, head_dim])
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
end

function SelfAttention(hidden::Int, n_heads::Int, n_kv_heads::Int, graph::Luminal.Graph, reg=nothing, prefix::String="self_attn")
    head_dim = div(hidden, n_heads)
    return SelfAttention(
        _linear(hidden, hidden, graph, reg, "$(prefix).q_proj"; bias=false),
        _linear(hidden, n_kv_heads * head_dim, graph, reg, "$(prefix).k_proj"; bias=false),
        _linear(hidden, n_kv_heads * head_dim, graph, reg, "$(prefix).v_proj"; bias=false),
        _linear(hidden, hidden, graph, reg, "$(prefix).o_proj"; bias=false),
        n_heads,
        n_kv_heads,
        head_dim
    )
end

function (sa::SelfAttention)(x::Luminal.GraphTensor, prev_seq::Int; rope_base=10000.0f0)
    # x: (batch, seq, hidden)
    batch, seq, hidden = Luminal.realized_dims(x.shape)
    
    queries = Luminal.reshape(sa.q_proj(x), [batch, seq, sa.n_heads, sa.head_dim])
    queries = Luminal.permute(queries, [1, 3, 2, 4]) # (B, H, S, D)
    
    keys = Luminal.reshape(sa.k_proj(x), [batch, seq, sa.n_kv_heads, sa.head_dim])
    keys = Luminal.permute(keys, [1, 3, 2, 4]) # (B, KV_H, S, D)
    
    values = Luminal.reshape(sa.v_proj(x), [batch, seq, sa.n_kv_heads, sa.head_dim])
    values = Luminal.permute(values, [1, 3, 2, 4]) # (B, KV_H, S, D)
    
    # RoPE
    queries = apply_rotary_embeddings(queries, prev_seq; base=rope_base)
    keys = apply_rotary_embeddings(keys, prev_seq; base=rope_base)
    
    # Attention: (Q @ K.T) / sqrt(D)
    # GQA: Repeat KV heads to match Q heads
    if sa.n_kv_heads < sa.n_heads
        groups = div(sa.n_heads, sa.n_kv_heads)
        keys = repeat_kv(keys, groups)
        values = repeat_kv(values, groups)
    end
    
    # (B, H, S, D) @ (B, H, D, S) -> (B, H, S, S)
    weights = Luminal.matmul(queries, Luminal.permute(keys, [1, 2, 4, 3])) * (1.0f0 / sqrt(Float32(sa.head_dim)))
    
    # Mask
    if seq > 1
        mask = Luminal.triu(x.graph_ref, seq, 1) * -1f9
        # Expand mask to (B, H, S, S)
        mask_expanded = Luminal.expand(Luminal.expand(mask, 1, sa.n_heads), 1, batch)
        weights = weights + mask_expanded
    end
    
    probs = Luminal.softmax(weights, 4)
    
    # (B, H, S, S) @ (B, H, S, D) -> (B, H, S, D)
    out = Luminal.matmul(probs, values)
    out = Luminal.permute(out, [1, 3, 2, 4]) # (B, S, H, D)
    out = Luminal.reshape(out, [batch, seq, hidden])
    
    return sa.o_proj(out)
end

# Transformer Block
struct TransformerBlock
    attention::SelfAttention
    attention_norm::LayerNorm
    feed_forward::Mlp
    feed_forward_norm::LayerNorm
end

function TransformerBlock(hidden::Int, n_heads::Int, n_kv_heads::Int, intermediate::Int, graph::Luminal.Graph, reg=nothing, prefix::String="block")
    return TransformerBlock(
        SelfAttention(hidden, n_heads, n_kv_heads, graph, reg, "$(prefix).self_attn"),
        _rmsnorm(hidden, graph, reg, "$(prefix).input_layernorm"),
        Mlp(hidden, intermediate, graph, reg, "$(prefix).mlp"),
        _rmsnorm(hidden, graph, reg, "$(prefix).post_attention_layernorm")
    )
end

function (tb::TransformerBlock)(x::Luminal.GraphTensor, prev_seq::Int; rope_base=10000.0f0)
    normed_x = tb.attention_norm(x)
    attn_out = tb.attention(normed_x, prev_seq; rope_base=rope_base)
    x = x + attn_out
    
    normed_x = tb.feed_forward_norm(x)
    ff_out = tb.feed_forward(normed_x)
    return x + ff_out
end

# Top-level Llama Model
struct Llama
    embedding::Embedding
    layers::Vector{TransformerBlock}
    norm::LayerNorm
    head::Linear
end

function Llama(graph::Luminal.Graph, reg=nothing; 
               vocab_size=128256, 
               hidden=4096, 
               n_layers=32, 
               n_heads=32, 
               n_kv_heads=8, 
               intermediate=14336)
    
    pfx = "model"
    layers = [TransformerBlock(hidden, n_heads, n_kv_heads, intermediate, graph, reg, "$(pfx).layers.$(i-1)") for i in 1:n_layers]
    
    emb_weight = Luminal.tensor(graph, [vocab_size, hidden])
    if reg !== nothing
        register_weight!(reg, "$(pfx).embed_tokens.weight", emb_weight)
    end
    
    return Llama(
        Embedding(emb_weight),
        layers,
        _rmsnorm(hidden, graph, reg, "$(pfx).norm"),
        _linear(hidden, vocab_size, graph, reg, "lm_head"; bias=false)
    )
end

function (l::Llama)(input::Luminal.GraphTensor, prev_seq::Int)
    x = l.embedding(input)
    for layer in l.layers
        x = layer(x, prev_seq; rope_base=500000.0f0)
    end
    x = l.norm(x)
    return l.head(x)
end

# ──────────────────────────────────────────────────────────────────────────────
# Top-level Phi-3 Model
# ──────────────────────────────────────────────────────────────────────────────

struct Phi3
    embedding::Embedding
    layers::Vector{TransformerBlock}
    norm::LayerNorm
    head::Linear
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
               intermediate=8192)
    
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
        _linear(hidden, vocab_size, graph, reg, "lm_head"; bias=false)
    )
end

function (l::Phi3)(input::Luminal.GraphTensor, prev_seq::Int)
    x = l.embedding(input)
    for layer in l.layers
        # Phi-3 mini uses RoPE base 10000.0
        x = layer(x, prev_seq; rope_base=10000.0f0)
    end
    x = l.norm(x)
    return l.head(x)
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

Host-side storage for past K/V tensors for one decode session.
- `self_cache[i]` = `(K, V)` arrays for layer i, shape (batch, n_kv_heads, max_seq, head_dim)
- `step_pos`: current 0-indexed decode position
"""
mutable struct LlamaKVCacheState
    step_pos::Int
    max_seq::Int
    self_cache::Vector{Tuple{Array{Float32,4}, Array{Float32,4}}}
end

"""
    LlamaKVCacheState(n_layers, n_kv_heads, head_dim; batch=1, max_seq=2048)
"""
function LlamaKVCacheState(n_layers::Int, n_kv_heads::Int, head_dim::Int;
                            batch::Int=1, max_seq::Int=2048)
    self = [(zeros(Float32, batch, n_kv_heads, max_seq, head_dim),
             zeros(Float32, batch, n_kv_heads, max_seq, head_dim))
            for _ in 1:n_layers]
    return LlamaKVCacheState(0, max_seq, self)
end


"""
    llama_self_attn_cached(sa, x, step_pos, past_k, past_v; rope_base=500000f0)

Single-token cached self-attention for Llama decoder-only models.
- `x` : (batch, 1, hidden)
- `step_pos` : current 0-indexed decode position
- `past_k`, `past_v` : (batch, n_kv_heads, max_seq, head_dim)
Returns `(output, new_k, new_v)`.
"""
function llama_self_attn_cached(sa::SelfAttention,
                                 x::Luminal.GraphTensor,
                                 step_pos::Int,
                                 past_k::Luminal.GraphTensor,
                                 past_v::Luminal.GraphTensor;
                                 rope_base::Float32=500000f0)
    batch, _, hidden = Luminal.realized_dims(x.shape)

    # Project current token: (batch, 1, hidden) → (batch, H/KV_H, 1, head_dim)
    q_raw = Luminal.reshape(sa.q_proj(x), [batch, 1, sa.n_heads, sa.head_dim])
    q     = Luminal.permute(q_raw, [1, 3, 2, 4])   # (B, H, 1, D)

    k_raw = Luminal.reshape(sa.k_proj(x), [batch, 1, sa.n_kv_heads, sa.head_dim])
    k_new = Luminal.contiguous(Luminal.permute(k_raw, [1, 3, 2, 4]))  # (B, KV_H, 1, D)

    v_raw = Luminal.reshape(sa.v_proj(x), [batch, 1, sa.n_kv_heads, sa.head_dim])
    v_new = Luminal.permute(v_raw, [1, 3, 2, 4])   # (B, KV_H, 1, D)

    # Apply RoPE to q and k_new at position step_pos
    q     = apply_rotary_embeddings(q,     step_pos; base=rope_base)
    k_new = apply_rotary_embeddings(k_new, step_pos; base=rope_base)

    # Scatter new k/v into the cache (same helper as Whisper KV scatter)
    max_seq = Luminal.realized_dims(past_k.shape)[3]
    suffix_len = max_seq - step_pos - 1

    function _kv_scatter_llama(past, slot)
        if step_pos == 0
            return Luminal.concat_along(slot,
                       Luminal.slice_along(past, 3, 1, max_seq), 3)
        elseif suffix_len == 0
            return Luminal.concat_along(
                       Luminal.slice_along(past, 3, 0, step_pos), slot, 3)
        else
            return Luminal.concat_along(
                       Luminal.concat_along(
                           Luminal.slice_along(past, 3, 0, step_pos), slot, 3),
                       Luminal.slice_along(past, 3, step_pos + 1, max_seq), 3)
        end
    end

    new_k = _kv_scatter_llama(past_k, k_new)
    new_v = _kv_scatter_llama(past_v, v_new)

    # Attend over [0 : step_pos + 1]
    k_ctx   = Luminal.slice_along(new_k, 3, 0, step_pos + 1)   # (B, KV_H, ctx, D)
    v_ctx   = Luminal.slice_along(new_v, 3, 0, step_pos + 1)   # (B, KV_H, ctx, D)

    # GQA: repeat KV heads to match Q heads
    if sa.n_kv_heads < sa.n_heads
        groups = div(sa.n_heads, sa.n_kv_heads)
        k_ctx = repeat_kv(k_ctx, groups)
        v_ctx = repeat_kv(v_ctx, groups)
    end

    k_ctx_t = Luminal.contiguous(Luminal.permute(k_ctx, [1, 2, 4, 3]))   # (B, H, D, ctx)
    weights  = Luminal.matmul(q, k_ctx_t) * (1.0f0 / sqrt(Float32(sa.head_dim)))
    probs    = Luminal.softmax(weights, 4)
    out      = Luminal.matmul(probs, v_ctx)   # (B, H, 1, D)
    out      = Luminal.reshape(Luminal.permute(out, [1, 3, 2, 4]), [batch, 1, hidden])

    return sa.o_proj(out), new_k, new_v
end


"""
    LlamaDecodeGraph

Node IDs for driving one step of the incremental Llama decode graph.
"""
struct LlamaDecodeGraph
    token_input_id::Int
    self_k_ids::Vector{Int}
    self_v_ids::Vector{Int}
    logits_id::Int
    new_self_k_ids::Vector{Int}
    new_self_v_ids::Vector{Int}
    step_pos::Int
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
                                   step_pos::Int;
                                   max_seq::Int=2048,
                                   batch::Int=1,
                                   rope_base::Float32=500000f0)
    n_layers  = length(model.layers)
    first_attn = model.layers[1].attention
    n_kv_heads = first_attn.n_kv_heads
    head_dim   = first_attn.head_dim

    # ── Inputs ────────────────────────────────────────────────────────────────
    token_in = Luminal.tensor(graph, [batch, 1])

    self_k_tensors = [Luminal.tensor(graph, [batch, n_kv_heads, max_seq, head_dim])
                      for _ in 1:n_layers]
    self_v_tensors = [Luminal.tensor(graph, [batch, n_kv_heads, max_seq, head_dim])
                      for _ in 1:n_layers]

    # ── Embedding ─────────────────────────────────────────────────────────────
    x = model.embedding(token_in)  # (batch, 1, hidden)

    # ── Decoder layers ────────────────────────────────────────────────────────
    new_k_tensors = Luminal.GraphTensor[]
    new_v_tensors = Luminal.GraphTensor[]

    for (i, layer) in enumerate(model.layers)
        normed = layer.attention_norm(x)
        attn_out, nk, nv = llama_self_attn_cached(
            layer.attention, normed, step_pos,
            self_k_tensors[i], self_v_tensors[i];
            rope_base=rope_base)
        x = x + attn_out
        push!(new_k_tensors, nk)
        push!(new_v_tensors, nv)

        normed_ff = layer.feed_forward_norm(x)
        x = x + layer.feed_forward(normed_ff)
    end

    # ── Head ─────────────────────────────────────────────────────────────────
    out    = model.norm(x)   # (batch, 1, hidden)
    logits = model.head(out) # (batch, 1, vocab)

    return LlamaDecodeGraph(
        token_in.id,
        [t.id for t in self_k_tensors],
        [t.id for t in self_v_tensors],
        logits.id,
        [t.id for t in new_k_tensors],
        [t.id for t in new_v_tensors],
        step_pos)
end


"""
    llama_decode_step!(exec_fn, idg, cache, token_id; device=get_device())

Execute one cached decode step.
- `exec_fn`: compiled execution function
- `idg`: LlamaDecodeGraph for this step_pos
- `cache`: LlamaKVCacheState (mutated in place)
- `token_id`: scalar Int (0-indexed vocab index)

Returns `logits::Array{Float32,3}` of shape (batch, 1, vocab_size).
"""
function llama_decode_step!(exec_fn,
                             idg::LlamaDecodeGraph,
                             cache::LlamaKVCacheState,
                             token_id::Int;
                             device=Luminal.get_device())
    inputs = Dict{Int, Any}()
    inputs[idg.token_input_id] = Float32[token_id;;]  # (1,1)

    for (i, (k_id, v_id)) in enumerate(zip(idg.self_k_ids, idg.self_v_ids))
        inputs[k_id] = Luminal.to_device(Dict(k_id => cache.self_cache[i][1]), device)[k_id]
        inputs[v_id] = Luminal.to_device(Dict(v_id => cache.self_cache[i][2]), device)[v_id]
    end

    results = exec_fn(inputs, device)
    logits  = results[idg.logits_id]

    # Update cache
    for (i, (nk_id, nv_id)) in enumerate(zip(idg.new_self_k_ids, idg.new_self_v_ids))
        cache.self_cache[i] = (
            Array{Float32,4}(results[nk_id]),
            Array{Float32,4}(results[nv_id]))
    end

    cache.step_pos += 1
    return logits
end

export LlamaKVCacheState, LlamaDecodeGraph, build_llama_decode_step!, llama_decode_step!, llama_self_attn_cached

include("Whisper.jl")
export WhisperSelfAttention, WhisperCrossAttention, EncoderTransformerBlock, AudioEncoder,
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
