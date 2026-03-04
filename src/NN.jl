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
function apply_rotary_embeddings(input::Luminal.GraphTensor, prev_seq; base=10000.0f0)
    # input: D, S, H, B
    dims = Luminal.realized_dims(input.shape)
    head_dim, seq, n_heads, batch = dims[1], dims[2], dims[3], dims[4]
    
    graph = input.graph_ref
    
    # Get freqs
    half_dim = div(head_dim, 2)
    freqs = Luminal.arange(graph, half_dim) * 2.0f0 / Float32(head_dim)
    # inv_freqs = 1.0 / base^(2i/d)
    # Using exp2(-x) to avoid overflow in intermediate exp2(x) when base=500k
    inv_freqs = Luminal.exp2(-freqs * log2(Float32(base)))
    
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
    
    # Align emb to (half_dim, seq) for broadcasting over (half_dim, seq, H, B)
    emb_t = Luminal.permute(emb, [2, 1])
    
    # Split input into halves along first dimension (rotate half)
    x0 = Luminal.slice_along(input, 1, 0, half_dim)
    x1 = Luminal.slice_along(input, 1, half_dim, head_dim)
    
    # Expand emb_t to (half_dim, seq, n_heads, batch)
    emb_expanded = Luminal.expand(Luminal.expand(emb_t, 3, n_heads), 4, batch)
    
    sin_emb = Luminal.sin(emb_expanded)
    cos_emb = Luminal.cos(emb_expanded)
    
    # Standard Llama RoPE: 
    # out_0 = x0 * cos - x1 * sin
    # out_1 = x1 * cos + x0 * sin
    x0_out = x0 * cos_emb - x1 * sin_emb
    x1_out = x1 * cos_emb + x0 * sin_emb
    
    return Luminal.concat_along(x0_out, x1_out, 1)
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

function (sa::SelfAttention)(x::Luminal.GraphTensor, prev_seq::Int; rope_base=10000.0f0, return_kv::Bool=false)
    # x: (hidden, seq, batch)
    hidden, seq, batch = Luminal.realized_dims(x.shape)
    
    # Project queries, keys, values
    # Llama weights are packed as (Heads * HeadDim, In). 
    # In Julia column-major matrix W(Out, In), HeadDim is the faster dimension.
    # W * x -> (Heads * HeadDim, Seq, Batch)
    queries = Luminal.reshape(sa.q_proj(x), [sa.head_dim, sa.n_heads, seq, batch])
    queries = Luminal.permute(queries, [1, 3, 2, 4]) # (D, S, H, B)
    
    keys = Luminal.reshape(sa.k_proj(x), [sa.head_dim, sa.n_kv_heads, seq, batch])
    keys = Luminal.permute(keys, [1, 3, 2, 4]) # (D, S, KV_H, B)
    
    values = Luminal.reshape(sa.v_proj(x), [sa.head_dim, sa.n_kv_heads, seq, batch])
    values = Luminal.permute(values, [1, 3, 2, 4]) # (D, S, KV_H, B)
    
    # RoPE
    queries = apply_rotary_embeddings(queries, prev_seq; base=rope_base)
    keys = apply_rotary_embeddings(keys, prev_seq; base=rope_base)
    
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
    weights = Luminal.matmul(q_t, keys) * (1.0f0 / sqrt(Float32(sa.head_dim)))
    
    # Mask
    if seq > 1
        mask = Luminal.triu(x.graph_ref, seq, 1) * -9f9
        # Expand mask (S, S) to (S, S, H, B)
        mask_expanded = Luminal.expand(Luminal.expand(mask, 3, sa.n_heads), 4, batch)
        weights = weights + mask_expanded
    end
    
    probs = Luminal.softmax(weights, 2) # Softmax over dimension 2 (columns S_k)
    
    # (S, S, H, B) @ (S, D, H, B) -> (S, D, H, B) 
    # Wait, (S, S) * (S, D) -> (S, D). Correct!
    v_t = Luminal.permute(values, [2, 1, 3, 4])
    out = Luminal.matmul(probs, v_t)
    
    # (S, D, H, B) -> (D, H, S, B) -> (Hidden, S, Batch)
    out = Luminal.permute(out, [2, 3, 1, 4])
    out = Luminal.reshape(out, [hidden, seq, batch])
    
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

function TransformerBlock(hidden::Int, n_heads::Int, n_kv_heads::Int, intermediate::Int, graph::Luminal.Graph, reg=nothing, prefix::String="block")
    return TransformerBlock(
        SelfAttention(hidden, n_heads, n_kv_heads, graph, reg, "$(prefix).self_attn"),
        _rmsnorm(hidden, graph, reg, "$(prefix).input_layernorm"),
        Mlp(hidden, intermediate, graph, reg, "$(prefix).mlp"),
        _rmsnorm(hidden, graph, reg, "$(prefix).post_attention_layernorm")
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
end

function Llama(graph::Luminal.Graph, reg=nothing; 
               vocab_size=128256, 
               hidden=4096, 
               n_layers=32, 
               n_heads=32, 
               n_kv_heads=8, 
               intermediate=14336,
               rope_base=500000.0f0)
    
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
        _linear(hidden, vocab_size, graph, reg, "lm_head"; bias=false),
        rope_base
    )
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

Host-side storage for past K/V tensors for one decode session.
- `self_cache[i]` = `(K, V)` arrays for layer i, shape (batch, n_kv_heads, max_seq, head_dim)
- `step_pos`: current 0-indexed decode position
"""
mutable struct LlamaKVCacheState
    step_pos::Int
    max_seq::Int
    self_cache::Vector{Tuple{Array{Float16,4}, Array{Float16,4}}}
end

"""
    LlamaKVCacheState(n_layers, n_kv_heads, head_dim; batch=1, max_seq=2048)
"""
function LlamaKVCacheState(n_layers::Int, n_kv_heads::Int, head_dim::Int;
                            batch::Int=1, max_seq::Int=2048)
    self = [(zeros(Float16, head_dim, max_seq, n_kv_heads, batch),
             zeros(Float16, head_dim, max_seq, n_kv_heads, batch))
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
                                 step_pos::Luminal.DimType,
                                 step_pos_tensor::Luminal.GraphTensor,
                                 past_k::Luminal.GraphTensor,
                                 past_v::Luminal.GraphTensor;
                                 rope_base::Float32=500000f0)
    hidden, _, batch = Luminal.realized_dims(x.shape)

    # Project current token: (Hidden, 1, B) -> (D, H, 1, B)
    q_raw = Luminal.reshape(sa.q_proj(x), [sa.head_dim, sa.n_heads, 1, batch])
    q     = Luminal.permute(q_raw, [1, 3, 2, 4])   # (D, 1, H, B)

    k_raw = Luminal.reshape(sa.k_proj(x), [sa.head_dim, sa.n_kv_heads, 1, batch])
    k_new = Luminal.permute(k_raw, [1, 3, 2, 4])  # (D, 1, KV_H, B)

    v_raw = Luminal.reshape(sa.v_proj(x), [sa.head_dim, sa.n_kv_heads, 1, batch])
    v_new = Luminal.permute(v_raw, [1, 3, 2, 4])   # (D, 1, KV_H, B)

    # Apply RoPE
    q     = apply_rotary_embeddings(q,     step_pos_tensor; base=rope_base)
    k_new = apply_rotary_embeddings(k_new, step_pos_tensor; base=rope_base)

    max_seq = Luminal.realized_dims(past_k.shape)[2]

    function _kv_scatter_llama(past, slot)
        return Luminal.concat_along(
                   Luminal.concat_along(
                       Luminal.slice_along(past, 2, 0, step_pos), slot, 2),
                   Luminal.slice_along(past, 2, step_pos + 1, max_seq), 2)
    end

    new_k = _kv_scatter_llama(past_k, k_new)
    new_v = _kv_scatter_llama(past_v, v_new)

    # Attend over [0 : step_pos + 1]
    k_ctx   = Luminal.slice_along(new_k, 2, 0, step_pos + 1)   # (D, ctx, KV_H, B)
    v_ctx   = Luminal.slice_along(new_v, 2, 0, step_pos + 1)   # (D, ctx, KV_H, B)

    # GQA: repeat KV heads to match Q heads
    if sa.n_kv_heads < sa.n_heads
        groups = div(sa.n_heads, sa.n_kv_heads)
        k_ctx = repeat_kv(k_ctx, groups)
        v_ctx = repeat_kv(v_ctx, groups)
    end

    # weights: (1, D, H, B) * (D, ctx, H, B) -> (1, ctx, H, B)
    q_t = Luminal.permute(q, [2, 1, 3, 4]) # (1, D, H, B)
    weights  = Luminal.matmul(q_t, k_ctx) * (1.0f0 / sqrt(Float32(sa.head_dim)))
    probs    = Luminal.softmax(weights, 2) # Softmax over ctx
    
    # out: (1, ctx, H, B) * (ctx, D, H, B) -> (1, D, H, B)
    v_ctx_t = Luminal.permute(v_ctx, [2, 1, 3, 4])
    out      = Luminal.matmul(probs, v_ctx_t)   # (1, D, H, B)
    
    # Back to (Hidden, 1, Batch)
    out      = Luminal.reshape(Luminal.permute(out, [2, 3, 1, 4]), [hidden, 1, batch])

    return sa.o_proj(out), new_k, new_v
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
    new_self_k_ids::Vector{Int}
    new_self_v_ids::Vector{Int}
    step_pos::Luminal.DimType
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
    n_layers  = length(model.layers)
    first_attn = model.layers[1].attention
    n_kv_heads = first_attn.n_kv_heads
    head_dim   = first_attn.head_dim

    # ── Inputs ────────────────────────────────────────────────────────────────
    token_in = Luminal.tensor(graph, [1, batch])
    pos_tensor = Luminal.tensor(graph, [1])

    self_k_tensors = [Luminal.tensor(graph, [head_dim, max_seq, n_kv_heads, batch])
                      for _ in 1:n_layers]
    self_v_tensors = [Luminal.tensor(graph, [head_dim, max_seq, n_kv_heads, batch])
                      for _ in 1:n_layers]

    # ── Embedding ─────────────────────────────────────────────────────────────
    x = model.embedding(token_in)  # (batch, 1, hidden)

    # ── Decoder layers ────────────────────────────────────────────────────────
    new_k_tensors = Luminal.GraphTensor[]
    new_v_tensors = Luminal.GraphTensor[]

    for (i, layer) in enumerate(model.layers)
        normed = layer.attention_norm(x)
        attn_out, nk, nv = llama_self_attn_cached(
            layer.attention, normed, step_pos, pos_tensor,
            self_k_tensors[i], self_v_tensors[i];
            rope_base=rope_base)
        x = x + attn_out
        push!(new_k_tensors, nk)
        push!(new_v_tensors, nv)

        normed_ff = layer.feed_forward_norm(x)
        x = x + layer.feed_forward(normed_ff)
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
                             sym_vals::Dict{Symbol, Int}=Dict{Symbol, Int}(),
                             device=Luminal.get_device())
    inputs = Dict{Int, Any}()
    inputs[idg.token_input_id] = Float32[token_id;;]  # (1,1)
    inputs[idg.pos_input_id] = Float32[cache.step_pos] # (1,)

    for (i, (k_id, v_id)) in enumerate(zip(idg.self_k_ids, idg.self_v_ids))
        inputs[k_id] = Luminal.to_device(Dict(k_id => cache.self_cache[i][1]), device)[k_id]
        inputs[v_id] = Luminal.to_device(Dict(v_id => cache.self_cache[i][2]), device)[v_id]
    end

    results = exec_fn(inputs; sym_vals=sym_vals, device=device)
    logits  = results[idg.logits_id]

    # Update cache
    for (i, (nk_id, nv_id)) in enumerate(zip(idg.new_self_k_ids, idg.new_self_v_ids))
        cache.self_cache[i] = (
            Array{Float16,4}(results[nk_id]),
            Array{Float16,4}(results[nv_id]))
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
