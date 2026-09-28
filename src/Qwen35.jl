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
    (dn::GatedDeltaNet)(x, conv_state, rec_state; fresh) -> (hidden, S, B)

The layer on `x` (hidden, S, B), reading and updating the state tensors (graph
inputs bound to the state buffers; see `deltanet_state_shapes`). `fresh=true`
for a prefill from empty states, `false` to continue (decode).
"""
function (dn::GatedDeltaNet)(x::Luminal.GraphTensor, conv_state::Luminal.GraphTensor,
                             rec_state::Luminal.GraphTensor; fresh::Bool)
    g_ = x.graph_ref
    _, S, B = Luminal.realized_dims(x.shape)
    nk, nv, dk, dv = dn.nk, dn.nv, dn.dk, dn.dv
    C = 2nk * dk + nv * dv

    # projections, then the causal convolution over q, k and v together
    qkv = dn.in_proj_qkv(x)                                                   # (C, S, B)
    conv_ins = [(t.id, 0, t.shape) for t in (qkv, conv_state, dn.conv_w)]
    qkv = Luminal.add_op!(g_, Luminal.CausalConv(fresh), conv_ins, Luminal.ShapeTracker([C, S, B]))

    q = Luminal.reshape(Luminal.slice_along(qkv, 1, 0, nk * dk), [dk, nk, S, B])
    k = Luminal.reshape(Luminal.slice_along(qkv, 1, nk * dk, 2nk * dk), [dk, nk, S, B])
    v = Luminal.reshape(Luminal.slice_along(qkv, 1, 2nk * dk, C), [dv, nv, S, B])
    q, k = _l2norm(q), _l2norm(k)

    # gates: beta = sigmoid(b), g = -exp(A_log) softplus(a + dt_bias), per value head
    beta = Luminal.sigmoid(dn.in_proj_b(x))                                   # (nv, S, B)
    a = dn.in_proj_a(x) + Luminal.expand(Luminal.expand(dn.dt_bias, 2, S), 3, B)
    g = Luminal.expand(Luminal.expand(-exp(dn.A_log), 2, S), 3, B) * Luminal.softplus(a)

    ins = [(t.id, 0, t.shape) for t in (q, k, v, g, beta, rec_state)]
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
