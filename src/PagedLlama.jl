# Batched Llama decode step over a shared paged KV pool (after upstream's `model_zoo::paged_llama3`).
#
# One fixed-shape graph serves any mix of sequences: `rows` token rows (decode tokens of different sequences, or a prefill chunk), a
# pool of `slots` KV slots per layer shared by all of them, and `context` gathered key slots. The host (see `PageTable` and
# `plan_step`) decides which slots each row writes and which the step reads; the graph only builds the position / sequence bias from
# the ids it is given, so one captured graph is reused for every step.
module PagedLlama

using ..Luminal
using ..Luminal.NN
using ..PagedAttention
import ..Luminal: GraphTensor, Graph, realized_dims

export PagedLlamaGraph, PagedKvPool, build_paged_llama_step!, paged_llama_step!

"Node ids of a paged decode graph (see `build_paged_llama_step!`)."
struct PagedLlamaGraph
    rows::Int
    slots::Int
    context::Int
    token_id::Int
    position_id::Int
    write_slots_id::Int
    row_seq_id::Int
    read_slots_id::Int
    key_position_id::Int
    key_seq_id::Int
    cache_in_ids::Vector{Tuple{Int,Int}}    # per layer, the (keys, values) pool inputs
    cache_out_ids::Vector{Tuple{Int,Int}}   # per layer, the pools after this step's writes
    logits_id::Int                          # (vocab, 1, rows)
end

"""
    build_paged_llama_step!(model, graph; rows, slots, context, rope_base) -> PagedLlamaGraph

The decode graph of a `Llama` (Llama 2/3, TinyLlama, Qwen3: full attention, no output gate or window) over a paged pool. Inputs, all
Float32 vectors: `tokens` (1, rows), `positions` (rows) the tokens' logical positions (RoPE and causality), `write_slots` (rows) the pool slot
each token's K/V goes to (>= `slots` drops the write), `row_seq` (rows) and `key_seq` (context) sequence ids (rows attend only to keys of
their own sequence; give padding rows and padding keys ids that match nothing), `read_slots` (context) the pool slots attention reads, and
`key_positions` (context) their logical positions; plus a (width, slots) key and value pool per layer. The step writes every row's K/V before
reading, so a row sees its own and earlier tokens of the same step.
"""
function build_paged_llama_step!(model::Llama, g::Graph; rows::Int, slots::Int, context::Int,
                                 rope_base::Float32=model.rope_base)
    rows > 0 && slots > 0 && context > 0 || throw(ArgumentError("rows, slots and context must be positive"))
    for l in model.layers
        a = l.attention
        (a.output_gate || a.window != 0 || a.rotary_dim ∉ (0, a.head_dim)) &&
            throw(ArgumentError("paged decode supports plain full-attention Llama layers only"))
    end
    tokens = Luminal.tensor(g, [1, rows])
    positions = Luminal.tensor(g, [rows])
    write_slots = Luminal.tensor(g, [rows])
    row_seq = Luminal.tensor(g, [rows])
    read_slots = Luminal.tensor(g, [context])
    key_positions = Luminal.tensor(g, [context])
    key_seq = Luminal.tensor(g, [context])
    bias = causal_bias(positions, key_positions) + sequence_isolation_bias(row_seq, key_seq)     # (context, rows)
    access = CacheAccess(write_slots, read_slots)

    caches_in = Tuple{GraphTensor,GraphTensor}[]
    caches_out = Tuple{GraphTensor,GraphTensor}[]
    x = model.embedding(tokens)                                                                  # (hidden, 1, rows)
    for layer in model.layers
        a = layer.attention
        width = a.n_kv_heads * a.head_dim
        kin, vin = Luminal.tensor(g, [width, slots]), Luminal.tensor(g, [width, slots])
        push!(caches_in, (kin, vin))
        geo = AttentionGeometry(a.n_heads, a.n_kv_heads, a.head_dim, a.scale)

        h = layer.attention_norm(x)
        D, H, KV = a.head_dim, a.n_heads, a.n_kv_heads
        # rows play the batch: (D, 1, heads, rows), each with its own RoPE position
        heads(t, n) = Luminal.permute(Luminal.reshape(t, [D, n, 1, rows]), [1, 3, 2, 4])
        base = NN._rope_base(a, rope_base)
        q = NN._rope(a, NN._qk_norm(a.q_norm, heads(a.q_proj(h), H)), positions, base)
        k = NN._rope(a, NN._qk_norm(a.k_norm, heads(a.k_proj(h), KV)), positions, base)
        flat(t, n) = Luminal.reshape(Luminal.contiguous(Luminal.permute(t, [1, 3, 2, 4])), [D * n, rows])
        v = Luminal.reshape(a.v_proj(h), [D * KV, rows])
        out, cache = paged_attention(flat(q, H), flat(k, KV), v, KvCache(kin, vin), access, bias, geo)
        push!(caches_out, (cache.keys, cache.values))
        x = x + a.o_proj(Luminal.reshape(out, [D * H, 1, rows]))
        x = x + layer.feed_forward(layer.feed_forward_norm(x))
    end
    logits = model.head(model.norm(x))                                                           # (vocab, 1, rows)
    return PagedLlamaGraph(rows, slots, context, tokens.id, positions.id, write_slots.id, row_seq.id, read_slots.id,
                           key_positions.id, key_seq.id,
                           [(k.id, v.id) for (k, v) in caches_in], [(k.id, v.id) for (k, v) in caches_out], logits.id)
end

"""
    PagedKvPool(model, slots; device=CPUDevice())

The per-layer (keys, values) pools, (kv_heads * head_dim, slots) each, zero-filled, on `device`. `paged_llama_step!` updates them.
"""
struct PagedKvPool
    layers::Vector{Tuple{Any,Any}}
end
function PagedKvPool(model::Llama, slots::Int; device::Luminal.AbstractDevice=Luminal.CPUDevice())
    return PagedKvPool([(Luminal.zero_tensor(device, Float32, l.attention.n_kv_heads * l.attention.head_dim, slots),
                         Luminal.zero_tensor(device, Float32, l.attention.n_kv_heads * l.attention.head_dim, slots))
                        for l in model.layers])
end

"""
    paged_llama_step!(exec, pg, pool, plan; device=get_device()) -> logits (vocab, rows)

Run one step. `plan` is a `plan_step` result (any named tuple with `tokens`, `positions`, `write_slots`, `row_seq`, `read_slots`,
`key_positions`, `key_seq`). `exec` must have been compiled with the pool outputs retained (`retain=paged_retain_ids(pg)`); their
contents are copied back into `pool`.
"""
function paged_llama_step!(exec, pg::PagedLlamaGraph, pool::PagedKvPool, plan; device=Luminal.get_device())
    f(v) = Float32.(v)
    inputs = Dict{Int,Any}(pg.token_id => Float32.(Base.reshape(plan.tokens, 1, :)),
                           pg.position_id => f(plan.positions), pg.write_slots_id => f(plan.write_slots),
                           pg.row_seq_id => f(plan.row_seq), pg.read_slots_id => f(plan.read_slots),
                           pg.key_position_id => f(plan.key_positions), pg.key_seq_id => f(plan.key_seq))
    for ((kin, vin), (k, v)) in zip(pg.cache_in_ids, pool.layers)
        inputs[kin] = k; inputs[vin] = v
    end
    res = exec(inputs; device=device)
    for ((kout, vout), (k, v)) in zip(pg.cache_out_ids, pool.layers)
        copyto!(k, res[kout]); copyto!(v, res[vout])
    end
    return Base.reshape(res[pg.logits_id], :, pg.rows)
end

paged_retain_ids(pg::PagedLlamaGraph) =
    vcat(pg.logits_id, [i for (k, v) in pg.cache_out_ids for i in (k, v)])
export paged_retain_ids

end # module PagedLlama
