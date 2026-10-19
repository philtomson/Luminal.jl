# Mixture-of-experts routing and layer (after upstream `luminal_nn::moe`, `model_zoo::qwen3_moe`).
#
# Routing policy stays with the caller: scores in, expert ids and weights out. Julia is column-major, so upstream's `[tokens.., K]` is
# `(K, tokens..)` here (the route-slot axis first), payload axes come first (`dispatch` of (features, tokens) gives (features, K, tokens)),
# and an expert parameter bank keeps the PyTorch order `(E, params..)`: `select` of an (E, a, b) bank gives (a, b, K, tokens..).
# Expert ids are 0-based, as Float32 values (what `topk_indexes` returns), like every coordinate in `gather`.
module MoE

using ..Luminal
import ..Luminal: GraphTensor, realized_dims, expand, gather, scatter, iota, cast, select, contiguous, reshape, concat_along, constant,
                  silu, softmax

export TopKRoutes, Routes, topk_routes, normalize_routes, dispatch, combine, select_experts, with_weights, topk_indexes, expert_select,
       SparseMoE

_dims(x::GraphTensor) = realized_dims(x.shape)
_f32(x::GraphTensor) = cast(x, Float32)

function _zeros(g::Luminal.Graph, dims)
    t = constant(g, 0f0)
    for (d, n) in enumerate(dims)
        t = expand(t, d, n)
    end
    return contiguous(t)
end

# `x` with the `dims` axes (each its own size) inserted in front
function _expand_front(x::GraphTensor, dims)
    for (d, n) in enumerate(dims)
        x = expand(x, d, n)
    end
    return x
end

"""
    topk_indexes(x, k) -> (k, rest..)

The 0-based indices of the `k` largest entries along dim 1 of `x` (E, rest..), largest first; ties go to the lower index. Built from
`ArgMax` and a mask, k passes.
"""
function topk_indexes(x::GraphTensor, k::Int)
    d = _dims(x)
    1 <= k <= d[1] || throw(ArgumentError("k = $k out of range for $(d[1]) entries"))
    rest = d[2:end]
    t = length(rest) == 0 ? 1 : prod(rest)
    pos = expand(iota(x.graph_ref, [d[1]], i -> i), 2, t)                       # (E, T) the entry index
    cur = reshape(contiguous(x), [d[1], t])
    picks = GraphTensor[]
    for i in 1:k
        c3 = reshape(cur, [d[1], t, 1])
        am = Luminal.add_op!(x.graph_ref, Luminal.ArgMax(), [(c3.id, 0, c3.shape)], Luminal.ShapeTracker([t, 1]))
        idx = reshape(am, [t])                                                  # (T,)
        push!(picks, reshape(idx, [1, rest...]))
        i == k || (cur = select(pos == expand(idx, 1, d[1]), -Inf32, cur))
    end
    return length(picks) == 1 ? picks[1] : foldl((a, b) -> concat_along(a, b, 1), picks)
end

# the coordinates (payload dims first, then the route/token dims of `ids`) used to gather from or scatter into `bank`
function expert_select(ids::GraphTensor, bank::GraphTensor)
    bd, rd = _dims(bank), _dims(ids)
    np = length(bd) - 1
    out = [bd[2:end]..., rd...]
    coords = GraphTensor[_expand_front(_f32(ids), bd[2:end])]
    for a in 1:np
        push!(coords, iota(ids.graph_ref, out, (c...) -> c[a]))
    end
    return gather(bank, coords)
end

"""
    TopKRoutes(expert_ids, weights)

A fixed number K of experts per token: `expert_ids` and `weights` are (K, tokens..).
"""
struct TopKRoutes
    expert_ids::GraphTensor
    weights::GraphTensor
    function TopKRoutes(expert_ids::GraphTensor, weights::GraphTensor)
        expert_ids.graph_ref === weights.graph_ref || throw(ArgumentError("TopKRoutes tensors must belong to one graph"))
        isequal(_dims(expert_ids), _dims(weights)) ||
            throw(DimensionMismatch("expert ids $(_dims(expert_ids)) and weights $(_dims(weights)) differ"))
        return new(expert_ids, weights)
    end
end

"""
    topk_routes(scores, expert_ids) -> TopKRoutes

Route weights picked from `scores` (E, tokens..) at `expert_ids` (K, tokens..); no activation or normalisation.
"""
function topk_routes(scores::GraphTensor, expert_ids::GraphTensor)
    sd, rd = _dims(scores), _dims(expert_ids)
    isequal(sd[2:end], rd[2:end]) || throw(DimensionMismatch("scores $sd and expert ids $rd must share their token axes"))
    coords = GraphTensor[_f32(expert_ids)]
    for a in 2:length(rd)
        push!(coords, iota(scores.graph_ref, rd, (c...) -> c[a]))
    end
    return TopKRoutes(expert_ids, gather(scores, coords))
end

with_weights(r::TopKRoutes, w::GraphTensor) = TopKRoutes(r.expert_ids, w)
"Scale each token's weights to sum to one."
function normalize_routes(r::TopKRoutes)
    k = _dims(r.weights)[1]
    return with_weights(r, r.weights / expand(sum(r.weights, 1), 1, k))
end

"`input` (payload.., tokens..) broadcast across the K slots: (payload.., K, tokens..)."
function dispatch(r::TopKRoutes, input::GraphTensor)
    rd, id = _dims(r.expert_ids), _dims(input)
    nt = length(rd) - 1
    length(id) >= nt && isequal(id[end-nt+1:end], rd[2:end]) ||
        throw(DimensionMismatch("input $id does not end in the route token axes $(rd[2:end])"))
    return expand(input, length(id) - nt + 1, rd[1])
end

"An expert parameter bank (E, params..) selected per route: (params.., K, tokens..)."
select_experts(r::TopKRoutes, bank::GraphTensor) = expert_select(r.expert_ids, bank)

"Weight the routed output (payload.., K, tokens..) and sum the K slots: (payload.., tokens..)."
function combine(r::TopKRoutes, routed::GraphTensor)
    rd, od = _dims(r.expert_ids), _dims(routed)
    isequal(od[end-length(rd)+1:end], rd) || throw(DimensionMismatch("routed output $od does not end in the route axes $rd"))
    np = length(od) - length(rd)
    w = _expand_front(cast(r.weights, Luminal.dtype(routed)), od[1:np])
    return sum(routed * w, np + 1)
end

"""
    Routes(token_ids, expert_ids, slot_ids, weights, token_count, max_routes_per_token)

The general sparse table: one entry per route (vectors of equal length R). `(token, slot)` pairs must be unique with slot in
`0:max_routes_per_token-1`, which lets `combine` scatter by assignment and sum.
"""
struct Routes
    token_ids::GraphTensor
    expert_ids::GraphTensor
    slot_ids::GraphTensor
    weights::GraphTensor
    token_count::Int
    max_routes_per_token::Int
    function Routes(token_ids, expert_ids, slot_ids, weights, token_count::Int, max_routes_per_token::Int)
        for t in (token_ids, expert_ids, slot_ids, weights)
            length(_dims(t)) == 1 || throw(ArgumentError("route tensors must be vectors"))
            _dims(t) == _dims(token_ids) || throw(DimensionMismatch("route table columns differ in length"))
        end
        return new(token_ids, expert_ids, slot_ids, weights, token_count, max_routes_per_token)
    end
end

"The flat route table of structured routes `r` (token-major: token t, slot s is route `s + K t`)."
function Routes(r::TopKRoutes)
    d = _dims(r.expert_ids)
    K, T = d[1], length(d) == 1 ? 1 : prod(d[2:end])
    g = r.expert_ids.graph_ref
    flat(x) = reshape(contiguous(x), [K * T])
    tok = iota(g, d, (c...) -> begin                                 # the flattened token index (first token axis fastest)
        id, stride = 0, 1
        for a in 2:length(d)
            id += c[a] * stride; stride *= d[a]
        end
        id
    end)
    slot = iota(g, d, (c...) -> c[1])
    return Routes(flat(tok), flat(r.expert_ids), flat(slot), flat(r.weights), T, K)
end

with_weights(r::Routes, w::GraphTensor) = Routes(r.token_ids, r.expert_ids, r.slot_ids, w, r.token_count, r.max_routes_per_token)

"`input` (payload.., T) gathered into route order: (payload.., R)."
function dispatch(r::Routes, input::GraphTensor)
    id = _dims(input)
    id[end] == r.token_count || throw(DimensionMismatch("input has $(id[end]) tokens, the routes expect $(r.token_count)"))
    np = length(id) - 1
    out = [id[1:np]..., _dims(r.token_ids)[1]]
    coords = GraphTensor[iota(r.token_ids.graph_ref, out, (c...) -> c[a]) for a in 1:np]
    push!(coords, _expand_front(_f32(r.token_ids), id[1:np]))
    return gather(input, coords)
end

select_experts(r::Routes, bank::GraphTensor) = expert_select(r.expert_ids, bank)

"Weight the routed output (payload.., R), scatter it to (token, slot) and sum the slots: (payload.., T)."
function combine(r::Routes, routed::GraphTensor)
    od = _dims(routed)
    od[end] == _dims(r.token_ids)[1] || throw(DimensionMismatch("routed output has $(od[end]) routes, the table has $(_dims(r.token_ids)[1])"))
    np = length(od) - 1
    g = routed.graph_ref
    weighted = routed * _expand_front(cast(r.weights, Luminal.dtype(routed)), od[1:np])
    dest = _zeros(g, [od[1:np]..., r.max_routes_per_token, r.token_count])
    coords = GraphTensor[iota(g, od, (c...) -> c[a]) for a in 1:np]
    push!(coords, _expand_front(_f32(r.slot_ids), od[1:np]))
    push!(coords, _expand_front(_f32(r.token_ids), od[1:np]))
    return sum(scatter(dest, weighted, coords), np + 1)
end

"""
    SparseMoE(router, gate_up, down, top_k, intermediate; norm_topk_prob=true)

Qwen3's routed SwiGLU feed-forward over `x` (hidden, tokens): the router `Linear` (hidden -> experts), softmax, the top-k experts per token
(weights renormalised when `norm_topk_prob`), then per route `down[e] * (silu(gate) * up)` with `gate; up = gate_up[e] * x`. Banks are
(E, 2*intermediate, hidden) and (E, hidden, intermediate), as in the checkpoints. The per-route weights are gathered, so this form
suits few tokens (decode, small batches); it is the reference formulation, not a grouped-GEMM kernel.
"""
struct SparseMoE
    router::Any
    gate_up::GraphTensor
    down::GraphTensor
    top_k::Int
    intermediate::Int
    norm_topk_prob::Bool
end
SparseMoE(router, gate_up, down, top_k, intermediate; norm_topk_prob::Bool=true) =
    SparseMoE(router, gate_up, down, top_k, intermediate, norm_topk_prob)

function (m::SparseMoE)(x::GraphTensor)
    hidden, T = _dims(x)
    probs = softmax(m.router(x), 1)                                        # (E, T)
    routes = topk_routes(probs, topk_indexes(probs, m.top_k))
    m.norm_topk_prob && (routes = normalize_routes(routes))
    I = m.intermediate
    xs = dispatch(routes, x)                                               # (hidden, K, T)
    gu = select_experts(routes, m.gate_up)                                 # (2I, hidden, K, T)
    proj = sum(gu * expand(xs, 1, 2I), 2)                                  # (2I, K, T)
    h = silu(Luminal.slice_along(proj, 1, 0, I)) * Luminal.slice_along(proj, 1, I, 2I)    # (I, K, T)
    dn = select_experts(routes, m.down)                                    # (hidden, I, K, T)
    out = sum(dn * expand(h, 1, hidden), 2)                                # (hidden, K, T)
    return combine(routes, out)                                            # (hidden, T)
end

end # module MoE
