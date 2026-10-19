# Paged attention over a shared key/value pool (after upstream Luminal's `luminal_nn::{cache, attention}`).
#
# The cache is functional graph state: one (width, slots) matrix of keys and one of values per layer, `slots` being a
# flat physical slot space shared by every sequence. Allocation and page-table policy stay with the caller, who
# passes, per step, the physical slot each new token is written to, the physical slots to read as the attention
# context (in any order), and an additive score bias built from the tokens' logical positions and sequence ids. Several
# sequences can share one step (the bias isolates them), and a sequence's slots need not be contiguous.
#
# Layout: Julia column-major, so upstream's [rows, features] is (features, rows) here; scores are (context, queries).
module PagedAttention

using ..Luminal
import ..Luminal: GraphTensor, Graph, realized_dims, cast, expand, select, iota, gather, scatter, matmul, softmax1,
                  permute, reshape, contiguous, constant, dtype

export KvCache, CacheAccess, AttentionGeometry, gather_rows, scatter_rows, causal_bias, sequence_isolation_bias,
       sliding_window_bias, packed_sequence_ids, grouped_query_attention, paged_attention, MASKED_SCORE

"Additive score of a masked key (finite, so a fully masked row softmaxes to a uniform row instead of NaN)."
const MASKED_SCORE = -1f9

_f32(x::GraphTensor) = cast(x, Float32)

function _vector(x::GraphTensor, name)
    length(realized_dims(x.shape)) == 1 || throw(ArgumentError("$name must be a vector"))
    return x
end

# The coordinate pair (feature, row) of a gather / scatter over the rows of a (width, rows) matrix: `indices` (n,)
# become a (width, n) row coordinate, the feature coordinate counts 0:width-1. Coordinates are Float32 (exact up to 2^24 rows).
function _row_coords(g::Graph, indices::GraphTensor, width::Int)
    n = only(realized_dims(indices.shape))
    feat = iota(g, [width, n], (f, _) -> f)
    return [feat, expand(_f32(_vector(indices, "row indices")), 1, width)]
end

"""
    gather_rows(data, indices) -> (width, n)

Columns `indices` (0-based slots, an Int or Float32 vector) of `data` (width, rows), in that order. Out-of-range reads give 0.
"""
function gather_rows(data::GraphTensor, indices::GraphTensor)
    dims = realized_dims(data.shape)
    length(dims) == 2 || throw(ArgumentError("gather_rows needs a (width, rows) matrix, got $dims"))
    return gather(data, _row_coords(data.graph_ref, indices, dims[1]))
end

"""
    scatter_rows(src, indices, dest) -> (width, rows)

A copy of `dest` (width, rows) whose columns `indices` are replaced by the columns of `src` (width, n). Out-of-range
slots are dropped; with a repeated slot, which write survives is unspecified on the GPU.
"""
function scatter_rows(src::GraphTensor, indices::GraphTensor, dest::GraphTensor)
    sd, dd = realized_dims(src.shape), realized_dims(dest.shape)
    length(sd) == 2 && length(dd) == 2 && sd[1] == dd[1] ||
        throw(DimensionMismatch("scatter_rows: source $sd and destination $dd must be (width, n) and (width, rows)"))
    only(realized_dims(indices.shape)) == sd[2] || throw(DimensionMismatch("scatter_rows: one slot per source column"))
    return scatter(dest, src, _row_coords(dest.graph_ref, indices, sd[1]))
end

"""
    KvCache(keys, values)

One layer's cache value: (width, slots) matrices with `width = kv_heads * head_dim`.
`write(cache, slots, new_keys, new_values)` returns the cache with those slots overwritten; `read(cache, slots)` gathers
the context `(keys, values)` of the given slots.
"""
struct KvCache
    keys::GraphTensor
    values::GraphTensor
    function KvCache(keys::GraphTensor, values::GraphTensor)
        realized_dims(keys.shape) == realized_dims(values.shape) && length(realized_dims(keys.shape)) == 2 ||
            throw(DimensionMismatch("KvCache: keys $(realized_dims(keys.shape)) and values $(realized_dims(values.shape)) must be equal (width, slots) matrices"))
        dtype(keys) === dtype(values) || throw(ArgumentError("KvCache: key and value dtypes differ"))
        return new(keys, values)
    end
end

write(c::KvCache, slots::GraphTensor, new_keys::GraphTensor, new_values::GraphTensor) =
    KvCache(scatter_rows(new_keys, slots, c.keys), scatter_rows(new_values, slots, c.values))
read(c::KvCache, slots::GraphTensor) = (keys=gather_rows(c.keys, slots), values=gather_rows(c.values, slots))

"The physical slots one step uses: `write_slots` (one per new token) and `read_slots` (the context, any order)."
struct CacheAccess
    write_slots::GraphTensor
    read_slots::GraphTensor
end

"The head layout and score scale of grouped-query attention (`query_heads` a multiple of `kv_heads`)."
struct AttentionGeometry
    query_heads::Int
    kv_heads::Int
    head_dim::Int
    scale::Float32
    function AttentionGeometry(query_heads, kv_heads, head_dim, scale=1f0 / sqrt(Float32(head_dim)))
        query_heads > 0 && kv_heads > 0 && head_dim > 0 || throw(ArgumentError("head counts and head_dim must be positive"))
        query_heads % kv_heads == 0 || throw(ArgumentError("query_heads must be divisible by kv_heads"))
        return new(query_heads, kv_heads, head_dim, Float32(scale))
    end
end

# --- additive biases, (keys, queries) ---------------------------------------------------------------------------

# (nk, nq) pair of broadcast vectors: the queries' values along dim 2, the keys' along dim 1
function _pair(q::GraphTensor, k::GraphTensor)
    nq, nk = only(realized_dims(_vector(q, "query vector").shape)), only(realized_dims(_vector(k, "key vector").shape))
    return expand(q, 1, nk), expand(k, 2, nq)
end

"""
    causal_bias(query_positions, key_positions) -> (keys, queries)

0 where the key's logical position is at most the query's, `MASKED_SCORE` otherwise. Positions need not match the
physical slots or the context order.
"""
function causal_bias(qpos::GraphTensor, kpos::GraphTensor)
    q, k = _pair(_f32(qpos), _f32(kpos))
    return select(k > q, MASKED_SCORE, 0f0)
end

"""
    sequence_isolation_bias(query_sequence_ids, key_sequence_ids) -> (keys, queries)

0 where query and key belong to the same sequence, `MASKED_SCORE` otherwise; add it to a causal or sliding-window bias
when the gathered context interleaves several sequences.
"""
function sequence_isolation_bias(qseq::GraphTensor, kseq::GraphTensor)
    q, k = _pair(_f32(qseq), _f32(kseq))
    return select(k == q, 0f0, MASKED_SCORE)
end

"""
    sliding_window_bias(query_positions, key_positions, window) -> (keys, queries)

Causal, and only the `window` most recent positions (the query's own included).
"""
function sliding_window_bias(qpos::GraphTensor, kpos::GraphTensor, window::Int)
    window > 0 || throw(ArgumentError("attention window must be positive"))
    q, k = _pair(_f32(qpos), _f32(kpos))
    return select((k > q) + (k < q - Float32(window - 1)) > 0f0, MASKED_SCORE, 0f0)
end

"""
    packed_sequence_ids(indptr, item_count) -> (item_count,)

The sequence each item of a packed batch belongs to: `indptr` (batch + 1,) holds the cumulative boundaries.
"""
function packed_sequence_ids(indptr::GraphTensor, item_count::Int)
    nb = only(realized_dims(_vector(indptr, "indptr").shape))
    items = expand(cast(iota(indptr.graph_ref, [item_count], i -> i), Float32), 2, nb)      # (items, boundaries)
    bounds = expand(_f32(indptr), 1, item_count)
    return sum(cast(bounds <= items, Float32), 2) - 1f0
end

# --- attention ------------------------------------------------------------------------------------------------------

# (hd*heads, n) -> (hd, n, heads)
_split_heads(x, hd, heads) = permute(reshape(x, [hd, heads, realized_dims(x.shape)[2]]), [1, 3, 2])

"""
    grouped_query_attention(query, keys, values, score_bias, geometry) -> (query_heads * head_dim, queries)

`query` (query_heads*head_dim, queries); `keys`, `values` (kv_heads*head_dim, context); `score_bias` (context, queries)
or (context, queries, query_heads), added to the scaled scores. Query head `h` reads key/value head `h ÷ groups`.
"""
function grouped_query_attention(query::GraphTensor, keys::GraphTensor, values::GraphTensor, bias::GraphTensor,
                                 geo::AttentionGeometry)
    hd, H, KV = geo.head_dim, geo.query_heads, geo.kv_heads
    groups = H ÷ KV
    qd, kd, vd = realized_dims(query.shape), realized_dims(keys.shape), realized_dims(values.shape)
    qd[1] == hd * H || throw(DimensionMismatch("query width $(qd[1]) != query_heads * head_dim = $(hd * H)"))
    kd[1] == hd * KV && vd[1] == hd * KV && kd[2] == vd[2] ||
        throw(DimensionMismatch("keys $kd / values $vd do not match kv_heads * head_dim = $(hd * KV)"))
    nq, nk = qd[2], kd[2]
    function kv_heads(x)                                    # (hd*KV, nk) -> (hd, nk, H), head h = kvh * groups + g
        t = permute(reshape(x, [hd, KV, nk]), [1, 3, 2])    # (hd, nk, KV)
        t = reshape(expand(t, 3, groups), [hd, nk, H])      # (hd, nk, groups, KV) -> heads, group fastest
        return contiguous(t)
    end
    q = _split_heads(query, hd, H)                                                       # (hd, nq, H)
    k = kv_heads(keys); v = kv_heads(values)
    scores = matmul(permute(k, [2, 1, 3]), q) * geo.scale                                # (nk, nq, H)
    bd = realized_dims(bias.shape)
    b = length(bd) == 2 ? expand(bias, 3, H) : bias
    realized_dims(b.shape) == [nk, nq, H] || throw(DimensionMismatch("score bias $bd does not match (context, queries[, heads]) = ($nk, $nq)"))
    out = matmul(v, softmax1(scores + cast(b, dtype(scores))))                           # (hd, nq, H)
    return reshape(contiguous(permute(out, [1, 3, 2])), [hd * H, nq])
end

"""
    paged_attention(query, new_keys, new_values, cache, access, score_bias, geometry) -> (output, cache)

Write the new keys and values into `cache` at `access.write_slots`, read the context at `access.read_slots` (so the
new tokens are part of it when their slots are listed), and run grouped-query attention under `score_bias`
(context, queries). Returns the output (query_heads*head_dim, queries) and the updated cache value.
"""
function paged_attention(query::GraphTensor, new_keys::GraphTensor, new_values::GraphTensor, cache::KvCache,
                         access::CacheAccess, bias::GraphTensor, geo::AttentionGeometry)
    only(realized_dims(access.write_slots.shape)) == realized_dims(query.shape)[2] ||
        throw(DimensionMismatch("one cache write slot is required per query"))
    cache = write(cache, access.write_slots, new_keys, new_values)
    ctx = read(cache, access.read_slots)
    return (output=grouped_query_attention(query, ctx.keys, ctx.values, bias, geo), cache=cache)
end

end # module PagedAttention
