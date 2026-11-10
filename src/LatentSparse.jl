# Latent Sparse Attention (D-LSA, flat design), CPU reference in plain Julia.
#
# A port of the plain-torch reference of https://github.com/Prefrontal-AI/latent-sparse-attention (Apache-2.0; paper arXiv 2610.42147) -
# `src/lsa/core`: `reference.py` (build_buckets, read), `ward.py` (the Ward merge) and `index_space.py` (the index vectors) - pool, Ward,
# bucket, score, pick, expand. It is the oracle the GPU kernels (LatentSparseSelect.jl, DecodeAttnGather.jl) and the decode wiring
# (LSADecode.jl) are tested against, so it favours being obviously the same algorithm over speed. Positions in tables and id lists are
# 0-based, as the oracle's and the fixtures' are, and -1 pads; Julia arrays are indexed from 1 inside.
#
# Depends on nothing in Luminal, so it also loads on its own (the tests include it that way); Luminal.jl includes it as `Luminal.LatentSparse`.

module LatentSparse

using LinearAlgebra

export BucketTable, build_buckets, read_selection, select_from_scores, ward_labels, index_vectors

"The buckets of a cache's closed cluster blocks, block after block."
struct BucketTable
    "Normalized key sum of each bucket (rows, index_dim)"
    centroids::Matrix{Float32}
    "0-based positions of each bucket's keys (rows, bucket_cap), -1 padded"
    members::Matrix{Int32}
    "Keys in each bucket (rows,)"
    sizes::Vector{Int32}
    "Buckets a cluster block ends with"
    block_bucket_count::Int
    "Keys in a cluster block"
    cluster_block_size::Int
end

const NORM_TOLERANCE = 1f-4

function _check_unit_rows(name::AbstractString, rows::AbstractMatrix{Float32})
    all(isfinite, rows) || throw(ArgumentError("$name holds a non-finite value"))
    off = isempty(rows) ? 0f0 : maximum(abs(norm(@view rows[i, :]) - 1f0) for i in 1:size(rows, 1))
    off <= NORM_TOLERANCE || throw(ArgumentError("$name must be unit rows - a row is $off off unit length"))
    return nothing
end

function _check_sizes(pool_size, bucket_size, bucket_cap, block_size)
    min(pool_size, bucket_size, bucket_cap, block_size) >= 1 ||
        throw(ArgumentError("pool_size, bucket_size, bucket_cap and block_size must be positive"))
    (bucket_size % pool_size == 0 && block_size % bucket_size == 0) ||
        throw(ArgumentError("pool_size $pool_size must divide bucket_size $bucket_size, which must divide block_size $block_size"))
    bucket_cap >= 2 * bucket_size ||
        throw(ArgumentError("bucket_cap must be at least twice bucket_size $bucket_size - got $bucket_cap"))
    return nothing
end

_normalize_rows(a::AbstractMatrix) = a ./ max.(sqrt.(sum(abs2, a; dims = 2)), 1f-30)

"""
    index_vectors(q, k, query_projection, key_projection, rotary_dim) -> (index_queries, index_keys)

The index space of the key-and-query input. `q` (head_dim, S, heads, 1) and `k` (head_dim, S, kv_heads, 1) are the tensors the
attention op receives - after the q/k norm and RoPE, the output gate split off. The first `rotary_dim` dims of each head are dropped
(RoPE touched only those, so the rest are the same before and after it), each kv head's remaining dims sit side by side, and a
group's query heads are summed first. The projections are (kv_heads * (head_dim - rotary_dim), index_dim), and the result is
(S, index_dim) unit rows each.
"""
function index_vectors(q::AbstractArray{Float32,4}, k::AbstractArray{Float32,4}, query_projection::AbstractMatrix{Float32},
                       key_projection::AbstractMatrix{Float32}, rotary_dim::Integer)
    head_dim, S, heads = size(q, 1), size(q, 2), size(q, 3)
    kv_heads = size(k, 3)
    (size(q, 4) == 1 && size(k, 4) == 1) || throw(ArgumentError("q and k must have a batch of 1"))
    (size(k, 1) == head_dim && size(k, 2) == S) || throw(ArgumentError("q $(size(q)) and k $(size(k)) disagree on head_dim or length"))
    heads % kv_heads == 0 || throw(ArgumentError("$heads query heads do not group over $kv_heads kv heads"))
    0 <= rotary_dim < head_dim || throw(ArgumentError("rotary_dim must be in 0:$(head_dim - 1) - got $rotary_dim"))
    width = kv_heads * (head_dim - rotary_dim)
    (size(query_projection, 1) == width && size(key_projection, 1) == width) ||
        throw(ArgumentError("projections must read $width dims - got $(size(query_projection, 1)) and $(size(key_projection, 1))"))
    group = heads ÷ kv_heads
    kept = rotary_dim+1:head_dim
    key_input = reduce(hcat, [permutedims(k[kept, :, h, 1]) for h in 1:kv_heads])
    query_input = reduce(hcat, [permutedims(dropdims(sum(q[kept, :, (h-1)*group+1:h*group, 1]; dims = 3); dims = 3)) for h in 1:kv_heads])
    return _normalize_rows(query_input * query_projection), _normalize_rows(key_input * key_projection)
end

"""
    ward_labels(pools, pool_size, bucket_count, bucket_cap; max_rounds = 256) -> Vector{Int}

Merge one block's pool keys `pools` (P, index_dim) down to `bucket_count` buckets and return each pool's bucket, numbered from 1
in the order of its first pool. A bucket is its key sum, merging two costs the sum of their norms less the norm of their sum, and
each round every pair that is each other's cheapest partner under `bucket_cap` merges into the lower index, cheapest first in the
round that reaches the count. A tie goes to the lower index. Throws if the cap leaves no pair before the count is reached.
"""
function ward_labels(pools::Matrix{Float32}, pool_size::Integer, bucket_count::Integer, bucket_cap::Integer; max_rounds::Integer = 256)
    count_pools = size(pools, 1)
    sums = copy(pools)
    sizes = fill(Float32(pool_size), count_pools)
    alive = trues(count_pools)
    parent = collect(1:count_pools)
    best = zeros(Int, count_pools)
    best_cost = fill(Inf32, count_pools)
    for _ in 1:max_rounds
        merges_left = count(alive) - bucket_count
        merges_left > 0 || break
        gram = sums * sums'
        squares = [gram[i, i] for i in 1:count_pools]
        norms = sqrt.(max.(squares, 0f0))
        fill!(best, 0)
        fill!(best_cost, Inf32)
        for i in 1:count_pools
            alive[i] || continue
            for j in 1:count_pools
                (j != i && alive[j] && sizes[i] + sizes[j] <= bucket_cap) || continue
                merged = sqrt(max(squares[i] + squares[j] + 2 * gram[i, j], 0f0))
                cost = norms[i] + norms[j] - merged
                # Strict, so the lowest j wins a tie
                if cost < best_cost[i]
                    best_cost[i] = cost
                    best[i] = j
                end
            end
        end
        # A pair that is each other's cheapest merges into its lower pool
        mutual = [i for i in 1:count_pools if best_cost[i] < Inf32 && best[best[i]] == i && i < best[i]]
        isempty(mutual) && break
        # Cheapest first, a tie to the lower pool, and no more than the block still needs
        sort!(mutual; by = i -> (best_cost[i], i))
        for i in @view mutual[1:min(merges_left, length(mutual))]
            j = best[i]
            sums[i, :] .+= @view sums[j, :]
            sizes[i] += sizes[j]
            alive[j] = false
            parent[j] = i
        end
    end
    count(alive) <= bucket_count ||
        throw(ArgumentError("a block stopped short of $bucket_count buckets under the cap $bucket_cap"))
    number = cumsum(alive)
    root(p) = (while parent[p] != p
        p = parent[p]
    end; p)
    return [number[root(p)] for p in 1:count_pools]
end

"""
    build_buckets(index_keys, pool_size, bucket_size, bucket_cap, cluster_block_size) -> BucketTable

Cluster every closed cluster block of the unit-row `index_keys` (T, index_dim) into buckets with Ward. A trailing partial
block is still open - its keys are in no bucket, and a read takes them from the window.
"""
function build_buckets(index_keys::Matrix{Float32}, pool_size::Integer, bucket_size::Integer, bucket_cap::Integer,
                       cluster_block_size::Integer)
    _check_unit_rows("index_keys", index_keys)
    _check_sizes(pool_size, bucket_size, bucket_cap, cluster_block_size)
    index_dim = size(index_keys, 2)
    block_count = size(index_keys, 1) ÷ cluster_block_size
    per_block = cluster_block_size ÷ bucket_size
    pools_per_block = cluster_block_size ÷ pool_size
    rows = block_count * per_block
    centroids = zeros(Float32, rows, index_dim)
    members = fill(Int32(-1), rows, bucket_cap)
    sizes = zeros(Int32, rows)
    for block in 0:block_count-1
        first_key = block * cluster_block_size
        pools = zeros(Float32, pools_per_block, index_dim)
        for pool in 1:pools_per_block, k in 1:pool_size
            pools[pool, :] .+= @view index_keys[first_key + (pool - 1) * pool_size + k, :]
        end
        labels = ward_labels(pools, pool_size, per_block, bucket_cap)
        for local_key in 0:cluster_block_size-1
            row = block * per_block + labels[local_key ÷ pool_size + 1]
            sizes[row] += 1
            members[row, sizes[row]] = Int32(first_key + local_key)
            centroids[row, :] .+= @view index_keys[first_key + local_key + 1, :]
        end
    end
    for row in 1:rows
        centroids[row, :] ./= max(norm(@view centroids[row, :]), 1f-30)
    end
    return BucketTable(centroids, members, sizes, per_block, cluster_block_size)
end

"""
    select_from_scores(scores, position, table, top_p, top_k, sink_size, window_size) -> Vector{Int32}

One query's keys from its `scores` of the buckets it may see (the first `length(scores)` of the table). The `top_p` best buckets, a tie
to the lower bucket, expanded best bucket first to their keys outside the sinks and the window, `top_k` at most - then the window, then
the sinks. 0-based ids, the expansion first.
"""
function select_from_scores(scores::AbstractVector{Float32}, position::Integer, table::BucketTable, top_p::Integer, top_k::Integer,
                            sink_size::Integer, window_size::Integer)
    # Best score first - the sort is stable, so a tie keeps the lower bucket. Adding 0 turns a -0.0 into +0.0, which Julia's `isless` orders
    # apart but the oracle's comparison (torch's) and a plain `==` call a tie
    picked = sortperm(scores .+ 0f0; rev = true, alg = MergeSort)
    resize!(picked, min(top_p, length(picked)))
    window_start = max(0, position + 1 - window_size)
    keys = Int32[]
    for bucket in picked
        length(keys) < top_k || break
        for slot in 1:size(table.members, 2)
            key = table.members[bucket, slot]
            (key >= sink_size && key < window_start) || continue
            push!(keys, key)
            length(keys) == top_k && break
        end
    end
    append!(keys, window_start:position)
    append!(keys, 0:min(sink_size, window_start, position + 1)-1)
    return keys
end

"""
    read_selection(index_queries, positions, table, top_p, top_k, sink_size, window_size) -> (selected, lengths)

Select the keys each query reads. A query scores every bucket of the cluster blocks closed before its own, picks the `top_p`
best, a tie going to the lower bucket, and reads their keys best bucket first until `top_k` are held - a key in the sinks or
the window is not counted, because the read takes them anyway. `selected` is (Tq, top_k + window_size + sink_size), 0-based key
ids - the expansion, then the window, then the sinks, -1 padded - and `lengths` the ids each row holds.
"""
function read_selection(index_queries::Matrix{Float32}, positions::AbstractVector{<:Integer}, table::BucketTable,
                        top_p::Integer, top_k::Integer, sink_size::Integer, window_size::Integer)
    _check_unit_rows("index_queries", index_queries)
    size(index_queries, 2) == size(table.centroids, 2) ||
        throw(ArgumentError("index_queries have $(size(index_queries, 2)) dims and the buckets $(size(table.centroids, 2))"))
    length(positions) == size(index_queries, 1) || throw(ArgumentError("positions must be one a query"))
    all(>=(0), positions) || throw(ArgumentError("positions must be non-negative"))
    (min(top_p, top_k, window_size) >= 1 && sink_size >= 0) ||
        throw(ArgumentError("top_p, top_k and window_size must be positive and sink_size non-negative"))
    width = top_k + window_size + sink_size
    selected = fill(Int32(-1), size(index_queries, 1), width)
    lengths = zeros(Int32, size(index_queries, 1))
    rows = size(table.centroids, 1)
    for row in axes(index_queries, 1)
        position = Int(positions[row])
        visible = min(position ÷ table.cluster_block_size * table.block_bucket_count, rows)
        scores = view(table.centroids, 1:visible, :) * view(index_queries, row, :)
        keys = select_from_scores(scores, position, table, top_p, top_k, sink_size, window_size)
        selected[row, 1:length(keys)] .= keys
        lengths[row] = length(keys)
    end
    return selected, lengths
end

end # module
