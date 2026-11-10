# lsa_gather_attention.jl -- stage 1 of the LSA port: decode attention over a selected key set, with no new kernel.
#
# One captured layer's K and V stay on the GPU as (kv_heads * head_dim, slots) matrices (PagedAttention's layout). A decode token
# reads the slots LatentSparse selected through `gather_rows` and runs `grouped_query_attention` under a bias that masks the
# list's padding. Checked against a CPU fp32 softmax over the same keys, and against dense attention - exactly, when the list
# holds every key. Then timed.
#
# Two ways to gather are run. Luminal's generic `gather_rows` is correct but takes about a second for one 1024 x 4612 gather, so it
# is kept as the reference path and timed three times only. The fast path gathers outside the graph with a plain ROCArray index
# (about 0.15 ms) and feeds the context to the same attention graph.
#
# Usage: julia -t auto --project=. benchmarks/lsa_gather_attention.jl <capture.jls> <maps_dir> [top_k=4096] [top_p=1024]
#   <capture.jls> from lsa_capture_layer3.jl, <maps_dir> the D-LSA maps as safetensors. Uses the GPU (stop the model server first).

using Luminal, Luminal.PagedAttention
using AMDGPU, Serialization, SafeTensors, Printf, Statistics, LinearAlgebra
include(joinpath(@__DIR__, "..", "src", "LatentSparse.jl"))
using .LatentSparse

const POOL, BUCKET, CAP, BLOCK, SINK, WINDOW, LAYER = 4, 8, 16, 512, 4, 512, 3

function softmax_cols(L)
    E = exp.(L .- maximum(L; dims = 1))
    return E ./ sum(E; dims = 1)
end

# One token's attention output (head_dim, heads) over the keys `idx` (1-based slots), or over every key up to it
function cpu_attention(q, k, v, scale, p, idx)
    D, H, KV = size(q, 1), size(q, 3), size(k, 3)
    group = H ÷ KV
    out = zeros(Float32, D, H)
    for h in 1:KV
        rows = idx === nothing ? (1:p+1) : idx
        K, V = k[:, rows, h, 1], v[:, rows, h, 1]
        Q = q[:, p+1, (h-1)*group+1:h*group, 1]
        out[:, (h-1)*group+1:h*group] = V * softmax_cols(scale .* (K' * Q))
    end
    return out
end

rel(a, b) = norm(a - b) / norm(b)

function main()
    cap = deserialize(ARGS[1]); maps_dir = ARGS[2]
    top_k = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 4096
    top_p = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1024
    q, k, v, scale, rotary = cap["q"], cap["k"], cap["v"], cap["scale"], cap["rotary_dim"]
    D, T, H, KV = size(q, 1), size(q, 2), size(q, 3), size(k, 3)
    width = top_k + WINDOW + SINK
    println(cap["weights"], ": $T slots, $H query heads on $KV kv heads of $D dims, read width $width")

    # The selection, from the distilled map
    maps = SafeTensors.load_safetensors(joinpath(maps_dir, "qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.safetensors"))
    index_q, index_k = index_vectors(q, k, maps["$LAYER.W_Q_idx"], maps["$LAYER.W_K_idx"], rotary)
    table = build_buckets(index_k, POOL, BUCKET, CAP, BLOCK)

    # The graph - the caches are resident tensors, the token's query, slot list and pad bias are the inputs
    device = get_device()
    g = Graph()
    key_cache, value_cache = tensor(g, [KV * D, T]), tensor(g, [KV * D, T])
    query, slots, bias = tensor(g, [H * D, 1]), tensor(g, [width]), tensor(g, [width, 1])
    context_keys, context_values = gather_rows(key_cache, slots), gather_rows(value_cache, slots)
    attended = grouped_query_attention(query, context_keys, context_values, bias, AttentionGeometry(H, KV, D, scale))
    cache_matrix(x) = Base.reshape(permutedims(x[:, :, :, 1], (1, 3, 2)), KV * D, T)      # [d, kv, slot] -> (kv * d, slot)
    g.tensors[(key_cache.id, 1)] = to_device(cache_matrix(k), device)
    g.tensors[(value_cache.id, 1)] = to_device(cache_matrix(v), device)
    t_compile = @elapsed exec = compile(g; device = device, retain = [attended.id])
    @printf("compiled in %.1f s\n", t_compile)

    function run_token(p, ids)
        n = length(ids)
        slot_list = zeros(Float32, width); slot_list[1:n] .= ids                    # padding reads slot 0 and is masked
        pad_bias = fill(MASKED_SCORE, width, 1); pad_bias[1:n, 1] .= 0f0
        pins = Dict{Int,Any}(query.id => Base.reshape(q[:, p+1, :, 1], H * D, 1), slots.id => slot_list, bias.id => pad_bias)
        return Base.reshape(Array{Float32}(exec(pins; device = device)[attended.id]), D, H)
    end

    # Fast path - the attention graph alone, over a context gathered with a plain ROCArray index
    g2 = Graph()
    query2, context_keys2, context_values2, bias2 = tensor(g2, [H * D, 1]), tensor(g2, [KV * D, width]), tensor(g2, [KV * D, width]), tensor(g2, [width, 1])
    attended2 = grouped_query_attention(query2, context_keys2, context_values2, bias2, AttentionGeometry(H, KV, D, scale))
    exec2 = compile(g2; device = device, retain = [attended2.id])
    key_dev, value_dev = ROCArray(cache_matrix(k)), ROCArray(cache_matrix(v))

    function run_token_fast(p, ids)
        n = length(ids)
        slot_list = ones(Int, width); slot_list[1:n] .= ids .+ 1                    # padding reads slot 1 and is masked
        pad_bias = fill(MASKED_SCORE, width, 1); pad_bias[1:n, 1] .= 0f0
        slot_dev = ROCArray(slot_list)
        pins = Dict{Int,Any}(query2.id => Base.reshape(q[:, p+1, :, 1], H * D, 1), bias2.id => pad_bias,
                             context_keys2.id => key_dev[:, slot_dev], context_values2.id => value_dev[:, slot_dev])
        return Base.reshape(Array{Float32}(exec2(pins; device = device)[attended2.id]), D, H)
    end

    println("\nanchor - the list holds every key up to the token, so the read must equal dense attention")
    for p in (0, 1000, 2047, width - 1)
        dense = cpu_attention(q, k, v, scale, p, nothing)
        @printf("  token %5d: %5d keys, relative difference from dense %.2e (gather_rows), %.2e (ROCArray gather)\n", p, p + 1,
                rel(run_token(p, collect(0:p)), dense), rel(run_token_fast(p, collect(0:p)), dense))
    end

    println("\nselected - the LatentSparse list ($top_k expansion keys at most, window $WINDOW, sinks $SINK)")
    positions = [2048, 4700, 6000, 9000, 12000, T - 1]
    selected, lengths = read_selection(index_q[positions .+ 1, :], positions, table, top_p, top_k, SINK, WINDOW)
    for (r, p) in enumerate(positions)
        ids = Int.(selected[r, 1:lengths[r]])
        sparse = cpu_attention(q, k, v, scale, p, ids .+ 1)
        dense = cpu_attention(q, k, v, scale, p, nothing)
        slow, fast = run_token(p, ids), run_token_fast(p, ids)
        @printf("  token %5d: %5d keys (%4.1f%%), vs CPU sparse %.2e (gather_rows) %.2e (ROCArray), vs dense %.4f (CPU sparse vs dense %.4f)\n",
                p, length(ids), 100 * length(ids) / (p + 1), rel(slow, sparse), rel(fast, sparse), rel(fast, dense), rel(sparse, dense))
    end

    println("\ntiming - the last token, caches resident, median after warm-up; each call includes the host round trip of query, slots and output")
    p = T - 1
    ids = Int.(selected[end, 1:lengths[end]])
    run_token(p, ids)
    slow = [@elapsed(run_token(p, ids)) for _ in 1:3]
    @printf("  gather_rows in the graph, %d keys: %9.2f ms per layer per token (median of 3)\n", length(ids), 1000 * median(slow))
    for _ in 1:5; run_token_fast(p, ids); end
    fast = [@elapsed(run_token_fast(p, ids)) for _ in 1:30]
    @printf("  ROCArray gather + attention graph: %9.2f ms per layer per token (median of 30)\n", 1000 * median(fast))
end

main()
