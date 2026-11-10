# lsa_select_kernel.jl -- Latent Sparse Attention's selection on the GPU (src/LatentSparseSelect.jl): correctness, then time.
#
# A  the pick kernel on scores full of ties, exactly against the CPU reference `select_from_scores` - tie-breaking, the cut at top_k,
#    top_p larger than the visible buckets, no visible bucket at all.
# B  with LSA_FIXTURES (the torch oracle's fixtures, see make_fixtures.py): the pick kernel on CPU-computed scores - exact - and the
#    whole GPU path (index query given, scores computed on the GPU) against the oracle's lists, where two buckets whose scores differ
#    by under an fp32 ulp may swap.
# C  with a capture and maps: layer 3's real queries through all three kernels, then the fused attention, against the CPU.
# D  timing per layer per decode token, each kernel and the whole selection, then the fused attention, then dense where the dense kernel
#    reaches (up to 262,144 keys).
#
# Usage: LSA_FIXTURES=<dir> julia -t auto --project=. benchmarks/lsa_select_kernel.jl [<capture.jls> <maps_dir>]      (uses the GPU)
using Luminal, AMDGPU, Printf, Statistics, Random, Serialization, LinearAlgebra, SafeTensors
isdefined(Luminal, :decode_attention_gather!) || Base.include(Luminal, joinpath(@__DIR__, "..", "src", "DecodeAttnGather.jl"))
isdefined(Luminal, :lsa_select!) || Base.include(Luminal, joinpath(@__DIR__, "..", "src", "LatentSparseSelect.jl"))
include(joinpath(@__DIR__, "..", "src", "LatentSparse.jl"))
using .LatentSparse

const D, H, KVH = 256, 24, 4
const POOL, BUCKET, CAP, BLOCK, SINK, WINDOW = 4, 8, 16, 512, 4, 512
const TOP_K, TOP_P = 4096, 1024
const WIDTH = TOP_K + WINDOW + SINK
const PER_BLOCK = BLOCK ÷ BUCKET
const SCALE = Float32(1 / sqrt(D))

function timeit(f; reps = 40)
    f(); f(); AMDGPU.synchronize()
    ts = [begin AMDGPU.synchronize(); @elapsed(begin f(); AMDGPU.synchronize() end) end for _ in 1:reps]
    return 1e3 * median(ts)
end

softmax_cols(L) = (E = exp.(L .- maximum(L; dims = 1)); E ./ sum(E; dims = 1))
function cpu_attention(q, k, v, idx)                      # q (D, H), k and v (D, T, KVH) Float32, idx the slots read (1-based)
    out = zeros(Float32, D, H); group = H ÷ KVH
    for h in 1:KVH
        out[:, (h-1)*group+1:h*group] = v[:, idx, h] * softmax_cols(SCALE .* (k[:, idx, h]' * q[:, (h-1)*group+1:h*group]))
    end
    return out
end
rel(a, b) = norm(a - b) / norm(b)

unit_rows(rng, n, d) = (a = randn(rng, Float32, n, d); a ./ sqrt.(sum(abs2, a; dims = 2)))
members_device(table) = ROCArray(permutedims(table.members))               # (bucket_cap, buckets)

struct Out
    ids::ROCArray{Int32,2}
    lastpos::ROCArray{Float32,1}
end
Out() = Out(AMDGPU.zeros(Int32, WIDTH, 1), AMDGPU.zeros(Float32, 1))
list(o::Out) = (n = Int(Array(o.lastpos)[1]) + 1; Int32.(Array(o.ids)[1:n, 1]))
posv(p) = ROCArray(Float32[p])

pick!(o, scores_dev, members_dev, p; top_p = TOP_P, top_k = TOP_K) =
    Luminal.lsa_pick_expand!(o.ids, o.lastpos, scores_dev, members_dev, posv(p); top_p, top_k, sink_size = SINK, window_size = WINDOW,
                             cluster_block_size = BLOCK, block_bucket_count = PER_BLOCK)

function part_a()
    println("A  pick kernel against the CPU reference, on scores full of ties")
    rng = MersenneTwister(3)
    keys = unit_rows(rng, 8192, 64)
    table = build_buckets(keys, POOL, BUCKET, CAP, BLOCK)
    nb = size(table.members, 1)
    members_dev = members_device(table)
    o = Out(); bad = 0; total = 0
    for levels in (2, 4, 16, 1_000_000), (top_p, top_k) in ((16, 64), (100, 4096), (1024, 4096), (7, 5))
        scores = round.(randn(rng, Float32, nb) .* levels) ./ levels          # at most a few distinct values - ties everywhere
        scores_dev = ROCArray(scores)
        for p in (0, 3, 511, 512, 700, 1500, 4097, 8191)
            visible = min(p ÷ BLOCK * PER_BLOCK, nb)
            pick!(o, scores_dev, members_dev, p; top_p, top_k)
            ref = select_from_scores(scores[1:visible], p, table, top_p, top_k, SINK, WINDOW)
            total += 1
            list(o) == ref || (bad += 1; bad <= 3 && @printf("   MISMATCH levels %d top_p %d top_k %d position %d: %d vs %d ids\n", levels, top_p, top_k, p, length(list(o)), length(ref)))
        end
    end
    @printf("   %d of %d cases identical to the CPU lists\n\n", total - bad, total)
end

function part_b()
    dir = get(ENV, "LSA_FIXTURES", "")
    for name in ("small", "design")
        path = joinpath(dir, "$name.safetensors")
        (isempty(dir) || !isfile(path)) && (println("B  skipping fixture $name - set LSA_FIXTURES\n"); continue)
        f = SafeTensors.load_safetensors(path)
        pool, bucket, cap, block, sink, window, top_p, top_k, _ = Int.(f["params"])
        queries, positions = f["index_queries"], Int.(f["positions"])
        table = LatentSparse.BucketTable(f["bucket_centroids"], f["bucket_members"], f["bucket_sizes"], block ÷ bucket, block)
        nb = size(table.members, 1)
        centroids_dev, members_dev = ROCArray(table.centroids), ROCArray(permutedims(table.members))
        scores_dev = AMDGPU.zeros(Float32, nb); qidx_dev = AMDGPU.zeros(Float32, 64)
        ids = AMDGPU.zeros(Int32, top_k + window + sink, 1); lastpos = AMDGPU.zeros(Float32, 1)
        exact_cpu_scores = 0; exact_gpu = 0; same_set = 0; vs_oracle = 0
        rows = 1:length(positions)
        for r in rows
            p = positions[r]; visible = min(p ÷ block * (block ÷ bucket), nb)
            # The CPU's own scores, so the ties and the cut are tested apart from the scoring
            cpu_scores = zeros(Float32, nb); cpu_scores[1:visible] = view(table.centroids, 1:visible, :) * view(queries, r, :)
            Luminal.lsa_pick_expand!(ids, lastpos, ROCArray(cpu_scores), members_dev, posv(p); top_p, top_k, sink_size = sink,
                                     window_size = window, cluster_block_size = block, block_bucket_count = block ÷ bucket)
            n = Int(Array(lastpos)[1]) + 1; got = Int32.(Array(ids)[1:n, 1])
            ref = select_from_scores(cpu_scores[1:visible], p, table, top_p, top_k, sink, window)
            exact_cpu_scores += got == ref
            # The whole GPU path from the index query
            copyto!(qidx_dev, queries[r, :])
            Luminal.lsa_score!(scores_dev, centroids_dev, qidx_dev, posv(p); cluster_block_size = block, block_bucket_count = block ÷ bucket)
            Luminal.lsa_pick_expand!(ids, lastpos, scores_dev, members_dev, posv(p); top_p, top_k, sink_size = sink, window_size = window,
                                     cluster_block_size = block, block_bucket_count = block ÷ bucket)
            n = Int(Array(lastpos)[1]) + 1; gpu = Int32.(Array(ids)[1:n, 1])
            oracle = Int32.(f["selected"][r, 1:f["lengths"][r]])
            exact_gpu += gpu == ref; same_set += sort(gpu) == sort(ref); vs_oracle += gpu == oracle
        end
        N = length(rows)
        @printf("B  fixture %s (%d queries, top_p %d, top_k %d)\n", name, N, top_p, top_k)
        @printf("   pick kernel on the CPU's scores: %d of %d identical to the CPU lists\n", exact_cpu_scores, N)
        @printf("   scores on the GPU too: %d identical to the CPU lists, %d with the same key set, %d identical to the torch oracle's\n\n",
                exact_gpu, same_set, vs_oracle)
    end
end

function part_c(capture, maps_dir)
    cap = deserialize(capture)
    q, k, v, rotary = cap["q"], cap["k"], cap["v"], cap["rotary_dim"]
    T = size(k, 2)
    maps = SafeTensors.load_safetensors(joinpath(maps_dir, "qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.safetensors"))
    index_q, index_k = index_vectors(q, k, maps["3.W_Q_idx"], maps["3.W_K_idx"], rotary)
    table = build_buckets(index_k, POOL, BUCKET, CAP, BLOCK)
    centroids_dev, members_dev = ROCArray(table.centroids), members_device(table)
    wq_t = ROCArray(permutedims(maps["3.W_Q_idx"]))
    k16, v16 = Float16.(k), Float16.(v)
    pk, pv = ROCArray(k16), ROCArray(v16)
    out = AMDGPU.zeros(Float32, D, H, 1, 1)
    qidx = AMDGPU.zeros(Float32, 64); scores = AMDGPU.zeros(Float32, size(table.centroids, 1)); o = Out()
    println("C  layer 3, real data - $T tokens")
    worst_q = 0.0; exact = 0; same = 0; attn_err = Float64[]; list_err = Float64[]
    positions = round.(Int, range(1024, T - 1, length = 24))
    for p in positions
        qd = ROCArray(q[:, p+1:p+1, :, :])
        Luminal.lsa_select!(o.ids, o.lastpos, qidx, scores, qd, wq_t, centroids_dev, members_dev, posv(p); rotary_dim = rotary, top_p = TOP_P,
                            top_k = TOP_K, sink_size = SINK, window_size = WINDOW, cluster_block_size = BLOCK, block_bucket_count = PER_BLOCK)
        worst_q = max(worst_q, rel(Array(qidx), index_q[p+1, :]))
        got = list(o)
        sel, len = read_selection(index_q[p+1:p+1, :], [p], table, TOP_P, TOP_K, SINK, WINDOW)
        ref = Int32.(sel[1, 1:len[1]])
        exact += got == ref; same += sort(got) == sort(ref)
        # The fused attention over the GPU's own list, against a CPU softmax over the CPU's list
        Luminal.decode_attention_gather!(out, qd, pk, pv, o.ids, o.lastpos, SCALE) || error("fused kernel declined")
        cpu = cpu_attention(q[:, p+1, :, 1], Float32.(k16[:, :, :, 1]), Float32.(v16[:, :, :, 1]), Int.(ref) .+ 1)
        push!(attn_err, rel(Array(out)[:, :, 1, 1], cpu))
    end
    @printf("   index query vs CPU: worst relative difference %.1e\n", worst_q)
    @printf("   lists, %d positions: %d identical to the CPU's, %d with the same key set\n", length(positions), exact, same)
    @printf("   fused attention over the GPU's list vs CPU softmax over the CPU's: median %.1e, worst %.1e\n\n", median(attn_err), maximum(attn_err))
end

# A selection-shaped bucket table for T keys - every cluster block's keys split at random into buckets of 8
function synthetic_table(rng, T)
    nb = T ÷ BLOCK * PER_BLOCK
    centroids = unit_rows(rng, nb, 64)
    members = fill(Int32(-1), CAP, nb)
    for blk in 0:T÷BLOCK-1
        perm = randperm(rng, BLOCK) .- 1 .+ blk * BLOCK
        for b in 0:PER_BLOCK-1
            members[1:BUCKET, blk * PER_BLOCK + b + 1] = perm[b*BUCKET+1:(b+1)*BUCKET]
        end
    end
    return centroids, members
end

function part_d()
    println("D  timing per layer per decode token (median ms, caches and tables resident)")
    @printf("%9s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n", "keys T", "dense", "idx query", "score", "pick", "select", "fused", "sel+fused", "vs dense", "buckets")
    rng = MersenneTwister(5)
    q = ROCArray(randn(Float32, D, 1, H, 1)); kn = ROCArray(randn(Float32, D, 1, KVH, 1)); vn = ROCArray(randn(Float32, D, 1, KVH, 1))
    wq_t = ROCArray(randn(Float32, 64, KVH * (D - 64)) ./ 30)
    out = AMDGPU.zeros(Float32, D, H, 1, 1)
    for T in (16384, 32768, 65536, 131072, 262144, 524288, 1048576)
        centroids, members = synthetic_table(rng, T)
        centroids_dev, members_dev = ROCArray(centroids), ROCArray(members)
        pk = ROCArray(Float16.(randn(Float32, D, T, KVH, 1))); pv = ROCArray(Float16.(randn(Float32, D, T, KVH, 1)))
        p = T - 1; pos = posv(p)
        qidx = AMDGPU.zeros(Float32, 64); scores = AMDGPU.zeros(Float32, size(centroids, 1)); o = Out()
        kw = (; top_p = TOP_P, top_k = TOP_K, sink_size = SINK, window_size = WINDOW, cluster_block_size = BLOCK, block_bucket_count = PER_BLOCK)
        t_dense = T <= 262144 ? timeit(() -> Luminal.decode_attention_split!(out, q, pk, pv, kn, vn, pos, SCALE, false)) : NaN
        t_idx = timeit(() -> Luminal.lsa_index_query!(qidx, q, wq_t; rotary_dim = 64))
        t_score = timeit(() -> Luminal.lsa_score!(scores, centroids_dev, qidx, pos; cluster_block_size = BLOCK, block_bucket_count = PER_BLOCK))
        t_pick = timeit(() -> Luminal.lsa_pick_expand!(o.ids, o.lastpos, scores, members_dev, pos; kw...))
        t_sel = timeit(() -> Luminal.lsa_select!(o.ids, o.lastpos, qidx, scores, q, wq_t, centroids_dev, members_dev, pos; rotary_dim = 64, kw...))
        t_fused = timeit(() -> Luminal.decode_attention_gather!(out, q, pk, pv, o.ids, o.lastpos, SCALE))
        t_all = timeit(() -> (Luminal.lsa_select!(o.ids, o.lastpos, qidx, scores, q, wq_t, centroids_dev, members_dev, pos; rotary_dim = 64, kw...);
                              Luminal.decode_attention_gather!(out, q, pk, pv, o.ids, o.lastpos, SCALE)))
        dense_s = isnan(t_dense) ? "      -" : @sprintf("%7.3f", t_dense)
        ratio = isnan(t_dense) ? "      -" : @sprintf("%6.1fx", t_dense / t_all)
        @printf("%9d %9s %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9s %9d\n", T, dense_s, t_idx, t_score, t_pick, t_sel, t_fused, t_all, ratio, size(centroids, 1))
        pk = pv = centroids_dev = members_dev = nothing; GC.gc()
    end
end

part_a()
part_b()
length(ARGS) >= 2 && part_c(ARGS[1], ARGS[2])
part_d()
