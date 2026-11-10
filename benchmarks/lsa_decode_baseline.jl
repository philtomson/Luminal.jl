# lsa_decode_baseline.jl -- dense split-K decode attention against the sparse read's parts, per layer, at Qwen3.8-27B's shape.
#
# Dense - `decode_attention_split!` over a T-key Float16 cache (D = 256, 24 query heads, 4 KV heads, one new token).
# Sparse - what a Latent Sparse Attention decode token costs once its keys are chosen: gather the 4612 selected slots
# (4096 expansion + 512 window + 4 sinks) out of the cache with a plain ROCArray index, then run the *same* kernel over that
# contiguous context. Selection itself is not on the GPU yet, so it is reported as headroom - dense minus (gather + attention) is the
# time the selection (q_idx, centroid scan, top-1024) may take before the sparse read stops winning. The centroid scan and a naive host
# top-k are timed as the two parts that are measurable now.
#
# The kernel takes at most 262,144 keys (cld(T, DA_CH) <= 1024), so 512k cannot be timed this way.
#
# With a capture and maps, first checks the gather + kernel path on real data: layer 3's real selection at the last token of a
# captured 16k text, against a CPU fp32 softmax over the same keys, and the dense kernel against CPU dense.
#
# Usage: julia -t auto --project=. benchmarks/lsa_decode_baseline.jl [<capture.jls> <maps_dir>]      (uses the GPU; stop the model server first)
using Luminal, AMDGPU, Printf, Statistics, Random, Serialization, LinearAlgebra, SafeTensors
include(joinpath(@__DIR__, "..", "src", "LatentSparse.jl"))
using .LatentSparse

const D, H, KVH = 256, 24, 4
const TOP_K, TOP_P, SINK, WINDOW = 4096, 1024, 4, 512
const WIDTH = TOP_K + WINDOW + SINK

function timeit(f; reps = 40)
    f(); f(); AMDGPU.synchronize()
    ts = [begin AMDGPU.synchronize(); @elapsed(begin f(); AMDGPU.synchronize() end) end for _ in 1:reps]
    return 1e3 * median(ts)
end

# One token over a (D, n, KVH, 1) Float16 cache that holds `n` valid keys - the new token is already in it
function decode!(out, q, pk, pv, kn, vn, n, scale)
    posv = ROCArray(Float32[n - 1])
    Luminal.decode_attention_split!(out, q, pk, pv, kn, vn, posv, scale, false) || error("the split kernel did not take this shape")
    return out
end

softmax_cols(L) = (E = exp.(L .- maximum(L; dims = 1)); E ./ sum(E; dims = 1))
function cpu_attention(q, k, v, scale, idx)                      # q (D, H), k and v (D, n, KVH) Float32, idx the keys read
    out = zeros(Float32, D, H); group = H ÷ KVH
    for h in 1:KVH
        out[:, (h-1)*group+1:h*group] = v[:, idx, h] * softmax_cols(scale .* (k[:, idx, h]' * q[:, (h-1)*group+1:h*group]))
    end
    return out
end
rel(a, b) = norm(a - b) / norm(b)

function check_real(capture, maps_dir)
    cap = deserialize(capture)
    q, k, v, scale, rotary = cap["q"], cap["k"], cap["v"], cap["scale"], cap["rotary_dim"]
    T = size(k, 2); p = T - 1
    maps = SafeTensors.load_safetensors(joinpath(maps_dir, "qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.safetensors"))
    index_q, index_k = index_vectors(q, k, maps["3.W_Q_idx"], maps["3.W_K_idx"], rotary)
    table = build_buckets(index_k, 4, 8, 16, 512)
    selected, lengths = read_selection(index_q[p+1:p+1, :], [p], table, TOP_P, TOP_K, SINK, WINDOW)
    ids = Int.(selected[1, 1:lengths[1]]) .+ 1
    # The caches the kernel reads are Float16, so the CPU reference reads the same rounded values
    k16, v16 = Float16.(k[:, :, :, 1]), Float16.(v[:, :, :, 1])
    pk, pv = ROCArray(Base.reshape(k16, D, T, KVH, 1)), ROCArray(Base.reshape(v16, D, T, KVH, 1))
    qd = ROCArray(Base.reshape(q[:, p+1:p+1, :, 1], D, 1, H, 1))
    kn, vn = AMDGPU.zeros(Float32, D, 1, KVH, 1), AMDGPU.zeros(Float32, D, 1, KVH, 1)
    out = AMDGPU.zeros(Float32, D, H, 1, 1)
    qh = q[:, p+1, :, 1]
    ref_k, ref_v = Float32.(k16), Float32.(v16)
    dense = cpu_attention(qh, ref_k, ref_v, scale, 1:T)
    sparse = cpu_attention(qh, ref_k, ref_v, scale, ids)
    got_dense = Array(decode!(out, qd, pk, pv, kn, vn, T, scale))[:, :, 1, 1]
    slot = ROCArray(ids)
    gk, gv = pk[:, slot, :, :], pv[:, slot, :, :]
    got_sparse = Array(decode!(out, qd, gk, gv, kn, vn, length(ids), scale))[:, :, 1, 1]
    println("real data - layer 3, token $p of a captured text, $(length(ids)) of $T keys selected")
    @printf("  dense kernel vs CPU dense:                    %.2e\n", rel(got_dense, dense))
    @printf("  gather + kernel vs CPU softmax over the list: %.2e\n", rel(got_sparse, sparse))
    @printf("  the list's own distance from dense:           %.4f\n\n", rel(sparse, dense))
end

function sweep()
    scale = Float32(1 / sqrt(D))
    q = ROCArray(randn(Float32, D, 1, H, 1)); kn = ROCArray(randn(Float32, D, 1, KVH, 1)); vn = ROCArray(randn(Float32, D, 1, KVH, 1))
    out = AMDGPU.zeros(Float32, D, H, 1, 1)
    qi = ROCArray(randn(Float32, 64))
    @printf("%8s %10s %10s %10s %12s %12s %12s %12s\n", "keys T", "dense", "gather", "attn 4612", "gather+attn", "headroom", "centroid", "host top-k")
    @printf("%8s %10s %10s %10s %12s %12s %12s %12s\n", "", "ms", "ms", "ms", "ms", "ms", "scan ms", "ms")
    for T in (16384, 32768, 65536, 131072, 262144)
        Random.seed!(T)
        pk = ROCArray(Float16.(randn(Float32, D, T, KVH, 1))); pv = ROCArray(Float16.(randn(Float32, D, T, KVH, 1)))
        dense = timeit(() -> decode!(out, q, pk, pv, kn, vn, T, scale))

        # A selection's shape - a scatter of expansion keys, then the window and the sinks
        ids = vcat(rand(SINK+1:T-WINDOW, TOP_K), T-WINDOW+1:T, 1:SINK)
        slot = ROCArray(ids)
        gather = timeit(() -> (pk[:, slot, :, :]; pv[:, slot, :, :]))
        gk, gv = pk[:, slot, :, :], pv[:, slot, :, :]
        attn = timeit(() -> decode!(out, q, gk, gv, kn, vn, WIDTH, scale))
        both = timeit(() -> (gk = pk[:, slot, :, :]; gv = pv[:, slot, :, :]; decode!(out, q, gk, gv, kn, vn, WIDTH, scale)))

        # The centroid scan over T / 8 buckets, and a naive selection of the best 1024 of its scores on the host
        centroids = ROCArray(randn(Float32, T ÷ 8, 64))
        scan = timeit(() -> centroids * qi)
        host = timeit(() -> (s = Array(centroids * qi); partialsortperm(s, 1:TOP_P; rev = true)); reps = 20)

        @printf("%8d %10.3f %10.3f %10.3f %12.3f %12.3f %12.3f %12.3f\n", T, dense, gather, attn, both, dense - both, scan, host)
        pk = pv = gk = gv = nothing; GC.gc()
    end
end

length(ARGS) >= 2 && check_real(ARGS[1], ARGS[2])
sweep()
