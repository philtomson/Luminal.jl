# lsa_gather_kernel.jl -- the fused gather decode attention (src/DecodeAttnGather.jl): correctness, then time against dense.
#
# Correctness - on random Float16 caches, `decode_attention_gather!` over a list of slots against (a) the same list gathered into a
# contiguous cache and run through the existing `decode_attention_split!` - the same chunks in the same order, so the results should agree to
# rounding - and (b) a CPU fp32 softmax over the same keys. List lengths straddle the kernel's 256-entry chunks, and a batch of two has two
# lengths. With a capture and maps, also the real layer-3 selection at the last token of a captured text.
# Timing - dense, gather + kernel (the stage 1 path) and fused, per layer per token, at Qwen3.8-27B's shape.
#
# Usage: julia -t auto --project=. benchmarks/lsa_gather_kernel.jl [<capture.jls> <maps_dir>]      (uses the GPU; stop the model server first)
using Luminal, AMDGPU, Printf, Statistics, Random, Serialization, LinearAlgebra, SafeTensors
isdefined(Luminal, :decode_attention_gather!) || Base.include(Luminal, joinpath(@__DIR__, "..", "src", "DecodeAttnGather.jl"))
include(joinpath(@__DIR__, "..", "src", "LatentSparse.jl"))
using .LatentSparse

const D, H, KVH = 256, 24, 4
const TOP_K, TOP_P, SINK, WINDOW = 4096, 1024, 4, 512
const WIDTH = TOP_K + WINDOW + SINK
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

# The existing path - gather the slots into a contiguous cache, run the dense kernel over it as a cache of `count` keys
function gather_then_dense(q, pk, pv, slots, count)
    gk, gv = pk[:, slots, :, :], pv[:, slots, :, :]
    out = AMDGPU.zeros(Float32, D, H, 1, 1)
    z = AMDGPU.zeros(Float32, D, 1, KVH, 1)
    Luminal.decode_attention_split!(out, q, gk, gv, z, z, ROCArray(Float32[count - 1]), SCALE, false) || error("dense kernel declined")
    return Array(out)[:, :, 1, 1]
end

function fused(q, pk, pv, ids, counts)
    B = size(ids, 2)
    out = AMDGPU.zeros(Float32, D, H, 1, B)
    Luminal.decode_attention_gather!(out, q, pk, pv, ROCArray(Int32.(ids)), ROCArray(Float32.(counts .- 1)), SCALE) || error("fused kernel declined")
    return Array(out)
end

function correctness()
    println("correctness - random Float16 caches, T = 20000, list width $WIDTH")
    Random.seed!(1)
    T = 20000
    k16, v16 = Float16.(randn(Float32, D, T, KVH, 2)), Float16.(randn(Float32, D, T, KVH, 2))
    pk, pv = ROCArray(k16), ROCArray(v16)
    q = randn(Float32, D, 1, H, 2); qd = ROCArray(q)
    kf, vf = Float32.(k16), Float32.(v16)
    worst = 0.0
    for count in (1, 2, 255, 256, 257, 1000, 4096, WIDTH)
        ids = fill(-1, WIDTH, 2)
        ids[1:count, 1] .= randperm(T)[1:count] .- 1                       # 0-based, distinct, unsorted
        ids[1:count, 2] .= randperm(T)[1:count] .- 1
        got = fused(qd, pk, pv, ids, [count, count])
        for b in 1:2
            slots = ROCArray(ids[1:count, b] .+ 1)
            qb = ROCArray(q[:, :, :, b:b])
            old = gather_then_dense(qb, pk[:, :, :, b:b], pv[:, :, :, b:b], slots, count)
            ref = cpu_attention(q[:, 1, :, b], kf[:, :, :, b], vf[:, :, :, b], ids[1:count, b] .+ 1)
            e_old, e_cpu = rel(got[:, :, 1, b], old), rel(got[:, :, 1, b], ref)
            worst = max(worst, e_cpu)
            b == 1 && @printf("  count %5d: fused vs gather+dense kernel %.2e, vs CPU softmax %.2e\n", count, e_old, e_cpu)
        end
    end
    # A batch of two with different counts in one launch
    ids = fill(-1, WIDTH, 2); ids[1:300, 1] .= randperm(T)[1:300] .- 1; ids[1:WIDTH, 2] .= randperm(T)[1:WIDTH] .- 1
    got = fused(qd, pk, pv, ids, [300, WIDTH])
    for (b, n) in ((1, 300), (2, WIDTH))
        e = rel(got[:, :, 1, b], cpu_attention(q[:, 1, :, b], kf[:, :, :, b], vf[:, :, :, b], ids[1:n, b] .+ 1))
        worst = max(worst, e)
        @printf("  batch of two, sequence %d with %4d keys: vs CPU softmax %.2e\n", b, n, e)
    end
    @printf("  worst vs CPU: %.2e\n\n", worst)
end

function check_real(capture, maps_dir)
    cap = deserialize(capture)
    q, k, v, rotary = cap["q"], cap["k"], cap["v"], cap["rotary_dim"]
    T = size(k, 2); p = T - 1
    maps = SafeTensors.load_safetensors(joinpath(maps_dir, "qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.safetensors"))
    index_q, index_k = index_vectors(q, k, maps["3.W_Q_idx"], maps["3.W_K_idx"], rotary)
    table = build_buckets(index_k, 4, 8, 16, 512)
    selected, lengths = read_selection(index_q[p+1:p+1, :], [p], table, TOP_P, TOP_K, SINK, WINDOW)
    ids = Base.reshape(Int32.(selected[1, :]), :, 1)
    count = lengths[1]
    k16, v16 = Float16.(k), Float16.(v)
    got = fused(ROCArray(q[:, p+1:p+1, :, :]), ROCArray(k16), ROCArray(v16), ids, [count])
    ref = cpu_attention(q[:, p+1, :, 1], Float32.(k16[:, :, :, 1]), Float32.(v16[:, :, :, 1]), Int.(ids[1:count, 1]) .+ 1)
    dense = cpu_attention(q[:, p+1, :, 1], Float32.(k16[:, :, :, 1]), Float32.(v16[:, :, :, 1]), 1:T)
    println("real data - layer 3, token $p of a captured text, $count of $T keys selected")
    @printf("  fused vs CPU softmax over the list: %.2e   (the list's own distance from dense: %.4f)\n\n", rel(got[:, :, 1, 1], ref), rel(ref, dense))
end

function sweep()
    q = ROCArray(randn(Float32, D, 1, H, 1)); kn = ROCArray(randn(Float32, D, 1, KVH, 1)); vn = ROCArray(randn(Float32, D, 1, KVH, 1))
    out = AMDGPU.zeros(Float32, D, H, 1, 1)
    @printf("%8s %10s %14s %10s %12s\n", "keys T", "dense", "gather+kernel", "fused", "fused vs dense")
    @printf("%8s %10s %14s %10s %12s\n", "", "ms", "ms", "ms", "")
    for T in (16384, 32768, 65536, 131072, 262144)
        Random.seed!(T)
        pk = ROCArray(Float16.(randn(Float32, D, T, KVH, 1))); pv = ROCArray(Float16.(randn(Float32, D, T, KVH, 1)))
        dense = timeit(() -> Luminal.decode_attention_split!(out, q, pk, pv, kn, vn, ROCArray(Float32[T - 1]), SCALE, false))
        ids = vcat(rand(SINK+1:T-WINDOW, TOP_K), T-WINDOW+1:T, 1:SINK) .- 1       # 0-based, a selection's shape
        idsd = ROCArray(Base.reshape(Int32.(ids), :, 1)); slots = ROCArray(ids .+ 1)
        last = ROCArray(Float32[WIDTH - 1])
        old = timeit(() -> (gk = pk[:, slots, :, :]; gv = pv[:, slots, :, :];
                            Luminal.decode_attention_split!(out, q, gk, gv, kn, vn, last, SCALE, false)))
        new = timeit(() -> Luminal.decode_attention_gather!(out, q, pk, pv, idsd, last, SCALE))
        @printf("%8d %10.3f %14.3f %10.3f %11.1fx\n", T, dense, old, new, dense / new)
        pk = pv = nothing; GC.gc()
    end
end

correctness()
length(ARGS) >= 2 && check_real(ARGS[1], ARGS[2])
sweep()
