# decode_attn_chunk.jl -- how does the split-K decode attention depend on its chunk size DA_CH (keys per workgroup)?
#
# DA_CH is a compile-time constant of src/DecodeAttn.jl, so each value is a separate checkout (see decode_attn_chunk.sh); this script times
# `decode_attention_split!` for the running checkout's DA_CH at Qwen3.8-27B's attention shape (D = 256, 24 heads, 4 KV heads, Float16 cache),
# for 1 and 4 new tokens (plain decode; DFlash / MTP verify) at 32k and 128k cached keys, and compares the output with the first run's
# (the reference is written to `refdir` by the first label).
#
# Usage: julia --project=<checkout> benchmarks/decode_attn_chunk.jl <label> <refdir>      (uses the GPU)
using Luminal, AMDGPU, Printf, Statistics, Random, Serialization

label, refdir = ARGS[1], ARGS[2]
mkpath(refdir)
const D, H, KVH = 256, 24, 4
println("== DA_CH = ", Luminal.DA_CH, " (", label, ")")
function timeit(f; reps = 40)
    f(); f(); AMDGPU.synchronize()
    ts = [begin AMDGPU.synchronize(); @elapsed(begin f(); AMDGPU.synchronize() end) end for _ in 1:reps]
    return 1e3 * median(ts)
end
for T in (32768, 131072), S in (1, 4)
    Random.seed!(42)
    q = ROCArray(randn(Float32, D, S, H, 1)); kn = ROCArray(randn(Float32, D, S, KVH, 1)); vn = ROCArray(randn(Float32, D, S, KVH, 1))
    pk = ROCArray(Float16.(randn(Float32, D, T, KVH, 1))); pv = ROCArray(Float16.(randn(Float32, D, T, KVH, 1)))
    posv = ROCArray(Float32[T - S])
    out = AMDGPU.zeros(Float32, D, H, S, 1)
    run() = Luminal.decode_attention_split!(out, q, pk, pv, kn, vn, posv, Float32(1 / sqrt(D)), false)
    ok = try run() catch e; println("  T=$T S=$S: failed: ", sprint(showerror, e)[1:min(end, 120)]); continue end
    ok || (println("  T=$T S=$S: not taken by the split kernel"); continue)
    AMDGPU.synchronize()
    got = Array(out)
    file = joinpath(refdir, "ref_T$(T)_S$(S).jls")
    isfile(file) || serialize(file, got)
    ref = deserialize(file)
    err = maximum(abs.(got .- ref)) / max(maximum(abs.(ref)), 1f-9)
    ms = timeit(run)
    gbs = 2 * T * D * KVH * 2 / (ms * 1e-3) / 1e9           # K and V read once (Float16)
    @printf("  T=%6d S=%d: %7.3f ms/layer   %6.1f GB/s   rel diff vs ref %.1e\n", T, S, ms, gbs, err)
    pk = nothing; pv = nothing; GC.gc()
end
