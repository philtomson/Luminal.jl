# flash_window.jl -- sliding-window prefill attention (flash_prefill! with a window) against full causal attention
#
# Times `flash_prefill!` for several windows at long sequences (the kernels skip the key tiles before a block's first visible key, so a
# window of W should cost about W / S of the full causal time at large S), and checks a long-sequence window against the host reference
# on a few heads (so that the tile skipping is exercised with a large first tile).
#
# Usage: julia --project=. benchmarks/flash_window.jl      (stop the model server first: this uses the GPU)

using Luminal, AMDGPU, Printf, Statistics

timed(f, n) = (t = time(); for _ in 1:n; f(); end; AMDGPU.synchronize(); (time() - t) / n)
relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

println("== correctness at a long sequence (S = 2048, 2 heads sharing a KV head)")
for D in (64, 128, 256), window in (256, 700)
    S, H, KVH = 2048, 2, 1
    q = randn(Float32, D, S, H, 1); k = randn(Float32, D, S, KVH, 1); v = randn(Float32, D, S, KVH, 1)
    out = AMDGPU.zeros(Float32, D, S, H, 1)
    Luminal.flash_prefill!(out, ROCArray(q), ROCArray(k), ROCArray(v), Float32(1 / sqrt(D)), window)
    @printf("  D=%-3d window=%-4d  rel err vs host reference %.1e\n", D, window, relerr(Array(out), Luminal.causal_attention_ref(q, k, v, Float32(1 / sqrt(D)), window)))
end

println("\n== time (ms) per call, 8 heads / 4 KV heads")
for D in (256, 128, 64), S in (2048, 8192, 16384)
    H, KVH = 8, 4
    q = ROCArray(randn(Float32, D, S, H, 1)); k = ROCArray(randn(Float32, D, S, KVH, 1)); v = ROCArray(randn(Float32, D, S, KVH, 1))
    out = AMDGPU.zeros(Float32, D, S, H, 1)
    sc = Float32(1 / sqrt(D))
    ws = (0, 4096, 1024, 256)
    fs = [() -> Luminal.flash_prefill!(out, q, k, v, sc, w) for w in ws]
    for f in fs; f(); f(); end
    AMDGPU.synchronize()
    ts = [Float64[] for _ in ws]
    for _ in 1:5, (i, f) in enumerate(fs); push!(ts[i], 1e3 * timed(f, 3)); end
    m = [median(t) for t in ts]
    @printf("  D=%-3d S=%-5d  %s\n", D, S, join((@sprintf("window %-5s %8.2f ms (%.2fx)", w == 0 ? "full" : string(w), m[i], m[1] / m[i]) for (i, w) in enumerate(ws)), "   "))
end
