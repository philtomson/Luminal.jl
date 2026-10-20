# attn_unmasked.jl -- is FlashAttentionOp (the only fused op for unmasked attention) faster than the composite form?
#
# Unmasked multi-head self-attention q, k, v (D, S, H, B), as Whisper's encoder spells it (scores (Sk, Sq, H, B), the fused SoftmaxOp,
# V * probs) against one FlashAttentionOp(scale, false) node, on the GPU. Median ms over interleaved rounds, and the max difference from a
# Float64 CPU reference on a small slice of the heads.
#
# Usage: julia --project=. benchmarks/attn_unmasked.jl      (stop the model server first: this uses the GPU)

using Luminal
using AMDGPU, Printf, Statistics

const DEV = get_device()
timed(run, n) = (t = time(); for _ in 1:n; run(); end; (time() - t) / n)
nsteps(run; budget=0.3) = (run(); run(); t = (t0 = time(); run(); time() - t0); clamp(round(Int, budget / max(t, 1e-6)), 1, 100))
function interleaved(runs, rounds)
    ns = [nsteps(r) for r in runs]
    ts = [Float64[] for _ in runs]
    for _ in 1:rounds, (i, r) in enumerate(runs)
        push!(ts[i], timed(r, ns[i]))
    end
    return [1e3 * median(t) for t in ts]
end

function composite(q, k, v)
    d = Luminal.realized_dims(q.shape)[1]
    scores = Luminal.matmul(Luminal.permute(k, [2, 1, 3, 4]), q) * (1f0 / sqrt(Float32(d)))
    return Luminal.matmul(v, Luminal.softmax1(scores))
end
fused(q, k, v) = Luminal.flash_attention(q, k, v; causal=false)

ref_attention(q, k, v) = begin
    D, S, H, B = size(q); out = zeros(Float64, D, S, H, B)
    for b in 1:B, h in 1:H
        s = (Float64.(k[:, :, h, b])' * Float64.(q[:, :, h, b])) ./ sqrt(D)
        s .-= maximum(s, dims=1); p = exp.(s); p ./= sum(p, dims=1)
        out[:, :, h, b] = Float64.(v[:, :, h, b]) * p
    end
    out
end

for (D, S, H, B) in ((64, 256, 8, 1), (64, 750, 12, 1), (64, 1500, 20, 1), (128, 512, 8, 1), (64, 1500, 20, 2))
    ins = (randn(Float32, D, S, H, B), randn(Float32, D, S, H, B), randn(Float32, D, S, H, B))
    runs = Base.Function[]; outs = Any[]
    for build in (composite, fused)
        g = Graph()
        ts = [Luminal.tensor(g, [D, S, H, B]) for _ in 1:3]
        o = build(ts...)
        inputs = Dict{Int,Any}(t.id => Luminal.to_device(a, DEV) for (t, a) in zip(ts, ins))
        exec = compile(g; device=DEV, retain=[o.id], free_intermediates=false)
        r = () -> exec(inputs; device=DEV)
        push!(outs, Float64.(Array(r()[o.id])))
        push!(runs, () -> (r(); AMDGPU.synchronize()))
    end
    t = interleaved(runs, 7)
    nh = min(H, 2)
    ex = ref_attention(ins[1][:, :, 1:nh, :], ins[2][:, :, 1:nh, :], ins[3][:, :, 1:nh, :])
    err(o) = maximum(abs.(o[:, :, 1:nh, :] .- ex)) / maximum(abs.(ex))
    @printf("D=%-3d S=%-4d H=%-2d B=%d   composite %9.3f ms   FlashAttentionOp %9.3f ms   (%.2fx)   rel err: composite %.1e, fused %.1e\n",
            D, S, H, B, t[1], t[2], t[1] / t[2], err(outs[1]), err(outs[2]))
end
