# Fused causal prefill attention (src/FlashAttn.jl, the CausalAttention op): the matrix-core
# kernel against the host reference, then through a compiled graph on every device.
using Test
using Luminal
using AMDGPU

relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

function naive(q, k, v, scale)   # the composite form: scores, mask, softmax, weighted sum
    D, S, H, B = size(q); G = H ÷ size(k, 3)
    out = zeros(Float32, D, S, H, B)
    for b in 1:B, h in 1:H
        kv = (h - 1) ÷ G + 1
        for i in 1:S
            s = [scale * sum(q[:, i, h, b] .* k[:, j, kv, b]) for j in 1:i]
            p = exp.(s .- maximum(s)); p ./= sum(p)
            out[:, i, h, b] = sum(p[j] * v[:, j, kv, b] for j in 1:i)
        end
    end
    return out
end

@testset "causal attention reference" begin
    q = randn(Float32, 8, 12, 4, 2); k = randn(Float32, 8, 12, 2, 2); v = randn(Float32, 8, 12, 2, 2)
    @test relerr(Luminal.causal_attention_ref(q, k, v, 0.35f0), naive(q, k, v, 0.35f0)) < 1e-5
end

if Luminal.wmma_supported()
    @testset "flash_prefill! against the reference" begin
        for (D, S, H, KVH, B) in ((256, 150, 8, 2, 1), (64, 16, 4, 4, 2), (128, 70, 6, 3, 1), (32, 65, 2, 1, 1))
            q = randn(Float32, D, S, H, B); k = randn(Float32, D, S, KVH, B); v = randn(Float32, D, S, KVH, B)
            out = AMDGPU.zeros(Float32, D, S, H, B)
            Luminal.flash_prefill!(out, ROCArray(q), ROCArray(k), ROCArray(v), Float32(1 / sqrt(D)))
            o = Array(out)
            @test !any(isnan, o)
            @test relerr(o, Luminal.causal_attention_ref(q, k, v, Float32(1 / sqrt(D)))) < 2e-3
        end
    end
end

@testset "CausalAttention op in a graph ($(nameof(typeof(dev))))" for dev in
        Any[CPUDevice(), (Luminal.get_device() isa CPUDevice ? () : (Luminal.get_device(),))...]
    D, S, H, KVH = 32, 40, 4, 2
    g = Luminal.Graph()
    q = Luminal.tensor(g, [D, S, H, 1]); k = Luminal.tensor(g, [D, S, KVH, 1]); v = Luminal.tensor(g, [D, S, KVH, 1])
    o = Luminal.add_op!(g, Luminal.CausalAttention(0.2f0), [(t.id, 0, t.shape) for t in (q, k, v)],
                        Luminal.ShapeTracker([D, S, H, 1]))
    qa = randn(Float32, D, S, H, 1); ka = randn(Float32, D, S, KVH, 1); va = randn(Float32, D, S, KVH, 1)
    exec = Luminal.compile(g; device = dev, retain = [o.id])
    res = exec(Dict{Int,Any}(q.id => qa, k.id => ka, v.id => va); device = dev)
    @test relerr(Array(res[o.id]), naive(qa, ka, va, 0.2f0)) < 2e-3
end
