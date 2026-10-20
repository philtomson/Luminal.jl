# Fused causal prefill attention (src/FlashAttn.jl, the CausalAttention op): the matrix-core
# kernel against the host reference, then through a compiled graph on every device.
using Test
using Luminal
using AMDGPU

relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

function naive(q, k, v, scale, window = 0)   # the composite form: scores, mask, softmax, weighted sum; window > 0: the last `window` keys only
    D, S, H, B = size(q); G = H ÷ size(k, 3)
    out = zeros(Float32, D, S, H, B)
    for b in 1:B, h in 1:H
        kv = (h - 1) ÷ G + 1
        for i in 1:S
            js = window > 0 ? (max(1, i - window + 1):i) : (1:i)
            s = [scale * sum(q[:, i, h, b] .* k[:, j, kv, b]) for j in js]
            p = exp.(s .- maximum(s)); p ./= sum(p)
            out[:, i, h, b] = sum(p[t] * v[:, j, kv, b] for (t, j) in enumerate(js))
        end
    end
    return out
end

@testset "causal attention reference" begin
    q = randn(Float32, 8, 12, 4, 2); k = randn(Float32, 8, 12, 2, 2); v = randn(Float32, 8, 12, 2, 2)
    @test relerr(Luminal.causal_attention_ref(q, k, v, 0.35f0), naive(q, k, v, 0.35f0)) < 1e-5
    for window in (1, 3, 5, 12, 30)       # a window of 1 is the query alone; one of at least S is plain causal attention
        @test relerr(Luminal.causal_attention_ref(q, k, v, 0.35f0, window), naive(q, k, v, 0.35f0, window)) < 1e-5
    end
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
    ow = Luminal.add_op!(g, Luminal.CausalAttention(0.2f0, 7), [(t.id, 0, t.shape) for t in (q, k, v)],
                         Luminal.ShapeTracker([D, S, H, 1]))
    o = Luminal.add_op!(g, Luminal.CausalAttention(0.2f0), [(t.id, 0, t.shape) for t in (q, k, v)],
                        Luminal.ShapeTracker([D, S, H, 1]))
    qa = randn(Float32, D, S, H, 1); ka = randn(Float32, D, S, KVH, 1); va = randn(Float32, D, S, KVH, 1)
    exec = Luminal.compile(g; device = dev, retain = [o.id, ow.id])
    res = exec(Dict{Int,Any}(q.id => qa, k.id => ka, v.id => va); device = dev)
    @test relerr(Array(res[o.id]), naive(qa, ka, va, 0.2f0)) < 2e-3
    @test relerr(Array(res[ow.id]), naive(qa, ka, va, 0.2f0, 7)) < 2e-3          # sliding window of 7
end

# chunk_attention! (the prefill-chunk attention over the KV cache): the 128-row kernel reads the Float32 query and writes the (D, H, S, B)
# output itself (direct path); the 64-row kernel goes through a Float16 copy of q and a permute. Both against a host reference.
if Luminal.wmma_supported()
    @testset "chunk_attention! against the host reference" begin
        D, H, KVH, S, pos, maxs = 256, 8, 2, 40, 100, 192
        scale = Float32(1 / sqrt(D))
        qh = randn(Float32, D, S, H, 1); knh = Float16.(randn(Float32, D, S, KVH, 1)); vnh = Float16.(randn(Float32, D, S, KVH, 1))
        pkh = Float16.(randn(Float32, D, maxs, KVH, 1)); pvh = Float16.(randn(Float32, D, maxs, KVH, 1))
        kfull = copy(pkh); vfull = copy(pvh); kfull[:, pos+1:pos+S, :, :] = knh; vfull[:, pos+1:pos+S, :, :] = vnh
        ref = zeros(Float32, D, H, S, 1)
        for h in 1:H, i in 1:S
            kv = (h - 1) ÷ (H ÷ KVH) + 1; nk = pos + i
            s = [Float64(scale) * sum(Float64.(qh[:, i, h, 1]) .* Float64.(kfull[:, j, kv, 1])) for j in 1:nk]
            p = exp.(s .- maximum(s)); p ./= sum(p)
            ref[:, h, i, 1] = sum(p[j] * Float64.(vfull[:, j, kv, 1]) for j in 1:nk)
        end
        for rows in ("128", "64")
            withenv("LUMINAL_FLASH_ROWS" => rows) do
                out = AMDGPU.zeros(Float32, D * H, S, 1)
                ok = Luminal.chunk_attention!(out, ROCArray(qh), ROCArray(copy(pkh)), ROCArray(copy(pvh)), ROCArray(knh), ROCArray(vnh), ROCArray(Float32[pos]), scale)
                @test ok
                @test relerr(Base.reshape(Array(out), D, H, S, 1), ref) < 3e-3
            end
        end
    end
end
