# The gather decode attention (src/DecodeAttnGather.jl) against the split-K kernel over the same keys gathered into a contiguous cache, and the
# CPU reference: list lengths around the 256-entry chunk edges, padded lists, several head layouts, and a batch with a length a sequence.
using Test
using Luminal
using AMDGPU
using Random

if Luminal.get_device() isa CPUDevice
    @info "no GPU: skipping the gather decode attention test"
else
    function cpu_reference(q, k, v, scale, slots)                # q (D, H), k and v (D, T, KVH) Float32, slots 1-based
        D, H, KVH = size(q, 1), size(q, 2), size(k, 3)
        group = H ÷ KVH
        out = zeros(Float32, D, H)
        for h in 1:KVH, g in 1:group
            scores = scale .* (k[:, slots, h]' * q[:, (h-1)*group+g])
            p = exp.(scores .- maximum(scores)); p ./= sum(p)
            out[:, (h-1)*group+g] = v[:, slots, h] * p
        end
        return out
    end
    rel(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), 1f-9)

    @testset "gather decode attention" begin
        Random.seed!(7)
        for (D, H, KVH) in ((256, 24, 4), (256, 8, 2), (256, 12, 2)), count in (1, 2, 255, 256, 257, 700)
            T, W, B = 1500, 700, 2
            scale = Float32(1 / sqrt(D))
            k16, v16 = Float16.(randn(Float32, D, T, KVH, B)), Float16.(randn(Float32, D, T, KVH, B))
            q = randn(Float32, D, 1, H, B)
            counts = [count, max(count ÷ 2, 1)]
            ids = fill(Int32(-1), W, B)
            for b in 1:B
                ids[1:counts[b], b] .= Int32.(randperm(T)[1:counts[b]] .- 1)
            end
            out = AMDGPU.zeros(Float32, D, H, 1, B)
            @test Luminal.decode_attention_gather!(out, ROCArray(q), ROCArray(k16), ROCArray(v16), ROCArray(ids), ROCArray(Float32.(counts .- 1)), scale)
            got = Array(out)
            for b in 1:B
                slots = Int.(ids[1:counts[b], b]) .+ 1
                ref = cpu_reference(q[:, 1, :, b], Float32.(k16[:, :, :, b]), Float32.(v16[:, :, :, b]), scale, slots)
                @test rel(got[:, :, 1, b], ref) < 2e-3
                # The same keys gathered, through the existing kernel - identical chunks, so the same arithmetic
                gk, gv = ROCArray(k16[:, slots, :, b:b]), ROCArray(v16[:, slots, :, b:b])
                o2 = AMDGPU.zeros(Float32, D, H, 1, 1); z = AMDGPU.zeros(Float32, D, 1, KVH, 1)
                @test Luminal.decode_attention_split!(o2, ROCArray(q[:, :, :, b:b]), gk, gv, z, z, ROCArray(Float32[counts[b] - 1]), scale, false)
                @test rel(got[:, :, 1, b], Array(o2)[:, :, 1, 1]) < 1e-5
            end
        end
    end

    @testset "gather decode attention refuses what it does not take" begin
        D, H, KVH, T, W = 256, 24, 4, 600, 300
        out = AMDGPU.zeros(Float32, D, H, 1, 1)
        q, k, v = ROCArray(randn(Float32, D, 1, H, 1)), ROCArray(Float16.(randn(Float32, D, T, KVH, 1))), ROCArray(Float16.(randn(Float32, D, T, KVH, 1)))
        ids, last = ROCArray(zeros(Int32, W, 1)), ROCArray(Float32[W - 1])
        @test Luminal.decode_attention_gather!(out, q, k, v, ids, last, 0.0625f0)
        @test !Luminal.decode_attention_gather!(out, ROCArray(randn(Float32, D, 2, H, 1)), k, v, ids, last, 0.0625f0)          # two tokens
        @test !Luminal.decode_attention_gather!(out, q, ROCArray(randn(Float32, D, T, KVH, 1)), v, ids, last, 0.0625f0)        # Float32 cache
        @test !Luminal.decode_attention_gather!(out, q, k, v, ROCArray(zeros(Int, W, 1)), last, 0.0625f0)                      # Int64 ids
        @test !Luminal.decode_attention_gather!(out, ROCArray(randn(Float32, 128, 1, H, 1)), k, v, ids, last, 0.0625f0)        # head_dim 128
    end
end
