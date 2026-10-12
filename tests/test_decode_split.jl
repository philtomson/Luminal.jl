# The split-K decode attention (src/DecodeAttn.jl) against the one-workgroup-per-head kernel and the CPU
# reference, on a Float16 cache: several KV groups, 1-4 new tokens, batches, positions around chunk edges.
using Test
using Luminal
using AMDGPU

if Luminal.get_device() isa CPUDevice
    @info "no GPU: skipping the split-K decode attention test"
else
    rel(a, b) = maximum(abs.(Array(a) .- Array(b))) / max(maximum(abs.(Array(b))), 1f-9)
    @testset "split-K decode attention" begin
        for (D, H, KVH, S, B, T, positions) in ((256, 24, 4, 1, 1, 1024, [100]), (256, 24, 4, 3, 1, 1024, [700]),
                                                (256, 24, 4, 2, 2, 1024, [253, 900]), (128, 16, 4, 3, 1, 2048, [1500]),
                                                (256, 8, 2, 2, 1, 600, [0]), (256, 24, 4, 3, 1, 1024, [254]),
                                                (256, 24, 4, 4, 1, 1024, [700]), (256, 24, 4, 4, 1, 1024, [253]), (256, 28, 4, 3, 1, 1024, [600]),
                                                (256, 24, 4, 2, 1, 1024, [300]))
            q = ROCArray(randn(Float32, D, S, H, B)); kn = ROCArray(randn(Float32, D, S, KVH, B)); vn = ROCArray(randn(Float32, D, S, KVH, B))
            pk = ROCArray(Float16.(randn(Float32, D, T, KVH, B))); pv = ROCArray(Float16.(randn(Float32, D, T, KVH, B)))
            pk2 = copy(pk); pv2 = copy(pv)
            posv = ROCArray(Float32.(positions))
            op = Luminal.DecodeAttention(Float32(1 / sqrt(D)), true, 0)
            o1 = AMDGPU.zeros(Float32, D, H, S, B); o2 = AMDGPU.zeros(Float32, D, H, S, B)
            withenv("LUMINAL_DECODE_SPLIT" => "0") do
                Luminal.execute_op!(o1, op, q, pk, pv, kn, vn, posv)
            end
            Luminal.execute_op!(o2, op, q, pk2, pv2, kn, vn, posv)
            @test rel(o2, o1) < 1e-2
            @test Array(pk) == Array(pk2) && Array(pv) == Array(pv2)
        end
    end

    # int8 cache: quantize a Float16 reference cache's rows with the kernel's own quantizer (prefill path: `quantize_kv!`), then
    # the attention over the int8 cache (new tokens quantized on write) must match the Float16 kernel over the dequantized cache.
    @testset "int8 KV cache" begin
        for (D, H, KVH, S, B, T, positions) in ((256, 24, 4, 1, 1, 1024, [100]), (256, 24, 4, 3, 1, 1024, [700]),
                                                (256, 24, 4, 4, 1, 1024, [253]), (128, 16, 4, 2, 2, 1024, [253, 900]),
                                                (256, 24, 4, 3, 1, 1024, [0]), (256, 24, 4, 40, 1, 1024, [100]), (256, 24, 4, 64, 1, 1024, [0]))   # (the last two: chunk steps)
            q = ROCArray(randn(Float32, D, S, H, B)); kn = ROCArray(randn(Float32, D, S, KVH, B)); vn = ROCArray(randn(Float32, D, S, KVH, B))
            xk = ROCArray(randn(Float32, D, T, KVH, B)); xv = ROCArray(randn(Float32, D, T, KVH, B))
            pk8 = AMDGPU.zeros(Int8, D + Luminal.DA_Q8_PAD, T, KVH, B); pv8 = AMDGPU.zeros(Int8, D + Luminal.DA_Q8_PAD, T, KVH, B)
            for b in 1:B
                Luminal.quantize_kv!(pk8, xk, T, b); Luminal.quantize_kv!(pv8, xv, T, b)
            end
            # the dequantized cache, on the host
            deq(p8) = begin
                a = Array(p8); sc = Array(reinterpret(Float32, p8))[D ÷ 4 + 1, :, :, :]
                Float16.(Float32.(a[1:D, :, :, :]) .* Base.reshape(sc, 1, T, KVH, B))
            end
            pk = ROCArray(deq(pk8)); pv = ROCArray(deq(pv8))
            @test maximum(abs.(Array(pk) .- Float16.(Array(xk)))) < 0.05            # 8 bits of a unit-variance row
            posv = ROCArray(Float32.(positions))
            op = Luminal.DecodeAttention(Float32(1 / sqrt(D)), true, 0)
            o1 = AMDGPU.zeros(Float32, D, H, S, B); o2 = AMDGPU.zeros(Float32, D, H, S, B)
            Luminal.execute_op!(o1, op, q, pk, pv, kn, vn, posv)
            Luminal.execute_op!(o2, op, q, pk8, pv8, kn, vn, posv)
            @test rel(o2, o1) < 1.5e-2        # (the 8-bit rounding of the new tokens, which the Float16 reference keeps exact, dominates at pos 0)
            # the new tokens' slots were written quantized: dequantized they match the tokens to 8 bits
            for b in 1:B
                pos = positions[b]
                got = Float32.(Array(pk8)[1:D, pos + 1:pos + S, :, b]) .* Array(reinterpret(Float32, pk8))[D ÷ 4 + 1:D ÷ 4 + 1, pos + 1:pos + S, :, b]
                @test maximum(abs.(got .- Array(kn)[:, 1:S, :, b])) < 0.05
            end
        end
    end

    @testset "V prefix transpose (chunk attention)" begin
        for (D, T, KVH, NK) in ((256, 700, 4, 600), (128, 300, 2, 257), (64, 129, 1, 1), (32, 64, 3, 64))
            pv = ROCArray(Float16.(randn(Float32, D, T, KVH, 1)))
            Spad = cld(NK, 64) * 64
            vt = AMDGPU.zeros(Float16, Spad, D, KVH, 1)
            @roc groupsize=(32, 8) gridsize=(cld(NK, 32), cld(D, 32), KVH) Luminal._vtranspose_kernel!(vt, pv, NK)
            ref = zeros(Float16, Spad, D, KVH, 1)
            ref[1:NK, :, :, :] .= PermutedDimsArray(Array(pv)[:, 1:NK, :, :], (2, 1, 3, 4))
            @test Array(vt) == ref
        end
    end
end
