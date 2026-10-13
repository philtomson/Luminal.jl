using Test
using Luminal
using AMDGPU

# AttnPrep (per-head RMSNorm with a weight + partial rotate-half RoPE, written as (D, S, H, B)) and GatedAttnOut (attention output times
# sigmoid of the gate half of q_proj's rows) against a host reference.
dev = get_device()
@testset "AttnPrep / GatedAttnOut" begin
    todev(x) = dev isa Luminal.AMDDevice ? Luminal.to_device(x, dev) : copy(x)
    for (D, H, rd, rowstride, S, B, perseq) in ((256, 4, 64, 512, 40, 1, false), (256, 3, 64, 256, 17, 2, true), (128, 2, 64, 256, 16, 1, false), (256, 2, 128, 512, 33, 2, false))
        eps = 1f-6
        x = randn(Float32, rowstride * H, S, B); w = 1f0 .+ randn(Float32, D) .* 0.1f0
        half = rd ÷ 2
        ang = randn(Float32, half, S, perseq ? B : 1)
        cs = cos.(ang); sn = sin.(ang)
        ref = zeros(Float32, D, S, H, B)
        for b in 1:B, h in 1:H, s in 1:S
            v = x[(h-1)*rowstride+1:(h-1)*rowstride+D, s, b]
            y = v .* (1f0 / sqrt(sum(abs2, v) / D + eps)) .* w
            cb = perseq ? b : 1
            z = copy(y)
            for i in 1:half
                z[i] = y[i] * cs[i, s, cb] - y[i+half] * sn[i, s, cb]
                z[i+half] = y[i+half] * cs[i, s, cb] + y[i] * sn[i, s, cb]
            end
            ref[:, s, h, b] = z
        end
        out = todev(zeros(Float32, D, S, H, B))
        Luminal.execute_op!(out, Luminal.AttnPrep(eps, D, H, rd, rowstride), todev(x), todev(w), todev(cs), todev(sn))
        @test maximum(abs.(Array(out) .- ref)) / maximum(abs.(ref)) < 2e-5
    end
    for (D, H, S, B) in ((256, 4, 40, 1), (128, 3, 17, 2))
        a = randn(Float32, D * H, S, B); qp = randn(Float32, 2D * H, S, B)
        ref = similar(a)
        for b in 1:B, s in 1:S, h in 0:H-1, d in 1:D
            g = qp[h * 2D + D + d, s, b]
            ref[h * D + d, s, b] = a[h * D + d, s, b] / (1f0 + exp(-g))
        end
        out = todev(zeros(Float32, D * H, S, B))
        Luminal.execute_op!(out, Luminal.GatedAttnOut(D), todev(a), todev(qp))
        @test Array(out) ≈ ref rtol = 1e-5
        if dev isa Luminal.AMDDevice
            o16 = Luminal.to_device(zeros(Float16, D * H, S, B), dev)
            Luminal.execute_op!(o16, Luminal.GatedAttnOut(D), todev(a), todev(qp))
            @test Float32.(Array(o16)) ≈ ref rtol = 2e-3 atol = 1e-3
        end
    end
end
