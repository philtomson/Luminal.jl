using Test
using Luminal
using AMDGPU

# ConvDeltaRule (convolution + SiLU + L2 norm of q / k in one kernel, then the row-split recurrence) against CausalConv + DeltaRuleQKV:
# lengths that are not a multiple of the 16-token chunk, fewer tokens than the conv width, fresh and continuing states, batch 2.
rel(a, b) = maximum(abs.(Array(a) .- Array(b))) / max(maximum(abs.(Array(a))), 1f-9)
dev = get_device()
if dev isa Luminal.AMDDevice
    @testset "ConvDeltaRule" begin
        todev(x) = Luminal.to_device(x, dev)
        for (nk, nv, S, B, lenv, fresh) in ((2, 6, 70, 1, [70], true), (2, 4, 33, 2, [33, 5], false), (2, 4, 16, 2, [2, 16], true), (1, 3, 40, 1, [1], false),
                                            (2, 4, 48, 2, [0, 48], false))
            dk = dv = 128; K = 4; C = 2nk * dk + nv * dv
            x = randn(Float32, C, S, B); cs0 = randn(Float32, C, K, B) .* 0.5f0; w = randn(Float32, C, 1, K) .* 0.5f0
            g = -abs.(randn(Float32, nv, S, B)) .* 0.1f0; beta = rand(Float32, nv, S, B); st0 = randn(Float32, dv, dk, nv, B) .* 0.1f0
            lens = Float32.(lenv); scale = Float32(1 / sqrt(dk))
            function run(kind)
                o = todev(zeros(Float32, dv, nv, S, B)); cs = todev(cs0); st = todev(st0)
                if kind == :fused
                    Luminal.execute_op!(o, Luminal.ConvDeltaRule(scale, fresh, 0, nk, dk, 1f-6), todev(x), cs, todev(w), todev(g), todev(beta), st, todev(lens))
                else
                    conv = todev(zeros(Float32, C, S, B))
                    Luminal.execute_op!(conv, Luminal.CausalConv(fresh, 0), todev(x), cs, todev(w), todev(lens))
                    Luminal.execute_op!(o, Luminal.DeltaRuleQKV(scale, fresh, 0, nk, dk, 1f-6), conv, todev(g), todev(beta), st, todev(lens))
                end
                AMDGPU.synchronize(); (Array(o), Array(cs), Array(st))
            end
            (o0, c0, s0) = run(:separate); (o1, c1, s1) = run(:fused)
            @test rel(o0, o1) < 2e-5
            @test rel(c0, c1) < 1e-6          # the conv state afterwards
            @test rel(s0, s1) < 2e-5
        end
    end
end
