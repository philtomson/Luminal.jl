using Test
using Luminal
using AMDGPU

# DeltaRuleQKV (the recurrence straight from the convolution's output) against slicing, L2-normalizing and DeltaRule: the
# row-split path, the fallback path (other head dims, snapshots) and the CPU path.
nrm(x) = x ./ sqrt.(sum(abs2, x; dims = 1) .+ 1f-6)
rel(a, b) = maximum(abs.(Array(a) .- Array(b))) / max(maximum(abs.(Array(a))), 1f-9)
for dev in (get_device(),)
    @testset "DeltaRuleQKV on $(typeof(dev))" begin
        for (dk, dv, nk, nv, S, B, lenv, fresh, snap_t) in ((128, 128, 2, 6, 70, 1, [70], true, 0), (128, 128, 2, 4, 33, 2, [33, 5], false, 0),
                                                            (64, 96, 2, 4, 40, 2, [40, 17], true, 0), (128, 128, 2, 4, 20, 1, [20], false, 3))
            C = 2nk * dk + nv * dv
            qkv = randn(Float32, C, S, B); g = -abs.(randn(Float32, nv, S, B)) .* 0.1f0; beta = rand(Float32, nv, S, B)
            st0 = randn(Float32, dv, dk, nv, B) .* 0.1f0; lens = Float32.(lenv); scale = Float32(1 / sqrt(dk))
            todev(x) = dev isa Luminal.AbstractGPUDevice ? Luminal.to_device(x, dev) : copy(x)
            function ref()
                q = nrm(Base.reshape(qkv[1:nk*dk, :, :], dk, nk, S, B)); k = nrm(Base.reshape(qkv[nk*dk+1:2nk*dk, :, :], dk, nk, S, B))
                v = Base.reshape(qkv[2nk*dk+1:C, :, :], dv, nv, S, B)
                o = todev(zeros(Float32, dv, nv, S, B)); s = todev(st0); sn = todev(zeros(Float32, dv, dk, nv, B * max(snap_t, 1)))
                args = [todev(q), todev(k), todev(v), todev(g), todev(beta), s, todev(lens)]
                Luminal.execute_op!(o, Luminal.DeltaRule(scale, fresh, snap_t), args..., sn); (Array(o), Array(s), Array(sn))
            end
            function new()
                o = todev(zeros(Float32, dv, nv, S, B)); s = todev(st0); sn = todev(zeros(Float32, dv, dk, nv, B * max(snap_t, 1)))
                Luminal.execute_op!(o, Luminal.DeltaRuleQKV(scale, fresh, snap_t, nk, dk, 1f-6), todev(qkv), todev(g), todev(beta), s, todev(lens), sn)
                (Array(o), Array(s), Array(sn))
            end
            (o0, s0, n0) = ref(); (o1, s1, n1) = new()
            @test rel(o0, o1) < 2e-5
            @test rel(s0, s1) < 2e-5
            snap_t > 0 && @test rel(n0, n1) < 2e-5
        end
    end
end
