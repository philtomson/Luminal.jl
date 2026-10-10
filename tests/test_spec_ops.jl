# The ops a speculative decoding step needs: DecodeAttention over several new tokens per sequence,
# and the state snapshots of CausalConv / DeltaRule (the state after the first `snap_t` tokens,
# which a rejected draft token rolls back to). CPU, and the GPU kernels where there is a GPU.
using Test
using Luminal
using AMDGPU

devs = Any[CPUDevice()]
Luminal.get_device() isa CPUDevice || push!(devs, Luminal.get_device())
nrm(x) = x ./ sqrt.(sum(abs2, x; dims = 1))
rel(a, b) = maximum(abs.(Array(a) .- Array(b))) / max(maximum(abs.(Array(b))), 1f-9)
dv_(dev, x) = Luminal.to_device(x, dev)

@testset "DecodeAttention with S new tokens ($(nameof(typeof(dev))))" for dev in devs
    for (D, H, KVH, S, B, maxseq, positions) in ((64, 4, 2, 2, 1, 40, [13]), (32, 6, 3, 3, 2, 50, [7, 30]),
                                                 (128, 2, 1, 2, 1, 20, [0]))
        q = randn(Float32, D, S, H, B); kn = randn(Float32, D, S, KVH, B); vn = randn(Float32, D, S, KVH, B)
        pk0 = randn(Float32, D, maxseq, KVH, B); pv0 = randn(Float32, D, maxseq, KVH, B)
        posv = Float32.(positions)
        scale = Float32(1 / sqrt(D))
        # S new tokens at once...
        pk = dv_(dev, copy(pk0)); pv = dv_(dev, copy(pv0))
        out = Luminal.zero_tensor(dev, Float32, D, H, S, B)
        Luminal.execute_op!(out, Luminal.DecodeAttention(scale, true, 0), dv_(dev, q), pk, pv, dv_(dev, kn), dv_(dev, vn), dv_(dev, posv))
        # ...against one token at a time, each at its own position
        pk1 = dv_(dev, copy(pk0)); pv1 = dv_(dev, copy(pv0))
        for t in 1:S
            o1 = Luminal.zero_tensor(dev, Float32, D, H, B)
            Luminal.execute_op!(o1, Luminal.DecodeAttention(scale, true, 0), dv_(dev, q[:, t:t, :, :]), pk1, pv1,
                                dv_(dev, kn[:, t:t, :, :]), dv_(dev, vn[:, t:t, :, :]), dv_(dev, posv .+ (t - 1)))
            @test rel(Array(out)[:, :, t, :], Array(o1)) < 1e-4
        end
        @test rel(pk, pk1) == 0 && rel(pv, pv1) == 0          # the cache got every token's K/V
    end
end

@testset "state snapshots ($(nameof(typeof(dev))))" for dev in devs
    # CausalConv: snapshot after token 1 == the state of a run that consumes only 1 token
    C, S, B, K = 24, 3, 2, 4
    x = randn(Float32, C, S, B); w = randn(Float32, C, 1, K); st0 = randn(Float32, C, K, B)
    for snap_t in (1, 2)
        lens = dv_(dev, Float32.([3, 1]))
        st = dv_(dev, copy(st0)); snap = dv_(dev, zeros(Float32, C, K, B)); out = Luminal.zero_tensor(dev, Float32, C, S, B)
        Luminal.execute_op!(out, Luminal.CausalConv(false, snap_t), dv_(dev, x), st, dv_(dev, w), lens, snap)
        for b in 1:B
            ref = dv_(dev, copy(st0)); o = Luminal.zero_tensor(dev, Float32, C, S, B)
            Luminal.execute_op!(o, Luminal.CausalConv(false), dv_(dev, x), ref, dv_(dev, w),
                                dv_(dev, Float32.([min(snap_t, [3, 1][b]), min(snap_t, [3, 1][b])])))
            @test rel(Array(snap)[:, :, b], Array(ref)[:, :, b]) < 1e-6
        end
        # and the state itself is the same as without a snapshot
        st2 = dv_(dev, copy(st0)); o2 = Luminal.zero_tensor(dev, Float32, C, S, B)
        Luminal.execute_op!(o2, Luminal.CausalConv(false), dv_(dev, x), st2, dv_(dev, w), lens)
        @test rel(st, st2) == 0 && rel(out, o2) == 0
    end
    # DeltaRule
    dk, dv, nk, nv, S, B = 32, 32, 2, 4, 3, 2
    q = nrm(randn(Float32, dk, nk, S, B)); k = nrm(randn(Float32, dk, nk, S, B))
    v = randn(Float32, dv, nv, S, B); g = -abs.(randn(Float32, nv, S, B)) .* 0.1f0
    beta = rand(Float32, nv, S, B); st0 = randn(Float32, dv, dk, nv, B) .* 0.1f0
    for snap_t in (1, 2), lenv in ([3, 3], [3, 1])
        lens = dv_(dev, Float32.(lenv))
        st = dv_(dev, copy(st0)); snap = dv_(dev, zeros(Float32, dv, dk, nv, B)); out = Luminal.zero_tensor(dev, Float32, dv, nv, S, B)
        Luminal.execute_op!(out, Luminal.DeltaRule(Float32(1 / sqrt(dk)), false, snap_t), dv_(dev, q), dv_(dev, k), dv_(dev, v),
                            dv_(dev, g), dv_(dev, beta), st, lens, snap)
        for b in 1:B
            ref = dv_(dev, copy(st0)); o = Luminal.zero_tensor(dev, Float32, dv, nv, S, B)
            Luminal.execute_op!(o, Luminal.DeltaRule(Float32(1 / sqrt(dk)), false), dv_(dev, q), dv_(dev, k), dv_(dev, v),
                                dv_(dev, g), dv_(dev, beta), ref, dv_(dev, Float32.([min(snap_t, lenv[1]), min(snap_t, lenv[2])])))
            @test rel(Array(snap)[:, :, :, b], Array(ref)[:, :, :, b]) < 1e-5
        end
    end
end

@testset "ArgMax op ($(nameof(typeof(dev))))" for dev in devs
    for (V, S, B) in ((1000, 2, 1), (257, 3, 2), (248320, 2, 1), (5, 1, 1))
        x = randn(Float32, V, S, B)
        x[V ÷ 2, 1, 1] = 100f0; x[V, 1, 1] = 100f0           # a tie: the first one wins
        out = Luminal.zero_tensor(dev, Float32, S, B)
        Luminal.execute_op!(out, Luminal.ArgMax(), dv_(dev, x))
        ref = Float32[argmax(view(x, :, s, b)) - 1 for s in 1:S, b in 1:B]
        @test Array(out) == ref
    end
end
