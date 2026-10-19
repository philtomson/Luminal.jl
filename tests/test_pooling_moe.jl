using Test
using Luminal
using Luminal.NN
using Luminal.Pooling
using Luminal.MoE
using Random
import Luminal: expand

const DEV = get(ENV, "LUMINAL_TEST_GPU", "0") == "1" ? Luminal.get_device() : CPUDevice()
function run1(build, inputs...)
    g = Graph(); ts = [tensor(g, collect(size(a))) for a in inputs]
    out = build(ts...)
    Array(execute(g, [out.id], Dict(t.id => a for (t, a) in zip(ts, inputs)), DEV)[out.id])
end

# plain-Julia pooling of one (H, W) channel
function ref_pool(x, k, s)
    oh, ow = (size(x, 1) - k[1]) ÷ s[1] + 1, (size(x, 2) - k[2]) ÷ s[2] + 1
    [sum(x[(i-1)*s[1]+1:(i-1)*s[1]+k[1], (j-1)*s[2]+1:(j-1)*s[2]+k[2]]) / prod(k) for i in 1:oh, j in 1:ow]
end

@testset "pooling" begin
    # (C, H, W): channel 1 holds 1..16 row-major
    x = zeros(Float32, 1, 4, 4); x[1, :, :] = Float32.(Base.reshape(1:16, 4, 4)')
    @test run1(t -> avg_pool_2d(t, (2, 2)), x)[1, :, :] ≈ Float32[3.5 5.5; 11.5 13.5]
    @test size(run1(t -> avg_pool_2d(t, (2, 2)), x)) == (1, 2, 2)
    y = randn(Float32, 3, 7, 9, 2)
    for (k, s) in (((2, 2), (2, 2)), ((3, 2), (1, 2)), ((3, 3), (2, 3)), ((1, 1), (1, 1)))
        out = run1(t -> avg_pool_2d(t, k, s), y)
        for c in 1:3, n in 1:2
            @test out[c, :, :, n] ≈ ref_pool(y[c, :, :, n], k, s) rtol=1e-5
        end
    end
    @test_throws ArgumentError avg_pool_2d(tensor(Graph(), [1, 2, 2]), (3, 3))
    # upstream's overlapping windows when the sizes do not divide: 5 -> 2 gives kernel 3, stride 2
    z = zeros(Float32, 1, 5, 5); z[1, :, :] = Float32.(Base.reshape(1:25, 5, 5)')
    @test run1(t -> adaptive_avg_pool_2d(t, (2, 2)), z)[1, :, :] ≈ Float32[7 9; 17 19]
    w = randn(Float32, 2, 6, 8)                                  # divides: plain 2x2 / 3x2 windows
    out = run1(t -> adaptive_avg_pool_2d(t, (3, 2)), w)
    @test size(out) == (2, 3, 2) && all(out[c, :, :] ≈ ref_pool(w[c, :, :], (2, 4), (2, 4)) for c in 1:2)
end

@testset "topk_indexes" begin
    x = Float32[0.1 5 3; 0.9 1 3; 0.5 2 9; 0.7 4 1]               # (E=4, T=3)
    ids = run1(t -> topk_indexes(t, 2), x)
    @test size(ids) == (2, 3)
    @test ids[:, 1] == [1, 3] && ids[:, 2] == [0, 3] && ids[:, 3] == [2, 0]    # ties (column 3: 3 and 3) go to the lower index
    @test run1(t -> topk_indexes(t, 1), x) == Float32[1 0 2]
    @test size(run1(t -> topk_indexes(t, 3), randn(Float32, 6, 2, 3))) == (3, 2, 3)
end

@testset "TopKRoutes" begin
    E, K, T, F, O = 5, 2, 3, 4, 3
    scores = randn(Float32, E, T); x = randn(Float32, F, T); bank = randn(Float32, E, O, F)
    ids = [3 0 4; 1 2 2]                                         # (K, T), 0-based
    idsf = Float32.(ids)
    # weights picked from scores, normalised, dispatched, per-route bank matmul and combine
    function layer(s, xx, b, i)
        r = normalize_routes(topk_routes(s, i))
        w = select_experts(r, b)                                 # (O, F, K, T)
        y = sum(w * expand(dispatch(r, xx), 1, O), 2)            # (O, K, T)
        return combine(r, y)                                     # (O, T)
    end
    out = run1(layer, scores, x, bank, idsf)
    ref = zeros(Float32, O, T)
    for t in 1:T
        w = [scores[ids[k, t] + 1, t] for k in 1:K]; w ./= sum(w)
        for k in 1:K
            ref[:, t] += w[k] * bank[ids[k, t] + 1, :, :] * x[:, t]
        end
    end
    @test out ≈ ref rtol=1e-4
    # the same through the flat route table
    function layer2(s, xx, b, i)
        r = Routes(normalize_routes(topk_routes(s, i)))
        w = select_experts(r, b)                                 # (O, F, R)
        y = sum(w * expand(dispatch(r, xx), 1, O), 2)            # (O, R)
        return combine(r, y)
    end
    @test run1(layer2, scores, x, bank, idsf) ≈ ref rtol=1e-4
    # token axes beyond one: (K, 2, 2)
    s2 = randn(Float32, E, 2, 2); ids2 = Float32.(rand(0:E-1, K, 2, 2))
    got = run1((s, i) -> topk_routes(s, i).weights, s2, ids2)
    @test got == [s2[Int(ids2[k, a, b]) + 1, a, b] for k in 1:K, a in 1:2, b in 1:2]
end

@testset "SparseMoE == per-token reference" begin
    H, I, E, K, T = 6, 5, 4, 2, 3
    x = randn(Float32, H, T); rw = randn(Float32, E, H); gu = 0.5f0 .* randn(Float32, E, 2I, H); dn = 0.5f0 .* randn(Float32, E, H, I)
    for norm in (true, false)
        g = Graph(); xt = tensor(g, [H, T])
        router = Linear(H, E, g; bias=false)
        moe = SparseMoE(router, tensor(g, [E, 2I, H]), tensor(g, [E, H, I]), K, I; norm_topk_prob=norm)
        out = moe(xt)
        res = execute(g, [out.id], Dict(xt.id => x, router.weight.id => rw, moe.gate_up.id => gu, moe.down.id => dn), DEV)
        ref = zeros(Float32, H, T)
        for t in 1:T
            lg = rw * x[:, t]; p = exp.(lg .- maximum(lg)); p ./= sum(p)
            top = sortperm(p; rev=true)[1:K]
            w = p[top]; norm && (w ./= sum(w))
            for (wk, e) in zip(w, top)
                z = gu[e, :, :] * x[:, t]
                h = (z[1:I] ./ (1 .+ exp.(-z[1:I]))) .* z[I+1:2I]
                ref[:, t] += wk * dn[e, :, :] * h
            end
        end
        @test Array(res[out.id]) ≈ ref rtol=1e-4
    end
end
