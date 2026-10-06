# MappingGather: discrete column gather forward, straight-through softmax
# surrogate backward. Interpreter, compile() on CPU and GPU, gradients.
using Test
using Luminal
using Luminal: compile

devices = Any[CPUDevice()]
Luminal.get_device() isa CPUDevice || push!(devices, Luminal.get_device())

function run_all(g, outs, inputs; retain=[o.id for o in outs])
    ids = [o.id for o in outs]
    rs = Any[execute(g, ids, inputs, CPUDevice())]
    for dev in devices
        cg = compile(g; device=dev, retain=retain, free_intermediates=false)
        r = cg(inputs; device=dev)
        push!(rs, Dict(id => Array(r[id]) for id in ids))
    end
    return rs
end

@testset "mapping_gather forward" begin
    g = Graph()
    x = tensor(g, [3, 4]); W = tensor(g, [4, 5]); idx = tensor(g, [5]; dtype=Int64)
    y = mapping_gather(x, W, idx; tau=0.01)
    xv = Float32[1 2 3 4; 5 6 7 8; 9 10 11 12]
    wv = randn(Float32, 4, 5)
    idv = Int64[2, 1, 4, 4, 3]
    for r in run_all(g, [y], Dict(x.id => xv, W.id => wv, idx.id => idv))
        @test r[y.id] == xv[:, idv]
    end
end

@testset "mapping_gather gradients" begin
    B, In, C = 4, 6, 5
    tau = 0.5f0
    g = Graph()
    x = tensor(g, [B, In]); W = tensor(g, [In, C]); idx = tensor(g, [C]; dtype=Int64)
    R = tensor(g, [B, C])
    y = mapping_gather(x, W, idx; tau=tau)
    loss = sum(sum(y * R, 1), 1)  # scalar
    mark_trainable!(x)
    mark_trainable!(W)
    grads = backward(loss)

    xv = randn(Float32, B, In)
    wv = randn(Float32, In, C)
    idv = Int64[3, 1, 6, 6, 2]          # 1-based, with a duplicate
    rv = randn(Float32, B, C)

    outs = [loss, grads[x.id], grads[W.id]]
    inputs = Dict(x.id => xv, W.id => wv, idx.id => idv, R.id => rv)
    for r in run_all(g, outs, inputs)
        # references: forward gather, then the straight-through surrogate
        sm = exp.(wv ./ tau) ./ sum(exp.(wv ./ tau), dims=1)
        @test r[loss.id][1] ≈ sum(rv .* xv[:, idv]) rtol = 1e-5
        @test r[grads[x.id].id] ≈ rv * sm' rtol = 1e-4
        @test r[grads[W.id].id] ≈ (2 .* xv .- 1)' * rv rtol = 1e-4
    end
end

@testset "mapping_gather chained into a graph" begin
    # gradients flow through the gather into upstream producers of x
    g = Graph()
    a = tensor(g, [2, 3]); W = tensor(g, [3, 4]); idx = tensor(g, [4]; dtype=Int64)
    y = mapping_gather(a * 2f0, W, idx; tau=1.0)
    loss = sum(sum(y * y, 1), 1)
    mark_trainable!(a)
    grads = backward(loss)
    av = Float32[1 2 3; 4 5 6]
    wv = randn(Float32, 3, 4)
    idv = Int64[1, 3, 2, 3]
    sm = exp.(wv) ./ sum(exp.(wv), dims=1)
    r = execute(g, [loss.id, grads[a.id].id],
                Dict(a.id => av, W.id => wv, idx.id => idv), CPUDevice())
    yv = (2 .* av)[:, idv]
    dy = 2 .* yv
    @test r[loss.id][1] ≈ sum(yv .^ 2) rtol = 1e-5
    @test r[grads[a.id].id] ≈ 2 .* (dy * sm') rtol = 1e-4   # d/da (2a) * router
end
