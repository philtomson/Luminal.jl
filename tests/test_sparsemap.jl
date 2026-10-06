# SparseMapGather: compact top-K router. Forward vs host gather; gradients vs
# host references (straight-through dS, softmax-weighted scatter dx);
# interpreter, compile() on CPU and GPU.
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
        # HIP-graph capture regression: ScatterND's init copy must stay
        # capture-safe (a null-stream memcpy aborts capture)
        dev isa CPUDevice && continue
        cg2 = compile(g; device=dev, retain=retain, free_intermediates=false,
                      capture=true)
        r2 = cg2(inputs; device=dev)
        push!(rs, Dict(id => Array(r2[id]) for id in ids))
    end
    return rs
end

@testset "sparse_map_gather forward (K=8)" begin
    K, C, B, In = 8, 12, 4, 30
    tau = 0.5f0
    g = Graph()
    x = tensor(g, [B, In]); S = tensor(g, [C, K]); cand = tensor(g, [C, K]; dtype=Int64)
    y = sparse_map_gather(x, S, cand; K=K, tau=tau)
    xv = rand(Float32, B, In)
    sv = randn(Float32, C, K)
    cv = Int64.(rand(1:In, C, K))                     # 1-based; duplicates allowed
    for r in run_all(g, [y], Dict(x.id => xv, S.id => sv, cand.id => cv))
        for b in 1:B, c in 1:C
            bestk = argmax(sv[c, :])
            @test r[y.id][b, c] ≈ xv[b, cv[c, bestk]] rtol = 1e-6
        end
    end
end

@testset "sparse_map_gather gradients (K=4)" begin
    K, C, B, In = 4, 10, 5, 24
    tau = 0.7f0
    g = Graph()
    x = tensor(g, [B, In]); S = tensor(g, [C, K]); cand = tensor(g, [C, K]; dtype=Int64)
    R = tensor(g, [B, C])
    y = sparse_map_gather(x, S, cand; K=K, tau=tau)
    loss = sum(sum(y * R, 1), 1)
    mark_trainable!(x); mark_trainable!(S)
    grads = backward(loss)

    xv = 0.1f0 .+ 0.8f0 .* rand(Float32, B, In)
    sv = randn(Float32, C, K)
    cv = Int64.(rand(1:In, C, K))
    rv = randn(Float32, B, C)

    # host references
    p = zeros(Float32, C, K)
    dS_ref = zeros(Float32, C, K)
    dx_ref = zeros(Float32, B, In)
    loss_ref = 0.0f0
    for c in 1:C
        srow = sv[c, :] ./ tau
        e = exp.(srow .- maximum(srow))
        p[c, :] .= e ./ sum(e)
    end
    for b in 1:B, c in 1:C
        bestk = argmax(p[c, :])
        yv = xv[b, cv[c, bestk]]
        loss_ref += rv[b, c] * yv
        for k in 1:K
            dS_ref[c, k] += (2 * xv[b, cv[c, k]] - 1) * rv[b, c]
            dx_ref[b, cv[c, k]] += rv[b, c] * p[c, k]
        end
    end

    outs = [loss, grads[x.id], grads[S.id]]
    inputs = Dict(x.id => xv, S.id => sv, cand.id => cv, R.id => rv)
    for r in run_all(g, outs, inputs)
        @test r[loss.id][1] ≈ loss_ref rtol = 1e-5
        @test r[grads[x.id].id] ≈ dx_ref rtol = 1e-4
        @test r[grads[S.id].id] ≈ dS_ref rtol = 1e-4
    end
end
