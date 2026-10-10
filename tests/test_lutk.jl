# LutKOp: fused K-input LUT layer. Forward vs the brute-force multilinear
# basis, pin/table gradients vs independent basis-derivative references;
# interpreter, compile() on CPU and GPU; both input forms (gathered pins and
# indexed x).
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

# phi[e-1] = prod_k (v_k if bit k of e-1 is 1 else 1-v_k), v1 the MSB — the
# same entry order as the interpolation tree and the Verilog export.
function ref_basis(vs::NTuple{K,T}) where {K,T}
    if K == 1
        return T[(1 - vs[1]), vs[1]]
    end
    rest = ref_basis(Base.tail(vs))
    return vcat((1 - vs[1]) .* rest, vs[1] .* rest)
end
ref_lut(vs::NTuple{K,T}, w) where {K,T} = sum(ref_basis(vs) .* w)

@testset "lut_layer forward (K=4, gathered)" begin
    K, out_dim, B = 4, 5, 3
    g = Graph()
    pins = tensor(g, [B, K * out_dim]); W = tensor(g, [out_dim, 2^K])
    y = lut_layer(pins, W; K=K)
    pv = 0.1f0 .+ 0.8f0 .* rand(Float32, B, K * out_dim)
    wv = randn(Float32, out_dim, 2^K)
    for r in run_all(g, [y], Dict(pins.id => pv, W.id => wv))
        for b in 1:B, n in 1:out_dim
            vs = ntuple(k -> pv[b, (k - 1) * out_dim + n], Val(K))
            @test r[y.id][b, n] ≈ ref_lut(vs, view(wv, n, :)) rtol = 1e-5 atol = 1e-5
        end
    end
end

@testset "lut_layer forward (K=6, indexed, binary x)" begin
    K, out_dim, B, In = 6, 4, 3, 20
    g = Graph()
    x = tensor(g, [B, In]); W = tensor(g, [out_dim, 2^K]); idx = tensor(g, [K * out_dim]; dtype=Int64)
    y = lut_layer(x, W, idx; K=K)
    xv = Float32.(rand(B, In) .> 0.5)
    wv = randn(Float32, out_dim, 2^K)
    idv = Int64.(rand(1:In, K * out_dim))
    for r in run_all(g, [y], Dict(x.id => xv, W.id => wv, idx.id => idv))
        for b in 1:B, n in 1:out_dim
            vs = ntuple(k -> xv[b, idv[(k - 1) * out_dim + n]], Val(K))
            @test r[y.id][b, n] ≈ ref_lut(vs, view(wv, n, :)) rtol = 1e-5
        end
    end
end

@testset "lut_layer gradients (K=4, gathered)" begin
    K, out_dim, B = 4, 6, 4
    g = Graph()
    pins = tensor(g, [B, K * out_dim]); W = tensor(g, [out_dim, 2^K])
    R = tensor(g, [B, out_dim])
    y = lut_layer(pins, W; K=K)
    loss = sum(sum(y * R, 1), 1)
    mark_trainable!(pins); mark_trainable!(W)
    grads = backward(loss)

    pv = 0.1f0 .+ 0.8f0 .* rand(Float32, B, K * out_dim)  # away from 0/1 for the basis-derivative reference
    wv = randn(Float32, out_dim, 2^K)
    rv = randn(Float32, B, out_dim)

    dpins_ref = zeros(Float32, B, K * out_dim)
    dW_ref = zeros(Float32, out_dim, 2^K)
    loss_ref = 0.0f0
    for b in 1:B, n in 1:out_dim
        vs = ntuple(k -> pv[b, (k - 1) * out_dim + n], Val(K))
        phi = ref_basis(vs)
        loss_ref += rv[b, n] * sum(phi .* view(wv, n, :))
        for e in 1:2^K
            dW_ref[n, e] += rv[b, n] * phi[e]
        end
        for k in 1:K
            # dphi_e/dv_k = +phi_e/v_k (bit 1) or -phi_e/(1-v_k) (bit 0)
            total = 0.0f0
            for e in 1:2^K
                bit = ((e - 1) >> (K - k)) & 1
                total += wv[n, e] * (bit == 1 ? phi[e] / vs[k] : -phi[e] / (1 - vs[k]))
            end
            dpins_ref[b, (k - 1) * out_dim + n] = rv[b, n] * total
        end
    end

    outs = [loss, grads[pins.id], grads[W.id]]
    inputs = Dict(pins.id => pv, W.id => wv, R.id => rv)
    for r in run_all(g, outs, inputs)
        @test r[loss.id][1] ≈ loss_ref rtol = 1e-4
        @test r[grads[pins.id].id] ≈ dpins_ref rtol = 1e-3
        @test r[grads[W.id].id] ≈ dW_ref rtol = 1e-3
    end
end

@testset "lut_layer gradients (K=2, indexed, scatter to x)" begin
    K, out_dim, B, In = 2, 6, 4, 9
    g = Graph()
    x = tensor(g, [B, In]); W = tensor(g, [out_dim, 2^K]); idx = tensor(g, [K * out_dim]; dtype=Int64)
    R = tensor(g, [B, out_dim])
    y = lut_layer(x, W, idx; K=K)
    loss = sum(sum(y * R, 1), 1)
    mark_trainable!(x); mark_trainable!(W)
    grads = backward(loss)

    xv = 0.1f0 .+ 0.8f0 .* rand(Float32, B, In)
    wv = randn(Float32, out_dim, 2^K)
    idv = Int64.(rand(1:In, K * out_dim))
    rv = randn(Float32, B, out_dim)

    dx_ref = zeros(Float32, B, In)
    dW_ref = zeros(Float32, out_dim, 2^K)
    for b in 1:B, n in 1:out_dim
        vs = ntuple(k -> xv[b, idv[(k - 1) * out_dim + n]], Val(K))
        phi = ref_basis(vs)
        for e in 1:2^K
            dW_ref[n, e] += rv[b, n] * phi[e]
        end
        for k in 1:K
            total = 0.0f0
            for e in 1:2^K
                bit = ((e - 1) >> (K - k)) & 1
                total += wv[n, e] * (bit == 1 ? phi[e] / vs[k] : -phi[e] / (1 - vs[k]))
            end
            dx_ref[b, idv[(k - 1) * out_dim + n]] += rv[b, n] * total
        end
    end

    outs = [loss, grads[x.id], grads[W.id]]
    inputs = Dict(x.id => xv, W.id => wv, idx.id => idv, R.id => rv)
    for r in run_all(g, outs, inputs)
        @test r[grads[x.id].id] ≈ dx_ref rtol = 1e-3
        @test r[grads[W.id].id] ≈ dW_ref rtol = 1e-3
    end
end
