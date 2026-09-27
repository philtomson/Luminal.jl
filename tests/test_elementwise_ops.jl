# Rounding (floor, ceil, round, trunc), select, exact division and exp: interpreter, compile() with
# fusion (CPU and GPU), and gradients.
using Test
using Luminal
using Luminal: compile

devices = Any[CPUDevice()]
Luminal.get_device() isa CPUDevice || push!(devices, Luminal.get_device())

@testset "rounding and select" begin
    g = Graph()
    x = tensor(g, [8, 5]); c = tensor(g, [8, 1]); y = tensor(g, [8, 5])
    outs = Dict(
        :floor => floor(x), :ceil => ceil(x), :round => round(x), :trunc => trunc(x),
        :select => select(c, x, y),                       # condition broadcast along dim 2
        :select_num => select(x < 0f0, 0f0, x),           # a relu, number branch
        :chain => floor(x * 2.5f0) + select(c, sqrt(abs(x)), y * y),
    )
    xv = Float32.(round.(randn(8, 5) .* 4; digits=1)); xv[1:4] .= Float32[2.5, -2.5, 0.5, -1.5]   # ties
    cv = Float32.(rand(0:1, 8, 1)); yv = randn(Float32, 8, 5)
    ref = Dict(
        :floor => floor.(xv), :ceil => ceil.(xv), :round => round.(xv), :trunc => trunc.(xv),
        :select => ifelse.(cv .!= 0, xv, yv), :select_num => ifelse.(xv .< 0, 0f0, xv),
        :chain => floor.(xv .* 2.5f0) .+ ifelse.(cv .!= 0, sqrt.(abs.(xv)), yv .* yv),
    )
    @test round(2.5f0) == 2f0 && round(-2.5f0) == -2f0        # half to even
    inputs = Dict(x.id => xv, c.id => cv, y.id => yv)
    ids = [t.id for t in values(outs)]
    res = execute(g, ids, inputs, CPUDevice())
    for (k, t) in outs
        @test res[t.id] ≈ ref[k]
    end
    for dev in devices
        cg = compile(g; device=dev)
        r = cg(inputs; device=dev)
        for (k, t) in outs
            @test Array(r[t.id]) ≈ ref[k]
        end
    end

    # a chain of the new ops fuses into one kernel
    g2 = Graph(); a = tensor(g2, [64]); b = tensor(g2, [64]); m = tensor(g2, [64])
    z = select(m, floor(a) * ceil(b), round(a) + trunc(b))
    av, bv, mv = randn(Float32, 64) .* 3, randn(Float32, 64) .* 3, Float32.(rand(0:1, 64))
    zr = ifelse.(mv .!= 0, floor.(av) .* ceil.(bv), round.(av) .+ trunc.(bv))
    for dev in devices
        cg = compile(g2; device=dev)
        @test length(cg.steps) == 1
        @test Array(cg(Dict(a.id => av, b.id => bv, m.id => mv); device=dev)[z.id]) ≈ zr
    end

    # an Inf in the branch not taken does not leak (it would through c*a + (1-c)*b)
    g3 = Graph(); p = tensor(g3, [3])
    q = select(p < 1f0, p, Luminal.reciprocal(p - p))
    @test execute(g3, q.id, Dict(p.id => Float32[0, 0.5, 2]), CPUDevice()) == Float32[0, 0.5, Inf]
end

@testset "div and exp" begin
    g = Graph(); a = tensor(g, [8, 6]); b = tensor(g, [8, 1])
    q = a / b; e = exp(a); ch = exp(a / b) * (2f0 / b)
    av = randn(Float32, 8, 6) .* 3; bv = randn(Float32, 8, 1) .+ 3f0
    inputs = Dict(a.id => av, b.id => bv)
    r = execute(g, [q.id, e.id, ch.id], inputs, CPUDevice())
    @test r[q.id] == av ./ bv                                  # one rounding, not a * (1/b)
    @test r[e.id] ≈ exp.(av)
    @test g.nodes[q.id].op isa Div && g.nodes[e.id].op isa Exp
    for dev in devices
        cg = compile(g; device=dev, retain=[q.id, e.id, ch.id])
        rc = cg(inputs; device=dev)
        @test Array(rc[q.id]) ≈ av ./ bv
        @test Array(rc[e.id]) ≈ exp.(av)
        @test Array(rc[ch.id]) ≈ exp.(av ./ bv) .* (2f0 ./ bv)
    end
    g2 = Graph(); x = tensor(g2, [32]); y = tensor(g2, [32])
    z = exp(x / y) / (y * y)
    xv, yv = randn(Float32, 32), rand(Float32, 32) .+ 1f0
    for dev in devices
        cg = compile(g2; device=dev)
        @test length(cg.steps) == 1
        @test Array(cg(Dict(x.id => xv, y.id => yv); device=dev)[z.id]) ≈ exp.(xv ./ yv) ./ (yv .* yv)
    end
end

function fd_grad(f, x; h=1f-2)
    g = similar(x)
    for i in eachindex(x)
        xp = copy(x); xm = copy(x); xp[i] += h; xm[i] -= h
        g[i] = (f(xp) - f(xm)) / 2h
    end
    g
end

@testset "gradients" begin
    # div (with broadcasting) and exp against finite differences
    g = Graph(); a = tensor(g, [3, 4]); b = tensor(g, [3, 1])
    loss = sum(sum(exp(a * 0.3f0) / b, 2), 1)
    mark_trainable!(a); mark_trainable!(b); gr = backward(loss)
    av = randn(Float32, 3, 4); bv = Float32[1.5, -2.0, 3.0][:, :]
    r = execute(g, [gr[a.id].id, gr[b.id].id], Dict(a.id => av, b.id => bv), CPUDevice())
    @test r[gr[a.id].id] ≈ fd_grad(v -> sum(exp.(v .* 0.3f0) ./ bv), av) rtol=1e-2
    @test r[gr[b.id].id] ≈ fd_grad(v -> sum(exp.(av .* 0.3f0) ./ v), bv) rtol=1e-2

    # select: the gradient follows the chosen branch, broadcast back to each input
    g = Graph(); c = tensor(g, [4, 1]); a = tensor(g, [4, 3])
    b = tensor(g, [1, 3])
    loss = sum(sum(select(c, a * a, b * 3f0), 2), 1)
    mark_trainable!(a); mark_trainable!(b); gr = backward(loss)
    cv = Float32[1, 0, 1, 0][:, :]; av = randn(Float32, 4, 3); bv = randn(Float32, 1, 3)
    r = execute(g, [gr[a.id].id, gr[b.id].id], Dict(c.id => cv, a.id => av, b.id => bv), CPUDevice())
    @test r[gr[a.id].id] ≈ 2 .* av .* cv
    @test r[gr[b.id].id] ≈ fill(3f0 * 2, 1, 3)                 # two rows pick b

    # rounding: zero gradient, only the other path contributes
    g = Graph(); x = tensor(g, [5])
    loss = sum(floor(x) * x + ceil(x) + round(x) + trunc(x), 1)
    mark_trainable!(x); gr = backward(loss)
    xv = Float32[0.3, 1.7, -2.2, 3.5, -0.4]
    @test execute(g, gr[x.id].id, Dict(x.id => xv), CPUDevice()) ≈ floor.(xv)
end

@testset "e-graph search" begin
    # the new ops pass through the e-graph bridge and its extraction
    g = Graph(); x = tensor(g, [16, 4]); w = tensor(g, [8, 16])
    h = matmul(w, x)
    y = select(h < 0f0, floor(h), round(h * 0.5f0))
    xv, wv = randn(Float32, 16, 4), randn(Float32, 8, 16)
    ref = execute(g, y.id, Dict(x.id => xv, w.id => wv), CPUDevice())
    for dev in devices
        ex = compile(g; device=dev, retain=[y.id], search=:static)
        @test Array(ex(Dict(x.id => xv, w.id => wv); device=dev)[y.id]) ≈ ref
    end
end
