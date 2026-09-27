# Coordinate-form gather and scatter, and iota: interpreter, compile() on CPU and
# GPU, gradients, e-graph search.
using Test
using Luminal
using Luminal: compile

devices = Any[CPUDevice()]
Luminal.get_device() isa CPUDevice || push!(devices, Luminal.get_device())

# references (0-based Float32 coordinates)
inrange(A, c) = all(0 .<= c .< size(A))
function ref_gather(A, cs...)
    [inrange(A, Int.(getindex.(cs, i))) ? A[(Int.(getindex.(cs, i)) .+ 1)...] : 0f0 for i in CartesianIndices(cs[1])]
end
function ref_scatter(init, src, cs...; add=false)
    out = copy(init)
    for i in eachindex(src)
        c = Int.(getindex.(cs, i)); inrange(out, c) || continue
        add ? (out[(c .+ 1)...] += src[i]) : (out[(c .+ 1)...] = src[i])
    end
    out
end

function run_all(g, outs, inputs; retain=[o.id for o in outs])
    ids = [o.id for o in outs]
    rs = Any[execute(g, ids, inputs, CPUDevice())]
    for dev in devices
        cg = compile(g; device=dev, retain=retain)
        r = cg(inputs; device=dev)
        push!(rs, Dict(id => Array(r[id]) for id in ids))
    end
    return rs
end

@testset "iota" begin
    g = Graph()
    a = iota(g, [4], i -> i)
    b = iota(g, [3, 5], (i, j) -> i * 5 + j)
    c = b * 2f0
    for r in run_all(g, [a, b, c], Dict{Int,Any}())
        @test r[a.id] == Float32[0, 1, 2, 3]
        @test r[b.id] == Float32[i * 5 + j for i in 0:2, j in 0:4]
        @test r[c.id] == 2 .* r[b.id]
    end
end

@testset "gather" begin
    g = Graph()
    A = tensor(g, [5, 4]); i = tensor(g, [3, 2]); j = tensor(g, [3, 2])
    y = gather(A, [i, j])
    # take along dim 1: out[k, l] = A[idx[k, l], l], column coordinates from iota
    idx = tensor(g, [2, 4])
    col = iota(g, [2, 4], (k, l) -> l)
    t = gather(A, [idx, col])
    # the column coordinate as a broadcast (expanded) row vector
    rowc = tensor(g, [3]); colc = iota(g, [1, 2], (_, l) -> l + 1)
    u = gather(A, [Luminal.expand(rowc, 2, 2), Luminal.expand(Luminal.reshape(colc, [2]), 1, 3)])
    Av = randn(Float32, 5, 4)
    iv = Float32[0 4; 2 -1; 5 1]; jv = Float32[3 0; 1 1; 0 3]   # (2,2) and (3,1) out of range
    idv = Float32[4 0 2 1; 1 1 3 0]; rv = Float32[0, 2, 4]
    inputs = Dict{Int,Any}(A.id => Av, i.id => iv, j.id => jv, idx.id => idv, rowc.id => rv)
    for r in run_all(g, [y, t, u], inputs)
        @test r[y.id] == ref_gather(Av, iv, jv)
        @test r[y.id][2, 2] == 0 && r[y.id][3, 1] == 0
        @test r[t.id] == [Av[Int(idv[k, l]) + 1, l] for k in 1:2, l in 1:4]
        @test r[u.id] == [Av[Int(rv[k]) + 1, l + 1] for k in 1:3, l in 1:2]
    end
end

@testset "scatter" begin
    g = Graph()
    init = tensor(g, [4, 3]); src = tensor(g, [5]); ci = tensor(g, [5]); cj = tensor(g, [5])
    s_rep = scatter(init, src * 1f0, [ci, cj])                 # src an intermediate
    s_add = scatter(init, src, [ci, cj]; mode=:add)
    s_dup = scatter(init, src, [ci * 0f0, cj * 0f0]; mode=:add)   # every write to [0, 0]
    iv0 = randn(Float32, 4, 3); sv = Float32[1, 2, 3, 4, 5]
    civ = Float32[0, 3, 1, 9, 2]; cjv = Float32[0, 2, 1, 0, -1]      # last two out of range
    inputs = Dict{Int,Any}(init.id => iv0, src.id => sv, ci.id => civ, cj.id => cjv)
    for r in run_all(g, [s_rep, s_add, s_dup], inputs)
        @test r[s_rep.id] ≈ ref_scatter(iv0, sv, civ, cjv)
        @test r[s_add.id] ≈ ref_scatter(iv0, sv, civ, cjv; add=true)
        @test r[s_dup.id][1, 1] ≈ iv0[1, 1] + sum(sv)
        @test r[s_dup.id][2:end] == iv0[2:end]
    end
    # a large scatter-add with many collisions (atomics)
    g2 = Graph(); z = tensor(g2, [16]); v = tensor(g2, [4096]); c = tensor(g2, [4096])
    h = scatter(z, v, [c]; mode=:add)
    vv = rand(Float32, 4096); cv = Float32.(rand(0:15, 4096))
    for r in run_all(g2, [h], Dict{Int,Any}(z.id => zeros(Float32, 16), v.id => vv, c.id => cv))
        @test r[h.id] ≈ ref_scatter(zeros(Float32, 16), vv, cv; add=true) rtol=1e-4
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
    # gather with a repeated coordinate: its gradients add up
    g = Graph(); A = tensor(g, [4, 3]); i = tensor(g, [5]); j = tensor(g, [5]); w = tensor(g, [5])
    loss = sum(gather(A, [i, j]) * gather(A, [i, j]) * w, 1)
    mark_trainable!(A); gr = backward(loss)
    Av = randn(Float32, 4, 3); iv = Float32[0, 2, 0, 3, 7]; jv = Float32[1, 2, 1, 0, 0]; wv = Float32[1, 2, 3, 4, 5]
    f(a) = sum(ref_gather(a, iv, jv) .^ 2 .* wv)
    r = execute(g, gr[A.id].id, Dict(A.id => Av, i.id => iv, j.id => jv, w.id => wv), CPUDevice())
    @test r ≈ fd_grad(f, Av) rtol=1e-2
    @test r[1, 2] ≈ 2 * Av[1, 2] * (1 + 3)

    # scatter, both modes: gradients for init and src
    for mode in (:replace, :add)
        g = Graph(); init = tensor(g, [4]); src = tensor(g, [3]); c = tensor(g, [3]); w = tensor(g, [4])
        out = scatter(init, src, [c]; mode=mode)
        loss = sum(out * out * w, 1)
        mark_trainable!(init); mark_trainable!(src); gr = backward(loss)
        inv, sv, cv, wv = randn(Float32, 4), randn(Float32, 3), Float32[2, 0, 5], Float32[1, 2, 3, 4]
        fo(a, b) = sum(ref_scatter(a, b, cv; add=mode === :add) .^ 2 .* wv)
        r = execute(g, [gr[init.id].id, gr[src.id].id], Dict(init.id => inv, src.id => sv, c.id => cv, w.id => wv), CPUDevice())
        @test r[gr[init.id].id] ≈ fd_grad(a -> fo(a, sv), inv) rtol=1e-2 atol=1e-3
        @test r[gr[src.id].id] ≈ fd_grad(b -> fo(inv, b), sv) rtol=1e-2 atol=1e-3
    end
end

@testset "e-graph search" begin
    g = Graph(); x = tensor(g, [16, 4]); w = tensor(g, [8, 16]); c = tensor(g, [6])
    h = matmul(w, x)
    y = gather(h, [c, iota(g, [6], k -> k % 4)])
    z = scatter(h, y * 2f0, [c, iota(g, [6], k -> 3 - k % 4)]; mode=:add)
    xv, wv, cv = randn(Float32, 16, 4), randn(Float32, 8, 16), Float32[0, 7, 3, 3, 5, 1]
    inputs = Dict(x.id => xv, w.id => wv, c.id => cv)
    ref = execute(g, [y.id, z.id], inputs, CPUDevice())
    for dev in devices
        ex = compile(g; device=dev, retain=[y.id, z.id], search=:static)
        r = ex(inputs; device=dev)
        @test Array(r[y.id]) ≈ ref[y.id] rtol=1e-4
        @test Array(r[z.id]) ≈ ref[z.id] rtol=1e-4
    end
end
