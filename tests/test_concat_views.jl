using Test
using Luminal

# concat_along builds Add(Pad(a), Pad(b)); compile() runs it as one copy step (the
# Pads get none). Slices read only by elementwise ops may be strided views.
@testset "concat and strided slice views" begin
    for dev in unique([CPUDevice(), get_device()]), free in (true, false)
        g = Graph()
        x = Luminal.tensor(g, [8, 3, 2])
        y = Luminal.tensor(g, [5, 3, 2])
        c = Luminal.concat_along(x, y, 1)                        # (13, 3, 2)
        h1 = Luminal.slice_along(x, 1, 0, 4)                     # strided halves of x
        h2 = Luminal.slice_along(x, 1, 4, 8)
        r = Luminal.concat_along(h1 * 2f0, h2 + 1f0, 1)          # (8, 3, 2)
        xv, yv = randn(Float32, 8, 3, 2), randn(Float32, 5, 3, 2)
        ex = compile(g; device=dev, retain=[c.id, r.id], free_intermediates=free)
        res = ex(Dict{Int,Any}(x.id => xv, y.id => yv); device=dev)
        @test Array(res[c.id]) == cat(xv, yv; dims=1)
        @test Array(res[r.id]) ≈ cat(xv[1:4, :, :] .* 2f0, xv[5:8, :, :] .+ 1f0; dims=1)
        pads = count(n -> n.op isa Luminal.Pad, g.nodes)
        @test pads == 4
        @test count(st -> hasfield(typeof(st), :op) && getfield(st, :op) isa Luminal.Pad, ex.steps) == 0
    end
end
