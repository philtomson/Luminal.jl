using Test
using Luminal

# Regression: a slice whose bound is a symbolic dim (e.g. the decode position)
# must keep that bound. It was dropped by `_sym_min`, so the KV-cache update
# rebuilt a wrong-length cache with every slot shifted.
@testset "Symbolic slice bounds" begin
    p = Luminal.Sym{Int}(:pos)
    g = Graph()
    past = Luminal.tensor(g, [2, 6, 1, 1])
    @test Luminal.eval_dim(Luminal.realized_dims(Luminal.slice_along(past, 2, 0, p).shape)[2],
                           Dict(:pos => 3)) == 3

    for dev in unique([CPUDevice(), get_device()])
        g = Graph()
        past = Luminal.tensor(g, [2, 6, 1, 1])
        slot = Luminal.tensor(g, [2, 1, 1, 1])
        out = Luminal.concat_along(
                  Luminal.concat_along(Luminal.slice_along(past, 2, 0, p), slot, 2),
                  Luminal.slice_along(past, 2, p + 1, 6), 2)
        ex = compile(g; device=dev, retain=[out.id])
        pv = Float32.(Base.reshape(1:12, 2, 6, 1, 1))
        for pos in (0, 3, 5)
            r = ex(Dict{Int,Any}(past.id => pv, slot.id => fill(-1f0, 2, 1, 1, 1));
                   sym_vals=Dict(:pos => pos), device=dev)
            expected = copy(pv); expected[:, pos + 1, :, :] .= -1
            @test Array(r[out.id]) == expected
        end
    end
end
