using Test
using Luminal

# compile() records its fusions and zero-copy views as named lowering sites; each
# can be turned off (`lowering`), and every combination computes the same result.
@testset "lowering sites: recorded, and each can be turned off ($(nameof(typeof(dev))))" for dev in
        unique([CPUDevice(), Luminal.get_device()])
    g = Luminal.Graph()
    W = Luminal.tensor(g, [64, 32]); x = Luminal.tensor(g, [32, 3]); r = Luminal.tensor(g, [64, 3])
    a = Luminal.tensor(g, [16, 12])
    h = r + Luminal.matmul(W, x)                         # residual epilogue (GPU GEMV weights)
    y = Luminal.sigmoid(h) * 2.0f0 + h                   # elementwise fusion
    z = Luminal.permute(Luminal.reshape(a, [16, 12, 1]), [1, 3, 2])   # views
    s = Luminal.slice_along(y, 1, 8, 24)                 # contiguous-slice view
    c1 = Luminal.tensor(g, [8, 3]); c2 = Luminal.tensor(g, [5, 3])
    c = Luminal.concat_along(c1 * 2.0f0, c2 + 1.0f0, 1)  # run-time concatenation
    ins = Dict{Int,Any}(x.id => randn(Float32, 32, 3), r.id => randn(Float32, 64, 3),
                        a.id => randn(Float32, 16, 12),
                        c1.id => randn(Float32, 8, 3), c2.id => randn(Float32, 5, 3))
    g.tensors[(W.id, 1)] = Luminal.to_device(randn(Float32, 64, 32), dev)
    wd = dev isa CPUDevice ? Float32 : Int8
    retain = [y.id, z.id, s.id, c.id]
    run(lowering) = begin
        cg = compile(g; device=dev, retain=retain, weight_dtype=wd, free_intermediates=false,
                     lowering=lowering)
        res = cg(ins; device=dev)
        [Array(res[i]) for i in retain], cg.cache[:lowering_sites]
    end
    ref, sites = run(Dict{String,Bool}())
    @test any(startswith("fuse "), sites)
    @test any(startswith("view "), sites)
    @test any(startswith("concat "), sites)
    @test ref[4] ≈ vcat(2 .* ins[c1.id], ins[c2.id] .+ 1)
    dev isa CPUDevice || @test any(startswith("residual "), sites)
    for site in sites
        out, _ = run(Dict(site => false))
        @test all(isapprox.(out, ref; rtol=1e-5))
    end
    out, _ = run(Dict(k => false for k in sites))
    @test all(isapprox.(out, ref; rtol=1e-5))
end
