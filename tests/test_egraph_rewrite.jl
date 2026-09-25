using Test
using Luminal
using Luminal.EGraphRewrite

# Every graph the rewrite layer extracts must compute the same outputs as the
# original graph.
run(g, outs, inputs, dev; kw...) = begin
    ex = compile(g; device=dev, retain=outs, kw...)
    r = ex(inputs; device=dev)
    [Array(r[o]) for o in outs], ex
end
remap_inputs(inputs, m) = Dict{Int,Any}(m[k] => v for (k, v) in inputs)

@testset "EGraphRewrite" begin
    @testset "round trip and expand-to-broadcast" begin
        g = Graph()
        x = Luminal.tensor(g, [4, 3]); y = Luminal.tensor(g, [4, 5, 3])
        out = Luminal.expand(x, 2, 5) * y + 1.0f0
        inputs = Dict{Int,Any}(x.id => randn(Float32, 4, 3), y.id => randn(Float32, 4, 5, 3))
        ref, _ = run(g, [out.id], inputs, CPUDevice())

        rw = to_egraph(g, [out.id])
        ng, m = extract_graph(rw)
        @test run(ng, [m[out.id]], remap_inputs(inputs, m), CPUDevice())[1] == ref

        saturate_graph!(rw)
        ng, m = extract_graph(rw)
        @test !any(n -> n.op isa Luminal.Expand, ng.nodes)       # the copy is gone
        @test run(ng, [m[out.id]], remap_inputs(inputs, m), CPUDevice())[1] ≈ ref
    end

    for dev in unique([CPUDevice(), get_device()])
        @testset "merged projections on $(nameof(typeof(dev)))" begin
            g = Graph()
            x = Luminal.tensor(g, [8, 1, 1])
            ws = [Luminal.tensor(g, [r, 8]) for r in (16, 8, 8)]
            for w in ws
                g.tensors[(w.id, 1)] = Luminal.to_device(Float32.(Float16.(randn(Float32, Luminal.realized_dims(w.shape)...))), dev)
            end
            ys = [Luminal.matmul(w, x) for w in ws]
            outs = [y.id for y in ys]
            inputs = Dict{Int,Any}(x.id => randn(Float32, 8, 1, 1))
            ref, _ = run(g, outs, inputs, dev)

            for precision in (dev isa CPUDevice ? (false,) : (false, true))
                rw = to_egraph(g, outs)
                saturate_graph!(rw; precision=precision)
                @test merge_projections!(rw; precision=precision) == 1
                ds = decisions(rw)
                joint = only(d for d in ds if startswith(d[1], "merge"))
                ng, m = extract_graph(rw; choices=joint[2][2])
                @test count(n -> n.op isa Luminal.MatMul || n.op isa Luminal.MatMulF16, ng.nodes) == 1
                @test count(n -> n.op isa Luminal.Slice, ng.nodes) == 3
                got, ex = run(ng, [m[o] for o in outs], remap_inputs(inputs, m), dev; free_intermediates=false)
                @test all(isapprox.(got, ref; rtol=1e-5))
                # The concatenated weight is folded at compile time: no step computes it
                @test length(ex.steps) == 4        # merged matmul + 3 slices
            end
        end
    end
end

@testset "compile(...; search=...)" begin
    dev = get_device()
    g = Graph()
    x = Luminal.tensor(g, [64, 1, 1])
    ws = [Luminal.tensor(g, [r, 64]) for r in (128, 32, 32)]
    for w in ws
        g.tensors[(w.id, 1)] = Luminal.to_device(Float32.(Float16.(randn(Float32, Luminal.realized_dims(w.shape)...))), dev)
    end
    out = sum(Luminal.matmul(w, x) for w in ws[2:3]) * 0.5f0
    q = Luminal.matmul(ws[1], x)
    retain = [q.id, out.id, x.id]
    inputs = Dict{Int,Any}(x.id => randn(Float32, 64, 1, 1))
    ref = compile(g; device=dev, retain=retain)(inputs; device=dev)
    cache = mktempdir()
    for search in (:static, :measured, :measured)       # the second :measured run hits the cache
        ex = compile(g; device=dev, retain=retain, free_intermediates=false, search=search,
                     precision=dev isa Luminal.AbstractGPUDevice, search_inputs=inputs, search_cache=cache)
        @test ex isa Luminal.EGraphRewrite.RewrittenGraph
        r = ex(inputs; device=dev)
        @test Array(r[q.id]) ≈ Array(ref[q.id]) rtol = 1e-4
        @test Array(r[out.id]) ≈ Array(ref[out.id]) rtol = 1e-4
    end
    @test length(readdir(cache)) == 1
end
