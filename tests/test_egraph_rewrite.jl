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

@testset "kernel variants" begin
    for dev in unique([CPUDevice(), get_device()])
        @testset "MatMulT on $(nameof(typeof(dev)))" begin
            for (ta, tb) in ((true, false), (false, true), (true, true)), dims in ((5, 7, 3), (5, 7, 3, 4, 2))
                M, K, N = dims[1:3]
                batch = dims[4:end]
                A = randn(Float32, (ta ? (K, M) : (M, K))..., batch...)
                B = randn(Float32, (tb ? (N, K) : (K, N))..., batch...)
                sw(X, t) = t ? permutedims(X, (2, 1, 3:ndims(X)...)) : X
                ref = zeros(Float32, M, N, batch...)
                Luminal.batch_matmul!(ref, sw(A, ta), sw(B, tb))
                out = Luminal.zero_tensor(dev, Float32, M, N, batch...)
                Luminal.execute_op!(out, Luminal.MatMulT(ta, tb), Luminal.to_device(A, dev), Luminal.to_device(B, dev))
                @test Array(out) ≈ ref rtol = 1e-5
            end
        end
    end

    @testset "Permute + MatMul -> MatMulT rewrite" begin
        g = Graph()
        q = Luminal.tensor(g, [64, 1, 4, 1]); k = Luminal.tensor(g, [64, 16, 4, 1])
        s = Luminal.matmul(Luminal.permute(q, [2, 1, 3, 4]), k)          # (1, 16, 4, 1)
        inputs = Dict{Int,Any}(q.id => randn(Float32, 64, 1, 4, 1), k.id => randn(Float32, 64, 16, 4, 1))
        ref, _ = run(g, [s.id], inputs, CPUDevice())
        rw = to_egraph(g, [s.id]); saturate_graph!(rw)
        d = only(d for d in decisions(rw) if any(o -> any(occursin("MatMulT", v) for v in values(o)), d[2]))
        opt = only(o for o in d[2] if !isempty(o) && occursin("MatMulT", first(values(o))))
        ng, m = extract_graph(rw; choices=opt)
        @test any(n -> n.op isa Luminal.MatMulT, ng.nodes)
        @test !any(n -> n.op isa Luminal.Permute, ng.nodes)
        @test run(ng, [m[s.id]], remap_inputs(inputs, m), CPUDevice())[1][1] ≈ ref[1] rtol = 1e-5
    end

    dev = get_device()
    if dev isa Luminal.AbstractGPUDevice
        @testset "MatMulF16 group sizes" begin
            W = Float32.(Float16.(randn(Float32, 96, 64)))
            hw = Luminal.half_weight(Luminal.to_device(W, dev))
            x = randn(Float32, 64, 3)
            for grp in (64, 128, 256)
                out = Luminal.zero_tensor(dev, Float32, 96, 3)
                Luminal.execute_op!(out, Luminal.MatMulF16(grp), hw, Luminal.to_device(x, dev))
                @test Array(out) ≈ W * x rtol = 1e-5
            end
        end

        @testset "MatMulF16 gemm_ex (Float16 activations)" begin
            W = Float32.(Float16.(randn(Float32, 96, 64)))
            x = randn(Float32, 64, 5, 2)
            hn = Luminal.half_weight(Luminal.to_device(W, dev), Luminal.HalfWeightN)
            @test hn isa Luminal.HalfWeightN && size(hn) == (96, 64)
            out = Luminal.zero_tensor(dev, Float32, 96, 5, 2)
            Luminal.execute_op!(out, Luminal.MatMulF16(256, :gemm_ex), hn, Luminal.to_device(x, dev))
            ref = Base.reshape(W * Base.reshape(x, 64, :), 96, 5, 2)
            @test Array(out) ≈ ref rtol = 1e-3       # activations rounded to Float16

            # compile() stores a gemm_ex-only weight untransposed (HalfWeightN)
            g = Graph()
            w = Luminal.tensor(g, [96, 64]); xin = Luminal.tensor(g, [64, 5, 2])
            y = Luminal.add_op!(g, Luminal.MatMulF16(256, :gemm_ex), [(w.id, 0, w.shape), (xin.id, 0, xin.shape)],
                                Luminal.ShapeTracker([96, 5, 2]))
            g.tensors[(w.id, 1)] = Luminal.to_device(W, dev)
            ex = compile(g; device=dev, retain=[y.id])
            @test ex.results[w.id] isa Luminal.HalfWeightN
            @test Array(ex(Dict{Int,Any}(xin.id => x); device=dev)[y.id]) ≈ ref rtol = 1e-3
        end
    end
end

# precision=:int8 / :int4: the static extraction chooses the quantized kernels,
# and the measured search keeps only candidates that match it (the quantized
# results, not Float32) -- they must equal the graph run on the dequantized weights.
if get_device() isa Luminal.AMDDevice
    @testset "search with int8 / int4 weights" begin
        dev = get_device()
        for (prec, T) in ((:int8, Luminal.QuantWeight), (:int4, Luminal.Q4Weight))
            g = Graph()
            # weights above the cost model's GEMV size threshold (64K elements)
            x = Luminal.tensor(g, [256, 1, 1])
            w1 = Luminal.tensor(g, [512, 256]); w2 = Luminal.tensor(g, [256, 512])
            vals = [randn(Float32, 512, 256), randn(Float32, 256, 512)]
            g.tensors[(w1.id, 1)] = Luminal.to_device(vals[1], dev)
            g.tensors[(w2.id, 1)] = Luminal.to_device(vals[2], dev)
            out = Luminal.matmul(w2, Luminal.matmul(w1, x))
            inputs = Dict{Int,Any}(x.id => randn(Float32, 256, 1, 1))
            for search in (:static, :measured)
                ex = compile(g; device=dev, retain=[out.id, x.id], free_intermediates=false, search=search,
                             precision=prec, search_inputs=inputs, search_cache=mktempdir())
                got = Array(ex(inputs; device=dev)[out.id])[:, 1, 1]
                stored = [v for v in ex.cg.results if v isa T]
                @test length(stored) == 2
                D1, D2 = (Array(Luminal.dequantize(T(Luminal.to_device(v, dev)))) for v in vals)
                @test got ≈ D2 * (D1 * inputs[x.id][:, 1, 1]) rtol = 1e-4
            end
        end
    end
end
