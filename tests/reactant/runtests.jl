# Luminal graphs compiled through Reactant.jl / XLA (ext/LuminalReactantExt.jl),
# checked against Luminal's interpreter. A separate environment, so Reactant is
# never a dependency of Luminal's own:
#   julia --project=tests/reactant tests/reactant/runtests.jl
# (first: julia --project=tests/reactant -e 'using Pkg; Pkg.instantiate()').
# Runs on XLA's CPU backend; set LUMINAL_REACTANT_BACKEND=gpu to try the GPU.
using Test, Luminal, Reactant
using Luminal.NN
Reactant.set_default_backend(get(ENV, "LUMINAL_REACTANT_BACKEND", "cpu"))

dims(g, id) = Int.(Luminal.eval_dim.(Luminal.realized_dims(g.shapes[id])))
relerr(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), 1f-6)

# Compile `outputs` of `g` with Reactant, run on `vals`, compare with `execute`
function check(g, outputs, inputs, vals; tol=1f-5)
    outs = outputs isa AbstractVector ? outputs : [outputs]
    ids = [o.id for o in outs]
    ref = Luminal.execute(g, ids, Dict{Int,Any}(zip([i.id for i in inputs], vals)), CPUDevice())
    fc = reactant_compile(g, outputs, inputs, vals...)
    got = fc(Reactant.to_rarray.(vals)...)
    got = outputs isa AbstractVector ? collect(got) : [got]
    for (k, id) in enumerate(ids)
        @test size(Array(got[k])) == size(ref[id])
        @test relerr(Array(got[k]), ref[id]) < tol
    end
end

@testset "Reactant extension" begin
    @testset "matmul + softmax" begin
        g = Graph()
        x = tensor(g, [64, 8]); W = tensor(g, [32, 64])
        y = softmax(matmul(W, x), 1) * 2.0f0
        check(g, y, [x, W], Any[randn(Float32, 64, 8), randn(Float32, 32, 64)])
    end

    @testset "elementwise and reductions" begin
        g = Graph()
        a = tensor(g, [16, 12]); b = tensor(g, [16, 12])
        e = exp2(a * 0.25f0) + log2(abs(b) + 1f0) * sqrt(abs(a))
        s = sin(a) * cos(b)
        outs = [e, Luminal.sum(e, 2), Luminal.max_reduce(s, 1)]
        check(g, outs, [a, b], Any[randn(Float32, 16, 12), randn(Float32, 16, 12)])
    end

    @testset "rounding and select" begin
        g = Graph()
        a = tensor(g, [16, 12]); c = tensor(g, [16, 1])
        outs = [floor(a), ceil(a), round(a), trunc(a), select(c, a * a, a), select(a < 0f0, 0f0, a)]
        av = randn(Float32, 16, 12) .* 3; av[1:4] .= Float32[2.5, -2.5, 0.5, -1.5]
        check(g, outs, [a, c], Any[av, Float32.(rand(0:1, 16, 1))])
    end

    @testset "views: permute, slice, pad, concat" begin
        g = Graph()
        a = tensor(g, [6, 5, 4]); b = tensor(g, [6, 3, 4])
        p = Luminal.permute(a, [3, 1, 2])
        sl = Luminal.slice_along(a, 2, 1, 4)
        pd = Luminal.pad_along(b, 2, 1, 2)
        cc = Luminal.concat_along(a, b, 2)
        check(g, [p * 1f0, sl * 1f0, pd * 1f0, cc * 1f0], [a, b], Any[randn(Float32, 6, 5, 4), randn(Float32, 6, 3, 4)])
    end

    # A small Llama prefill: embedding (gather), RMSNorm, RoPE, grouped-query
    # attention (batched matmuls), SwiGLU MLP, lm_head
    cfg = (vocab_size=256, hidden=128, intermediate=256, n_layers=2, n_heads=4, n_kv_heads=2, rope_base=10000f0)
    S = 16
    toks = Float32.(Base.reshape(rand(0:255, S), S, 1))

    @testset "tiny Llama, weights as inputs" begin
        g = Graph(); reg = WeightRegistry()
        m = Llama(g, reg; cfg...)
        inp = tensor(g, [S, 1]); out = m(inp, 0)
        ws = [GraphTensor(id, g.shapes[id], g) for id in sort!(collect(values(reg.mapping)))]
        wv = [randn(Float32, dims(g, w.id)...) .* 0.05f0 for w in ws]
        check(g, out, [inp; ws], Any[toks, wv...]; tol=1f-4)
    end

    @testset "tiny Llama, loaded weights baked in; StableHLO" begin
        g = Graph(); reg = WeightRegistry()
        m = Llama(g, reg; cfg...)
        inp = tensor(g, [S, 1]); out = m(inp, 0)
        W = Dict{String,Any}(n => randn(Float32, dims(g, id)...) .* 0.05f0 for (n, id) in reg.mapping)
        load_weights!(g, reg, W; device=CPUDevice())
        check(g, out, [inp], Any[toks]; tol=1f-4)
        hlo = to_stablehlo(g, out, [inp], toks)
        @test occursin("stablehlo.dot_general", hlo)
        @test occursin("func.func @main", hlo)
    end
end
