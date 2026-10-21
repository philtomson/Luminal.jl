# gemma_window.jl -- does recognising Gemma3's sliding-window attention (C1: causal-attention-{gqa,mha}-window) pay off on a real model?
#
# Gemma3-4B prefill graphs in two forms, both with exact Float32 matmuls (precision=false, so only the fusion differs):
#   composite  the model as it is (windowed local layers: score matrix, mask from primitives, softmax), compiled as is (search=:none)
#   fused      the same graph through the e-graph's static extraction (the fusion rules turn each attention into one CausalAttention)
# Each is timed in interleaved rounds (median ms); the fused logits are compared with the composite's. The window mask only exists for
# prompts longer than the window (1024), so the default lengths are above it.
#
# Usage: julia --project=. benchmarks/gemma_window.jl [plen ...]      (stop the model server first: this uses the GPU)

using Luminal, Luminal.NN
using AMDGPU, Printf, Statistics

const DIR = joinpath(@__DIR__, "..", "gemma3_4b")
const LENGTHS = isempty(ARGS) ? [1280, 2048] : parse.(Int, ARGS)
const DEV = get_device()

model = model_template(DIR)
wd = load_weights_to_dict(DIR; device=DEV)
V = model.config.vocab_size

function build(plen)
    g = Graph(); reg = WeightRegistry()
    m = Luminal.Decoding._rebuild_model_like(model, g, reg; rope_base=model.rope_base)
    inp = Luminal.tensor(g, [plen, 1])
    out, kvs = m(inp, 0; return_kv=true)
    load_weights!(g, reg, wd; device=DEV)
    return g, inp, out, vcat(out.id, [k.id for (k, _) in kvs], [v.id for (_, v) in kvs], inp.id)
end

attn_nodes(g) = [n.op.window for n in g.nodes if n.op isa Luminal.CausalAttention]
timed(run, n) = (t = time(); for _ in 1:n; run(); end; (time() - t) / n)
nsteps(run; budget=0.5) = (run(); run(); t = (t0 = time(); run(); time() - t0); clamp(round(Int, budget / max(t, 1e-6)), 1, 10))
function interleaved(runs, rounds)
    ns = [nsteps(r) for r in runs]
    ts = [Float64[] for _ in runs]
    for _ in 1:rounds, (i, r) in enumerate(runs)
        push!(ts[i], timed(r, ns[i]))
    end
    return [1e3 * median(t) for t in ts]
end

for plen in LENGTHS
    println("\n== prefill, $plen tokens (window $(model.config.sliding_window))")
    ids = Luminal.to_device(Float32.(rand(0:V-1, plen, 1)), DEV)
    runs = Base.Function[]; names = String[]; logits = Any[]
    for (name, search) in (("composite", :none), ("fused", :static))
        g, inp, out, retain = build(plen)
        inputs = Dict{Int,Any}(inp.id => ids)
        t0 = time()
        exec = search === :none ? compile(g; device=DEV, retain=retain) :
               compile(g; device=DEV, retain=retain, search=:static, precision=false, search_inputs=inputs)
        ctime = time() - t0
        run1() = exec(inputs; device=DEV)
        r = run1()
        push!(logits, Array(r[out.id]))
        # (the compiled graph is the one that runs: count its fused attention nodes)
        gc = hasproperty(exec, :graph) ? exec.graph : g
        @printf("%-10s CausalAttention nodes (windows) in the source graph: %s, compile %.1f s\n", name, string(sort!(collect(Set(attn_nodes(g))))), ctime)
        push!(runs, () -> (run1(); AMDGPU.synchronize()))
        push!(names, name)
    end
    t = interleaved(runs, 5)
    ref = logits[1]
    for (i, name) in enumerate(names)
        err = maximum(abs.(logits[i] .- ref)) / max(maximum(abs.(ref)), 1f-6)
        @printf("  %-10s %9.2f ms   (%.2fx composite)   logits rel. diff vs composite %.1e   argmax agree %.4f\n", name, t[i], t[1] / t[i], err,
                mean(argmax(logits[i], dims=1) .== argmax(ref, dims=1)))
    end
end
