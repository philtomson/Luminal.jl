# prefill_search.jl — how the e-graph search does on TinyLlama *prefill* graphs.
#
# Prefill graphs have the prompt length baked in, so each length is its own
# graph. Prompts can be padded to length buckets (causal attention: tokens after
# the prompt cannot change earlier positions), so one searched graph per bucket
# serves every prompt up to that length. This times, per bucket, steady-state
# runs of the compiled (captured) prefill graph:
#   Float32 weights, Float16 weights, static search, measured search.
#
# Usage: julia --project=. examples/prefill_search.jl [model_dir] [lengths...]

using Luminal, Luminal.NN, Luminal.EGraphRewrite
using AMDGPU, Printf, Statistics

const DIR = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "tinyllama_chat")
const LENGTHS = length(ARGS) >= 2 ? parse.(Int, ARGS[2:end]) : [16, 64, 256]
const DEV = get_device()

model = NN.Llama(Graph(), WeightRegistry(); vocab_size=32000, hidden=2048, n_layers=22,
                 n_heads=32, n_kv_heads=4, intermediate=5632, rope_base=10000f0)
wd = load_weights_to_dict(DIR; device=DEV)

function prefill_graph(plen)
    g = Graph(); reg = WeightRegistry()
    m = Luminal.Decoding._rebuild_model_like(model, g, reg; rope_base=10000f0)
    inp = Luminal.tensor(g, [plen, 1])
    out, kvs = m(inp, 0; return_kv=true)
    load_weights!(g, reg, wd; device=DEV)
    retain = vcat(out.id, [k.id for (k, _) in kvs], [v.id for (_, v) in kvs], inp.id)
    return g, inp, out, retain
end

function steady_ms(ex, inputs; n=10)
    for _ in 1:3; ex(inputs; device=DEV); end          # warm-up, capture, replay
    AMDGPU.synchronize()
    ts = Float64[]
    for _ in 1:n
        t = time(); ex(inputs; device=DEV); AMDGPU.synchronize(); push!(ts, time() - t)
    end
    return 1e3 * median(ts)
end

for plen in LENGTHS
    g, inp, out, retain = prefill_graph(plen)
    ids = Luminal.to_device(Float32.(rand(0:31999, plen, 1)), DEV)   # on device: no copy per run
    inputs = Dict{Int,Any}(inp.id => ids)
    common = (device=DEV, retain=retain, free_intermediates=false, capture=true)
    ref = Array{Float32}(compile(g; common...)(inputs; device=DEV)[out.id])
    relerr(ex) = (r = Array{Float32}(ex(inputs; device=DEV)[out.id]); maximum(abs.(r .- ref)) / maximum(abs.(ref)))

    println("\n== prompt length $plen: $(length(g.nodes)) nodes")
    for (label, kw) in (("f32 weights", (;)), ("f16 weights", (weight_dtype=Float16,)),
                        ("static search (+precision)", (search=:static, precision=true)))
        t0 = time()
        ex = compile(g; common..., kw...)
        tc = time() - t0
        @printf("  %-28s %8.2f ms/run   relerr %.1e   (compile %.1f s)\n", label, steady_ms(ex, inputs), relerr(ex), tc)
        ex = nothing; GC.gc()
    end
    t0 = time()
    ex = compile(g; common..., search=:measured, precision=true, search_inputs=inputs)
    tc = time() - t0
    @printf("  %-28s %8.2f ms/run   relerr %.1e   (search + compile %.0f s)\n",
            "measured search (+precision)", steady_ms(ex, inputs), relerr(ex), tc)
    ex = nothing; GC.gc()
end
