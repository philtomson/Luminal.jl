# prefill_tolerance.jl -- how far is the static extraction of a TinyLlama prefill graph from the original graph?
#
# `LlamaSession(...; search=:measured)` compiles its prefill graph with `compile(g; search=:measured, precision=true, search_inputs=zeros)`; the
# search first checks the static extraction (Float16 weights, fused ops) against the original graph on the search inputs, relative to the largest
# reference value of each output, and demands less than `search_tolerance` (default 1e-3). This prints that error for the logits and for the K and
# V of the first and last layer, at several prompt lengths, for the zero inputs the search uses and for random tokens.
#
# Usage: julia --project=. benchmarks/prefill_tolerance.jl [len ...]      (stop the model server first: this uses the GPU)

using Luminal, Luminal.NN
using AMDGPU, Printf, Statistics

const DIR = joinpath(@__DIR__, "..", "tinyllama_chat")
const LENGTHS = isempty(ARGS) ? [6, 8, 15, 16, 32, 64] : parse.(Int, ARGS)
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
    return g, inp, out, kvs
end

relerr(got, want) = maximum(abs.(got .- want); init=0f0) / max(maximum(abs, want; init=0f0), 1f-6)

println("relative error of the static extraction (precision=true) against the original graph")
for plen in LENGTHS
    g, inp, out, kvs = prefill_graph(plen)
    retain = vcat(out.id, [k.id for (k, _) in kvs], [v.id for (_, v) in kvs], inp.id)
    ref_cg = compile(g; device=DEV, retain=retain)
    st_cg = compile(g; device=DEV, retain=retain, search=:static, precision=true)
    for (label, ids) in (("zero tokens", zeros(Float32, plen, 1)), ("random tokens", Float32.(rand(0:31999, plen, 1))))
        inputs = Dict{Int,Any}(inp.id => Luminal.to_device(ids, DEV))
        r = ref_cg(inputs; device=DEV); s = st_cg(inputs; device=DEV)
        e(id) = relerr(Array{Float32}(s[id]), Array{Float32}(r[id]))
        ek = [e(k.id) for (k, _) in kvs]; ev = [e(v.id) for (_, v) in kvs]
        @printf("  len %-3d %-13s logits %.1e   K layer 1 %.1e, 22 %.1e   V layer 1 %.1e, 22 %.1e   worst of all outputs %.1e\n",
                plen, label, e(out.id), ek[1], ek[end], ev[1], ev[end], maximum(vcat(e(out.id), ek, ev)))
    end
    Luminal.release!(ref_cg); Luminal.release!(st_cg.cg)
end
