# logits_trace.jl (derived from c1_fusion.jl) -- does recognising fused ops from primitives (C1) run as fast as the hand-fused ops?
#
# TinyLlama prefill graphs in five forms, all with exact Float32 matmuls (precision=false, so only the fusion differs):
#   hand         the model as it is: hand-fused RMSNormOp, SwiGLU as primitives (the default Mlp)
#   hand+swiglu  hand-fused RMSNormOp and a hand-built SwiGLUOp
#   prim         RMSNorm and SwiGLU as primitives, compiled as is (search=:none)
#   prim+C1      the same graph through the e-graph's static extraction (the fusion rules recognise RMSNorm and SwiGLU)
#   hand+C1      the `hand` graph through the static extraction (recognises SwiGLU)
# Each is timed in interleaved rounds (median ms) and its logits are compared with `hand`.
#
# Usage: julia --project=. benchmarks/c1_fusion.jl [plen ...]      (stop the model server first: this uses the GPU)

using Luminal, Luminal.NN
using AMDGPU, Printf, Statistics, Random, Serialization

const DIR = joinpath(@__DIR__, "..", "tinyllama_chat")
const LENGTHS = isempty(ARGS) ? [16, 64] : parse.(Int, ARGS)
const DEV = (ENV["TRACE_DEVICE"] == "gpu" ? get_device() : CPUDevice())
const PRIM_NORM = Ref(false)      # build RMSNorm from primitives
const SWIGLU = Ref(false)         # build silu(gate) * up as a SwiGLUOp

# Benchmark-only override of two model calls, to build the same model in the other spelling
function (ln::NN.LayerNorm)(x::Luminal.GraphTensor)
    if !PRIM_NORM[] && !ln.mean_norm && ln.weight !== nothing && ln.bias === nothing
        return Luminal.add_op!(x.graph_ref, Luminal.RMSNormOp(ln.epsilon),
                               [(x.id, 0, x.shape), (ln.weight.id, 0, ln.weight.shape)],
                               Luminal.ShapeTracker(Luminal.realized_dims(x.shape)))
    end
    out = ln.mean_norm ? Luminal.layer_norm(x, 1, ln.epsilon) : Luminal.std_norm(x, 1, ln.epsilon)
    ln.weight !== nothing && (out = out * ln.weight)
    ln.bias !== nothing && (out = out + ln.bias)
    return out
end
function (m::NN.Mlp)(x::Luminal.GraphTensor)
    g = m.gate_proj(x)
    if SWIGLU[] && m.act === :silu
        up = m.up_proj(x)
        h = Luminal.add_op!(x.graph_ref, Luminal.SwiGLUOp(), [(g.id, 0, g.shape), (up.id, 0, up.shape)],
                            Luminal.ShapeTracker(Luminal.realized_dims(g.shape)))
        return m.down_proj(h)
    end
    gate = m.act === :gelu_tanh ? Luminal.gelu(g; approximate=true) : Luminal.silu(g)
    return m.down_proj(gate * m.up_proj(x))
end

model = NN.Llama(Graph(), WeightRegistry(); vocab_size=32000, hidden=2048, n_layers=22,
                 n_heads=32, n_kv_heads=4, intermediate=5632, rope_base=10000f0)
wd = load_weights_to_dict(DIR; device=DEV)

function build(plen; prim_norm, swiglu)
    PRIM_NORM[] = prim_norm; SWIGLU[] = swiglu
    g = Graph(); reg = WeightRegistry()
    m = Luminal.Decoding._rebuild_model_like(model, g, reg; rope_base=10000f0)
    inp = Luminal.tensor(g, [plen, 1])
    out, kvs = m(inp, 0; return_kv=true)
    load_weights!(g, reg, wd; device=DEV)
    return g, inp, out, vcat(out.id, [k.id for (k, _) in kvs], [v.id for (_, v) in kvs], inp.id)
end

counts(g) = (count(n -> n.op isa Luminal.RMSNormOp, g.nodes), count(n -> n.op isa Luminal.SwiGLUOp, g.nodes), length(g.nodes))

timed(run, n) = (t = time(); for _ in 1:n; run(); end; (time() - t) / n)
nsteps(run; budget=0.3) = (run(); run(); t = (t0 = time(); run(); time() - t0); clamp(round(Int, budget / max(t, 1e-6)), 1, 20))
function interleaved(runs, rounds)
    ns = [nsteps(r) for r in runs]
    ts = [Float64[] for _ in runs]
    for _ in 1:rounds, (i, r) in enumerate(runs)
        push!(ts[i], timed(r, ns[i]))
    end
    return [1e3 * median(t) for t in ts]
end


# Usage: TRACE_DEVICE=cpu|gpu TRACE_OUT=file julia --project=. benchmarks/logits_trace.jl
# Logits of TinyLlama's 16-token prefill in four spellings (hand, hand+swiglu, prim, prim+C1), Float32 matmuls, fixed token ids;
# the hand graph is run twice (determinism). Saved to TRACE_OUT for `logits_trace_compare.jl`.
plen = 16
ids = Luminal.to_device(Float32.(rand(MersenneTwister(1), 0:31999, plen, 1)), DEV)
function logits_of(; prim_norm, swiglu, search=:none)
    g, inp, out, retain = build(plen; prim_norm=prim_norm, swiglu=swiglu)
    inputs = Dict{Int,Any}(inp.id => ids)
    exec = search === :none ? compile(g; device=DEV, retain=retain, free_intermediates=false) :
           compile(g; device=DEV, retain=retain, free_intermediates=false, search=:static, precision=false, search_inputs=inputs)
    return Float64.(Array(exec(inputs; device=DEV)[out.id]))
end
res = Dict{String,Any}()
res["hand"] = logits_of(prim_norm=false, swiglu=false)
res["hand2"] = logits_of(prim_norm=false, swiglu=false)
res["hand+swiglu"] = logits_of(prim_norm=false, swiglu=true)
res["prim"] = logits_of(prim_norm=true, swiglu=false)
res["prim+C1"] = logits_of(prim_norm=true, swiglu=false, search=:static)
serialize(ENV["TRACE_OUT"], res)
println("saved ", ENV["TRACE_OUT"], " on ", typeof(DEV))
