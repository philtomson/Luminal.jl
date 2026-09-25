# egraph_search.jl — explore equivalent TinyLlama decode graphs with the e-graph
# rewrite layer and pick the fastest by measurement.
#
#   decode Graph -> e-graph -> saturate (rules) -> choice groups
#     -> candidate graphs (one alternative per group) -> compile + HIP capture
#     -> verify against the original graph -> time -> keep the fastest
#
# Usage: julia --project=. examples/egraph_search.jl [model_dir]

using Luminal, Luminal.NN, Luminal.EGraphRewrite
using AMDGPU, Printf, Statistics

const DIR = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "tinyllama_chat")
const DEV = get_device()
const MAX_SEQ = 256

model = NN.Llama(Graph(), WeightRegistry(); vocab_size=32000, hidden=2048, n_layers=22,
                 n_heads=32, n_kv_heads=4, intermediate=5632, rope_base=10000f0)
wd = load_weights_to_dict(DIR; device=DEV)

function decode_graph()
    dg = Graph(); dreg = WeightRegistry()
    dm = Luminal.Decoding._rebuild_model_like(model, dg, dreg; rope_base=10000f0)
    idg = build_llama_decode_step!(dm, dg, Luminal.Sym{Int}(:pos); max_seq=MAX_SEQ, rope_base=10000f0)
    load_weights!(dg, dreg, wd; device=DEV)
    return dg, idg
end

remap(idg, m) = LlamaDecodeGraph(m[idg.token_input_id], m[idg.pos_input_id],
    [m[i] for i in idg.self_k_ids], [m[i] for i in idg.self_v_ids], m[idg.logits_id],
    [m[i] for i in idg.new_self_k_ids], [m[i] for i in idg.new_self_v_ids], idg.step_pos)

outputs(idg) = vcat(idg.logits_id, idg.new_self_k_ids, idg.new_self_v_ids)

# A fixed, realistic cache state (random K/V, first 40 slots filled), shared by all candidates.
const KV0 = [(Luminal.to_device(randn(Float32, 64, MAX_SEQ, 4, 1), DEV),
              Luminal.to_device(randn(Float32, 64, MAX_SEQ, 4, 1), DEV)) for _ in 1:22]

function prepare(g, idg; weight_dtype=Float32)
    retain = vcat(outputs(idg), idg.token_input_id, idg.pos_input_id, idg.self_k_ids, idg.self_v_ids)
    cg = compile(g; device=DEV, retain=retain, free_intermediates=false,
                 weight_dtype=weight_dtype, capture=true)
    cache = LlamaKVCacheState(22, 4, 64; max_seq=MAX_SEQ, device=DEV)
    for i in 1:22
        copyto!(cache.self_cache[i][1], KV0[i][1]); copyto!(cache.self_cache[i][2], KV0[i][2])
    end
    return cg, cache
end

# Run `n` steps at positions 40.. with a fixed token sequence; return logits per step.
function run_steps!(cg, idg, cache, n)
    cache.step_pos = 40
    ls = Vector{Vector{Float32}}()
    for s in 1:n
        l = llama_decode_step!(cg, idg, cache, 100 + 7s; device=DEV)
        push!(ls, vec(Array{Float32}(l)))
    end
    return ls
end

function time_steps!(cg, idg, cache; n=30)
    run_steps!(cg, idg, cache, 3)           # warm-up, capture
    ts = Float64[]
    cache.step_pos = 40
    for s in 1:n
        t = time()
        l = llama_decode_step!(cg, idg, cache, 100 + 7s; device=DEV)
        Array(l); push!(ts, time() - t)
    end
    return median(ts)
end

relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

# ── Reference: the original graph ──────────────────────────────────────────
dg, idg = decode_graph()
println("decode graph: ", length(dg.nodes), " nodes")
cg, cache = prepare(dg, idg)
REF = run_steps!(cg, idg, cache, 4)
t_f32 = time_steps!(cg, idg, cache)
cg = nothing; GC.gc()
cg, cache = prepare(dg, idg; weight_dtype=Float16)
t_f16 = time_steps!(cg, idg, cache)
cg = nothing; GC.gc()
@printf("original graph:   f32 weights %.2f ms   f16 weights %.2f ms\n", 1e3t_f32, 1e3t_f16)

# ── Candidates from the e-graph ────────────────────────────────────────────
# make_runner(choices): extract, compile with capture, verify against the
# original graph, and return a function that runs one synchronized step.
function runner_for(rw)
    return function (choices)
        ng, m = extract_graph(rw; choices=choices)
        nidg = remap(idg, m)
        cg, cache = prepare(ng, nidg)
        ls = run_steps!(cg, nidg, cache, 4)
        err = maximum(relerr.(ls, REF))
        err < 1e-3 || return nothing
        return () -> begin
            cache.step_pos = 40
            Array(llama_decode_step!(cg, nidg, cache, 107; device=DEV))
            nothing
        end
    end
end

for precision in (false, true)
    println("\n== rules: canonical + algebraic + merged projections",
            precision ? " + precision (Float16 weights)" : "")
    t0 = time()
    rw = to_egraph(dg, outputs(idg))
    rep = saturate_graph!(rw; precision=precision)
    nmerged = merge_projections!(rw; precision=precision)
    ds = decisions(rw)
    @printf("e-graph: %d e-classes, saturation %s, %d merged groups, %d decisions, %.1f s\n",
            length(rw.g.classes), rep.reason, nmerged, length(ds), time() - t0)
    t0 = time()
    choices, t = measured_search(rw, runner_for(rw))
    @printf("measured search: %.2f ms/step with %d forced classes (%.0f s)\n",
            1e3t, length(choices), time() - t0)
end
