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
function evaluate(rw, choices; label="")
    ng, m = extract_graph(rw; choices=choices)
    nidg = remap(idg, m)
    cg, cache = prepare(ng, nidg)
    ls = run_steps!(cg, nidg, cache, 4)
    err = maximum(relerr.(ls, REF))
    t = time_steps!(cg, nidg, cache)
    cg = nothing; GC.gc()
    @printf("  %-44s %4d nodes  %.2f ms  relerr %.1e\n", label, length(ng.nodes), 1e3t, err)
    return t, err
end

for precision in (false, true)
    println("\n== rules: canonical + algebraic", precision ? " + precision (Float16 weights)" : "")
    t0 = time()
    rw = to_egraph(dg, outputs(idg))
    rep = saturate_graph!(rw; precision=precision)
    groups = choice_groups(rw)
    @printf("e-graph: %d e-classes, saturation %s, %.1f s, %d choice groups\n",
            length(rw.g.classes), rep.reason, time() - t0, length(groups))
    for (key, cls, sigs) in groups
        println("  group x", length(cls), " ", key[2], ": ", join(sigs, "  |  "))
    end

    # Static cost model's pick
    t_static, e_static = evaluate(rw, Dict{Luminal.EGraphRewrite.Id,String}(); label="static-cost extraction")

    # Measured coordinate descent: per group, try each alternative; keep improvements.
    choices = Dict{Luminal.EGraphRewrite.Id,String}()
    best = t_static
    for (key, cls, sigs) in groups
        winner = nothing
        for sig in sigs
            trial = copy(choices)
            for c in cls; trial[c] = sig; end
            t, err = evaluate(rw, trial; label=string("group ", key[2], " -> ", sig))
            if err < 1e-3 && t < best * 0.98
                best, winner = t, sig
            end
        end
        if winner !== nothing
            for c in cls; choices[c] = winner; end
        end
    end
    @printf("measured search result: %.2f ms (static pick %.2f ms)\n", 1e3best, 1e3t_static)
    println("choices kept: ", unique(values(choices)))
end
