# search_headroom.jl -- is there headroom beyond coordinate descent in the e-graph search space?
#
# For a TinyLlama prefill graph: run `measured_search` (coordinate descent, the current strategy), then draw random joint
# assignments of every decision (e-graph choice groups, merged projections, lowering sites) for the same wall time, and re-time
# the incumbent and the best random candidates together in interleaved rounds. If the best random assignment is not at least a
# couple of percent faster than the coordinate-descent result, a genome search over the same space has little to find.
#
# Usage: julia --project=. benchmarks/search_headroom.jl [plen ...]      (stop the model server first: this uses the GPU)

using Luminal, Luminal.NN, Luminal.EGraphRewrite
using AMDGPU, Printf, Statistics, Random

const DIR = joinpath(@__DIR__, "..", "tinyllama_chat")
const LENGTHS = isempty(ARGS) ? [16, 64] : parse.(Int, ARGS)
const DEV = get_device()
const _Runner = Luminal.EGraphRewrite._Runner

model = NN.Llama(Graph(), WeightRegistry(); vocab_size=32000, hidden=2048, n_layers=22,
                 n_heads=32, n_kv_heads=4, intermediate=5632, rope_base=10000f0)
wd = load_weights_to_dict(DIR; device=DEV)

function prefill_graph(plen)
    g = Graph(); reg = WeightRegistry()
    m = Luminal.Decoding._rebuild_model_like(model, g, reg; rope_base=10000f0)
    inp = Luminal.tensor(g, [plen, 1])
    out, kvs = m(inp, 0; return_kv=true)
    load_weights!(g, reg, wd; device=DEV)
    return g, inp, out, vcat(out.id, [k.id for (k, _) in kvs], [v.id for (_, v) in kvs], inp.id)
end

release!(r) = r isa _Runner && Luminal.release!(r.r.cg)
timed(run, n) = (t = time(); for _ in 1:n; run(); end; (time() - t) / n)
nsteps(run; budget=0.3) = (run(); run(); t = (t0 = time(); run(); time() - t0); clamp(round(Int, budget / max(t, 1e-6)), 1, 10))

# Interleaved rounds: every candidate is timed once per round, so drift hits all alike. Returns median ms per candidate.
function interleaved(runs, rounds)
    ns = [nsteps(r) for r in runs]
    ts = [Float64[] for _ in runs]
    for _ in 1:rounds, (i, r) in enumerate(runs)
        push!(ts[i], timed(r, ns[i]))
    end
    return [1e3 * median(t) for t in ts]
end

for plen in LENGTHS
    g, inp, out, retain = prefill_graph(plen)
    ids = Luminal.to_device(Float32.(rand(0:31999, plen, 1)), DEV)
    inputs = Dict{Int,Any}(inp.id => ids)
    println("\n== prefill, $plen tokens: $(length(g.nodes)) nodes")
    t0 = time()
    sp = search_space(g; search=:measured, precision=:activations, search_inputs=inputs, search_tolerance=1e-2,
                      device=DEV, retain=retain, free_intermediates=false)   # no HIP graph capture: a Reshape copy in some candidates cannot be captured (fails the same way at 51b13c4)
    ds = decisions(sp.rw)
    @printf("setup %.0f s; %d e-graph decisions (options: %s)\n", time() - t0, length(ds), join((length(o) for (_, o) in ds), ","))

    # 1. coordinate descent
    t0 = time()
    cd_choices, _ = measured_search(sp.rw, sp.make_runner; extra=sp.lowering_decisions, release=release!, log=false)
    t_cd = time() - t0
    @printf("coordinate descent: %.0f s, %d forced choices\n", t_cd, length(cd_choices))

    # 2. random joint assignments for the same wall time
    rng = MersenneTwister(plen)
    function random_choices()
        c = Dict{Any,String}()
        for (_, opts) in ds
            merge!(c, rand(rng, opts))
        end
        for (_, opts) in sp.lowering_decisions(c)
            rand(rng, Bool) && merge!(c, opts[2])
        end
        return c
    end
    pool = Tuple{Float64,Any,Any}[]                       # (quick ms, choices, runner): the 5 fastest are kept
    nsamples = 0; nrejected = 0; quick = Float64[]
    t0 = time()
    while time() - t0 < t_cd
        c = random_choices()
        r = sp.make_runner(c)
        nsamples += 1
        if r === nothing
            nrejected += 1; continue
        end
        ms = interleaved([r], 3)[1]
        push!(quick, ms)
        push!(pool, (ms, c, r)); sort!(pool; by=first)
        while length(pool) > 5
            release!(pop!(pool)[3])
        end
    end
    @printf("random sampling: %d assignments in %.0f s (%d rejected by verification); quick times median %.2f ms, best %.2f ms\n",
            nsamples, time() - t0, nrejected, median(quick), minimum(quick))

    # 3. head to head: the original graph (plain compile, Float32 and Float16 weights), static extraction (cost model after the
    #    rewrite rules), coordinate descent, and the five best random assignments
    plain(; kw...) = (ex = compile(g; device=DEV, retain=retain, free_intermediates=false, kw...);
                    () -> (ex(inputs; device=DEV); AMDGPU.synchronize(); nothing))
    f32_run = plain(); f16_run = plain(weight_dtype=Float16)
    cd_run = sp.make_runner(cd_choices); static_run = sp.make_runner(Dict{Any,String}())
    runs = Any[f32_run, f16_run, static_run, cd_run, (p[3] for p in pool)...]
    names = ["original, f32 weights", "original, f16 weights", "static extraction", "coordinate descent",
             ("random #$i" for i in 1:length(pool))...]
    ms = interleaved(runs, 15)
    println("head to head (15 interleaved rounds, median ms per run; the search uses :activations precision, the originals are exact):")
    for (n, m) in zip(names, ms)
        @printf("  %-24s %8.2f ms   %+6.1f%% vs coordinate descent   %5.2fx vs original f32\n", n, m, 100 * (m / ms[4] - 1), ms[1] / m)
    end
    best = minimum(ms[5:end])
    @printf("RESULT %d tokens: original f32 %.2f ms, f16 %.2f ms, static %.2f ms, coordinate descent %.2f ms (%.2fx over f32); best random %.2f ms: %+.1f%% vs coordinate descent; fraction of quick samples faster than CD (quick-timed): %.2f\n",
            plen, ms[1], ms[2], ms[3], ms[4], ms[1] / ms[4], best, 100 * (best / ms[4] - 1), count(<(ms[4]), quick) / max(length(quick), 1))
    foreach(release!, runs); sp = nothing; GC.gc()
end
