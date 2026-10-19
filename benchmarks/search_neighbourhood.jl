# search_neighbourhood.jl -- is the coordinate-descent optimum beatable by a coordinated move?
#
# Coordinate descent changes one decision at a time, so it can miss improvements that need two decisions to change together. This
# runs it, then evaluates around its result: every single-decision change, and as many two-decision changes as fit in the same wall
# time as the search. Candidates are timed against the incumbent in interleaved rounds (cheap pass), then the best are re-timed
# in 15 interleaved rounds. A genome search is only worth building if some neighbour wins by 2% or more.
#
# Usage: julia --project=. benchmarks/search_neighbourhood.jl [plen ...]      (stop the model server first: this uses the GPU)

using Luminal, Luminal.NN, Luminal.EGraphRewrite
using AMDGPU, Printf, Statistics, Random

const DIR = joinpath(@__DIR__, "..", "tinyllama_chat")
const LENGTHS = isempty(ARGS) ? [16, 64] : parse.(Int, ARGS)
const DEV = get_device()
const _Runner = Luminal.EGraphRewrite._Runner

say(args...) = (println(args...); flush(stdout))

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
    say("\n== prefill, $plen tokens: $(length(g.nodes)) nodes")
    sp = search_space(g; search=:measured, precision=:activations, search_inputs=inputs, search_tolerance=1e-2,
                      device=DEV, retain=retain, free_intermediates=false)   # no HIP graph capture (see search_headroom.jl)
    ds = decisions(sp.rw)

    t0 = time()
    cd, _ = measured_search(sp.rw, sp.make_runner; extra=sp.lowering_decisions, release=release!, log=false)
    t_cd = time() - t0
    @printf("coordinate descent: %.0f s, %d forced choices\n", t_cd, length(cd)); flush(stdout)

    # The moves: each is a function choices -> choices' that changes one decision of the incumbent.
    moves = Tuple{String,Base.Function}[]
    for (label, opts) in ds
        keys_of_d = Set{Any}(k for o in opts for k in keys(o))
        for (i, o) in enumerate(opts)
            push!(moves, ("$(first(split(label, ':'))) opt $i", c -> (n = Dict{Any,String}(k => v for (k, v) in c if !(k in keys_of_d)); merge!(n, o); n)))
        end
    end
    for (label, opts) in sp.lowering_decisions(cd)
        site = only(keys(opts[2]))
        push!(moves, ("lowering $label", c -> (n = copy(c); get(n, site, "on") == "off" ? delete!(n, site) : (n[site] = "off"); n)))
    end
    # pairs touch two different decisions; singles are the moves themselves
    singles = [(l, m(cd)) for (l, m) in moves]
    filter!(x -> x[2] != cd, singles)
    unique!(x -> x[2], singles)
    say("neighbourhood: ", length(singles), " distinct single-decision changes")
    rng = MersenneTwister(plen)
    function random_pair()
        (la, ma), (lb, mb) = moves[rand(rng, eachindex(moves))], moves[rand(rng, eachindex(moves))]
        return ("$la + $lb", mb(ma(cd)))
    end

    inc = sp.make_runner(cd)
    results = Tuple{Float64,String,Any}[]                 # ratio candidate / incumbent (cheap pass), label, choices
    nsingle = npair = nrej = 0
    function evaluate(label, choices)
        r = sp.make_runner(choices)
        if r === nothing
            nrej += 1; return
        end
        ms = interleaved([inc, r], 3)
        push!(results, (ms[2] / ms[1], label, choices))
        release!(r)
    end
    for (l, c) in singles
        evaluate(l, c); nsingle += 1
    end
    t1 = time()
    while time() - t1 < t_cd
        l, c = random_pair()
        c == cd && continue
        evaluate(l, c); npair += 1
    end
    ratios = first.(results)
    @printf("evaluated %d single and %d pair neighbours (%d rejected by verification); cheap-pass ratio vs incumbent: median %.3f, best %.3f, %d below 0.98\n",
            nsingle, npair, nrej, median(ratios), minimum(ratios), count(<(0.98), ratios)); flush(stdout)

    # head to head for the best few
    top = sort(results; by=first)[1:min(6, length(results))]
    runs = Any[inc, (sp.make_runner(t[3]) for t in top)...]
    keep = [true; [r !== nothing for r in runs[2:end]]]
    runs = [r for (r, k) in zip(runs, keep) if k]; labels = ["coordinate descent (incumbent)"; [t[2] for (t, k) in zip(top, keep[2:end]) if k]]
    ms = interleaved(runs, 15)
    say("head to head (15 interleaved rounds, median ms per run):")
    for (n, m) in zip(labels, ms)
        @printf("  %-70s %8.2f ms  %+6.2f%% vs incumbent\n", first(n, 70), m, 100 * (m / ms[1] - 1)); flush(stdout)
    end
    best = minimum(ms[2:end])
    @printf("RESULT %d tokens: best neighbour %.2f ms vs coordinate descent %.2f ms: %+.2f%% (gate: -2%% or better)\n",
            plen, best, ms[1], 100 * (best / ms[1] - 1)); flush(stdout)
    foreach(release!, runs); sp = nothing; GC.gc()
end
