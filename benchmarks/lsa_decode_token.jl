# lsa_decode_token.jl -- time per decode token on the full Qwen3.8-27B, dense attention against Latent Sparse Attention, as the context fills.
#
# The method of examples/qwen35_context_bench.jl: the full-attention layers' KV cache is filled with random values (a step's time does not
# depend on them) and one plain decode step is timed at each length. For LSA the index keys and buckets of the random cache are then built
# by `lsa_after_prefill!`, once, up to the longest length - the state is valid for every shorter one. Dense first, then `lsa_enable!`.
#
# max_seq defaults to 262,144, the most the dense split-K kernel takes (cld(T, 256) <= 1024), so dense and LSA are timed on the same cache.
#
# Usage: julia -t auto,1 --project=. benchmarks/lsa_decode_token.jl <file.gguf> <maps.safetensors> [--lens=4096,16384,65536,131072,262000]
#     [--max-seq=262144] [--reps=20] [--top-k=4096] [--top-p=1024] [--min-avail=15] [--dense-only] [--lsa-only]
#   (uses the GPU - stop the model server first. About 15 GB of weights, 16 GiB of KV cache at 262,144 positions, and the index state)
using Luminal, Luminal.NN, Printf, Statistics, AMDGPU
using Luminal: LSAConfig, lsa_load_maps
using Luminal.Decoding: lsa_enable!, lsa_disable!

status(args...) = (println(args...); flush(stdout))
flag(name, default) = (i = findfirst(a -> startswith(a, "--$name="), ARGS); i === nothing ? default : split(ARGS[i], "=", limit = 2)[2])
meminfo_kb() = (d = Dict{String,Int}(); for l in eachline("/proc/meminfo"); m = match(r"^(\w+):\s+(\d+)", l); m === nothing || (d[m[1]] = parse(Int, m[2])); end; d)
avail_gb() = meminfo_kb()["MemAvailable"] / 1024^2
function start_watchdog(limit_gb)
    Threads.@spawn :interactive while true
        if avail_gb() < limit_gb
            status(@sprintf("WATCHDOG: MemAvailable %.1f GB under %d GB, exiting", avail_gb(), limit_gb)); ccall(:_exit, Cvoid, (Cint,), 3)
        end
        sleep(0.02)
    end
end

# Random Float16 K/V in a cache array, filled in blocks so the host never holds the whole thing
function fill_random!(a, scale)
    n = size(a, 2); block = 8192
    for i in 0:block:n-1
        m = min(block, n - i)
        view(a, :, i+1:i+m, :, :) .= ROCArray(Float16.(scale .* randn(Float32, size(a, 1), m, size(a, 3), size(a, 4))))
    end
    return a
end

function time_steps(s, lens, reps)
    dc = Luminal.Decoding._decode_graph(s, 1)
    cache, idg = dc.cache, dc.idg
    ms = Float64[]
    for L in lens
        step() = (cache.positions .= L; AMDGPU.synchronize(); t = @elapsed(begin
                      Luminal.NN.llama_decode_step!(dc.exec, idg, cache, 1000; device = s.device); AMDGPU.synchronize() end); t)
        step(); step()                                                       # the first calls warm up and then capture the HIP graph
        times = [step() for _ in 1:reps]
        push!(ms, 1000 * median(times))
        status(@sprintf("   L = %7d: %8.2f ms/token  (min %.2f, max %.2f)   MemAvailable %.1f GB", L, ms[end], 1000 * minimum(times), 1000 * maximum(times), avail_gb()))
    end
    return ms
end

function prepare_cache!(s)
    dc = Luminal.Decoding._decode_graph(s, 1)
    for (i, l) in enumerate(s.model.layers)
        Luminal.NN._is_state_layer(l) && continue
        for a in dc.cache.self_cache[i]
            fill_random!(a, 0.5f0)
        end
    end
    AMDGPU.synchronize()
    return dc
end

function main()
    args = filter(a -> !startswith(a, "--"), ARGS)
    path, maps_path = args[1], args[2]
    lens = parse.(Int, split(flag("lens", "4096,16384,65536,131072,262000"), ","))
    max_seq, reps = parse(Int, flag("max-seq", "262144")), parse(Int, flag("reps", "20"))
    top_k, top_p = parse(Int, flag("top-k", "4096")), parse(Int, flag("top-p", "1024"))
    start_watchdog(parse(Int, flag("min-avail", "15")))
    maximum(lens) < max_seq || error("the longest length must be under max_seq $max_seq")
    status(@sprintf("start: MemAvailable %.1f GB", avail_gb()))
    f = read_gguf(path); cfg = qwen35_config(f)
    tok = LlamaTokenizer(f)
    T = Luminal.Int4
    t = @elapsed W = load_qwen35_gguf(path, cfg; quantize = _ -> T)
    status(@sprintf("weights loaded in %.0f s, MemAvailable %.1f GB", t, avail_gb()))
    s = LlamaSession(Qwen35(Luminal.Graph(), nothing; cfg...), tok, path; max_seq = max_seq, weights = W, decode_weights = T, kv_dtype = Float16)
    n_full = count(l -> !Luminal.NN._is_state_layer(l), s.model.layers)
    dense_ms = nothing; lsa_ms = nothing

    if !("--lsa-only" in ARGS)
        status("\n== dense attention, $(cfg.n_layers) layers ($n_full full-attention)")
        t = @elapsed prepare_cache!(s)
        status(@sprintf("   graph built and cache filled in %.0f s, MemAvailable %.1f GB", t, avail_gb()))
        dense_ms = time_steps(s, lens, reps)
    end

    if !("--dense-only" in ARGS)
        status("\n== Latent Sparse Attention, top_k $top_k, top_p $top_p")
        maps = lsa_load_maps(maps_path)
        lsa_enable!(s, LSAConfig(maps = maps, top_k = top_k, top_p = top_p))
        t = @elapsed dc = prepare_cache!(s)
        status(@sprintf("   graph built and cache filled in %.0f s, MemAvailable %.1f GB", t, avail_gb()))
        st = Luminal.LSA_SESSIONS[s]
        t = @elapsed Luminal.lsa_after_prefill!(st, dc.cache, maximum(lens) + 1)
        status(@sprintf("   index keys and buckets of %d positions, %d layers: %.1f s", maximum(lens) + 1, length(st.layers), t))
        lsa_ms = time_steps(s, lens, reps)
    end

    if dense_ms !== nothing && lsa_ms !== nothing
        status("\n== time per token")
        status(@sprintf("%9s %12s %12s %9s", "context", "dense ms", "LSA ms", "speedup"))
        for (i, L) in enumerate(lens)
            status(@sprintf("%9d %12.2f %12.2f %8.2fx", L, dense_ms[i], lsa_ms[i], dense_ms[i] / lsa_ms[i]))
        end
    end
end

main()
