# Prefill and decode speed of a Qwen3.5 / Qwen3.6 GGUF at several prompt lengths.
#
# Usage:
#   julia -t auto,1 --project=. examples/qwen35_gguf_bench.jl <file.gguf> [--int4|--int8|--f16]
#       [--lengths=512,2048,4096] [--new=32] [--reps=3] [--max-seq=8191] [--kv=f16|f32]
#       [--min-avail=10] [--stop-after=read|load|session|compile-prefill|compile-decode] [--reclaim] [--buffers] [--cold-only]
#   LUMINAL_TRACE_MEM=1 traces the first (cold) call step by step: HIP pool bytes before/after each graph step.
#
# Memory: the GPU shares system RAM, so a stage log line ("[mem] ...") gives MemAvailable, the
# lowest value seen since the previous stage (sampled every 20 ms on an interactive thread, so run with `-t auto,1`), and the HIP pool's used
# bytes. A watchdog thread exits the process at once if MemAvailable falls below --min-avail
# GB (default 10), so the rest of the machine is not squeezed. --stop-after stops cleanly
# after a stage, to walk through the stages one at a time.
#
# For each prompt length (after one untimed call of each kind, which compile and capture): the time to the first token (prefill + one greedy pick, median
# of `reps` after a warm-up that also compiles the graph), and the time per generated
# token over `--new` tokens decoded at that depth. Prompts are fixed pseudo-random token
# ids. Decode attention caps the context at 8191 positions (DECODE_ATTN_MAX_CTX).
using Luminal
using Luminal.NN
using Printf
using Statistics
using AMDGPU

status(args...) = (println(args...); flush(stdout))   # (the output is usually redirected)

# --- memory log and watchdog ----------------------------------------------------------
meminfo_kb() = (d = Dict{String,Int}();
                for l in eachline("/proc/meminfo")
                    m = match(r"^(\w+):\s+(\d+)", l); m === nothing || (d[m[1]] = parse(Int, m[2]))
                end; d)
const MIN_AVAIL_KB = Ref(typemax(Int))     # lowest MemAvailable since the last stage line
const T0 = time()
function start_watchdog(limit_gb)
    Threads.@spawn :interactive while true
        av = meminfo_kb()["MemAvailable"]
        MIN_AVAIL_KB[] = min(MIN_AVAIL_KB[], av)
        if av < limit_gb * 1024 * 1024
            status(@sprintf("[mem] WATCHDOG: MemAvailable %.1f GB < %d GB, exiting", av / 1024^2, limit_gb))
            ccall(:_exit, Cvoid, (Cint,), 3)
        end
        sleep(0.02)
    end
end
function stage(label)
    d = meminfo_kb()
    pool = try AMDGPU.HIP.memory_pool(AMDGPU.device()); catch; nothing end
    pu = pool === nothing ? NaN : AMDGPU.HIP.used_memory(pool) / 1024^3        # live allocations
    pr = pool === nothing ? NaN : AMDGPU.HIP.reserved_memory(pool) / 1024^3    # held by the pool, live or freed
    status(@sprintf("[mem] %7.1fs %-34s avail %5.1f GB  (min since last %5.1f)  used %5.1f  shared %5.1f  hip pool used %5.1f reserved %5.1f GB",
                    time() - T0, label, d["MemAvailable"] / 1024^2, min(MIN_AVAIL_KB[], d["MemAvailable"]) / 1024^2,
                    (d["MemTotal"] - d["MemAvailable"]) / 1024^2, d["Shmem"] / 1024^2, pu, pr))
    MIN_AVAIL_KB[] = typemax(Int)
end

function main()
    args       = filter(a -> !startswith(a, "--"), ARGS)
    flag(name, default) = (i = findfirst(a -> startswith(a, "--$name="), ARGS); i === nothing ? default : split(ARGS[i], "=", limit=2)[2])
    path       = args[1]
    T          = "--int4" in ARGS ? Luminal.Int4 : "--f16" in ARGS ? Float16 : Int8
    lengths    = parse.(Int, split(flag("lengths", "512,2048,4096"), ","))
    n_new      = parse(Int, flag("new", "32"))
    reps       = parse(Int, flag("reps", "3"))
    max_seq    = parse(Int, flag("max-seq", "8191"))
    kv         = flag("kv", "f16") == "f32" ? Float32 : Float16
    min_avail  = parse(Int, flag("min-avail", "10"))
    stop_after = flag("stop-after", "")
    reclaim    = "--reclaim" in ARGS      # trim the HIP memory pool after every call
    start_watchdog(min_avail)
    stage("start")
    stopped(name) = (stage("after " * name); stop_after == name && (status("stopping after $name"); true))
    f   = read_gguf(path)
    cfg = qwen35_config(f)
    stopped("read") && return
    status("$(f.metadata["general.name"]): weights $T, KV cache $kv, max_seq $max_seq, WMMA ",
            get(ENV, "LUMINAL_WMMA", "1") != "0")
    t_load = @elapsed W = load_qwen35_gguf(path, cfg; quantize=_ -> T)
    status(@sprintf("load %.1f s", t_load))
    stopped("load") && return
    s = LlamaSession(Qwen35(Luminal.Graph(), nothing; cfg...), nothing, path; max_seq=max_seq,
                     weights=W, decode_weights=T, kv_dtype=kv)
    stopped("session") && return
    # build the graphs ahead of time, one stage each (generate would do the same on first use)
    Luminal.Decoding._prefill_graph(s, Luminal.Decoding._prefill_bucket(first(lengths)), 1)
    stopped("compile-prefill") && return
    Luminal.Decoding._decode_graph(s, 1)
    stopped("compile-decode") && return
    ids(n) = [1000 + (i * 7919) % 20000 for i in 1:n]
    # --buffers: after the first run of each prefill graph, what its result buffers hold
    function dump_buffers(n)
        cg = s.prefill[(Luminal.Decoding._prefill_bucket(n), 1)].exec
        # the model's own weights (shared with the graph, already counted after the load)
        wset = Set{UInt}()
        function addw!(x)
            x isa AbstractArray && push!(wset, objectid(x))
            x isa Union{Luminal.QuantWeight, Luminal.Q4Weight, Luminal.HalfWeight, Luminal.HalfWeightN} &&
                foreach(f -> addw!(getfield(x, f)), fieldnames(typeof(x)))
        end
        foreach(addw!, values(s.weights))
        seen = Set{UInt}(); by = Dict{String,Tuple{Int,Float64}}(); big = Tuple{Float64,String,Any}[]
        for id in eachindex(cg.results)
            isassigned(cg.results, id) || continue
            r = cg.results[id]
            (r isa AbstractArray && r isa Union{Array, AMDGPU.ROCArray}) || continue
            (objectid(r) in seen || objectid(r) in wset) && continue; push!(seen, objectid(r))
            b = length(r) * sizeof(eltype(r)); k = string(typeof(cg.graph.nodes[id].op).name.name)
            c, t = get(by, k, (0, 0.0)); by[k] = (c + 1, t + b); push!(big, (b, k, size(r)))
        end
        status(@sprintf("[buffers] prompt %d: %d result buffers outside the weights, %.2f GB", n, length(seen), sum(v[2] for v in values(by); init=0.0) / 1e9))
        for (k, (c, t)) in sort(collect(by); by = x -> -x[2][2])[1:min(6, end)]
            status(@sprintf("[buffers]   %-16s %5d buffers %7.2f GB", k, c, t / 1e9))
        end
        sort!(big; by = x -> -x[1])
        for (b, k, d) in big[1:min(4, end)]; status(@sprintf("[buffers]   largest: %7.0f MB %s %s", b / 1e6, k, d)); end
        stage("before forced GC")
        GC.gc(true); GC.gc(true); AMDGPU.HIP.reclaim()
        stage("after full GC + pool trim")
    end
    # one timed call, logged: memory growth per call shows up here
    function timed(label, n_tokens, p)
        t = @elapsed generate_ids(s, [p]; max_new_tokens=n_tokens)
        reclaim && Luminal.reclaim!(s.device)
        stage(label)
        return t
    end
    status(@sprintf("%8s %12s %14s %12s %12s", "prompt", "TTFT (ms)", "prefill tok/s", "TPOT (ms)", "decode tok/s"))
    for n in lengths
        n + n_new < max_seq || (status("skip $n (needs max_seq > $(n + n_new))"); continue)
        p = ids(n)
        t_cold = timed("prompt $n: cold (1 token)", 1, p)
        # the first decode call captures the HIP graph (seconds); keep it out of the timed runs
        n_new > 1 && timed("prompt $n: decode warm-up", n_new, p)
        get(ENV, "LUMINAL_TRACE_MEM", "") != "" && (ENV["LUMINAL_TRACE_MEM"] = "")   # trace the cold call only
        "--buffers" in ARGS && dump_buffers(n)
        "--cold-only" in ARGS && (status("stopping after the first call"); return)
        ttft = median([timed("prompt $n: ttft run $r", 1, p) for r in 1:reps])
        tfull = median([timed("prompt $n: $n_new tokens run $r", n_new, p) for r in 1:reps])
        tpot = (tfull - ttft) / (n_new - 1)
        status(@sprintf("%8d %12.1f %14.1f %12.2f %12.1f   (cold %.1f s)", n, 1e3 * ttft, n / ttft, 1e3 * tpot, 1 / tpot, t_cold))
    end
end

main()
