# Prefill and decode speed of a Qwen3.5 / Qwen3.6 GGUF at several prompt lengths.
#
# Usage:
#   julia -t auto,1 --project=. examples/qwen35_gguf_bench.jl <file.gguf> [--int4|--int8|--f16]
#       [--lengths=512,2048,4096] [--new=32] [--reps=3] [--max-seq=8191] [--kv=f16|f32]
#       [--min-avail=10] [--stop-after=read|load|session|compile-prefill|compile-decode] [--reclaim]
#
# Memory: the GPU shares system RAM, so a stage log line ("[mem] ...") gives MemAvailable, the
# lowest value seen since the previous stage (sampled every 20 ms on an interactive thread, so run with `-t auto,1`), and the HIP pool's used
# bytes. A watchdog thread exits the process at once if MemAvailable falls below --min-avail
# GB (default 10), so the rest of the machine is not squeezed. --stop-after stops cleanly
# after a stage, to walk through the stages one at a time.
#
# For each prompt length: the time to the first token (prefill + one greedy pick, median
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
    pool = try AMDGPU.HIP.used_memory() / 1024^3 catch; NaN end
    status(@sprintf("[mem] %7.1fs %-34s avail %5.1f GB  (min since last %5.1f)  used %5.1f  shared %5.1f  hip pool %5.1f GB",
                    time() - T0, label, d["MemAvailable"] / 1024^2, min(MIN_AVAIL_KB[], d["MemAvailable"]) / 1024^2,
                    (d["MemTotal"] - d["MemAvailable"]) / 1024^2, d["Shmem"] / 1024^2, pool))
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
        ttft = median([timed("prompt $n: ttft run $r", 1, p) for r in 1:reps])
        tfull = median([timed("prompt $n: $n_new tokens run $r", n_new, p) for r in 1:reps])
        tpot = (tfull - ttft) / (n_new - 1)
        status(@sprintf("%8d %12.1f %14.1f %12.2f %12.1f   (cold %.1f s)", n, 1e3 * ttft, n / ttft, 1e3 * tpot, 1 / tpot, t_cold))
    end
end

main()
