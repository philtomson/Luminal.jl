# Decode speed as the context fills up: time one plain decode step (1 token) and one speculative verify
# step (3 tokens) at several cache lengths on a real Qwen3.5 / Qwen3.6 GGUF. The KV cache is filled with
# random values (the step time does not depend on them), so no long prompt is run and memory stays at
# the weights plus the cache (65 KB per token on the 27B).
#
# Usage:
#   julia -t auto,1 --project=. examples/qwen35_context_bench.jl <base.gguf> [--int4|--int8] [--max-ctx=32768]
#       [--lens=128,1024,4096,16384,32000] [--reps=20] [--layers=N] [--min-avail=20] [--swap-gb=3]
using Luminal
using Luminal.NN
using Printf
using AMDGPU

status(args...) = (println(args...); flush(stdout))
meminfo_kb() = (d = Dict{String,Int}();
                for l in eachline("/proc/meminfo")
                    m = match(r"^(\w+):\s+(\d+)", l); m === nothing || (d[m[1]] = parse(Int, m[2]))
                end; d)
function start_watchdog(limit_gb; swap_gb = 1)
    m0 = meminfo_kb(); swap0 = m0["SwapTotal"] - m0["SwapFree"]
    Threads.@spawn :interactive while true
        m = meminfo_kb(); av = m["MemAvailable"]
        swap_used = m["SwapTotal"] - m["SwapFree"] - swap0
        if av < limit_gb * 1024 * 1024 || swap_used > swap_gb * 1024 * 1024
            status(@sprintf("[mem] WATCHDOG: MemAvailable %.1f GB (limit %d), swap use +%.2f GB, exiting", av / 1024^2, limit_gb, swap_used / 1024^2))
            ccall(:_exit, Cvoid, (Cint,), 3)
        end
        sleep(0.02)
    end
end


function main()
    args = filter(a -> !startswith(a, "--"), ARGS)
    flag(name, default) = (i = findfirst(a -> startswith(a, "--$name="), ARGS); i === nothing ? default : split(ARGS[i], "=", limit = 2)[2])
    base = args[1]
    T = "--int8" in ARGS ? Int8 : Luminal.Int4
    maxctx = parse(Int, flag("max-ctx", "32768")); reps = parse(Int, flag("reps", "20"))
    lens = parse.(Int, split(flag("lens", "128,1024,4096,16384,32000"), ","))
    start_watchdog(parse(Int, flag("min-avail", "20")); swap_gb = parse(Int, flag("swap-gb", "3")))
    f = read_gguf(base); cfg = qwen35_config(f)
    nl = parse(Int, flag("layers", string(cfg.n_layers)))
    nl < cfg.n_layers && (cfg = merge(cfg, (n_layers = nl, layer_types = cfg.layer_types[1:nl])))
    tok = LlamaTokenizer(f)
    W = load_qwen35_gguf(base, cfg; quantize = _ -> T)
    s = LlamaSession(Qwen35(Luminal.Graph(), nothing; cfg...), tok, base; max_seq = maxctx + 8, weights = W, decode_weights = T,
                     kv_dtype = Float16)
    dc = Luminal.Decoding._decode_graph(s, 1)
    vg = Luminal.Decoding._verify_graph(s, 1, 3)
    # random K/V for the full-attention layers (the recurrent layers' states stay zero)
    for (i, l) in enumerate(s.model.layers)
        Luminal.NN._is_state_layer(l) && continue
        for a in dc.cache.self_cache[i]
            copyto!(a, ROCArray(Float16.(0.5f0 .* randn(Float32, size(a)...))))
        end
    end
    function timed(f)
        f(); AMDGPU.synchronize()
        t = @elapsed (for _ in 1:reps; f(); end; AMDGPU.synchronize())
        return 1000 * t / reps
    end
    status("context  plain step   tok/s | verify3 step  (ms)   tokens/s at 2.2 tokens/step")
    for L in lens
        L + 4 <= s.max_seq || continue
        function plain()
            dc.cache.positions .= L
            Luminal.NN.llama_decode_step!(dc.exec, dc.idg, dc.cache, [100]; device = s.device)
        end
        function verify()
            vg.cache.positions .= L
            Luminal.Decoding._verify_step!(s, vg, [100; 101; 102;;])
        end
        tp = timed(plain); tv = timed(verify)
        status(@sprintf("%7d  %8.1f ms %6.2f | %8.1f ms  %6.2f", L, tp, 1000 / tp, tv, 2.2 * 1000 / tv))
    end
end
main()
