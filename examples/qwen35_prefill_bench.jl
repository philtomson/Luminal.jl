# Chunked prefill on a real Qwen3.5 / Qwen3.6 GGUF: prefill time (and tokens/s) for prompts of several lengths
# (real text repeated to length), plus a check against the one-shot prefill on a prompt that fits it.
#
# Usage:
#   julia -t auto,1 --project=. examples/qwen35_prefill_bench.jl <base.gguf> [--int4|--int8] [--max-ctx=33000]
#       [--lens=512,2048,8192,16384] [--chunk=512] [--layers=N] [--min-avail=20] [--swap-gb=3] [--no-check]
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



const TEXT = "The history of the steam engine begins long before the Industrial Revolution. Ancient engineers such as Hero of Alexandria described simple devices that used steam to produce motion, but it was not until the late seventeenth century that practical machines appeared. Thomas Savery patented a steam-powered water pump in 1698, and Thomas Newcomen built the first piston engine a few years later. James Watt then improved the design by adding a separate condenser. "

function main()
    args = filter(a -> !startswith(a, "--"), ARGS)
    flag(name, default) = (i = findfirst(a -> startswith(a, "--$name="), ARGS); i === nothing ? default : split(ARGS[i], "=", limit = 2)[2])
    base = args[1]
    T = "--int8" in ARGS ? Int8 : Luminal.Int4
    maxctx = parse(Int, flag("max-ctx", "33000")); chunk = parse(Int, flag("chunk", "512"))
    lens = parse.(Int, split(flag("lens", "512,2048,8192,16384"), ","))
    ENV["LUMINAL_PREFILL_CHUNK"] = string(chunk)
    start_watchdog(parse(Int, flag("min-avail", "20")); swap_gb = parse(Int, flag("swap-gb", "3")))
    f = read_gguf(base); cfg = qwen35_config(f)
    nl = parse(Int, flag("layers", string(cfg.n_layers)))
    nl < cfg.n_layers && (cfg = merge(cfg, (n_layers = nl, layer_types = cfg.layer_types[1:nl])))
    tok = LlamaTokenizer(f)
    ids = Int.(Luminal.LlamaTokenization.encode(tok, repeat(TEXT, 400); bos = false))
    W = load_qwen35_gguf(base, cfg; quantize = _ -> T)
    s = LlamaSession(Qwen35(Luminal.Graph(), nothing; cfg...), tok, base; max_seq = maxctx, weights = W, decode_weights = T,
                     kv_dtype = Float16)
    # pre-flight: weights are already loaded; the largest scratch is one chunk (or the one-shot check) at ~32 MB/token (27B)
    scale = nl / 64
    check_n = parse(Int, flag("check-n", "192")); check_chunk = parse(Int, flag("check-chunk", "64"))
    need_gb = 0.032 * scale * max(chunk, "--no-check" in ARGS ? 0 : check_n)
    avail_gb = meminfo_kb()["MemAvailable"] / 1024^2
    status(@sprintf("pre-flight: ~%.1f GB scratch needed, %.1f GB available", need_gb, avail_gb))
    avail_gb - need_gb >= parse(Int, flag("min-avail", "20")) + 8 || (status("refusing: not enough memory headroom"); exit(2))
    if !("--no-check" in ARGS)
        p = ids[1:check_n]
        withenv("LUMINAL_PREFILL_CHUNK" => "0") do
            ll0 = Vector{Float32}[]; g0 = only(generate_ids(s, [p]; max_new_tokens = 16, last_logits = ll0))
            withenv("LUMINAL_PREFILL_CHUNK" => string(check_chunk)) do
                ll1 = Vector{Float32}[]; g1 = only(generate_ids(s, [p]; max_new_tokens = 16, last_logits = ll1))
                e = maximum(abs.(only(ll1) .- only(ll0))) / maximum(abs.(only(ll0)))
                status(@sprintf("check (%d tokens, chunk %d vs one-shot): logits rel err %.2e, first 16 tokens identical: %s", length(p), check_chunk, e, g0 == g1))
            end
        end
    end
    status("prompt tokens   prefill s   tokens/s")
    for n in lens
        n + chunk <= s.max_seq || continue
        p = ids[1:n]
        Luminal.Decoding._prefill_chunked!(s, p[1:min(n, 2 * chunk)])      # compiles the chunk graph on the first call
        AMDGPU.synchronize()
        t = @elapsed (Luminal.Decoding._prefill_chunked!(s, p); AMDGPU.synchronize())
        status(@sprintf("%12d  %10.2f  %9.1f", n, t, n / t))
    end
end
main()
