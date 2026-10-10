# Speculative decoding (the MTP head) against plain greedy decoding on a real Qwen3.5 / Qwen3.6 GGUF.
#
# Usage:
#   julia -t auto,1 --project=. examples/qwen35_spec_bench.jl <base.gguf> <mtp.gguf> [--int4|--int8]
#       [--new=128] [--reps=2] [--min-avail=20] [--swap-gb=3] [--layers=N] [--oracle=P] [--draft-vocab=N] [--prompt="text"]
# The base weights come from <base.gguf>, the MTP head (`blk.<n>.nextn.*`) from <mtp.gguf> (a GGUF that has it).
# Prints tokens/s for both loops, the acceptance rate, and whether the two outputs are identical.
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

const DEFAULT_PROMPT = "The history of the steam engine begins long before the Industrial Revolution. " *
    "Ancient engineers such as Hero of Alexandria described simple devices that used steam to produce motion, " *
    "but it was not until the late seventeenth century that practical machines appeared. Thomas Savery patented " *
    "a steam-powered water pump in 1698, and Thomas Newcomen built the first piston engine a few years later. " *
    "James Watt then"

function main()
    args = filter(a -> !startswith(a, "--"), ARGS)
    flag(name, default) = (i = findfirst(a -> startswith(a, "--$name="), ARGS); i === nothing ? default : split(ARGS[i], "=", limit = 2)[2])
    base, mtp = args[1], args[2]
    T = "--int8" in ARGS ? Int8 : Luminal.Int4
    # (the GGUF reader drops each tensor from the page cache after reading it; see LUMINAL_GGUF_KEEP_CACHE)
    nnew = parse(Int, flag("new", "128")); reps = parse(Int, flag("reps", "2"))
    oracle = parse(Float64, flag("oracle", "-1"))
    start_watchdog(parse(Int, flag("min-avail", "20")); swap_gb = parse(Int, flag("swap-gb", "3")))
    prompt = flag("prompt", DEFAULT_PROMPT)
    f = read_gguf(base); cfg = qwen35_config(f)
    nblocks = cfg.n_layers                       # (the MTP head is block number n_layers of the full model)
    nl = parse(Int, flag("layers", string(cfg.n_layers)))
    nl < cfg.n_layers && (cfg = merge(cfg, (n_layers = nl, layer_types = cfg.layer_types[1:nl])))
    tok = LlamaTokenizer(f)
    ids = Luminal.LlamaTokenization.encode(tok, prompt; bos = false)
    status("prompt: $(length(ids)) tokens; weights $T")
    t_load = @elapsed begin
        W = load_qwen35_gguf(base, cfg; quantize = _ -> T)
        merge!(W, load_qwen35_mtp(mtp, cfg; quantize = _ -> T, block = nblocks))
    end
    status(@sprintf("load %.1f s", t_load))
    s = LlamaSession(Qwen35(Luminal.Graph(), nothing; cfg...), tok, base; max_seq = 1024, weights = W, decode_weights = T,
                     kv_dtype = Float16)
    d = MTPDrafter(s; draft_vocab = parse(Int, flag("draft-vocab", "0")))
    prompt_ids = Int.(ids)
    # warm-ups compile the graphs (plain decode, MTP, verify) and capture the HIP graphs
    generate_ids(s, [prompt_ids]; max_new_tokens = 4)
    generate_ids_speculative(s, d, prompt_ids; max_new_tokens = 8)
    generate_ids_speculative(s, d, prompt_ids; max_new_tokens = 8)
    plain = nothing; spec = nothing; tp = Inf; ts = Inf; st = Dict{Symbol,Int}()
    for _ in 1:reps
        t = @elapsed plain = only(generate_ids(s, [prompt_ids]; max_new_tokens = nnew))
        tp = min(tp, t)
        # --oracle=P: draft the true next token with probability P (a sliced model's MTP head is useless; this fixes the
        # acceptance rate to measure the cost of the cycle)
        hook = oracle < 0 ? nothing : (n, draft) -> (rand() < oracle && n + 1 <= length(plain) ? plain[n + 1] : mod(draft + 1, cfg.vocab_size))
        st1 = Dict{Symbol,Int}()
        t = @elapsed spec1 = generate_ids_speculative(s, d, prompt_ids; max_new_tokens = nnew, stats = st1, draft_hook = hook)
        if t < ts
            ts = t; st = st1; spec = spec1          # the stats of the fastest run
        end
    end
    # (both include the prefill and first token; subtract it by also timing a 1-token run)
    t1 = @elapsed generate_ids(s, [prompt_ids]; max_new_tokens = 1)
    n = length(plain)
    @printf("plain greedy : %d tokens in %.2f s  -> %.2f tok/s (decode only %.2f tok/s)\n", n, tp, n / tp, (n - 1) / (tp - t1))
    @printf("speculative  : %d tokens in %.2f s  -> %.2f tok/s (decode only %.2f tok/s)\n", length(spec), ts, length(spec) / ts, (length(spec) - 1) / (ts - t1))
    @printf("acceptance   : %d of %d drafts (%.0f%%), %.2f tokens per verify step\n", st[:accepted], st[:cycles],
            100 * st[:accepted] / max(st[:cycles], 1), (length(spec) - 1) / max(st[:cycles], 1))
    @printf("phases (ms, whole run): verify %d, MTP steps %d, rollback %d (%d cycles)\n",
            st[:ms_verify], st[:ms_mtp], st[:ms_rollback], st[:cycles])
    same = plain[1:min(length(plain), length(spec))] == spec[1:min(length(plain), length(spec))]
    status("outputs identical over $(min(length(plain), length(spec))) tokens: $same")
    first_diff = findfirst(i -> plain[i] != spec[i], 1:min(length(plain), length(spec)))
    first_diff === nothing || status("first difference at token $first_diff")
    status("text: ", Luminal.LlamaTokenization.decode(tok, spec))
end
main()
