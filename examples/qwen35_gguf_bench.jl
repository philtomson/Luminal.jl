# Prefill and decode speed of a Qwen3.5 / Qwen3.6 GGUF at several prompt lengths.
#
# Usage:
#   julia -t auto --project=. examples/qwen35_gguf_bench.jl <file.gguf> [--int4|--int8|--f16]
#       [--lengths=512,2048,4096] [--new=32] [--reps=3] [--max-seq=8191] [--kv=f16|f32]
#
# For each prompt length: the time to the first token (prefill + one greedy pick, median
# of `reps` after a warm-up that also compiles the graph), and the time per generated
# token over `--new` tokens decoded at that depth. Prompts are fixed pseudo-random token
# ids. Decode attention caps the context at 8191 positions (DECODE_ATTN_MAX_CTX).
using Luminal
using Luminal.NN
using Printf
using Statistics

status(args...) = (println(args...); flush(stdout))   # (the output is usually redirected)

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
    f   = read_gguf(path)
    cfg = qwen35_config(f)
    status("$(f.metadata["general.name"]): weights $T, KV cache $kv, max_seq $max_seq, WMMA ",
            get(ENV, "LUMINAL_WMMA", "1") != "0")
    t_load = @elapsed begin
        W = load_qwen35_gguf(path, cfg; quantize=_ -> T)
        s = LlamaSession(Qwen35(Luminal.Graph(), nothing; cfg...), nothing, path; max_seq=max_seq,
                         weights=W, decode_weights=T, kv_dtype=kv)
    end
    status(@sprintf("load %.1f s", t_load))
    ids(n) = [1000 + (i * 7919) % 20000 for i in 1:n]
    status(@sprintf("%8s %12s %14s %12s %12s", "prompt", "TTFT (ms)", "prefill tok/s", "TPOT (ms)", "decode tok/s"))
    for n in lengths
        n + n_new < max_seq || (status("skip $n (needs max_seq > $(n + n_new))"); continue)
        p = ids(n)
        t_cold = @elapsed generate_ids(s, [p]; max_new_tokens=1)               # compiles the graphs
        ttft = median([@elapsed(generate_ids(s, [p]; max_new_tokens=1)) for _ in 1:reps])
        tfull = median([@elapsed(generate_ids(s, [p]; max_new_tokens=n_new)) for _ in 1:reps])
        tpot = (tfull - ttft) / (n_new - 1)
        status(@sprintf("%8d %12.1f %14.1f %12.2f %12.1f   (cold %.1f s)", n, 1e3 * ttft, n / ttft, 1e3 * tpot, 1 / tpot, t_cold))
    end
end

main()
