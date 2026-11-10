# lsa_decode_quality.jl -- what Latent Sparse Attention costs the full Qwen3.8-27B in quality, as the context grows.
#
# One prompt a length: real text (the docs of a latent-sparse-attention checkout, LSA_REPO, then its source files) with three facts buried at
# 20%, 50% and 80% of the way in, and one question at the end asking for all three. The prompt goes through the chunked prefill.
#   dense   decodes `--new` greedy tokens - the reference answer.
#   LSA     at each budget is fed the dense model's own tokens, so every context is identical, and its logits at each step are compared:
#           the KL divergence from dense, how often the argmax agrees, the rank of dense's token, and whether the three facts survive in
#           the tokens LSA's argmax would have written.
# One process loads the weights once and runs every length and budget; max_seq is the longest prompt + 1024, and at most 262,144,
# the most the dense split-K kernel takes.
#
# Usage: julia -t auto,1 --project=. benchmarks/lsa_decode_quality.jl <file.gguf> <maps.safetensors> [--prompts=64000,128000]
#     [--budgets=4096:1024,2048:512] [--new=64] [--min-avail=15]       (uses the GPU - stop the model server first)
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

softmax(x) = (e = exp.(x .- maximum(x)); e ./ sum(e))

const FACTS = [("the vault passphrase", "amber-falcon-4417"), ("the locker code", "58-triton-209"), ("the harbor of departure", "Port Selmara")]
const DEPTHS = (0.2, 0.5, 0.8)

# At least `n` tokens of real text - the docs, then the source files
function haystack(tok, repo, n)
    files = vcat([joinpath(repo, "README.md")], sort(readdir(joinpath(repo, "docs/guide"); join = true)), [joinpath(repo, "docs/reference/notation.md")],
                 sort([joinpath(r, f) for (r, _, fs) in walkdir(joinpath(repo, "src")) for f in fs if endswith(f, ".py")]))
    ids = Int[]
    for file in filter(isfile, files)
        append!(ids, encode(tok, read(file, String) * "\n\n"))
        length(ids) >= n && return ids
    end
    error("only $(length(ids)) tokens of text for a prompt of $n")
end

# `n` tokens - the haystack with the facts at their depths and the question last
function needle_prompt(tok, hay, n)
    needles = [encode(tok, "\n\nNote for the record: $what is $value.\n\n") for (what, value) in FACTS]
    question = encode(tok, "\n\nQuestion: According to the text above, what is the vault passphrase, what is the locker code, and what is the harbor of departure?\nAnswer: The vault passphrase is")
    body = hay[1:n - sum(length, needles) - length(question)]
    cuts = [round(Int, d * length(body)) for d in DEPTHS]
    return vcat(body[1:cuts[1]], needles[1], body[cuts[1]+1:cuts[2]], needles[2], body[cuts[2]+1:cuts[3]], needles[3], body[cuts[3]+1:end], question), cuts
end

# Prefill `prompt`, then `steps` decode steps - the given tokens if `forced`, else greedy. Returns the 0-based tokens, the logits that chose
# each (the first from the prefill), and the prefill and decode seconds.
function run!(s, prompt, steps; forced = nothing)
    t_prefill = @elapsed (pf, res, dc, plens, logits) = Luminal.Decoding._prefill_into_cache!(s, [prompt])
    cur = vec(Array{Float32}(logits))
    tokens, all_logits = Int[], Vector{Float32}[]
    t_decode = @elapsed for i in 1:steps
        push!(all_logits, cur)
        tok = forced === nothing ? argmax(cur) - 1 : forced[i]
        push!(tokens, tok)
        i == steps && break
        cur = vec(Array{Float32}(Luminal.NN.llama_decode_step!(dc.exec, dc.idg, dc.cache, tok; device = s.device)))
    end
    return tokens, all_logits, t_prefill, t_decode
end

facts_found(text) = [occursin(value, text) for (_, value) in FACTS]

function main()
    args = filter(a -> !startswith(a, "--"), ARGS)
    path, maps_path = args[1], args[2]
    prompts = parse.(Int, split(flag("prompts", "64000"), ","))
    budgets = [Tuple(parse.(Int, split(b, ":"))) for b in split(flag("budgets", "4096:1024,2048:512"), ",")]
    steps = parse(Int, flag("new", "64"))
    start_watchdog(parse(Int, flag("min-avail", "15")))
    max_seq = maximum(prompts) + 1024
    max_seq <= 262144 || error("the dense kernel takes at most 262,144 positions - the longest prompt is at most 261,120 tokens")
    f = read_gguf(path); cfg = qwen35_config(f); tok = LlamaTokenizer(f)
    repo = get(ENV, "LSA_REPO", joinpath(homedir(), "devel/AI/latent-sparse-attention"))
    hay = haystack(tok, repo, maximum(prompts))
    status("$(length(hay)) tokens of text; prompts $prompts, budgets $budgets, $steps decoded tokens, max_seq $max_seq")
    T = Luminal.Int4
    W = load_qwen35_gguf(path, cfg; quantize = _ -> T)
    s = LlamaSession(Qwen35(Luminal.Graph(), nothing; cfg...), tok, path; max_seq = max_seq, weights = W, decode_weights = T, kv_dtype = Float16)
    maps = lsa_load_maps(maps_path)
    status(@sprintf("weights loaded, MemAvailable %.1f GB", avail_gb()))

    for n in prompts
        prompt, cuts = needle_prompt(tok, hay, n)
        status("\n######## prompt $n tokens, facts at tokens $cuts of the text")
        t0 = time()
        dense_tokens, dense_logits, tp, td = run!(s, prompt, steps)
        dense_text = decode(tok, dense_tokens)
        status(@sprintf("== dense: prefill %.0f s, %d decode steps %.1f s (%.0f ms/token), MemAvailable %.1f GB", tp, steps, td, 1000 * td / steps, avail_gb()))
        status("   answer: ", repr(dense_text))
        status("   facts recovered: ", facts_found(dense_text), " (vault passphrase, locker code, harbor)")
        for (top_k, top_p) in budgets
            lsa_enable!(s, LSAConfig(maps = maps, top_k = top_k, top_p = top_p))
            _, lsa_logits, tp, td = run!(s, prompt, steps; forced = dense_tokens)
            kls = [sum(softmax(d) .* (log.(softmax(d) .+ 1f-30) .- log.(softmax(l) .+ 1f-30))) for (d, l) in zip(dense_logits, lsa_logits)]
            agree = [argmax(d) == argmax(l) for (d, l) in zip(dense_logits, lsa_logits)]
            ranks = [1 + count(>(l[argmax(d)]), l) for (d, l) in zip(dense_logits, lsa_logits)]
            lsa_text = decode(tok, [argmax(l) - 1 for l in lsa_logits])
            status(@sprintf("== LSA top_k %d top_p %d (%.1f%% of the context read at most): prefill %.0f s, decode %.0f ms/token, MemAvailable %.1f GB",
                            top_k, top_p, 100 * (top_k + 516) / n, tp, 1000 * td / steps, avail_gb()))
            status(@sprintf("   KL from dense: mean %.4f, median %.4f, max %.4f;  argmax agrees %d of %d;  dense token's rank: mean %.2f, worst %d",
                            mean(kls), median(kls), maximum(kls), count(agree), steps, mean(ranks), maximum(ranks)))
            status("   facts in the argmax stream: ", facts_found(lsa_text), "   ", repr(first(lsa_text, 90)))
        end
        lsa_disable!(s)
        status(@sprintf("-- %d tokens done in %.0f s, MemAvailable %.1f GB", n, time() - t0, avail_gb()))
    end
end

main()
