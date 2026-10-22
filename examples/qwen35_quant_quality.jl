# What does re-quantizing a Qwen3.5 / Qwen3.6 GGUF's weights cost? Next-token logits of the same prompts under two weight formats, compared.
#
#   julia -t auto,1 --project=. examples/qwen35_quant_quality.jl run <base.gguf> <int4|int8|native> <out.jls> [--prompts=100] [--ctx=256]
#   julia --project=. examples/qwen35_quant_quality.jl compare <ref.jls> <test.jls>
#   julia -t auto,1 --project=. examples/qwen35_quant_quality.jl kv <base.gguf> <int4|int8|native> <out.jls> [--prompts=20] [--ctx=16384] [--new=32]
#
# `kv` asks what the int8 KV cache costs: the same prompts under a Float16 and an Int8 K/V cache (same weights, one process), reporting the same
# agreement / KL / log-probability numbers as `compare` (Float16 cache = reference), plus how many greedy continuation tokens agree. A long `--ctx`
# runs the chunked prefill, which reads the earlier chunks from the (quantized) cache. `--more` adds this repo's TODO.md, docs and sources to the text
# (the windows need `ctx` tokens of real text).
#
# `run` takes real English text (this repo's README and gufo's docs when present), cuts `--prompts` windows of `--ctx` tokens at random positions
# (fixed seed, so every format sees the same ones), prefill each and saves the logits of the last position plus the true next token.
# `compare` reports, for test against reference: top-1 agreement, the mean KL divergence KL(ref || test) of the next-token distributions,
# and the mean log-probability each assigns to the true next token (their difference is the perplexity cost of the format).
# int8 is the reference for the GGUF's own values: its error is ~1/250 of a group's range against ~1/15 for int4.
using Luminal
using Luminal.NN
using Printf
using Random
using Serialization
using Statistics

function logsoftmax(x)
    m = maximum(x); lse = m + log(sum(exp.(Float64.(x) .- m)))
    return Float64.(x) .- lse
end

function load_text_ids(tok; more = "--more" in ARGS)
    root = joinpath(@__DIR__, "..")
    files = [joinpath(root, "README.md"); sort(filter(isfile, readdir(expanduser("~/devel/AI/gufo/docs"); join = true)))]
    more && append!(files, [joinpath(root, "TODO.md"); sort(filter(isfile, readdir(joinpath(root, "docs"); join = true))); sort(filter(isfile, readdir(joinpath(root, "src"); join = true)))])
    text = join((read(p, String) for p in files if endswith(p, ".md") || endswith(p, ".jl")), "\n\n")
    ids = Int.(Luminal.LlamaTokenization.encode(tok, text; bos = false))
    println("text: $(length(ids)) tokens from $(length(files)) files")
    return ids
end

function run_mode(base, fmt, out; nprompts, ctx)
    T = fmt == "int8" ? Int8 : Luminal.Int4          # "native": the file's own values where a native layout exists (else int4)
    ENV["LUMINAL_PREFILL_CHUNK"] = "256"
    f = read_gguf(base); cfg = qwen35_config(f)
    tok = LlamaTokenizer(f)
    ids = load_text_ids(tok)
    rng = MersenneTwister(1234)
    cuts = sort(rand(rng, ctx+1:length(ids)-1, nprompts))
    W = load_qwen35_gguf(base, cfg; quantize = _ -> T, native = fmt == "native")
    s = LlamaSession(Qwen35(Luminal.Graph(), nothing; cfg...), tok, base; max_seq = 1024, weights = W, decode_weights = T, kv_dtype = Float16)
    logits = Vector{Vector{Float32}}(); targets = Int[]
    t0 = time()
    for (i, c) in enumerate(cuts)
        prompt = ids[c-ctx+1:c]
        ll = Vector{Float32}[]
        generate_ids(s, [prompt]; max_new_tokens = 1, last_logits = ll)
        push!(logits, copy(only(ll))); push!(targets, ids[c+1])
        i % 20 == 0 && (println("  $i / $nprompts prompts, $(round(time() - t0, digits = 1)) s"); flush(stdout))
    end
    serialize(out, (; fmt, targets, logits))
    println("saved $(length(logits)) logit vectors to $out")
end

compare_mode(refpath, testpath) = compare_logits(deserialize(refpath), deserialize(testpath))
function compare_logits(ref, tst)
    @assert ref.targets == tst.targets
    n = length(ref.targets)
    agree = 0; kls = Float64[]; lp_ref = Float64[]; lp_tst = Float64[]; same_top5 = 0
    for i in 1:n
        lr = logsoftmax(ref.logits[i]); lt = logsoftmax(tst.logits[i])
        agree += argmax(lr) == argmax(lt)
        same_top5 += Set(partialsortperm(lr, 1:5, rev = true)) == Set(partialsortperm(lt, 1:5, rev = true))
        push!(kls, sum(exp.(lr) .* (lr .- lt)))
        push!(lp_ref, lr[ref.targets[i] + 1]); push!(lp_tst, lt[ref.targets[i] + 1])
    end
    @printf("%s (test) against %s (reference), %d prompts\n", tst.fmt, ref.fmt, n)
    @printf("  top-1 agreement        %.1f %%\n", 100agree / n)
    @printf("  same top-5 set         %.1f %%\n", 100same_top5 / n)
    @printf("  mean KL(ref || test)   %.4f nats (median %.4f, max %.3f)\n", mean(kls), median(kls), maximum(kls))
    @printf("  true next-token logp   ref %.4f   test %.4f   (test - ref %.4f +- %.4f s.e.)\n", mean(lp_ref), mean(lp_tst), mean(lp_tst) - mean(lp_ref),
            std(lp_tst .- lp_ref) / sqrt(n))
    @printf("  perplexity ratio test / ref  %.4f\n", exp(mean(lp_ref) - mean(lp_tst)))
end

function kv_mode(base, fmt, out; nprompts, ctx, nnew)
    T = fmt == "int8" ? Int8 : Luminal.Int4
    ENV["LUMINAL_PREFILL_CHUNK"] = "256"
    f = read_gguf(base); cfg = qwen35_config(f)
    tok = LlamaTokenizer(f)
    ids = load_text_ids(tok)
    rng = MersenneTwister(1234)
    cuts = sort(rand(rng, ctx+1:length(ids)-nnew-1, nprompts))
    W = load_qwen35_gguf(base, cfg; quantize = _ -> T, native = fmt == "native")
    sess(kv) = LlamaSession(Qwen35(Luminal.Graph(), nothing; cfg...), tok, base; max_seq = ctx + 512, weights = W, decode_weights = T, kv_dtype = kv)
    res = Dict(kv => (; logits = Vector{Vector{Float32}}(), gen = Vector{Vector{Int}}()) for kv in (Float16, Int8))
    targets = Int[]
    for kv in (Float16, Int8)                                  # (one session at a time: each holds a K/V cache for ctx + 512 tokens)
        s = sess(kv); t0 = time()
        for (i, c) in enumerate(cuts)
            prompt = ids[c-ctx+1:c]
            ll = Vector{Float32}[]
            g = only(generate_ids(s, [prompt]; max_new_tokens = nnew, last_logits = ll))
            push!(res[kv].logits, copy(only(ll))); push!(res[kv].gen, g)
            kv === Float16 && push!(targets, ids[c+1])
            println("  $kv: prompt $i / $nprompts, $(round(time() - t0, digits = 1)) s"); flush(stdout)
        end
        s = nothing; GC.gc()
    end
    ref = (; fmt = "$fmt, Float16 KV", targets, logits = res[Float16].logits)
    tst = (; fmt = "$fmt, Int8 KV", targets, logits = res[Int8].logits)
    println("\n== ctx $ctx, $nprompts prompts")
    compare_logits(ref, tst)
    same = [res[Float16].gen[i] == res[Int8].gen[i] for i in 1:nprompts]
    common = [something(findfirst(j -> res[Float16].gen[i][j] != res[Int8].gen[i][j], 1:min(length(res[Float16].gen[i]), length(res[Int8].gen[i]))), nnew + 1) - 1 for i in 1:nprompts]
    @printf("  greedy continuations of %d tokens: identical for %d of %d prompts; mean agreeing prefix %.1f tokens\n", nnew, count(same), nprompts, mean(common))
    serialize(out, (; ref, tst, gens = res))
end

flag(name, default) = (i = findfirst(a -> startswith(a, "--$name="), ARGS); i === nothing ? default : split(ARGS[i], "=", limit = 2)[2])
args = filter(a -> !startswith(a, "--"), ARGS)
if args[1] == "run"
    run_mode(args[2], args[3], args[4]; nprompts = parse(Int, flag("prompts", "100")), ctx = parse(Int, flag("ctx", "256")))
elseif args[1] == "kv"
    kv_mode(args[2], args[3], args[4]; nprompts = parse(Int, flag("prompts", "20")), ctx = parse(Int, flag("ctx", "16384")), nnew = parse(Int, flag("new", "32")))
elseif args[1] == "compare"
    compare_mode(args[2], args[3])
end
