# quant_sensitivity.jl — which tensors of a model tolerate 4-bit weights?
#
# Perplexity over a text (in chunks, one prefill each) with a per-tensor weight
# policy (exact int4/int8 values, see `perplexity`):
#   - baselines: all Float32, all int8, all int4
#   - one tensor type (q/k/v/o/gate/up/down proj, lm_head) at int4, the rest Float32
#   - one layer (all its matmuls) at int4, the rest Float32
#   - mixed: all int4 except the most sensitive types / layers at int8
# Each row also gives the matmul weights' size in that configuration.
#
#   julia --project=. examples/quant_sensitivity.jl <model_dir> [text_file] [chunk_tokens] [n_chunks]
using Luminal, Luminal.NN, Luminal.LlamaTokenization
using AMDGPU, Printf

dir = ARGS[1]
text = read(get(ARGS, 2, joinpath(homedir(), "devel", "luminal", "docs", "architecture.md")), String)
chunk = parse(Int, get(ARGS, 3, "512"))
nchunks = parse(Int, get(ARGS, 4, "4"))
dev = get_device()

tok = LlamaTokenizer(dir)
cfg = llama_config(dir)
# Only as much text as the chunks need (~4 characters per token, with margin):
# the tokenizer is slow on very long inputs.
ids_all = LlamaTokenization.encode(tok, text[1:min(end, prevind(text, min(end, 8 * chunk * nchunks)))]; bos=false)
length(ids_all) >= nchunks * (chunk - 1) || error("text too short for $nchunks chunks of $chunk tokens")
chunks = [vcat(tok.bos_id, ids_all[(i - 1) * (chunk - 1) + 1 : i * (chunk - 1)]) for i in 1:nchunks]
W = load_weights_to_dict(dir; device=dev)
matmuls = sort!([k for k in keys(W) if endswith(k, ".weight") && ndims(W[k]) == 2 && !occursin("embed_tokens", k)])
@printf("%s: %d chunks of %d tokens, %d matmul weights\n", basename(abspath(dir)), nchunks, chunk, length(matmuls))

kind(name) = occursin("lm_head", name) ? "lm_head" : match(r"\.(\w+_proj)\.weight$", name)[1]
layer(name) = (m = match(r"layers\.(\d+)\.", name); m === nothing ? -1 : parse(Int, m[1]))
bytes_per(T) = T === Luminal.Int4 ? 0.5625 : T === Int8 ? 1.03 : T === Float16 ? 2.0 : 4.0   # int4: symmetric g32
size_gb(policy) = sum(length(W[k]) * bytes_per(policy(k)) for k in matmuls) / 2^30

# Perplexity with `policy(name) -> Type` for every matmul weight. Quantized tensors
# take their exact dequantized values (`dequantize` of the real int8 / 4-bit
# storage: what the kernels compute), so the model runs through the Float32 path
# (rocBLAS prefill): the same numbers as compile(...; weight_dtype=policy), much
# faster for 512-token chunks. The graph is compiled once; each configuration
# swaps its weights in.
storage(T) = T === Luminal.Int4 ? Luminal.Q4Weight : T === Int8 ? Luminal.QuantWeight :
             T === Float16 ? Luminal.HalfWeight : nothing
const G = Graph(); const REG = WeightRegistry(); const M = Llama(G, REG; cfg...)
const INP = Luminal.tensor(G, [chunk, 1]); const OUT = M(INP, 0)
load_weights!(G, REG, W; device=dev)
const EX = compile(G; device=dev, retain=[OUT.id], fold=false)

function perplexity(policy)
    swapped = Int[]
    for name in matmuls
        T = policy(name)
        T === Float32 && continue
        id = REG.mapping[name]
        q = storage(T)(W[name])
        EX.results[id] = Luminal.dequantize(q)
        foreach(f -> (x = getfield(q, f); x isa AMDGPU.ROCArray && AMDGPU.unsafe_free!(x)), fieldnames(typeof(q)))
        push!(swapped, id)
        Luminal.reclaim!(dev)        # leave no garbage between tensors (8B: 224 of them)
    end
    nll = 0.0; n = 0
    for ids in chunks
        l = Array{Float32}(EX(Dict{Int,Any}(INP.id => Float32.(Base.reshape(ids, chunk, 1))); device=dev)[OUT.id])
        for i in 1:chunk-1
            v = Float64.(view(l, :, i, 1)); mx = maximum(v)
            nll += mx + log(sum(exp.(v .- mx))) - v[ids[i+1] + 1]; n += 1
        end
    end
    for (name, id) in REG.mapping                     # restore the Float32 weights
        id in swapped || continue
        AMDGPU.unsafe_free!(EX.results[id]); EX.results[id] = W[name]
    end
    Luminal.reclaim!(dev)
    return exp(nll / n)
end

base = perplexity(_ -> Float32)
row(label, policy, p=perplexity(policy)) =
    (@printf("%-34s perplexity %8.4f  (%+6.2f%%)   weights %5.2f GB\n", label, p, 100 * (p / base - 1), size_gb(policy)); p)

println("\n== baselines")
row("all Float32", _ -> Float32, base)
row("all int8", _ -> Int8)
all4 = row("all int4", _ -> Luminal.Int4)

println("\n== one tensor type at int4 (rest Float32)")
kinds = sort!(unique(kind.(matmuls)))
kind_cost = Dict(k => row("int4: $k", n -> kind(n) == k ? Luminal.Int4 : Float32) / base - 1 for k in kinds)

println("\n== one layer at int4 (rest Float32)")
layers = sort!(unique(filter(>=(0), layer.(matmuls))))
layer_cost = Dict(l => row("int4: layer $l", n -> layer(n) == l ? Luminal.Int4 : Float32) / base - 1 for l in layers)

println("\n== sensitivity ranking")
for (k, c) in sort!(collect(kind_cost), by = last, rev = true)
    @printf("  %-10s %+6.2f%%\n", k, 100c)
end
worst_layers = first.(sort!(collect(layer_cost), by = last, rev = true))
@printf("  layers, most sensitive first: %s\n", join(worst_layers[1:min(8, end)], ", "))

println("\n== mixed: int4 except the most sensitive at int8")
ranked_kinds = first.(sort!(collect(kind_cost), by = last, rev = true))
for k in 1:3
    keep = Set(ranked_kinds[1:k])
    row("int8: $(join(sort!(collect(keep)), "+"))", n -> kind(n) in keep ? Int8 : Luminal.Int4)
end
for k in (2, 4)
    keep = Set(worst_layers[1:min(k, end)])
    row("int8: layers $(join(sort!(collect(keep)), ","))", n -> layer(n) in keep ? Int8 : Luminal.Int4)
end
keep_k = Set(ranked_kinds[1:1]); keep_l = Set(worst_layers[1:min(2, end)])
row("int8: $(ranked_kinds[1]) + layers $(join(sort!(collect(keep_l)), ","))",
    n -> (kind(n) in keep_k || layer(n) in keep_l) ? Int8 : Luminal.Int4)
