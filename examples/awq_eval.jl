# awq_eval.jl — does AWQ scaling improve 4-bit accuracy?
#
# Calibrates AWQ scales on one text (e.g. WikiText-2 train), then compares
# perplexity with and without them, for 4-bit and mixed int8/int4 weights, on
# evaluation texts it was not calibrated on. Quantized tensors take their exact
# dequantized values (what the kernels compute) through the Float32 path.
#
#   julia --project=. examples/awq_eval.jl <model_dir> <calib_text> <eval_text> [<eval_text2> ...]
using Luminal, Luminal.NN
using AMDGPU, Printf

dir, calib = ARGS[1], ARGS[2]
evals = ARGS[3:end]
chunk, nchunks = 512, 16
dev = get_device()
tok = LlamaTokenizer(dir)
cfg = llama_config(dir)
model = Llama(Graph(), nothing; cfg...)
W = load_weights_to_dict(dir; device=dev)
matmuls = sort!([k for k in keys(W) if endswith(k, ".weight") && ndims(W[k]) == 2 && !occursin("embed_tokens", k)])

function chunks_of(path)
    text = read(path, String)
    ids = Luminal.encode(tok, text[1:min(end, prevind(text, min(end, 8 * chunk * nchunks)))])
    [vcat(tok.bos_id, ids[(i - 1) * (chunk - 1) + 1 : i * (chunk - 1)]) for i in 1:nchunks]
end
eval_chunks = [(basename(p), chunks_of(p)) for p in evals]

# One compiled Float32 graph; each configuration swaps dequantized weights in.
const G = Graph(); const REG = WeightRegistry(); const M = Llama(G, REG; cfg...)
const INP = Luminal.tensor(G, [chunk, 1]); const OUT = M(INP, 0)
load_weights!(G, REG, W; device=dev)
const EX = compile(G; device=dev, retain=[OUT.id], fold=false)
storage(T) = T === Luminal.Int4 ? Luminal.Q4Weight : T === Int8 ? Luminal.QuantWeight : nothing

function perplexity(policy, chunks)
    swapped = Int[]
    for name in matmuls
        S = storage(policy(name)); S === nothing && continue
        id = REG.mapping[name]; q = S(W[name])
        EX.results[id] = Luminal.dequantize(q)
        foreach(f -> Luminal._free_now!(getfield(q, f)), fieldnames(typeof(q)))
        push!(swapped, id); Luminal.reclaim!(dev)
    end
    nll = 0.0; n = 0
    for ids in chunks
        l = Array{Float32}(EX(Dict{Int,Any}(INP.id => Float32.(Base.reshape(ids, chunk, 1))); device=dev)[OUT.id])
        for i in 1:chunk-1
            v = Float64.(view(l, :, i, 1)); mx = maximum(v)
            nll += mx + log(sum(exp.(v .- mx))) - v[ids[i+1] + 1]; n += 1
        end
    end
    for (name, id) in REG.mapping
        id in swapped || continue
        AMDGPU.unsafe_free!(EX.results[id]); EX.results[id] = W[name]
    end
    Luminal.reclaim!(dev)
    return exp(nll / n)
end

configs = [("Float32", _ -> Float32), ("int4", weight_preset(:int4)),
           ("int4_mixed", weight_preset(:int4_mixed))]
results = Dict{Tuple{String,String,Bool}, Float64}()
for (tname, ch) in eval_chunks, (cname, pol) in configs
    results[(tname, cname, false)] = perplexity(pol, ch)
end

t = @elapsed scales = awq_scales(model, tok, dir; text=read(calib, String), policy=:int4, weights=W, device=dev)
@printf("AWQ calibration: %.0f s, %d groups\n", t, length(scales))
awq_apply!(W, scales, model)                  # in place: the compiled graph sees it
for (tname, ch) in eval_chunks, (cname, pol) in configs
    results[(tname, cname, true)] = perplexity(pol, ch)
end

for (tname, _) in eval_chunks
    base = results[(tname, "Float32", false)]
    @printf("\n%s   (Float32 %.4f; after AWQ transform %.4f)\n", tname, base, results[(tname, "Float32", true)])
    for (cname, _) in configs[2:end]
        plain, awq = results[(tname, cname, false)], results[(tname, cname, true)]
        @printf("  %-12s plain %8.4f (%+5.2f%%)   AWQ %8.4f (%+5.2f%%)\n", cname,
                plain, 100(plain / base - 1), awq, 100(awq / base - 1))
    end
end
