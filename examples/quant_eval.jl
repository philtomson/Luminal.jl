# quant_eval.jl — accuracy of reduced-precision weights on TinyLlama.
#
# Runs the full model over a text passage (one prefill pass) with Float32,
# Float16 and int8 weights, and reports perplexity on the passage's next-token
# predictions, top-1 agreement with Float32 at every position, and the relative
# logit error.
#
# Usage: julia --project=. examples/quant_eval.jl [model_dir]

using Luminal, Luminal.NN, Luminal.LlamaTokenization
using Printf

const DIR = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "tinyllama_chat")
const DEV = get_device()

const PASSAGE = """
The harbor town woke slowly on winter mornings. Fishing boats rocked against the
pier while their crews checked nets and argued about the weather. Above the
market, the baker opened her shutters and set out bread that was still warm, and
children on their way to school stopped to stare at the gulls fighting over the
crumbs. By noon the fog had lifted from the water, and the old lighthouse keeper
climbed the stairs to wind the clock, as he had done every day for forty years.
He liked to say that the sea kept its own time, but the town needed a clock that
everyone could agree on.
"""

tok = LlamaTokenizer(DIR)
ids = LlamaTokenization.encode(tok, PASSAGE; bos=true)
n = length(ids)
println("passage: $n tokens")

model = NN.Llama(Graph(), WeightRegistry(); vocab_size=32000, hidden=2048, n_layers=22,
                 n_heads=32, n_kv_heads=4, intermediate=5632, rope_base=10000f0)
wd = load_weights_to_dict(DIR; device=DEV)

function logits_with(weight_dtype)
    g = Graph(); reg = WeightRegistry()
    m = Luminal.Decoding._rebuild_model_like(model, g, reg; rope_base=10000f0)
    inp = Luminal.tensor(g, [n, 1])
    out = m(inp, 0)
    load_weights!(g, reg, wd; device=DEV)
    ex = compile(g; device=DEV, retain=[out.id], weight_dtype=weight_dtype)
    return Array{Float32}(ex(Dict{Int,Any}(inp.id => Float32.(Base.reshape(ids, n, 1))); device=DEV)[out.id])
end

# Mean negative log-likelihood of token i+1 given the prefix, for i = 1..n-1
function perplexity(logits)
    nll = 0.0
    for i in 1:n-1
        l = Float64.(logits[:, i, 1])
        mx = maximum(l)
        nll += (mx + log(sum(exp.(l .- mx)))) - l[ids[i+1] + 1]
    end
    return exp(nll / (n - 1))
end

ref = logits_with(Float32)
@printf("%-10s perplexity %.4f\n", "Float32", perplexity(ref))
for (label, dt) in (("Float16", Float16), ("int8", Int8))
    l = logits_with(dt)
    top1 = count(i -> argmax(l[:, i, 1]) == argmax(ref[:, i, 1]), 1:n)
    relerr = maximum(abs.(l .- ref)) / maximum(abs.(ref))
    @printf("%-10s perplexity %.4f   top-1 agreement %d/%d   logit relerr %.1e\n",
            label, perplexity(l), top1, n, relerr)
end
