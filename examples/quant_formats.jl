# quant_formats.jl — choose a weight quantization format by its accuracy.
#
# "Fake quantization": every matmul weight (not the embedding table) is
# quantized to the candidate format and dequantized back to Float32, then the
# unchanged Float32 model runs. Perplexity over a text (in chunks) measures the
# format's accuracy alone, before any kernel exists for it.
#
#   julia --project=. examples/quant_formats.jl <model_dir> [text_file] [chunk_tokens] [n_chunks]
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
all_ids = LlamaTokenization.encode(tok, text[1:min(end, prevind(text, min(end, 8 * chunk * nchunks)))]; bos=false)
length(all_ids) >= nchunks * (chunk - 1) || error("text too short for $nchunks chunks of $chunk tokens")
chunks = [vcat(tok.bos_id, all_ids[(i - 1) * (chunk - 1) + 1 : i * (chunk - 1)]) for i in 1:nchunks]
println("$(basename(abspath(dir))): $nchunks chunks of $chunk tokens")

W = load_weights_to_dict(dir; device=dev)
is_matmul(k) = endswith(k, ".weight") && ndims(W[k]) == 2 && !occursin("embed_tokens", k)

# Quantize-dequantize one (Out, In) matrix with the real storage types, so the
# model computes exactly what the int8 / 4-bit kernels would.
function fakequant(Wm; storage, kwargs...)
    q = storage(Wm; kwargs...)
    d = Luminal.dequantize(q)
    for f in fieldnames(typeof(q))
        x = getfield(q, f); x isa AMDGPU.ROCArray && AMDGPU.unsafe_free!(x)   # (mn may be nothing)
    end
    return d
end

function perplexity(weights)
    g = Graph(); reg = WeightRegistry(); m = Llama(g, reg; cfg...)
    inp = Luminal.tensor(g, [chunk, 1]); out = m(inp, 0)
    load_weights!(g, reg, weights; device=dev)
    ex = compile(g; device=dev, retain=[out.id])
    nll = 0.0; n = 0
    for ids in chunks
        l = Array{Float32}(ex(Dict{Int,Any}(inp.id => Float32.(Base.reshape(ids, chunk, 1))); device=dev)[out.id])
        for i in 1:chunk-1
            v = Float64.(view(l, :, i, 1)); mx = maximum(v)
            nll += mx + log(sum(exp.(v .- mx))) - v[ids[i+1] + 1]; n += 1
        end
    end
    Luminal.release!(ex); Luminal.reclaim!(dev)
    return exp(nll / n)
end

formats = [("Float32", nothing),
           ("int8 g128 (QuantWeight)", (storage=Luminal.QuantWeight, group=128)),
           ("int4 g32 sym   0.5625 B/w", (storage=Luminal.Q4Weight, group=32, symmetric=true)),
           ("int4 g64 sym   0.5313 B/w", (storage=Luminal.Q4Weight, group=64, symmetric=true)),
           ("int4 g64 asym  0.5625 B/w", (storage=Luminal.Q4Weight, group=64, symmetric=false)),
           ("int4 g32 asym  0.6250 B/w", (storage=Luminal.Q4Weight, group=32, symmetric=false)),
           ("int4 g128 asym 0.5313 B/w", (storage=Luminal.Q4Weight, group=128, symmetric=false))]
filter!(f -> isempty(ARGS[5:end]) || any(p -> occursin(p, f[1]), ARGS[5:end]), formats)

base = nothing
for (label, fmt) in formats
    t = @elapsed begin
        weights = fmt === nothing ? W : Dict{String,Any}()
        if fmt !== nothing
            for (k, v) in W
                weights[k] = is_matmul(k) ? fakequant(v; fmt...) : v
                # the temporaries of each matrix, freed before the next (left to the
                # GC they pile up: 8B has 224 matrices, the largest 2 GB)
                is_matmul(k) && Luminal.reclaim!(dev)
            end
        end
        p = perplexity(weights)
        if fmt !== nothing
            for (k, v) in weights
                is_matmul(k) && AMDGPU.unsafe_free!(v)
            end
        end
        weights = nothing
    end
    global base = base === nothing ? p : base
    @printf("%-28s perplexity %8.4f  (%+6.2f%%)   [%.0f s]\n", label, p, 100 * (p / base - 1), t)
end
