# Text generation with a Qwen3.5 / Qwen3.6 llama.cpp GGUF file.
#
# Usage:
#   julia -t auto --project=. examples/qwen35_gguf.jl <file.gguf> [prompt] [max_tokens]
#       [--int4 | --int8 | --f16] [--max-seq=N]
#
# The architecture, weights and tokenizer all come from the GGUF file.
# The weights are converted to --int8 (default) or --int4 as they are read.
using Luminal
using Luminal.NN
using Printf

function main()
    args       = filter(a -> !startswith(a, "--"), ARGS)
    flag(name) = (i = findfirst(a -> startswith(a, "--$name="), ARGS); i === nothing ? nothing : split(ARGS[i], "=", limit=2)[2])
    path       = args[1]
    prompt     = length(args) >= 2 ? args[2] : "The capital of France is"
    max_tokens = length(args) >= 3 ? parse(Int, args[3]) : 40
    max_seq    = flag("max-seq") === nothing ? 512 : parse(Int, flag("max-seq"))
    T          = "--int4" in ARGS ? Luminal.Int4 : "--f16" in ARGS ? Float16 : Int8
    f   = read_gguf(path)
    cfg = qwen35_config(f)
    tok = LlamaTokenizer(f)
    println("$(f.metadata["general.name"]): $(cfg.n_layers) layers, hidden $(cfg.hidden), vocab $(cfg.vocab_size), weights $T")
    t0 = time()
    W = load_qwen35_gguf(path, cfg; quantize=_ -> T)
    t1 = time()
    session = LlamaSession(Qwen35(Luminal.Graph(), nothing; cfg...), tok, path; max_seq=max_seq,
                           weights=W, decode_weights=T)
    out = generate(session, [prompt]; max_new_tokens=max_tokens)
    t2 = time()
    println("-" ^ 60, "\n>> $prompt\n", out[1], "\n", "-" ^ 60)
    @printf "[load %.1f s, generate %.1f s incl. compilation]\n" (t1 - t0) (t2 - t1)
end

main()
