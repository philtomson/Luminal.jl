# Text generation with a Llama-family checkpoint (TinyLlama, Llama-3-8B-Instruct, …).
#
# Usage:
#   julia --project=. examples/llama_chat.jl <model_dir> [prompt] [max_tokens]
#       [--chat] [--int8 | --f32] [--search=static|measured] [--max-seq=N]
#       [--prompt=another --prompt=...] [--interactive]
#
# The architecture and RoPE base come from the checkpoint's config.json.
# --chat         wrap prompts in the model's chat format (Llama-3 or Zephyr/TinyLlama)
# --int8, --f32  decode weight storage (default Float16 on GPU)
# --prompt=...   extra prompts, generated together with `prompt` as one batch
# --interactive  then keep reading prompts from stdin, one per line, reusing the
#                session's weights and compiled graphs
#
#   julia --project=. examples/llama_chat.jl llama3_8b_instruct "Why is the sky blue?" 100 --chat --int8
using Luminal
using Luminal.NN
using Printf

function main()
    args       = filter(a -> !startswith(a, "--"), ARGS)   # positionals; flags anywhere
    flag(name) = (i = findfirst(a -> startswith(a, "--$name="), ARGS); i === nothing ? nothing : split(ARGS[i], "=", limit=2)[2])
    model_dir  = length(args) >= 1 ? args[1] : "tinyllama_chat"
    prompt     = length(args) >= 2 ? args[2] : "Once upon a time"
    max_tokens = length(args) >= 3 ? parse(Int, args[3]) : 100
    chat_mode  = "--chat" in ARGS
    weights    = "--int8" in ARGS ? Int8 : "--f32" in ARGS ? Float32 : nothing
    search     = flag("search") === nothing ? :none : Symbol(flag("search"))
    max_seq    = flag("max-seq") === nothing ? 1024 : parse(Int, flag("max-seq"))
    isdir(model_dir) || error("Model directory not found: $model_dir")

    tok = LlamaTokenizer(model_dir)
    wrap(p) = chat_mode ? chat_prompt(tok, p) : p
    prompts = [prompt; [a[length("--prompt=")+1:end] for a in ARGS if startswith(a, "--prompt=")]]

    cfg = llama_config(model_dir)
    println("Model $model_dir: $(cfg.n_layers) layers, hidden $(cfg.hidden), ",
            "$(cfg.n_heads)/$(cfg.n_kv_heads) heads, vocab $(cfg.vocab_size), rope base $(cfg.rope_base)")
    model = Llama(Graph(), nothing; cfg...)

    t0 = time()
    session = LlamaSession(model, tok, model_dir; max_seq=max_seq, search=search,
                           decode_weights=weights)
    t1 = time()
    responses = generate(session, wrap.(prompts); max_new_tokens=max_tokens)
    t2 = time()
    println("-" ^ 60)
    for (p, r) in zip(prompts, responses)
        println(">> $p\n$r")
        println("-" ^ 60)
    end
    n_gen = sum(r -> length(Luminal.encode(tok, r)), responses)
    @printf "[%d tokens; load %.1f s, first generate %.1f s incl. compilation]\n" n_gen (t1 - t0) (t2 - t1)

    if "--interactive" in ARGS
        println("Enter a prompt per line (empty line or EOF to quit).")
        while true
            print("\n>> ")
            line = readline()
            isempty(strip(line)) && break
            t = @elapsed r = generate(session, wrap(line); max_new_tokens=max_tokens)
            println(r)
            @printf "[%.2f s]\n" t
        end
    end
end

main()
