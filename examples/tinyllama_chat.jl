# TinyLlama End-to-End Text Generation
#
# Usage:
#   julia --project=. examples/tinyllama_chat.jl [model_dir] [prompt] [max_tokens] [rope_base]
#       [--chat] [--int8] [--search=static|measured] [--prompt=another --prompt=...]
#
# Extra `--prompt=` flags are generated together with `prompt` as one batch.
#
# Defaults:
#   model_dir = /devel/phil/Llama-3.2
#   prompt    = "Once upon a time"
#
# Example:
#   julia --project=. examples/tinyllama_chat.jl /devel/phil/Llama-3.2 "Tell me about Julia"

using Luminal
using Luminal.NN
using Printf

function main()
    args         = filter(a -> !startswith(a, "--"), ARGS)   # positionals; flags anywhere
    model_dir    = length(args) >= 1 ? args[1] : "/home/phil/devel/Luminal.jl/tinyllama_chat"
    prompt       = length(args) >= 2 ? args[2] : "Once upon a time"
    max_tokens   = length(args) >= 3 ? parse(Int, args[3]) : 100
    rope_base    = length(args) >= 4 ? parse(Float32, args[4]) : 10000.0f0
    chat_mode    = "--chat" in ARGS
    decode_weights = "--int8" in ARGS ? Int8 : nothing
    search_arg   = findfirst(a -> startswith(a, "--search="), ARGS)
    search       = search_arg === nothing ? :none : Symbol(split(ARGS[search_arg], "=")[2])

    prompts = [prompt; [a[length("--prompt=")+1:end] for a in ARGS if startswith(a, "--prompt=")]]

    # Apply chat template if requested
    if chat_mode
        prompts = ["<|user|>\n$p</s>\n<|assistant|>\n" for p in prompts]
    end
    prompt = prompts[1]

    println("=========================================")
    println("   Luminal.jl - TinyLlama Text Generation")
    println("=========================================")
    println("Model dir : $model_dir")
    println("Prompt    : \"$prompt\"")
    println("Max tokens: $max_tokens")
    println()

    isdir(model_dir) || error("Model directory not found: $model_dir")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    println("[1/3] Loading tokenizer …")
    tok = LlamaTokenizer(model_dir)
    println("  Vocab size: $(length(tok.vocab))")
    println("  BOS id: $(tok.bos_id)  EOS id: $(tok.eos_id)")

    # ── Model (TinyLlama-1.1B config) ─────────────────────────────────────────
    # TinyLlama: hidden=2048, 22 layers, 32 heads, 4 KV heads, intermediate=5632
    # Use CPUDevice since 4.4GB of weights + intermediates exceeds 8GB GPU VRAM
    # in the current Luminal memory system.
    # Use the best available device (CUDA if present)
    device = get_device()

    # Create dummy model architecture to register node IDs
    println("\n[2/3] Constructing TinyLlama architecture …")
    graph = Graph()
    reg = WeightRegistry()
    model = NN.Llama(graph, reg;
                     vocab_size=32000,
                     hidden=2048,
                     n_layers=22,
                     n_heads=32,
                     n_kv_heads=4,
                     intermediate=5632)
    println("  Parameters registered: ", length(reg.mapping))
    println("  Device: ", device)

    # ── Generate ──────────────────────────────────────────────────────────────
    println("\n[3/3] Generating …")
    println("-" ^ 40)

    t0 = time()
    responses = llama_generate(model, tok, prompts, model_dir;
                               max_new_tokens=max_tokens,
                               max_seq=256,
                               device=device,
                               rope_base=rope_base,  # Decided by command-line or default
                               search=search,
                               decode_weights=decode_weights)
    t1 = time()

    for (p, r) in zip(prompts, responses)
        println(">> PROMPT: $p\n>> RESPONSE: $r")
        println("-" ^ 40)
    end

    n_gen = sum(r -> length(Luminal.encode(tok, r)), responses)
    elapsed = t1 - t0
    @printf "\n[Stats] %d tokens in %.1fs, including weight loading and compilation\n" n_gen elapsed
    println("=========================================")
end

main()
