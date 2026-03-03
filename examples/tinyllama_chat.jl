# TinyLlama End-to-End Text Generation
#
# Usage:
#   julia --project=. examples/tinyllama_chat.jl [model_dir] [prompt]
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
    model_dir    = length(ARGS) >= 1 ? ARGS[1] : "/devel/phil/Llama-3.2"
    prompt       = length(ARGS) >= 2 ? ARGS[2] : "Once upon a time"
    max_tokens   = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 100

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
    print(">> PROMPT: $prompt\n>> RESPONSE: ")

    t0 = time()
    response = llama_generate(model, tok, prompt, model_dir;
                               max_new_tokens=max_tokens,
                               max_seq=256,
                               device=device,
                               rope_base=10000f0)  # TinyLlama uses Llama-2 RoPE base
    t1 = time()

    println(response)
    println("-" ^ 40)

    n_gen = length(encode(tok, response))
    elapsed = t1 - t0
    @printf "\n[Stats] %d tokens in %.1fs  (%.1f tok/s)\n" n_gen elapsed (n_gen / elapsed)
    println("=========================================")
end

main()
