
using Luminal
using Luminal.NN
using Luminal.LlamaTokenization
using Statistics
using Printf

function main()
    model_dir = "/devel/phil/Llama-3.2"
    prompt = "What is the capital of France?"
    
    println("Loading tokenizer...")
    tok = LlamaTokenizer(model_dir)
    
    device = get_device()
    println("Using device: ", device)
    
    # TinyLlama config
    vocab_size = 32000
    hidden = 2048
    n_layers = 22
    n_heads = 32
    n_kv_heads = 4
    intermediate = 5632
    rope_base = 500000.0f0 # Testing 500k instead of 10k
    
    graph = Graph()
    reg = WeightRegistry()
    model = NN.Llama(graph, reg;
                     vocab_size=vocab_size,
                     hidden=hidden,
                     n_layers=n_layers,
                     n_heads=n_heads,
                     n_kv_heads=n_kv_heads,
                     intermediate=intermediate,
                     rope_base=rope_base)
    
    prompt_ids = LlamaTokenization.encode(tok, prompt; bos=true)
    plen = length(prompt_ids)
    println("Prompt: ", prompt)
    println("Tokens: ", prompt_ids)
    
    # Prefill graph
    pfx_graph = Graph()
    pfx_reg = WeightRegistry()
    pfx_model = Luminal.Decoding._rebuild_model_like(model, pfx_graph, pfx_reg; rope_base=rope_base)
    pfx_input = Luminal.tensor(pfx_graph, [plen, 1])
    pfx_out = pfx_model(pfx_input, 0)
    
    println("Loading weights...")
    weights_dict = load_weights_to_dict(model_dir; device=device)
    load_weights!(pfx_graph, pfx_reg, weights_dict; device=device)
    
    println("Compiling...")
    exec = compile(pfx_graph; device=device)
    
    println("Running prefill...")
    inputs = Dict(pfx_input.id => Float32.(Base.reshape(prompt_ids, plen, 1)))
    results = exec(inputs; device=device)
    println("Prefill complete.")
    logits = results[pfx_out.id]
    # Top tokens at last position
    # logits is (Vocab, Seq, Batch)
    last_logits = view(logits, :, plen, 1)
    println("Logits stats: mean=$(mean(last_logits)), std=$(std(last_logits)), max=$(maximum(last_logits))")
    
    probs = exp.(last_logits .- maximum(last_logits))
    probs ./= sum(probs)
    
    probs_cpu = Array(probs)
    indices_cpu = Array(sortperm(probs, rev=true)[1:20])
    
    println("\nTop 20 predicted tokens:")
    for idx in indices_cpu
        token_id = idx - 1
        t = LlamaTokenization.decode(tok, [token_id])
        raw = get(tok.id_to_token, token_id, "???")
        @printf("ID %5d: %8.4f%%  |%-10s|  (raw: %s)\n", token_id, probs_cpu[idx]*100, t, raw)
    end
    
    # Check "Paris" (Llama-2 ID 7237, or search by text)
    paris_ids = LlamaTokenization.encode(tok, " Paris"; bos=false)
    for p_id in paris_ids
        p_prob = probs_cpu[p_id + 1]
        t_str = LlamaTokenization.decode(tok, [p_id])
        @printf("Token ID %5d (|%s|) prob: %.8f%%\n", p_id, t_str, p_prob*100)
    end
end

main()
