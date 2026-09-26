using Test
using Luminal
using Luminal.NN

# A small random Llama with grouped-query attention.
const CFG = (vocab_size=50, hidden=64, n_layers=2, n_heads=4, n_kv_heads=2, intermediate=128)
const ROPE = 10000f0
const MAX_SEQ = 16

function llama_weights()
    g = Graph(); reg = WeightRegistry()
    Llama(g, reg; CFG..., rope_base=ROPE)
    Dict{String,Any}(k => 0.3f0 .* randn(Float32, Tuple(Luminal.realized_dims(g.shapes[id]))...)
                     for (k, id) in reg.mapping)
end

function decode_exec(W, batch, dev; capture=false)
    g = Graph(); reg = WeightRegistry()
    m = Llama(g, reg; CFG..., rope_base=ROPE)
    idg = build_llama_decode_step!(m, g, 0; max_seq=MAX_SEQ, batch=batch, rope_base=ROPE)
    load_weights!(g, reg, W; device=dev)
    retain = vcat(idg.logits_id, idg.new_self_k_ids, idg.new_self_v_ids, idg.token_input_id,
                  idg.pos_input_id, idg.self_k_ids, idg.self_v_ids)
    exec = compile(g; device=dev, retain=retain, free_intermediates=false, capture=capture)
    cache = LlamaKVCacheState(CFG.n_layers, CFG.n_kv_heads, CFG.hidden ÷ CFG.n_heads;
                              batch=batch, max_seq=MAX_SEQ, device=dev)
    return exec, idg, cache
end

function prefill(W, ids::Matrix{Int}, dev)          # ids (S, B)
    g = Graph(); reg = WeightRegistry()
    m = Llama(g, reg; CFG..., rope_base=ROPE)
    inp = Luminal.tensor(g, collect(size(ids)))
    out, kvs = m(inp, 0; return_kv=true)
    load_weights!(g, reg, W; device=dev)
    retain = vcat(out.id, [k.id for (k, _) in kvs])
    res = compile(g; device=dev, retain=retain)(Dict{Int,Any}(inp.id => Float32.(ids)); device=dev)
    return Array(res[out.id]), [Array(res[k.id]) for (k, _) in kvs]
end

devices = Any[CPUDevice()]
Luminal.get_device() isa CPUDevice || push!(devices, Luminal.get_device())

for dev in devices
    name = nameof(typeof(dev))
    W = llama_weights()
    seqs = [[3, 14, 15, 9, 2, 6], [7, 1], [11, 30, 4, 8]]
    B = length(seqs)

    for capture in (dev isa Luminal.AMDDevice ? (false, true) : (false,))
    @testset "batched decode == batch-1 decode per sequence ($name, capture=$capture)" begin
        # Reference: each sequence alone, one token per step
        ref = map(seqs) do s
            exec, idg, cache = decode_exec(W, 1, dev)
            [Array(llama_decode_step!(exec, idg, cache, t; device=dev))[:, 1, 1] for t in s]
        end
        # Batched: every step, a sequence either advances to its next token or (while
        # it waits, so the batch's positions diverge) holds at its position.
        exec, idg, cache = decode_exec(W, B, dev; capture=capture)
        fed = zeros(Int, B)
        step = 0
        while any(fed .< length.(seqs))
            step += 1
            adv = [fed[b] < length(seqs[b]) && (b == 1 || step % b == 0) for b in 1:B]
            toks = [seqs[b][min(fed[b] + 1, length(seqs[b]))] for b in 1:B]
            @test cache.positions == fed
            logits = Array(llama_decode_step!(exec, idg, cache, toks; advance=adv, device=dev))
            for b in 1:B
                adv[b] || continue
                fed[b] += 1
                @test logits[:, 1, b] ≈ ref[b][fed[b]] rtol=1e-4
            end
        end
        @test cache.positions == length.(seqs)
        @test_throws ErrorException cache.step_pos       # positions differ
    end
    end

    @testset "right-padded batched prefill == individual prefills ($name)" begin
        L = maximum(length.(seqs))
        ids = zeros(Int, L, B)
        for (b, s) in enumerate(seqs)
            ids[1:length(s), b] .= s
        end
        logits, ks = prefill(W, ids, dev)
        for (b, s) in enumerate(seqs)
            n = length(s)
            l1, k1 = prefill(W, Base.reshape(s, n, 1), dev)
            @test logits[:, 1:n, b] ≈ l1[:, :, 1] rtol=1e-4
            @test ks[1][:, 1:n, :, b] ≈ k1[1][:, :, :, 1] rtol=1e-4
        end
    end
end

# A LlamaSession reuses its weights and compiled graphs across calls; batched and
# single-prompt generation agree. Needs the TinyLlama checkpoint (tinyllama_chat/).
const TINYLLAMA_DIR = get(ENV, "TINYLLAMA_DIR", joinpath(@__DIR__, "..", "tinyllama_chat"))
if isfile(joinpath(TINYLLAMA_DIR, "model.safetensors")) && !(Luminal.get_device() isa CPUDevice)
    @testset "LlamaSession reuse (TinyLlama)" begin
        tok = LlamaTokenizer(TINYLLAMA_DIR)
        model = Llama(Graph(), nothing; vocab_size=32000, hidden=2048, n_layers=22, n_heads=32,
                      n_kv_heads=4, intermediate=5632)
        s = LlamaSession(model, tok, TINYLLAMA_DIR; max_seq=128, rope_base=10000f0, decode_weights=Int8)
        chat(p) = "<|user|>\n$p</s>\n<|assistant|>\n"
        prompts = chat.(["What is the capital of France?", "List three prime numbers."])
        first = generate(s, prompts; max_new_tokens=12)
        @test length(s.prefill) == 1 && length(s.decode) == 1
        @test generate(s, prompts; max_new_tokens=12) == first      # reused cache and graphs
        @test length(s.prefill) == 1 && length(s.decode) == 1
        @test generate(s, prompts[2]; max_new_tokens=12) == first[2]
        @test occursin("Paris", first[1])
    end
else
    @info "Skipping LlamaSession test (no $TINYLLAMA_DIR or no GPU)"
end
