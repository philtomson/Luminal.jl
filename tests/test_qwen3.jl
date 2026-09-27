# Qwen3 architecture on a tiny random model: head_dim independent of hidden,
# QK-norm (per-head RMSNorm of q and k before RoPE), tied embeddings, eps 1e-6.
# Prefill logits against a plain-Julia reference forward pass; the cached decode
# step against prefill at every position (CPU and GPU); config parsing; and, when
# the Qwen3-0.6B checkpoint is present, the tokenizer's chat prompt.
using Test
using Luminal
using Luminal.NN

const QCFG = (vocab_size=50, hidden=32, n_layers=2, n_heads=4, n_kv_heads=2, intermediate=48,
              rope_base=1f6, head_dim=16, qk_norm=true, rms_eps=1f-6, tie_embeddings=true)
const MAXS = 16

devices = Any[CPUDevice()]
Luminal.get_device() isa CPUDevice || push!(devices, Luminal.get_device())

function qwen_weights()
    g = Graph(); reg = WeightRegistry()
    Llama(g, reg; QCFG...)
    W = Dict{String,Any}(k => 0.3f0 .* randn(Float32, Tuple(Luminal.realized_dims(g.shapes[id]))...)
                         for (k, id) in reg.mapping if k != "lm_head.weight")   # tied: not in the checkpoint
    for (k, v) in W
        endswith(k, "norm.weight") && (W[k] = 1f0 .+ 0.1f0 .* v)
    end
    return W
end

# ---- plain-Julia reference (Hugging Face Qwen3 semantics) ----
rms(x, w, eps) = x ./ sqrt.(sum(abs2, x; dims=1) ./ size(x, 1) .+ eps) .* w
function rope(x, pos, base)                    # x (D, S, H): rotate-half, positions pos
    D = size(x, 1); h = D ÷ 2
    inv = base .^ (-(0:h-1) .* 2 ./ D)
    ang = inv .* pos'                          # (h, S)
    c, s = cos.(ang), sin.(ang)
    x1, x2 = x[1:h, :, :], x[h+1:end, :, :]
    return cat(x1 .* c .- x2 .* s, x2 .* c .+ x1 .* s; dims=1)
end
function ref_forward(W, ids)
    c = QCFG; D = c.head_dim; S = length(ids); G = c.n_heads ÷ c.n_kv_heads
    E = W["model.embed_tokens.weight"]
    x = Float64.(permutedims(E[ids .+ 1, :]))              # (hidden, S)
    for l in 0:c.n_layers-1
        p = "model.layers.$l"
        h = rms(x, W["$p.input_layernorm.weight"], c.rms_eps)
        q = Base.reshape(W["$p.self_attn.q_proj.weight"] * h, D, c.n_heads, S)
        k = Base.reshape(W["$p.self_attn.k_proj.weight"] * h, D, c.n_kv_heads, S)
        v = Base.reshape(W["$p.self_attn.v_proj.weight"] * h, D, c.n_kv_heads, S)
        q = rope(permutedims(rms(q, W["$p.self_attn.q_norm.weight"], c.rms_eps), (1, 3, 2)), 0:S-1, c.rope_base)
        k = rope(permutedims(rms(k, W["$p.self_attn.k_norm.weight"], c.rms_eps), (1, 3, 2)), 0:S-1, c.rope_base)
        v = permutedims(v, (1, 3, 2))                         # (D, S, H)
        o = zeros(D, S, c.n_heads)
        for hq in 1:c.n_heads
            hk = (hq - 1) ÷ G + 1
            sc = (q[:, :, hq]' * k[:, :, hk]) ./ sqrt(D)      # (Sq, Sk)
            for i in 1:S, j in i+1:S; sc[i, j] = -Inf; end
            pr = exp.(sc .- maximum(sc; dims=2)); pr ./= sum(pr; dims=2)
            o[:, :, hq] = v[:, :, hk] * pr'
        end
        x = x .+ W["$p.self_attn.o_proj.weight"] * Base.reshape(permutedims(o, (1, 3, 2)), D * c.n_heads, S)
        h = rms(x, W["$p.post_attention_layernorm.weight"], c.rms_eps)
        gt, up = W["$p.mlp.gate_proj.weight"] * h, W["$p.mlp.up_proj.weight"] * h
        x = x .+ W["$p.mlp.down_proj.weight"] * (gt ./ (1 .+ exp.(-gt)) .* up)
    end
    return E * rms(x, W["model.norm.weight"], c.rms_eps)    # tied head
end

@testset "Qwen3 architecture" begin
    W = qwen_weights()
    ids = [3, 14, 15, 9, 2, 6, 40]
    ref = ref_forward(W, ids)
    for dev in devices
        # prefill
        g = Graph(); reg = WeightRegistry(); m = Llama(g, reg; QCFG...)
        @test Luminal.realized_dims(g.shapes[reg.mapping["model.layers.0.self_attn.q_proj.weight"]]) == [64, 32]
        @test haskey(reg.mapping, "model.layers.1.self_attn.k_norm.weight")
        inp = Luminal.tensor(g, [length(ids), 1]); out = m(inp, 0)
        load_weights!(g, reg, W; device=dev)
        # the tied head is its own node, loaded with the embedding's array
        @test g.tensors[(reg.mapping["lm_head.weight"], 1)] === g.tensors[(reg.mapping["model.embed_tokens.weight"], 1)]
        lg = Array(compile(g; device=dev, retain=[out.id])(Dict{Int,Any}(inp.id => Float32.(Base.reshape(ids, :, 1))); device=dev)[out.id])[:, :, 1]
        @test maximum(abs.(lg .- ref)) / maximum(abs.(ref)) < 1e-4

        # cached decode, one token per step, reproduces prefill at every position
        g = Graph(); reg = WeightRegistry(); m = Llama(g, reg; QCFG...)
        idg = build_llama_decode_step!(m, g, 0; max_seq=MAXS, batch=1, rope_base=QCFG.rope_base)
        load_weights!(g, reg, W; device=dev)
        retain = vcat(idg.logits_id, idg.new_self_k_ids, idg.new_self_v_ids, idg.token_input_id,
                      idg.pos_input_id, idg.self_k_ids, idg.self_v_ids)
        exec = compile(g; device=dev, retain=retain, free_intermediates=false)
        cache = LlamaKVCacheState(QCFG.n_layers, QCFG.n_kv_heads, QCFG.head_dim; batch=1, max_seq=MAXS, device=dev)
        steps = hcat([Array(llama_decode_step!(exec, idg, cache, t; device=dev))[:, 1, 1] for t in ids]...)
        @test maximum(abs.(steps .- ref)) / maximum(abs.(ref)) < 1e-4
    end
end

@testset "Qwen3 config" begin
    dir = mktempdir()
    write(joinpath(dir, "config.json"), """{"model_type": "qwen3", "vocab_size": 151936, "hidden_size": 1024,
        "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
        "intermediate_size": 3072, "rope_theta": 1000000, "rms_norm_eps": 1e-06, "tie_word_embeddings": true,
        "hidden_act": "silu", "attention_bias": false, "rope_scaling": null, "use_sliding_window": false}""")
    c = llama_config(dir)
    @test c.head_dim == 128 && c.qk_norm && c.rms_eps == 1f-6 && c.tie_embeddings && c.rope_base == 1f6
    write(joinpath(dir, "config.json"), """{"model_type": "mistral", "vocab_size": 10, "hidden_size": 8,
        "num_hidden_layers": 1, "num_attention_heads": 2, "intermediate_size": 16}""")
    @test_throws ErrorException llama_config(dir)
end

qdir = joinpath(@__DIR__, "..", "qwen3_0.6b")
if isfile(joinpath(qdir, "tokenizer.json"))
    @testset "Qwen3 tokenizer" begin
        tok = LlamaTokenizer(qdir)
        @test tok.bos_id == -1                                        # "bos_token": null
        @test 151645 in tok.eos_ids                                   # <|im_end|>
        ids = Luminal.encode(tok, chat_prompt(tok, "Hi"); bos=true)
        # <|im_start|>user\n ... <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
        @test ids[1:3] == [151644, 872, 198]
        @test ids[end-3:end] == [151667, 271, 151668, 271]          # <think> is one token
    end
end
