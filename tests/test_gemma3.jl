# Gemma3 architecture on a tiny random model, with a sliding window short enough
# to matter (window 3, every other layer global): prefill against a plain-Julia
# reference (scaled embeddings, (1 + w) RMSNorms, four norms per block, QK-norm,
# GELU-tanh MLP, local/global RoPE with linear scaling, windowed attention, tied
# head), and the cached decode step (DecodeAttention's window) against it at every
# position, CPU and GPU. Plus config parsing, and the tokenizer when `gemma3_4b/`
# is present.
using Test
using Luminal
using Luminal.NN

const GCFG = (vocab_size=40, hidden=32, n_layers=4, n_heads=4, n_kv_heads=2, head_dim=16, intermediate=48,
              rope_base=1f6, rope_local_base=1f4, rope_scaling=(type=:linear, factor=8.0),
              sliding_window=3, pattern=2, query_pre_attn_scalar=12, rms_eps=1f-6, prefix="model")

devices = Any[CPUDevice()]
Luminal.get_device() isa CPUDevice || push!(devices, Luminal.get_device())

function gemma_weights()
    g = Graph(); reg = WeightRegistry(); Gemma3(g, reg; GCFG...)
    Dict{String,Any}(k => 0.3f0 .* randn(Float32, Tuple(Luminal.realized_dims(g.shapes[id]))...)
                     for (k, id) in reg.mapping if k != "lm_head.weight")
end

grms(x, w, eps) = x ./ sqrt.(sum(abs2, x; dims=1) ./ size(x, 1) .+ eps) .* (1 .+ w)
gelu_tanh(x) = 0.5x * (1 + tanh(sqrt(2 / π) * (x + 0.044715x^3)))
function rope(x, pos, inv)                     # x (D, S, H), rotate-half
    h = length(inv); ang = inv .* pos'
    c, s = cos.(ang), sin.(ang)
    x1, x2 = x[1:h, :, :], x[h+1:end, :, :]
    cat(x1 .* c .- x2 .* s, x2 .* c .+ x1 .* s; dims=1)
end
function ref_forward(W, ids)
    c = GCFG; D = c.head_dim; S = length(ids); G = c.n_heads ÷ c.n_kv_heads; p0 = c.prefix
    E = W["$p0.embed_tokens.weight"]
    x = Float64.(permutedims(E[ids .+ 1, :])) .* Float32(sqrt(c.hidden))
    for l in 0:c.n_layers-1
        p = "$p0.layers.$l"
        glob = (l + 1) % c.pattern == 0
        inv = glob ? Float64.(NN.rope_inv_freqs(D, c.rope_base, c.rope_scaling)) : Float64.(NN.rope_inv_freqs(D, c.rope_local_base))
        h = grms(x, W["$p.input_layernorm.weight"], c.rms_eps)
        q = Base.reshape(W["$p.self_attn.q_proj.weight"] * h, D, c.n_heads, S)
        k = Base.reshape(W["$p.self_attn.k_proj.weight"] * h, D, c.n_kv_heads, S)
        v = permutedims(Base.reshape(W["$p.self_attn.v_proj.weight"] * h, D, c.n_kv_heads, S), (1, 3, 2))
        q = rope(permutedims(grms(q, W["$p.self_attn.q_norm.weight"], c.rms_eps), (1, 3, 2)), 0:S-1, inv)
        k = rope(permutedims(grms(k, W["$p.self_attn.k_norm.weight"], c.rms_eps), (1, 3, 2)), 0:S-1, inv)
        o = zeros(D, S, c.n_heads)
        for hq in 1:c.n_heads
            hk = (hq - 1) ÷ G + 1
            sc = (q[:, :, hq]' * k[:, :, hk]) ./ sqrt(c.query_pre_attn_scalar)
            for i in 1:S, j in 1:S
                (j > i || (!glob && i - j >= c.sliding_window)) && (sc[i, j] = -Inf)
            end
            pr = exp.(sc .- maximum(sc; dims=2)); pr ./= sum(pr; dims=2)
            o[:, :, hq] = v[:, :, hk] * pr'
        end
        a = W["$p.self_attn.o_proj.weight"] * Base.reshape(permutedims(o, (1, 3, 2)), D * c.n_heads, S)
        x = x .+ grms(a, W["$p.post_attention_layernorm.weight"], c.rms_eps)
        h = grms(x, W["$p.pre_feedforward_layernorm.weight"], c.rms_eps)
        f = W["$p.mlp.down_proj.weight"] * (gelu_tanh.(W["$p.mlp.gate_proj.weight"] * h) .* (W["$p.mlp.up_proj.weight"] * h))
        x = x .+ grms(f, W["$p.post_feedforward_layernorm.weight"], c.rms_eps)
    end
    return E * grms(x, W["$p0.norm.weight"], c.rms_eps)
end

@testset "Gemma3 architecture" begin
    W = gemma_weights()
    ids = [3, 14, 15, 9, 2, 6, 30, 11]          # longer than the window
    ref = ref_forward(W, ids)
    for dev in devices
        g = Graph(); reg = WeightRegistry(); m = Gemma3(g, reg; GCFG...)
        @test [l.attention.window for l in m.layers] == [3, 0, 3, 0]
        inp = Luminal.tensor(g, [length(ids), 1]); out = m(inp, 0)
        load_weights!(g, reg, W; device=dev)
        lg = Array(compile(g; device=dev, retain=[out.id])(Dict{Int,Any}(inp.id => Float32.(Base.reshape(ids, :, 1))); device=dev)[out.id])[:, :, 1]
        @test maximum(abs.(lg .- ref)) / maximum(abs.(ref)) < 1e-4

        g = Graph(); reg = WeightRegistry(); m = Gemma3(g, reg; GCFG...)
        idg = build_llama_decode_step!(m, g, 0; max_seq=16, batch=1, rope_base=GCFG.rope_base)
        load_weights!(g, reg, W; device=dev)
        retain = vcat(idg.logits_id, idg.new_self_k_ids, idg.new_self_v_ids, idg.token_input_id,
                      idg.pos_input_id, idg.self_k_ids, idg.self_v_ids)
        exec = compile(g; device=dev, retain=retain, free_intermediates=false)
        cache = LlamaKVCacheState(GCFG.n_layers, GCFG.n_kv_heads, GCFG.head_dim; batch=1, max_seq=16, device=dev)
        steps = hcat([Array(llama_decode_step!(exec, idg, cache, t; device=dev))[:, 1, 1] for t in ids]...)
        @test maximum(abs.(steps .- ref)) / maximum(abs.(ref)) < 1e-4
    end
end

@testset "Gemma3 config" begin
    dir = mktempdir()
    write(joinpath(dir, "config.json"), """{"model_type": "gemma3", "text_config": {"head_dim": 256,
        "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 2560, "intermediate_size": 10240,
        "num_attention_heads": 8, "num_hidden_layers": 34, "num_key_value_heads": 4, "query_pre_attn_scalar": 256,
        "rms_norm_eps": 1e-06, "rope_local_base_freq": 10000.0, "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
        "rope_theta": 1000000.0, "sliding_window": 1024, "sliding_window_pattern": 6, "vocab_size": 262208}}""")
    c = gemma3_config(dir)
    @test c.head_dim == 256 && c.sliding_window == 1024 && c.pattern == 6 && c.rope_scaling == (type=:linear, factor=8.0)
    @test c.prefix == "model"                                   # no index file: text-only naming
end

gdir = joinpath(@__DIR__, "..", "gemma3_4b")
if isfile(joinpath(gdir, "tokenizer.json"))
    @testset "Gemma3 tokenizer" begin
        tok = LlamaTokenizer(gdir)
        ids = Luminal.encode(tok, chat_prompt(tok, "Hi there"); bos=true)
        # <bos><start_of_turn>user\nHi there<end_of_turn>\n<start_of_turn>model\n
        @test ids[1:4] == [2, 105, 2364, 107] && ids[end-4:end] == [106, 107, 105, 4368, 107]
        @test !startswith(tok.id_to_token[Luminal.encode(tok, "Hi")[1]], "▁")
    end
end
