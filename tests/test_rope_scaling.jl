# RoPE scaling (Llama-3.1 / 3.2 "llama3", and "linear"): the scaled inverse
# frequencies against values from transformers, config parsing, and a tiny scaled
# model whose cached decode reproduces its prefill (CPU and GPU).
using Test
using Luminal
using Luminal.NN

devices = Any[CPUDevice()]
Luminal.get_device() isa CPUDevice || push!(devices, Luminal.get_device())

const L3 = (type=:llama3, factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0,
            original_max_position_embeddings=8192.0)          # Llama-3.2-1B/3B

@testset "scaled frequencies" begin
    inv = NN.rope_inv_freqs(64, 500000f0, L3)
    base = NN.rope_inv_freqs(64, 500000f0)
    # transformers' _compute_llama3_parameters for Llama-3.2-1B (head_dim 64)
    hf = Dict(1 => 1.0, 13 => 0.00729266507551074, 16 => 0.0012905480107292533,
              19 => 1.9461638657958247e-05, 22 => 5.687232260243036e-06,
              25 => 1.6619674170215148e-06, 32 => 9.418306490260875e-08)
    for (i, v) in hf
        @test inv[i] ≈ v rtol=1e-6
    end
    @test inv[1:15] == base[1:15]                             # short wavelengths unchanged
    @test inv[end] ≈ base[end] / 32 rtol=1e-6                 # long ones divided by the factor
    @test NN.rope_inv_freqs(64, 1f6, (type=:linear, factor=8.0)) ≈ NN.rope_inv_freqs(64, 1f6) ./ 8
end

@testset "config" begin
    dir = mktempdir()
    cfg(rs) = write(joinpath(dir, "config.json"), """{"vocab_size": 10, "hidden_size": 8, "num_hidden_layers": 1,
        "num_attention_heads": 2, "intermediate_size": 16, "rope_theta": 500000, "rope_scaling": $rs}""")
    cfg("""{"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192}""")
    @test llama_config(dir).rope_scaling == L3
    cfg("""{"type": "linear", "factor": 4.0}""")                 # older key name
    @test llama_config(dir).rope_scaling == (type=:linear, factor=4.0)
    cfg("""{"rope_type": "default"}""")
    @test llama_config(dir).rope_scaling === nothing
    cfg("""{"rope_type": "yarn", "factor": 4.0}""")
    @test_throws ErrorException llama_config(dir)
end

@testset "scaled model: decode == prefill" begin
    # small head_dim and base so the scaling changes angles at these positions
    C = (vocab_size=40, hidden=32, n_layers=2, n_heads=4, n_kv_heads=2, intermediate=48, rope_base=100f0,
         rope_scaling=(type=:llama3, factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0,
                       original_max_position_embeddings=16.0))
    g = Graph(); reg = WeightRegistry(); Llama(g, reg; C...)
    W = Dict{String,Any}(k => 0.3f0 .* randn(Float32, Tuple(Luminal.realized_dims(g.shapes[id]))...) for (k, id) in reg.mapping)
    ids = [3, 14, 15, 9, 2, 6, 30, 11]
    for dev in devices
        g = Graph(); reg = WeightRegistry(); m = Llama(g, reg; C...)
        inp = Luminal.tensor(g, [length(ids), 1]); out = m(inp, 0)
        load_weights!(g, reg, W; device=dev)
        pre = Array(compile(g; device=dev, retain=[out.id])(Dict{Int,Any}(inp.id => Float32.(Base.reshape(ids, :, 1))); device=dev)[out.id])[:, :, 1]

        g = Graph(); reg = WeightRegistry(); m = Llama(g, reg; C...)
        idg = build_llama_decode_step!(m, g, 0; max_seq=16, batch=1, rope_base=C.rope_base)
        load_weights!(g, reg, W; device=dev)
        retain = vcat(idg.logits_id, idg.new_self_k_ids, idg.new_self_v_ids, idg.token_input_id,
                      idg.pos_input_id, idg.self_k_ids, idg.self_v_ids)
        exec = compile(g; device=dev, retain=retain, free_intermediates=false)
        cache = LlamaKVCacheState(C.n_layers, C.n_kv_heads, 8; batch=1, max_seq=16, device=dev)
        steps = hcat([Array(llama_decode_step!(exec, idg, cache, t; device=dev))[:, 1, 1] for t in ids]...)
        @test maximum(abs.(steps .- pre)) / maximum(abs.(pre)) < 1e-4

        # and the scaling is in effect: the unscaled model differs
        g = Graph(); reg = WeightRegistry(); m = Llama(g, reg; C..., rope_scaling=nothing)
        inp = Luminal.tensor(g, [length(ids), 1]); out = m(inp, 0)
        load_weights!(g, reg, W; device=dev)
        plain = Array(compile(g; device=dev, retain=[out.id])(Dict{Int,Any}(inp.id => Float32.(Base.reshape(ids, :, 1))); device=dev)[out.id])[:, :, 1]
        @test maximum(abs.(plain .- pre)) / maximum(abs.(pre)) > 1e-3
    end
end
