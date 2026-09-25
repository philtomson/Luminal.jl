using Test
using Luminal
using Luminal.NN

# A small random Whisper decoder keeps these tests fast.
const CFG = (d_model=64, n_layers=2, n_heads=4, ffn_dim=128, vocab_size=97, max_positions=12)
const ENC_SEQ = 8

function random_weights(g, reg)
    Dict{String,Any}(k => 0.2f0 .* randn(Float32, Tuple(Luminal.realized_dims(g.shapes[id]))...)
                     for (k, id) in reg.mapping)
end

@testset "KVCacheState allocation" begin
    cache = NN.KVCacheState(CFG.n_layers, CFG.n_heads, 16, ENC_SEQ; max_seq=CFG.max_positions)
    @test cache.step_pos == 0
    @test cache.max_seq == CFG.max_positions
    @test length(cache.self_cache) == CFG.n_layers
    @test length(cache.cross_cache) == CFG.n_layers
    sk, sv = cache.self_cache[1]
    @test size(sk) == (16, CFG.max_positions, CFG.n_heads, 1)   # (D, max_seq, H, B)
    @test all(sk .== 0)
    @test size(cache.cross_cache[1][1]) == (16, ENC_SEQ, CFG.n_heads, 1)
end

@testset "IncrementalDecodeGraph structure" begin
    g = Graph()
    td = NN.TextDecoder(g; CFG...)
    idg = NN.build_decode_step!(td, g, ENC_SEQ; max_seq=CFG.max_positions)
    @test length(idg.cross_k_ids) == CFG.n_layers
    @test length(idg.self_k_ids) == CFG.n_layers
    @test length(idg.new_self_k_ids) == CFG.n_layers
    @test Luminal.realized_dims(g.shapes[idg.logits_id]) == [CFG.vocab_size, 1, 1]
    @test Luminal.realized_dims(g.shapes[idg.new_self_k_ids[1]]) == [16, 1, CFG.n_heads, 1]
end

# Cached decoding must reproduce the full-sequence decoder's logits at every position.
devices = Any[CPUDevice()]
Luminal.get_device() isa CPUDevice || push!(devices, Luminal.get_device())
for dev in devices
    @testset "cached decode == full decoder ($(nameof(typeof(dev))))" begin
        tokens = [3, 17, 5, 42, 8, 0]
        S = length(tokens)
        enc = randn(Float32, CFG.d_model, ENC_SEQ, 1)

        # Full-sequence reference
        g = Graph(); reg = WeightRegistry()
        td = NN.TextDecoder(g; reg=reg, CFG...)
        W = random_weights(g, reg)
        enc_in = Luminal.tensor(g, [CFG.d_model, ENC_SEQ, 1])
        tok_in = Luminal.tensor(g, [S, 1])
        logits = td(enc_in, tok_in)
        load_weights!(g, reg, W; device=dev)
        full = Array(compile(g; device=dev, retain=[logits.id])(
            Dict{Int,Any}(enc_in.id => enc, tok_in.id => Float32.(Base.reshape(tokens, S, 1)));
            device=dev)[logits.id])

        # Cross K/V
        cg = Graph(); creg = WeightRegistry()
        ctd = NN.TextDecoder(cg; reg=creg, CFG...)
        c_in = Luminal.tensor(cg, [CFG.d_model, ENC_SEQ, 1])
        kv = NN.project_cross_kv(ctd, c_in)
        load_weights!(cg, creg, W; device=dev)
        kv_res = compile(cg; device=dev, retain=[t.id for p in kv for t in p])(
            Dict{Int,Any}(c_in.id => enc); device=dev)
        cache = NN.KVCacheState(CFG.n_layers, CFG.n_heads, CFG.d_model ÷ CFG.n_heads, ENC_SEQ;
                                max_seq=CFG.max_positions, device=dev)
        for (i, (k, v)) in enumerate(kv)
            cache.cross_cache[i][1] .= kv_res[k.id]
            cache.cross_cache[i][2] .= kv_res[v.id]
        end

        # One compiled step graph, replayed per position
        dg = Graph(); dreg = WeightRegistry()
        dtd = NN.TextDecoder(dg; reg=dreg, CFG...)
        idg = NN.build_decode_step!(dtd, dg, ENC_SEQ; max_seq=CFG.max_positions)
        load_weights!(dg, dreg, W; device=dev)
        retain = vcat(idg.logits_id, idg.new_self_k_ids, idg.new_self_v_ids, idg.token_input_id,
                      idg.pos_input_id, idg.self_k_ids, idg.self_v_ids, idg.cross_k_ids, idg.cross_v_ids)
        exec = compile(dg; device=dev, retain=retain, free_intermediates=false)
        for (p, t) in enumerate(tokens)
            step = Array(NN.decode_step!(exec, idg, cache, [t]; device=dev))
            @test step[:, 1, 1] ≈ full[:, p, 1] rtol=1e-4
        end
        @test cache.step_pos == S
    end
end
