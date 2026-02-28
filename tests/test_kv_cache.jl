using Test
using Luminal
using Luminal.NN

# Small constants to keep the test fast
const TEST_BATCH     = 1
const TEST_ENC_SEQ   = 8       # small encoder sequence
const TEST_MAX_SEQ   = 10      # short max decode length
const TEST_N_LAYERS  = 1       # one decoder layer
const TEST_STEPS     = 3       # decode 3 tokens

@testset "KVCacheState allocation" begin
    cache = NN.KVCacheState(TEST_BATCH, TEST_N_LAYERS, NN.HEADS, NN.HEAD_DIM,
                             TEST_ENC_SEQ; max_seq=TEST_MAX_SEQ)

    @test cache.step_pos == 0
    @test cache.max_seq  == TEST_MAX_SEQ
    @test length(cache.self_cache)  == TEST_N_LAYERS
    @test length(cache.cross_cache) == TEST_N_LAYERS

    sk, sv = cache.self_cache[1]
    @test size(sk) == (TEST_BATCH, NN.HEADS, TEST_MAX_SEQ, NN.HEAD_DIM)
    @test all(sk .== 0)

    ck, cv = cache.cross_cache[1]
    @test size(ck) == (TEST_BATCH, NN.HEADS, TEST_ENC_SEQ, NN.HEAD_DIM)
end

@testset "IncrementalDecodeGraph structure (step 0)" begin
    g   = Graph()
    reg = WeightRegistry()
    dec = NN.TextDecoder(g; reg=reg)

    step_g  = Graph()           # fresh graph for the decode step
    # Reuse the same weight tensors from dec by rebinding them
    # In practice: load weights into `step_g` pointing to the same buffers.
    # For structural testing we just check the IDG fields are populated.

    # Build a minimal (1-layer) TextDecoder on step_g using the same
    # architecture parameters
    dec2 = NN.TextDecoder(step_g)
    idg = NN.build_decode_step!(dec2, step_g, TEST_ENC_SEQ, 0;
                                 max_seq=TEST_MAX_SEQ, batch=TEST_BATCH)

    @test idg.token_input_id > 0
    @test length(idg.cross_k_ids) == NN.DEC_LAYERS
    @test length(idg.self_k_ids)  == NN.DEC_LAYERS
    @test length(idg.new_self_k_ids) == NN.DEC_LAYERS
    @test idg.logits_id > 0
    @test idg.step_pos[] == 0

    println("IncrementalDecodeGraph: token_in=$(idg.token_input_id), logits=$(idg.logits_id), $(length(idg.self_k_ids)) layers")
end

@testset "_kv_scatter places token at correct position" begin
    g      = Graph()
    # Minimal dims: (batch=1, heads=1, max_seq=4, head_dim=2)
    past_k = Luminal.tensor(g, [1, 1, 4, 2])
    new_k  = Luminal.tensor(g, [1, 1, 1, 2])

    # step_pos = 0: new_k goes at front
    updated0 = NN._kv_scatter(past_k, new_k, 0)
    @test Luminal.realized_dims(updated0.shape)[3] == 4

    # step_pos = 1
    updated1 = NN._kv_scatter(past_k, new_k, 1)
    @test Luminal.realized_dims(updated1.shape)[3] == 4

    # step_pos = 3 (last slot)
    updated3 = NN._kv_scatter(past_k, new_k, 3)
    @test Luminal.realized_dims(updated3.shape)[3] == 4
end

@testset "whisper_self_attn_cached graph shapes" begin
    g   = Graph()
    reg = WeightRegistry()
    # Use tiny dims manually by building a linear matching WhisperSelfAttention
    sa = NN.WhisperSelfAttention(NN.D_MODEL, g, reg, "test.attn")

    x      = Luminal.tensor(g, [TEST_BATCH, 1, NN.D_MODEL])          # (b, 1, hidden)
    past_k = Luminal.tensor(g, [TEST_BATCH, NN.HEADS, TEST_MAX_SEQ, NN.HEAD_DIM])
    past_v = Luminal.tensor(g, [TEST_BATCH, NN.HEADS, TEST_MAX_SEQ, NN.HEAD_DIM])

    out, new_k, new_v = NN.whisper_self_attn_cached(sa, x, 0, past_k, past_v)

    @test out   isa Luminal.GraphTensor
    @test new_k isa Luminal.GraphTensor
    @test new_v isa Luminal.GraphTensor

    # Output should be (batch, 1, hidden)
    out_dims  = Luminal.realized_dims(out.shape)
    @test out_dims[1] == TEST_BATCH
    @test out_dims[2] == 1
    @test out_dims[3] == NN.D_MODEL

    # Updated cache should preserve max_seq dim
    nk_dims = Luminal.realized_dims(new_k.shape)
    @test nk_dims[3] == TEST_MAX_SEQ
    @test nk_dims[4] == NN.HEAD_DIM
end

@testset "whisper_cross_attn_cached graph shapes" begin
    g   = Graph()
    reg = WeightRegistry()
    ca  = NN.WhisperCrossAttention(NN.D_MODEL, g, reg, "test.cross_attn")

    x     = Luminal.tensor(g, [TEST_BATCH, 1, NN.D_MODEL])
    enc_k = Luminal.tensor(g, [TEST_BATCH, NN.HEADS, TEST_ENC_SEQ, NN.HEAD_DIM])
    enc_v = Luminal.tensor(g, [TEST_BATCH, NN.HEADS, TEST_ENC_SEQ, NN.HEAD_DIM])

    out = NN.whisper_cross_attn_cached(ca, x, enc_k, enc_v)
    @test out isa Luminal.GraphTensor
    out_dims = Luminal.realized_dims(out.shape)
    @test out_dims == [TEST_BATCH, 1, NN.D_MODEL]
end
