using Test
using Luminal
using Luminal.NN

@testset "WeightRegistry basic functionality" begin
    g   = Graph()
    reg = WeightRegistry()

    # Register a dummy tensor
    t = tensor(g, [4, 4])
    register_weight!(reg, "layer.weight", t)

    @test length(reg.mapping) == 1
    @test reg.mapping["layer.weight"] == t.id
end

@testset "AudioEncoder with WeightRegistry" begin
    g   = Graph()
    reg = WeightRegistry()

    enc = NN.AudioEncoder(g; reg=reg)

    # Should have registered conv1, conv2, post_ln and per-layer weights
    wkeys = collect(Base.keys(reg.mapping))
    @test any(occursin("conv1.weight", k) for k in wkeys)
    @test any(occursin("conv2.weight", k) for k in wkeys)
    @test any(occursin("layer_norm.weight", k) for k in wkeys)

    # Encoder layers should be registered with layer index
    @test any(occursin("layers.0", k) for k in wkeys)
    @test any(occursin("layers.0.attn.q_proj.weight", k) for k in wkeys)
    @test any(occursin("layers.0.mlp.fc1.weight", k) for k in wkeys)

    println("Registered $(length(reg.mapping)) encoder weight tensors")
end

@testset "TextDecoder with WeightRegistry" begin
    g   = Graph()
    reg = WeightRegistry()

    dec = NN.TextDecoder(g; reg=reg)

    wkeys = collect(Base.keys(reg.mapping))
    @test any(occursin("embed_tokens.weight", k) for k in wkeys)
    @test any(occursin("embed_positions.weight", k) for k in wkeys)
    @test any(occursin("layers.0.self_attn.q_proj.weight", k) for k in wkeys)
    @test any(occursin("layers.0.encoder_attn.q_proj.weight", k) for k in wkeys)
    @test any(occursin("layer_norm.weight", k) for k in wkeys)

    println("Registered $(length(reg.mapping)) decoder weight tensors")
end

@testset "Full Whisper WeightRegistry (AudioEncoder + TextDecoder)" begin
    g   = Graph()
    reg = WeightRegistry()

    enc = NN.AudioEncoder(g; reg=reg)
    dec = NN.TextDecoder(g; reg=reg)

    n = length(reg.mapping)
    println("Total Whisper-tiny parameters registered: $n")

    # Both encoder AND decoder should be in the registry
    enc_keys = filter(k -> startswith(k, "model.encoder"), collect(Base.keys(reg.mapping)))
    dec_keys = filter(k -> startswith(k, "model.decoder"), collect(Base.keys(reg.mapping)))
    @test length(enc_keys) > 0
    @test length(dec_keys) > 0
    println("  Encoder params: $(length(enc_keys)), Decoder params: $(length(dec_keys))")

    # The number of params registered should be > 100 (whisper-tiny has ~39M params)
    @test n > 50
end

@testset "load_weights! with dummy safetensors file" begin
    g   = Graph()
    reg = WeightRegistry()

    # Small linear for test
    l = NN.Linear(4, 8, g; bias=false)
    register_weight!(reg, "test.weight", l.weight)

    # Write a minimal valid safetensors binary by hand.
    # Format: 8 bytes (uint64le header_size) + JSON header + raw tensor data.
    # Tensor: float32, shape [8,4] = 32 floats = 128 bytes
    dummy_weight = collect(Float32, 1:32)
    # SafeTensors stores tensors row-major (C order): shape [8, 4]
    # Header JSON:
    header_json = bytes = codeunits("""{
        "test.weight": {"dtype":"F32","shape":[8,4],"data_offsets":[0,128]}
    }""")
    header_size = length(header_json)
    # Pad to 8-byte boundary
    pad_len = mod(-header_size, 8)
    header_padded = [header_json; fill(UInt8(' '), pad_len)]
    total_header_size = length(header_padded)

    tmp = tempname() * ".safetensors"
    open(tmp, "w") do io
        # 8-byte little-endian header size
        write(io, htol(UInt64(total_header_size)))
        write(io, header_padded)
        # Raw float32 data (the loader does column-permutation internally)
        write(io, reinterpret(UInt8, dummy_weight))
    end

    # Load it
    load_weights!(g, reg, tmp)

    # Check it ended up in the graph's tensor store
    @test haskey(g.tensors, (l.weight.id, 1))
    loaded = g.tensors[(l.weight.id, 1)]
    @test length(loaded) == 32

    rm(tmp)
end
