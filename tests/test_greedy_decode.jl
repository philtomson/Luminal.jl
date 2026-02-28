using Test
using Luminal
using Luminal.NN
using JSON3

@testset "Full Greedy Decode Flow (Mock)" begin
    # 1. Setup mock model directory
    tmp_dir = mktempdir()
    
    # Mock vocab.json
    sot_str = "\u003c\u007cstartoftranscript\u007c\u003e"
    eot_str = "\u003c\u007cendoftext\u007c\u003e"
    en_str  = "\u003c\u007cen\u007c\u003e"
    tr_str  = "\u003c\u007ctranscribe\u007c\u003e"
    nt_str  = "\u003c\u007cnotimestamps\u007c\u003e"

    vocab = Dict{String,Int}(
        "h"     => 0, "e" => 1, "l" => 2, "o" => 3, " " => 4,
        sot_str => 50257,
        eot_str => 50258,
        en_str  => 50259,
        tr_str  => 50260,
        nt_str  => 50261,
    )
    write(joinpath(tmp_dir, "vocab.json"), JSON3.write(vocab))
    write(joinpath(tmp_dir, "merges.txt"), "#version: 0.2\nh e\nhe l\nhel l\nhell o")
    
    # Mock weights (minimal set for TextDecoder)
    # We'll just create a dummy safetensors file with some random data for all keys
    # Actually, to make it easier, we'll just mock the directory for the tokenizer
    # and use the native load_weights! from memory in the test.
    
    tokenizer = WhisperTokenizer(tmp_dir)
    
    # 2. Setup Graph and TextDecoder
    g = Graph()
    reg = WeightRegistry()
    td = TextDecoder(g; reg=reg)
    
    # Mock enc_output
    batch, enc_seq = 1, 8
    enc_output = rand(Float32, batch, enc_seq, NN.D_MODEL)
    
    # 3. Mock weights_dict
    weights_dict = Dict{String, Array{Float32}}()
    for (name, id) in reg.mapping
        st = g.shapes[id]
        dims = Int[d for d in Luminal.realized_dims(st)]
        weights_dict[name] = rand(Float32, dims...)
    end
    
    # I'll create a dummy .safetensors file in tmp_dir
    function save_dummy_safetensors(path, dict)
        header = Dict{String, Any}("__metadata__" => Dict("format" => "pt"))
        offset = 0
        data = UInt8[]
        for (k, v) in dict
            # SafeTensors.jl load_safetensors PERMUTES them.
            # So if we want load_safetensors to give us what we want, 
            # we need to provide them "transposed" or know that load_safetensors 
            # will flip them back.
            # For random data, shape just needs to be consistent.
            b = reinterpret(UInt8, vec(v))
            append!(data, b)
            header[k] = Dict("dtype" => "F32", "shape" => reverse(size(v)), "data_offsets" => [offset, offset + length(b)])
            offset += length(b)
        end
        header_json = JSON3.write(header)
        header_size = length(header_json)
        pad = mod(-header_size, 8)
        header_padded = header_json * (" "^pad)
        total_header_size = length(header_padded)
        
        open(path, "w") do io
            write(io, htol(UInt64(total_header_size)))
            write(io, header_padded)
            write(io, data)
        end
    end
    
    save_dummy_safetensors(joinpath(tmp_dir, "model.safetensors"), weights_dict)
    
    # 4. Run Greedy Decode
    println("Starting greedy_decode with mock weights...")
    # Using a very small max_len for speed
    text, ids = greedy_decode(td, tokenizer, enc_output, tmp_dir; max_len=5)
    
    @test length(ids) > 0
    @test ids[1] == tokenizer.sot_id
    @test length(text) >= 0 
    
    println("Decoded text: ", repr(text))
    println("Token IDs: ", ids)
end
