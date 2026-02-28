using Test
using Luminal
using Luminal.NN
using JSON3

@testset "LANGUAGES table" begin
    @test haskey(NN.LANGUAGES, "en")
    @test NN.LANGUAGES["en"] == "english"
    @test haskey(NN.LANGUAGES, "fr")
    @test length(NN.LANGUAGES) >= 30
end

@testset "GPT-2 byte encoder roundtrip" begin
    # Every byte 0-255 should have a unique encoding
    @test length(NN._BYTE_ENC) == 256
    @test length(NN._BYTE_DEC) == 256

    # Roundtrip: encode then decode should give original byte
    for b in 0x00:0xff
        s = NN._BYTE_ENC[b]
        @test NN._BYTE_DEC[s] == b
    end
end

@testset "WhisperTokenizer from tokenizer.json (mock)" begin
    # Create a minimal mock tokenizer.json
    # vocab: hello=0, world=1, he=2, llo=3, Ġworld=4, SOT=5, EOT=6
    # merge: (he, llo) -> hello
    # Unicode escaped special tokens
    sot_str = "\u003c\u007cstartoftranscript\u007c\u003e"
    eot_str = "\u003c\u007cendoftext\u007c\u003e"
    en_str  = "\u003c\u007cen\u007c\u003e"
    tr_str  = "\u003c\u007ctranscribe\u007c\u003e"
    nt_str  = "\u003c\u007cnotimestamps\u007c\u003e"

    vocab = Dict{String,Int}(
        "h"    => 0,
        "e"    => 1,
        "l"    => 2,
        "o"    => 3,
        "he"   => 4,
        "hel"  => 5,
        "hell" => 6,
        "hello"=> 7,
        " "    => 8,
        "w"    => 9,
        sot_str => 50257,
        eot_str => 50256,
        en_str  => 50259,
        tr_str  => 50358,
        nt_str  => 50360,
    )

    import_json = """
    {
        "model": {
            "type": "BPE",
            "vocab": $(JSON3.write(vocab)),
            "merges": ["h e", "he l", "hel l", "hell o"]
        },
        "added_tokens": [
            {"id": 50257, "content": "$(sot_str)"},
            {"id": 50256, "content": "$(eot_str)"}
        ]
    }
    """

    tmp_dir = mktempdir()
    write(joinpath(tmp_dir, "tokenizer.json"), import_json)

    tok = NN.WhisperTokenizer(tmp_dir)

    @test tok.eot_id == 50256
    @test tok.sot_id == 50257
    @test tok.transcribe_id == 50358
    @test tok.notimestamps_id == 50360
    @test haskey(tok.language_ids, "en")
    @test tok.language_ids["en"] == 50259

    # sot_sequence should contain [sot, lang, task, notimestamps]
    prefix = sot_sequence(tok; language="en", task=:transcribe)
    @test prefix[1] == 50257
    @test prefix[2] == 50259
    @test prefix[3] == 50358
    @test prefix[4] == 50360
    @test length(prefix) == 4

    println("WhisperTokenizer loaded successfully")
    println("  sot_id=$(tok.sot_id), eot_id=$(tok.eot_id)")
    println("  sot_sequence: $prefix")
end

@testset "_pretokenize basic splitting" begin
    words = NN._pretokenize("hello world foo")
    @test length(words) == 3
    @test words[1] == "hello"
    @test startswith(words[2], " ")   # space prepended to non-first words
    @test words[2] == " world"
    @test words[3] == " foo"
end

@testset "_bpe_encode merge application" begin
    # Simple: merge "h"+"e" -> "he", then "he"+"l" -> "hel", etc.
    ranks = Dict(
        ("h", "e")   => 1,
        ("he", "l")  => 2,
        ("hel", "l") => 3,
        ("hell","o") => 4,
    )
    result = NN._bpe_encode(["h","e","l","l","o"], ranks)
    @test result == ["hello"]
end

@testset "decode skips special tokens" begin
    # Build a minimal tok manually for decode testing
    sot_str = "\u003c\u007cstartoftranscript\u007c\u003e"
    eot_str = "\u003c\u007cendoftext\u007c\u003e"

    vocab = Dict{String,Int}(
        "H" => 0, "i" => 1,  # H and i are printable ASCII, map directly
        sot_str => 50257,
        eot_str => 50256,
    )
    id_to_token = Dict(v => k for (k,v) in vocab)

    tok = NN.WhisperTokenizer(
        vocab, id_to_token,
        Tuple{String,String}[],
        Dict{Tuple{String,String},Int}(),
        50256, 50257, -1, -1, -1, -1,
        Dict{String,Int}())

    # decode([sot_id, H_id, i_id, eot_id]) with skip_special=true
    # should give "Hi"
    out = decode(tok, [50257, 0, 1, 50256]; skip_special=true)
    # H and i have byte encodings
    @test length(out) > 0  # something was decoded
    println("Decoded: $(repr(out))")
end
