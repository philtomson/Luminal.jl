using Test
using Luminal

# Token ids from Hugging Face `tokenizers` (encode(text, add_special_tokens=False))
# for TinyLlama (SentencePiece-style BPE) and Llama-3-8B-Instruct (byte-level
# BPE). Runs for the checkpoints that are present.
const TEXTS = ["Hello, world!", "Numbers: 1234567 and 3.14159", "Unicode: café, 日本語, Привет 🚀",
               "line1\n\n  indented\ttab", "Hi</s> bye", "<|user|>\nWhat is 2+2?</s>\n<|assistant|>\n",
               "<|start_header_id|>user<|end_header_id|>\n\nWhat is 2+2?<|eot_id|>"]
const HF_IDS = Dict(
    "tinyllama_chat" => [
        [15043, 29892, 3186, 29991],
        [11848, 2596, 29901, 29871, 29896, 29906, 29941, 29946, 29945, 29953, 29955, 322, 29871, 29941, 29889, 29896, 29946, 29896, 29945, 29929],
        [23862, 29901, 274, 28059, 29892, 29871, 30325, 30346, 30968, 29892, 7203, 7616, 29871, 243, 162, 157, 131],
        [1196, 29896, 13, 13, 29871, 1399, 14927, 12, 3891],
        [6324, 2, 491, 29872],
        [529, 29989, 1792, 29989, 29958, 13, 5618, 338, 29871, 29906, 29974, 29906, 29973, 2, 13, 29966, 29989, 465, 22137, 29989, 29958, 13],
        [529, 29989, 2962, 29918, 6672, 29918, 333, 29989, 29958, 1792, 29966, 29989, 355, 29918, 6672, 29918, 333, 29989, 29958, 13, 13, 5618, 338, 29871, 29906, 29974, 29906, 29973, 29966, 29989, 29872, 327, 29918, 333, 29989, 29958]],
    "llama3_8b_instruct" => [
        [9906, 11, 1917, 0],
        [28336, 25, 220, 4513, 10961, 22, 323, 220, 18, 13, 9335, 2946],
        [35020, 25, 53050, 11, 105180, 102158, 11, 80584, 28089, 8341, 11410, 248, 222],
        [1074, 16, 271, 220, 1280, 16243, 59249],
        [13347, 524, 82, 29, 54141],
        [27, 91, 882, 91, 397, 3923, 374, 220, 17, 10, 17, 27147, 82, 397, 27, 91, 78191, 91, 397],
        [128006, 882, 128007, 271, 3923, 374, 220, 17, 10, 17, 30, 128009]])

for (name, expected) in HF_IDS
    dir = joinpath(@__DIR__, "..", name)
    if !isfile(joinpath(dir, "tokenizer.json"))
        @info "Skipping $name tokenizer test (no $dir)"
        continue
    end
    @testset "LlamaTokenizer matches Hugging Face ($name)" begin
        tok = LlamaTokenizer(dir)
        for (text, ids) in zip(TEXTS, expected)
            @test Luminal.encode(tok, text) == ids
        end
        @test Luminal.encode(tok, "") == Int[]
        # Round trip (special tokens are dropped by decode)
        if tok.byte_level
            @test Luminal.decode(tok, Luminal.encode(tok, TEXTS[3])) == TEXTS[3]
            @test tok.eos_ids == [128009, 128001]
            @test Luminal.encode(tok, chat_prompt(tok, "Hi"); bos=true) ==
                  [128000, 128006, 882, 128007, 271, 13347, 128009, 128006, 78191, 128007, 271]
        else
            @test tok.eos_ids == [2]
        end
    end
end
