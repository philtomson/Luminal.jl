module LlamaTokenization

using JSON3

export LlamaTokenizer, encode, decode

"""
    LlamaTokenizer
    
A SentencePiece-style BPE tokenizer used by Llama-2, Llama-3, and Phi-3,
loaded from a HuggingFace `tokenizer.json`.
"""
struct LlamaTokenizer
    vocab::Dict{String,Int}                        # token => id
    id_to_token::Dict{Int,String}                  # id => token
    merges::Vector{Tuple{String,String}}           # ordered merge rules
    merge_ranks::Dict{Tuple{String,String},Int}    # (a,b) => rank
    
    # Common special token IDs
    bos_id::Int
    eos_id::Int
    unk_id::Int
    pad_id::Int
end

"""
    LlamaTokenizer(model_dir)

Load the tokenizer from a HuggingFace model directory containing `tokenizer.json`.
"""
function LlamaTokenizer(model_dir::String)
    tj = joinpath(model_dir, "tokenizer.json")
    !isfile(tj) && error("tokenizer.json not found in $model_dir")
    
    data = JSON3.read(read(tj, String))
    
    vocab = Dict{String,Int}()
    merges = Tuple{String,String}[]
    
    # Vocabulary
    raw_vocab = data["model"]["vocab"]
    for (tok, id) in pairs(raw_vocab)
        vocab[String(tok)] = Int(id)
    end
    
    # Merges
    for entry in data["model"]["merges"]
        parts = split(String(entry), ' ')
        if length(parts) == 2
            push!(merges, (parts[1], parts[2]))
        end
    end
    
    id_to_token = Dict(v => k for (k, v) in vocab)
    merge_ranks = Dict(p => i for (i, p) in enumerate(merges))
    
    # Resolve special tokens
    _id(n) = get(vocab, n, -1)
    bos_id = _id("<s>")
    eos_id = _id("</s>")
    unk_id = _id("<unk>")
    pad_id = _id("<pad>")
    
    # Handle added tokens if they are not in the main vocab
    if haskey(data, "added_tokens")
        for st in data["added_tokens"]
            content = String(st["content"])
            id = Int(st["id"])
            vocab[content] = id
            id_to_token[id] = content
        end
    end
    
    return LlamaTokenizer(vocab, id_to_token, merges, merge_ranks, bos_id, eos_id, unk_id, pad_id)
end

# ─────────────────────────────────────────────────────────────────────────────
# BPE Logic
# ─────────────────────────────────────────────────────────────────────────────

function _bpe_encode(word::Vector{String}, merge_ranks::Dict{Tuple{String,String},Int})
    symbols = copy(word)
    isempty(symbols) && return symbols

    while length(symbols) > 1
        best_rank = typemax(Int)
        best_idx  = -1
        for i in 1:(length(symbols) - 1)
            pair = (symbols[i], symbols[i+1])
            rank = get(merge_ranks, pair, typemax(Int))
            if rank < best_rank
                best_rank = rank
                best_idx  = i
            end
        end
        best_rank == typemax(Int) && break

        merged = symbols[best_idx] * symbols[best_idx + 1]
        new_symbols = String[]
        i = 1
        while i <= length(symbols)
            if i == best_idx
                push!(new_symbols, merged)
                i += 2
            else
                push!(new_symbols, symbols[i])
                i += 1
            end
        end
        symbols = new_symbols
    end
    return symbols
end

"""
    encode(tok, text; bos=false, eos=false) -> Vector{Int}

Encode text into Llama/Phi-3 token IDs.
"""
function encode(tok::LlamaTokenizer, text::String; bos::Bool=false, eos::Bool=false)
    # 1. Pre-processing: SentencePiece style
    # Replace space with U+2581 (lower one eighth block)
    # Llama-2 usually prepends a space if the string doesn't start with one.
    processed = replace(text, " " => "\u2581")
    if !startswith(processed, "\u2581")
        processed = "\u2581" * processed
    end
    
    # 2. Split into characters
    chars = [string(c) for c in processed]
    
    # 3. Apply BPE
    bpe_tokens = _bpe_encode(chars, tok.merge_ranks)
    
    # 4. Convert to IDs
    ids = Int[]
    bos && tok.bos_id != -1 && push!(ids, tok.bos_id)
    
    for t in bpe_tokens
        id = get(tok.vocab, t, -1)
        if id != -1
            push!(ids, id)
        else
            # Byte fallback or unknown
            # For Llama-2, unknown chars are often UTF-8 bytes like <0xXX>
            for b in codeunits(t)
                hex = uppercase(string(b, base=16, pad=2))
                byte_token = "<0x$hex>"
                push!(ids, get(tok.vocab, byte_token, tok.unk_id))
            end
        end
    end
    
    eos && tok.eos_id != -1 && push!(ids, tok.eos_id)
    return ids
end

"""
    decode(tok, ids) -> String

Decode token IDs back to a string.
"""
function decode(tok::LlamaTokenizer, ids::Vector{Int})
    text = ""
    for id in ids
        token = get(tok.id_to_token, id, "")
        if isempty(token) || id in [tok.bos_id, tok.eos_id, tok.pad_id]
            continue
        end
        
        # Byte fallback check: <0xXX>
        if length(token) == 6 && startswith(token, "<0x") && endswith(token, ">")
            try
                b = parse(UInt8, token[4:5], base=16)
                # This is tricky for multi-byte UTF-8. 
                # For now, we'll just push the byte if we can.
                # In a robust decoder, we'd collect bytes and convert to String.
                text *= Char(b)
            catch
                text *= token
            end
        else
            text *= token
        end
    end
    
    # Replace U+2581 back to space
    decoded = replace(text, "\u2581" => " ")
    
    # Handle the potential leading space if added by pre-processing
    # Actually, the leading block ' ' should just become a space.
    # If the original text didn't have it, we might want to trim it, 
    # but usually llama output includes it.
    
    return decoded
end

end # module LlamaTokenization
