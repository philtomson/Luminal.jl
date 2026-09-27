module LlamaTokenization

using JSON3
import ..Luminal: encode, decode

export LlamaTokenizer, encode, decode, chat_prompt

"""
    LlamaTokenizer

A BPE tokenizer loaded from a Hugging Face `tokenizer.json`, in either of the two
forms Llama-family models use:
- SentencePiece style (Llama-2, TinyLlama, Phi-3): spaces become `▁`, unknown
  characters fall back to `<0xXX>` byte tokens.
- Byte-level (Llama-3): text is pre-split by the tokenizer's regex, every byte is
  mapped to a printable character (GPT-2's byte encoder), then BPE is applied.
  Special tokens such as `<|eot_id|>` in the text are recognized as single tokens.
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

    eos_ids::Vector{Int}          # every id that ends generation (eos_id included)
    special_ids::Set{Int}         # special tokens, dropped by `decode`
    byte_level::Bool              # byte-level BPE (Llama-3) rather than SentencePiece
    pattern::Union{Regex,Nothing} # byte-level pre-tokenizer split
    ignore_merges::Bool           # a whole pre-token found in the vocab is one token
    specials::Union{Regex,Nothing}  # matches special tokens in the input text
end

# GPT-2's reversible byte <-> printable-character mapping.
function _byte_encoder()
    bs = vcat(Int('!'):Int('~'), Int('¡'):Int('¬'), Int('®'):Int('ÿ'))
    cs = copy(bs)
    n = 0
    for b in 0:255
        if !(b in bs)
            push!(bs, b); push!(cs, 256 + n); n += 1
        end
    end
    enc = Dict(UInt8(b) => Char(c) for (b, c) in zip(bs, cs))
    return enc, Dict(c => b for (b, c) in enc)
end
const BYTE_ENCODER, BYTE_DECODER = _byte_encoder()

_json_get(d, k, default) = haskey(d, k) && d[k] !== nothing ? d[k] : default

"""
    LlamaTokenizer(model_dir)

Load the tokenizer from a Hugging Face model directory containing `tokenizer.json`.
The end-of-sequence ids come from `tokenizer_config.json` and `generation_config.json`
when present (Llama-3-Instruct ends turns with `<|eot_id|>` as well as `<|end_of_text|>`).
"""
function LlamaTokenizer(model_dir::String)
    tj = joinpath(model_dir, "tokenizer.json")
    !isfile(tj) && error("tokenizer.json not found in $model_dir")

    data = JSON3.read(read(tj, String))
    model = data["model"]

    vocab = Dict{String,Int}()
    merges = Tuple{String,String}[]

    # Vocabulary
    for (tok, id) in pairs(model["vocab"])
        vocab[String(tok)] = Int(id)
    end

    # Merges: "a b" strings, or [a, b] pairs in newer tokenizer.json files
    # (pairs are indexed, not collected: collect on a JSON3.Array is so slow that
    # Qwen3's 151k pairs took over two minutes)
    for entry in model["merges"]
        if entry isa AbstractString
            parts = split(String(entry), ' ')
            length(parts) == 2 && push!(merges, (String(parts[1]), String(parts[2])))
        elseif length(entry) == 2
            push!(merges, (String(entry[1]), String(entry[2])))
        end
    end

    id_to_token = Dict(v => k for (k, v) in vocab)
    merge_ranks = Dict(p => i for (i, p) in enumerate(merges))

    # Added tokens (special tokens, and e.g. Qwen3's <think>, live here)
    special_ids = Set{Int}()
    special_strs = String[]
    for st in _json_get(data, "added_tokens", [])
        content = String(st["content"])
        id = Int(st["id"])
        vocab[content] = id
        id_to_token[id] = content
        # every added token is split out of the text before BPE, as Hugging Face
        # does, special or not (Qwen3's <think> is "special": false); only the
        # special ones are hidden when decoding
        push!(special_strs, content)
        _json_get(st, "special", false) && push!(special_ids, id)
    end

    # Byte-level pre-tokenization (Llama-3): a Split regex followed by ByteLevel
    pre = _json_get(data, "pre_tokenizer", nothing)
    pres = pre === nothing ? [] : pre["type"] == "Sequence" ? collect(pre["pretokenizers"]) : [pre]
    byte_level = any(p -> p["type"] == "ByteLevel", pres)
    pattern = nothing
    for p in pres
        if p["type"] == "Split" && haskey(p["pattern"], "Regex")
            # (*UCP): \s, \w etc. match Unicode, as in Hugging Face's regex engine
            pattern = Regex("(*UCP)" * String(p["pattern"]["Regex"]))
        end
    end
    if byte_level && pattern === nothing
        # GPT-2's pattern, for ByteLevel with use_regex
        pattern = r"(*UCP)'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    end
    specials = isempty(special_strs) ? nothing :
        Regex(join(map(t -> "\\Q" * t * "\\E", sort(special_strs; by=length, rev=true)), "|"))

    # Special token ids
    _id(n) = get(vocab, n, -1)
    tcfg_path = joinpath(model_dir, "tokenizer_config.json")
    tcfg = isfile(tcfg_path) ? JSON3.read(read(tcfg_path, String)) : Dict{Symbol,Any}()
    _name(x) = x isa AbstractString ? String(x) : x === nothing ? nothing : String(_json_get(x, "content", ""))
    bos_name = _name(_json_get(tcfg, "bos_token", nothing))
    eos_name = _name(_json_get(tcfg, "eos_token", nothing))
    # an explicit `"bos_token": null` means none (Qwen, whose vocab has "<s>" as text)
    no_bos = haskey(tcfg, :bos_token) && tcfg[:bos_token] === nothing
    bos_id = no_bos ? -1 : bos_name !== nothing && haskey(vocab, bos_name) ? vocab[bos_name] : _id("<s>")
    eos_id = eos_name !== nothing && haskey(vocab, eos_name) ? vocab[eos_name] : _id("</s>")
    unk_id = _id("<unk>")
    pad_id = _id("<pad>")

    eos_ids = eos_id == -1 ? Int[] : [eos_id]
    gcfg_path = joinpath(model_dir, "generation_config.json")
    if isfile(gcfg_path)
        e = _json_get(JSON3.read(read(gcfg_path, String)), "eos_token_id", nothing)
        e isa Integer && push!(eos_ids, e)
        e isa AbstractVector && append!(eos_ids, Int.(e))
    end
    unique!(eos_ids)
    eos_id == -1 && !isempty(eos_ids) && (eos_id = eos_ids[1])

    return LlamaTokenizer(vocab, id_to_token, merges, merge_ranks, bos_id, eos_id, unk_id, pad_id,
                          eos_ids, special_ids, byte_level, pattern,
                          Bool(_json_get(model, "ignore_merges", false)), specials)
end

# ─────────────────────────────────────────────────────────────────────────────
# BPE Logic
# ─────────────────────────────────────────────────────────────────────────────

# Byte-pair encoding of one pre-token: repeatedly merge the adjacent pair with the
# lowest merge rank (leftmost first on ties). Symbols form a linked list and
# candidate pairs sit in a min-heap keyed by (rank, position), so a long input is
# O(n log n) -- a SentencePiece-style tokenizer has no pre-split, and the whole
# text is one "word". Heap entries whose symbols have since changed are skipped.
function _bpe_encode(word::Vector{String}, merge_ranks::Dict{Tuple{String,String},Int})
    n = length(word)
    n <= 1 && return copy(word)
    sym = copy(word)
    nxt = collect(2:n+1); nxt[n] = 0
    prv = collect(0:n-1)
    alive = trues(n)
    heap = Tuple{Int,Int,String,String}[]       # (rank, left position, left, right)
    function push_pair!(i)
        j = nxt[i]
        j == 0 && return
        r = get(merge_ranks, (sym[i], sym[j]), 0)
        r > 0 && _heap_push!(heap, (r, i, sym[i], sym[j]))
    end
    for i in 1:n-1
        push_pair!(i)
    end
    while !isempty(heap)
        r, i, a, b = _heap_pop!(heap)
        (alive[i] && sym[i] == a && nxt[i] != 0 && sym[nxt[i]] == b) || continue   # stale
        j = nxt[i]
        sym[i] = a * b
        alive[j] = false
        nxt[i] = nxt[j]
        nxt[j] != 0 && (prv[nxt[j]] = i)
        prv[i] != 0 && push_pair!(prv[i])
        push_pair!(i)
    end
    out = String[]
    i = 1
    while i != 0
        push!(out, sym[i]); i = nxt[i]
    end
    return out
end

function _heap_push!(h, x)
    push!(h, x); i = length(h)
    while i > 1 && h[i] < h[i >> 1]
        h[i], h[i >> 1] = h[i >> 1], h[i]; i >>= 1
    end
end
function _heap_pop!(h)
    top = h[1]; last = pop!(h)
    if !isempty(h)
        h[1] = last; i = 1; n = length(h)
        while true
            l = 2i; r = l + 1; m = i
            l <= n && h[l] < h[m] && (m = l)
            r <= n && h[r] < h[m] && (m = r)
            m == i && break
            h[i], h[m] = h[m], h[i]; i = m
        end
    end
    return top
end

"""
    encode(tok, text; bos=false, eos=false) -> Vector{Int}

Encode text into token IDs. Byte-level tokenizers also recognize special tokens
written in `text` (e.g. a Llama-3 chat template).
"""
function encode(tok::LlamaTokenizer, text::String; bos::Bool=false, eos::Bool=false)
    ids = Int[]
    bos && tok.bos_id != -1 && push!(ids, tok.bos_id)
    for (i, (seg, special)) in enumerate(_split_specials(tok, text))
        if special
            push!(ids, tok.vocab[seg])
        elseif tok.byte_level
            _encode_byte_level!(ids, tok, seg)
        else
            # SentencePiece prepends "▁" to the text, not to text after a special token
            _encode_sentencepiece!(ids, tok, seg; prefix = i == 1)
        end
    end
    eos && tok.eos_id != -1 && push!(ids, tok.eos_id)
    return ids
end

# `text` as (segment, is special token) pieces: special tokens written in the text
# (e.g. a chat template's "<|eot_id|>" or "</s>") are single tokens, as in HF tokenizers.
function _split_specials(tok::LlamaTokenizer, text::String)
    segments = Tuple{String,Bool}[]
    pos = 1
    if tok.specials !== nothing
        for m in eachmatch(tok.specials, text)
            m.offset > pos && push!(segments, (text[pos:prevind(text, m.offset)], false))
            push!(segments, (m.match, true))
            pos = m.offset + ncodeunits(m.match)
        end
    end
    pos <= ncodeunits(text) && push!(segments, (text[pos:end], false))
    return segments
end

function _encode_byte_level!(ids::Vector{Int}, tok::LlamaTokenizer, text::String)
    for m in eachmatch(tok.pattern, text)
        word = String([BYTE_ENCODER[b] for b in codeunits(m.match)])
        if tok.ignore_merges && haskey(tok.vocab, word)
            push!(ids, tok.vocab[word])
            continue
        end
        for t in _bpe_encode([string(c) for c in word], tok.merge_ranks)
            push!(ids, get(tok.vocab, t, tok.unk_id))
        end
    end
    return ids
end

function _encode_sentencepiece!(ids::Vector{Int}, tok::LlamaTokenizer, text::String; prefix::Bool=true)
    isempty(text) && return ids
    # SentencePiece style: spaces become U+2581, and a leading one is prepended
    processed = replace(text, " " => "\u2581")
    if prefix && !startswith(processed, "\u2581")
        processed = "\u2581" * processed
    end
    bpe_tokens = _bpe_encode([string(c) for c in processed], tok.merge_ranks)
    for t in bpe_tokens
        id = get(tok.vocab, t, -1)
        if id != -1
            push!(ids, id)
        else
            # Byte fallback for unknown tokens: map each UTF-8 byte to its vocab ID
            for b in codeunits(t)
                hex = uppercase(string(b, base=16, pad=2))
                push!(ids, get(tok.vocab, "<0x$hex>", tok.unk_id))
            end
        end
    end
    return ids
end

"""
    decode(tok, ids) -> String

Decode token IDs back to a string, dropping special tokens.
"""
function decode(tok::LlamaTokenizer, ids::AbstractVector{<:Integer})
    tok.byte_level && return _decode_byte_level(tok, ids)
    text = ""
    for id in ids
        token = get(tok.id_to_token, id, "")
        if isempty(token) || id in (tok.bos_id, tok.eos_id, tok.pad_id) || id in tok.eos_ids
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
    return decoded
end

function _decode_byte_level(tok::LlamaTokenizer, ids::AbstractVector{<:Integer})
    bytes = UInt8[]
    for id in ids
        (id in tok.special_ids || id in tok.eos_ids) && continue
        for c in get(tok.id_to_token, id, "")
            push!(bytes, BYTE_DECODER[c])
        end
    end
    return String(bytes)
end

"""
    chat_prompt(tok, message; thinking=false) -> String

`message` as a single user turn in the model's chat format, ending where the
assistant's reply begins: ChatML for Qwen (with an empty think block unless
`thinking`, as Qwen3's template does), Llama-3's header format when the
vocabulary has its header tokens, otherwise the Zephyr format TinyLlama-Chat
uses. The BOS token is not included (`encode(...; bos=true)` and `generate` add
it, for models that have one).
"""
function chat_prompt(tok::LlamaTokenizer, message::AbstractString; thinking::Bool=false)
    if haskey(tok.vocab, "<|im_start|>")
        # ChatML (Qwen). Qwen3 thinks by default; with thinking off its template
        # opens the reply with an empty think block.
        think = haskey(tok.vocab, "<think>") && !thinking ? "<think>\n\n</think>\n\n" : ""
        return "<|im_start|>user\n$message<|im_end|>\n<|im_start|>assistant\n" * think
    end
    if haskey(tok.vocab, "<|start_header_id|>")
        return "<|start_header_id|>user<|end_header_id|>\n\n$(strip(message))<|eot_id|>" *
               "<|start_header_id|>assistant<|end_header_id|>\n\n"
    end
    return "<|user|>\n$message</s>\n<|assistant|>\n"
end

end # module LlamaTokenization
