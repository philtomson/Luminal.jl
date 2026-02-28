
# WhisperTokenizer.jl
#
# Self-contained byte-level BPE tokenizer for Whisper in pure Julia.
# Reads vocab.json + merges.txt from a HuggingFace model directory.
# No Python dependency.
#
# Special token strings are written using Julia unicode escapes, e.g.:
#   SOT = "\u003c\u007cstartoftranscript\u007c\u003e"
# which evaluates identically to the actual token string at runtime.
#
# Usage:
#   tok = WhisperTokenizer("/path/to/openai/whisper-tiny")
#   ids = encode(tok, "Hello!")
#   txt = decode(tok, ids)
#   prefix = sot_sequence(tok; language="en", task=:transcribe)

using JSON3

export WhisperTokenizer, encode, decode, sot_sequence, LANGUAGES

# ─────────────────────────────────────────────────────────────────────────────
# Language table (ISO 639-1 code => name)
# ─────────────────────────────────────────────────────────────────────────────

const LANGUAGES = Dict{String,String}(
    "en" => "english",  "zh" => "chinese",  "de" => "german",
    "es" => "spanish",  "ru" => "russian",  "ko" => "korean",
    "fr" => "french",   "ja" => "japanese", "pt" => "portuguese",
    "tr" => "turkish",  "pl" => "polish",   "nl" => "dutch",
    "ar" => "arabic",   "sv" => "swedish",  "it" => "italian",
    "id" => "indonesian","hi" => "hindi",   "fi" => "finnish",
    "vi" => "vietnamese","he" => "hebrew",  "uk" => "ukrainian",
    "el" => "greek",    "ms" => "malay",    "cs" => "czech",
    "ro" => "romanian", "da" => "danish",   "hu" => "hungarian",
    "ta" => "tamil",    "no" => "norwegian","th" => "thai",
    "hr" => "croatian", "bg" => "bulgarian","sk" => "slovak",
    "lt" => "lithuanian","la" => "latin",   "mi" => "maori",
    "cy" => "welsh",    "fa" => "persian",  "lv" => "latvian",
    "bn" => "bengali",  "sr" => "serbian",  "az" => "azerbaijani",
    "sl" => "slovenian","kn" => "kannada",  "et" => "estonian",
    "mk" => "macedonian","eu"=> "basque",   "is" => "icelandic",
    "hy" => "armenian", "ne" => "nepali",   "mn" => "mongolian",
    "bs" => "bosnian",  "kk" => "kazakh",   "sq" => "albanian",
    "sw" => "swahili",  "gl" => "galician", "mr" => "marathi",
    "haw"=> "hawaiian", "lo" => "lao",      "uz" => "uzbek",
    "tl" => "tagalog",
)

# ─────────────────────────────────────────────────────────────────────────────
# GPT-2 byte encoder (maps raw bytes 0-255 to printable unicode chars)
# This is identical to tiktoken / OpenAI's byte encoding.
# ─────────────────────────────────────────────────────────────────────────────

function _build_byte_encoder()
    bs = collect(Int('!'):Int('~'))
    append!(bs, collect(0x00a1:0x00ac))
    append!(bs, collect(0x00ae:0x00ff))
    cs = copy(bs)
    n = 0
    for b in 0x00:0xff
        if Int(b) ∉ bs
            push!(bs, Int(b))
            push!(cs, 256 + n)
            n += 1
        end
    end
    enc = Dict{UInt8, String}()
    dec = Dict{String, UInt8}()
    for (b, c) in zip(bs, cs)
        s = string(Char(c))
        enc[UInt8(b)] = s
        dec[s] = UInt8(b)
    end
    return enc, dec
end

const _BYTE_ENC, _BYTE_DEC = _build_byte_encoder()

# ─────────────────────────────────────────────────────────────────────────────
# Struct
# ─────────────────────────────────────────────────────────────────────────────

"""
    WhisperTokenizer

Byte-level BPE tokenizer for Whisper loaded from a HuggingFace model directory.
"""
struct WhisperTokenizer
    vocab::Dict{String,Int}                        # token => id (0-indexed)
    id_to_token::Dict{Int,String}                  # id => token
    merges::Vector{Tuple{String,String}}           # ordered merge rules
    merge_ranks::Dict{Tuple{String,String},Int}    # (a,b) => rank
    # Resolved special token IDs (-1 if not present in vocab)
    eot_id::Int
    sot_id::Int
    nospeech_id::Int
    notimestamps_id::Int
    transcribe_id::Int
    translate_id::Int
    language_ids::Dict{String,Int}  # lang code => token id
end

# ─────────────────────────────────────────────────────────────────────────────
# Constructor
# ─────────────────────────────────────────────────────────────────────────────

"""
    WhisperTokenizer(model_dir)

Load vocabulary and merge rules from a HuggingFace Whisper model directory.
Supports `tokenizer.json` (fast tokenizer) or `vocab.json` + `merges.txt`.
"""
function WhisperTokenizer(model_dir::String)
    vocab = Dict{String,Int}()
    merges = Tuple{String,String}[]

    tj = joinpath(model_dir, "tokenizer.json")
    vj = joinpath(model_dir, "vocab.json")
    mt = joinpath(model_dir, "merges.txt")

    if isfile(tj)
        _load_tokenizer_json!(tj, vocab, merges)
    elseif isfile(vj) && isfile(mt)
        _load_vocab_merges!(vj, mt, vocab, merges)
    else
        error("No tokenizer files found in: $model_dir\n" *
              "Expected tokenizer.json or vocab.json + merges.txt")
    end

    id_to_token = Dict(v => k for (k, v) in vocab)
    merge_ranks = Dict(p => i for (i, p) in enumerate(merges))

    # Resolve special token IDs using unicode-escaped strings
    # (avoids angle brackets in source, identical at runtime)
    _id(n) = get(vocab, n, -1)

    eot_id          = _id("\u003c\u007cendoftext\u007c\u003e")
    sot_id          = _id("\u003c\u007cstartoftranscript\u007c\u003e")
    nospeech_id     = _id("\u003c\u007cnospeech\u007c\u003e")
    notimestamps_id = _id("\u003c\u007cnotimestamps\u007c\u003e")
    transcribe_id   = _id("\u003c\u007ctranscribe\u007c\u003e")
    translate_id    = _id("\u003c\u007ctranslate\u007c\u003e")

    lang_ids = Dict{String,Int}()
    for lang_code in keys(LANGUAGES)
        tok = "\u003c\u007c$(lang_code)\u007c\u003e"
        id  = _id(tok)
        id != -1 && (lang_ids[lang_code] = id)
    end

    return WhisperTokenizer(
        vocab, id_to_token, merges, merge_ranks,
        eot_id, sot_id, nospeech_id, notimestamps_id,
        transcribe_id, translate_id, lang_ids)
end

# ─────────────────────────────────────────────────────────────────────────────
# File loaders
# ─────────────────────────────────────────────────────────────────────────────

function _load_tokenizer_json!(path, vocab, merges)
    data = JSON3.read(read(path, String))
    # vocabulary
    raw_vocab = data["model"]["vocab"]
    for (tok, id) in pairs(raw_vocab)
        vocab[String(tok)] = Int(id)
    end
    # merge rules: each entry is a string "a b"
    for entry in data["model"]["merges"]
        parts = split(String(entry), ' ')
        length(parts) == 2 && push!(merges, (parts[1], parts[2]))
    end
    # also add added_tokens (special tokens) to vocab
    if haskey(data, "added_tokens")
        for st in data["added_tokens"]
            vocab[String(st["content"])] = Int(st["id"])
        end
    end
end

function _load_vocab_merges!(vocab_path, merges_path, vocab, merges)
    # vocab.json: {"token": id, ...}
    raw = JSON3.read(read(vocab_path, String))
    for (tok, id) in pairs(raw)
        vocab[String(tok)] = Int(id)
    end
    # merges.txt: first line is a comment (#version: ...), then "a b" per line
    for line in eachline(merges_path)
        startswith(line, '#') && continue
        isempty(strip(line))  && continue
        parts = split(line, ' ')
        length(parts) == 2 && push!(merges, (parts[1], parts[2]))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# BPE core
# ─────────────────────────────────────────────────────────────────────────────

"""
    _bpe_encode(word, merge_ranks) -> Vector{String}

Apply BPE merges to a pre-tokenized word (already byte-encoded).
"""
function _bpe_encode(word::Vector{String}, merge_ranks::Dict{Tuple{String,String},Int})
    # Start with each character as its own token
    symbols = copy(word)
    isempty(symbols) && return symbols

    while length(symbols) > 1
        # Find the highest-priority merge pair
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
        best_rank == typemax(Int) && break   # no more applicable merges

        # Apply the merge
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

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

"""
    encode(tok, text; add_special_tokens=false) -> Vector{Int}

Encode `text` to a sequence of token ids using byte-level BPE.
"""
function encode(tok::WhisperTokenizer, text::String; add_special_tokens::Bool=false)
    ids = Int[]

    # GPT-2 / tiktoken regex pattern for pre-tokenization:
    # Split on word boundaries, punctuation, and spaces.
    # We use a simplified version: split on spaces but keep the space with
    # the following word (prepend space marker), matching tiktoken behavior.
    words = _pretokenize(text)

    for word in words
        # Byte-encode the word
        byte_chars = [_BYTE_ENC[b] for b in codeunits(word)]
        bpe_tokens = _bpe_encode(byte_chars, tok.merge_ranks)
        for t in bpe_tokens
            id = get(tok.vocab, t, -1)
            if id != -1
                push!(ids, id)
            else
                # Unknown token: encode byte-by-byte
                for bc in byte_chars
                    push!(ids, get(tok.vocab, bc, tok.eot_id))
                end
            end
        end
    end

    return ids
end

"""
    decode(tok, ids; skip_special=true) -> String

Decode a sequence of token ids back to a UTF-8 string.
"""
function decode(tok::WhisperTokenizer, ids::Vector{Int}; skip_special::Bool=true)
    # Special token IDs to skip
    special_ids = Set{Int}(filter(x -> x >= 0, [
        tok.eot_id, tok.sot_id, tok.nospeech_id,
        tok.notimestamps_id, tok.transcribe_id, tok.translate_id,
        values(tok.language_ids)...
    ]))

    bytes = UInt8[]
    for id in ids
        skip_special && id in special_ids && continue
        token = get(tok.id_to_token, id, "")
        isempty(token) && continue
        # Timestamp tokens (e.g. <|0.00|>) have no byte representation; skip
        if startswith(token, "\u003c\u007c") && endswith(token, "\u007c\u003e")
            continue
        end
        # Decode each char in the BPE token via byte decoder
        for ch in token
            s = string(ch)
            if haskey(_BYTE_DEC, s)
                push!(bytes, _BYTE_DEC[s])
            end
        end
    end

    return String(bytes)
end

"""
    sot_sequence(tok; language="en", task=:transcribe, notimestamps=true) -> Vector{Int}

Build the start-of-transcript token sequence that Whisper's decoder expects.
Returns `[sot_id, language_id, task_id, notimestamps_id]`.
"""
function sot_sequence(tok::WhisperTokenizer;
                      language::String="en",
                      task::Symbol=:transcribe,
                      notimestamps::Bool=true)
    seq = Int[tok.sot_id]

    lang_id = get(tok.language_ids, language, -1)
    lang_id != -1 && push!(seq, lang_id)

    if task == :transcribe
        tok.transcribe_id != -1 && push!(seq, tok.transcribe_id)
    elseif task == :translate
        tok.translate_id != -1 && push!(seq, tok.translate_id)
    end

    notimestamps && tok.notimestamps_id != -1 && push!(seq, tok.notimestamps_id)
    return seq
end

# ─────────────────────────────────────────────────────────────────────────────
# Pre-tokenizer (simplified GPT-2 / tiktoken splitting)
# ─────────────────────────────────────────────────────────────────────────────

"""
    _pretokenize(text) -> Vector{String}

Split text into words in the same manner as tiktoken's GPT-2 encoding:
each word other than the first is prefixed with a space.
"""
function _pretokenize(text::String)
    # Naive split: split on whitespace, prepend space to each non-first word.
    # This approximates the GPT-2 regex without requiring a regex library.
    words = String[]
    i = firstindex(text)
    buf = IOBuffer()
    in_space = true  # track whether we are at the start or after a space

    for ch in text
        if ch == ' '
            s = String(take!(buf))
            if !isempty(s)
                push!(words, s)
            end
            in_space = true
        else
            if in_space && !isempty(words)
                write(buf, ' ')
            end
            write(buf, ch)
            in_space = false
        end
    end

    s = String(take!(buf))
    !isempty(s) && push!(words, s)
    return words
end
