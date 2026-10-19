# Next-Concept Modeling (NCM, Kumbong et al. 2026, https://github.com/HazyResearch/ncm): text generated coarse to fine.
#
# A multiscale residual VQ tokenizer turns a text into a hierarchy of discrete codes (level k has levels[k] positions,
# each from a codebook), and a transformer predicts the codes of one level, all positions in parallel, from the
# running sum of the levels before it. A sample therefore costs one forward pass per level, not one per token.
#
# This file holds the generator (the transformer and its codebook-tied output heads). One compiled graph per level:
# the prefix token's embedding and the level's input vectors go through the shared stack, the prefix position
# attending only to itself and the level's positions to the prefix and to each other (the repo's
# `_decode_level_logits`). RoPE positions restart at 0 for the level, so the tables are inputs of the graph.
module NCM

using ..Luminal
import ..Luminal: Graph, WeightRegistry, register_weight!, load_weights!, load_weights_to_dict, compile
import ..Luminal.NN: Linear, LayerNorm, _linear, _layernorm, _rmsnorm

using Random
using LinearAlgebra

export NCMConfig, NCMGenerator, level_logits, generate_codes, decode_codes, detokenize, generate_text

"""
    NCMConfig(; n_embd, n_head, n_layer, embed_dim, levels, rope_base, eps)

`levels`: the positions of every level of the tokenizer (`levels[1]` is the seed level, sampled from a unigram
table; the generator predicts levels 2:end). `embed_dim`: the width of the VQ codebook vectors.
"""
Base.@kwdef struct NCMConfig
    n_embd::Int = 768
    n_head::Int = 12
    n_layer::Int = 12
    embed_dim::Int = 384
    vocab::Int = 4096
    levels::Vector{Int} = [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]
    rope_base::Float64 = 100000.0
    eps::Float32 = 1f-6
    resi_ratio::Float32 = 0.5f0              # the tokenizer's quant_resi blend
end

struct NCMBlock
    ln_1::LayerNorm
    c_attn::Linear
    q_norm::LayerNorm
    k_norm::LayerNorm
    c_proj::Linear
    ln_2::LayerNorm
    c_fc::Linear
    mlp_proj::Linear
end

function NCMBlock(cfg::NCMConfig, g::Graph, reg, prefix::String)
    C, hd = cfg.n_embd, cfg.n_embd ÷ cfg.n_head
    return NCMBlock(
        _layernorm(C, g, reg, "$prefix.ln_1"; epsilon=cfg.eps),
        _linear(C, 3C, g, reg, "$prefix.attn.c_attn"),
        _rmsnorm(hd, g, reg, "$prefix.attn.q_norm"; epsilon=cfg.eps),
        _rmsnorm(hd, g, reg, "$prefix.attn.k_norm"; epsilon=cfg.eps),
        _linear(C, C, g, reg, "$prefix.attn.c_proj"),
        _layernorm(C, g, reg, "$prefix.ln_2"; epsilon=cfg.eps),
        _linear(C, 4C, g, reg, "$prefix.mlp.c_fc"),
        _linear(4C, C, g, reg, "$prefix.mlp.c_proj"))
end

# (Hidden, S, B) -> (HeadDim, S, Heads, B)
function _heads(x, n_heads)
    hidden, s, b = Luminal.realized_dims(x.shape)
    return Luminal.permute(Luminal.reshape(x, [hidden ÷ n_heads, n_heads, s, b]), [1, 3, 2, 4])
end

# rotate-half RoPE with explicit cos / sin tables (half_dim, S)
_rope(t, cos, sin) = Luminal.add_op!(t.graph_ref, Luminal.RotaryEmbed(),
    [(t.id, 0, t.shape), (cos.id, 0, cos.shape), (sin.id, 0, sin.shape)],
    Luminal.ShapeTracker(collect(Luminal.realized_dims(t.shape))))

function (b::NCMBlock)(x, cos, sin, mask, n_heads)
    C = Luminal.realized_dims(x.shape)[1]
    qkv = b.c_attn(b.ln_1(x))
    q = _heads(Luminal.slice_along(qkv, 1, 0, C), n_heads)
    k = _heads(Luminal.slice_along(qkv, 1, C, 2C), n_heads)
    v = _heads(Luminal.slice_along(qkv, 1, 2C, 3C), n_heads)
    q = _rope(b.q_norm(q), cos, sin)
    k = _rope(b.k_norm(k), cos, sin)
    d, sq, h, bt = Luminal.realized_dims(q.shape)
    scores = Luminal.matmul(Luminal.permute(k, [2, 1, 3, 4]), q) * (1.0f0 / sqrt(Float32(d)))   # (Sk, Sq, H, B)
    scores = scores + Luminal.expand(Luminal.expand(mask, 3, h), 4, bt)
    out = Luminal.matmul(v, Luminal.softmax1(scores))                                           # (D, Sq, H, B)
    out = Luminal.reshape(Luminal.permute(out, [1, 3, 2, 4]), [d * h, sq, bt])
    x = x + b.c_proj(out)
    return x + b.mlp_proj(Luminal.gelu(b.c_fc(b.ln_2(x)); approximate=false))
end

# The graph of one level: `L` positions, batch `B`.
struct NCMLevelGraph
    exec::Any
    x_id::Int
    prefix_id::Int
    cos_id::Int
    sin_id::Int
    mask_id::Int
    out_id::Int
    L::Int
    B::Int
end

"""
    NCMGenerator(dir; device=get_device(), config=NCMConfig())

The generator of an NCM checkpoint converted by `examples/ncm_convert.py` (`generator.safetensors` and
`tokenizer.safetensors` in `dir`). The output heads are tied to the tokenizer's codebooks: a head's logit for code `e`
is `(2 z·e - |e|²) exp(s) / mean|e|²` for `z = proj(h)`, folded here into one Linear with the codebook as its weight.
"""
mutable struct NCMGenerator
    cfg::NCMConfig
    weights::Dict{String, Any}
    device::Any
    level0_unigram::Vector{Float32}
    embeddings::Vector{Matrix{Float32}}      # the tokenizer's codebooks (vocab, embed_dim)
    graphs::Dict{Tuple{Int,Int}, NCMLevelGraph}
    tk::Dict{String, Any}                    # tokenizer weights (CPU): quantizer convs, encoder.wte, decoder
    vocab::Vector{Vector{UInt8}}             # BPE id -> bytes (empty without bpe_vocab.bin)
end

function NCMGenerator(dir::String; device=Luminal.get_device(), config::NCMConfig=NCMConfig())
    cfg = config
    w = load_weights_to_dict(joinpath(dir, "generator.safetensors"); device=device)
    tk = load_weights_to_dict(joinpath(dir, "tokenizer.safetensors"); device=Luminal.CPUDevice())
    emb = [Array{Float32}(tk["vq.embeddings.$i.weight"]) for i in 0:length(cfg.levels) - 1]
    for k in 1:length(cfg.levels) - 1                      # head k predicts level k + 1 (1-based), tied to its codebook
        E = emb[k + 1]
        logT = only(Array(w["lm_heads.$(k - 1).log_inv_T"]))
        s = exp(logT) / max(sum(abs2, E) / size(E, 1), 1f-12)
        w["head.$k.weight"] = Luminal.to_device(E .* (2f0 * s), device)
        w["head.$k.bias"] = Luminal.to_device(vec(-sum(abs2, E; dims=2) .* s), device)
    end
    return NCMGenerator(cfg, w, device, vec(Array{Float32}(Array(w["level0_unigram"]))), emb,
                        Dict{Tuple{Int,Int}, NCMLevelGraph}(), tk, _load_vocab(joinpath(dir, "bpe_vocab.bin")))
end

# bpe_vocab.bin (examples/ncm_convert.py): int32 n, int32 offsets[n + 1], the token bytes
function _load_vocab(path::String)
    isfile(path) || return Vector{UInt8}[]
    io = IOBuffer(read(path))
    n = Int(read(io, Int32)); off = [Int(read(io, Int32)) for _ in 0:n]
    blob = read(io)
    return [blob[off[i] + 1:off[i + 1]] for i in 1:n]
end

# cos / sin tables (half_dim, 1 + L) for RoPE positions [0, 0 .. L-1]: the prefix and the level each count from 0
function _rope_tables(cfg::NCMConfig, L::Int)
    hd = cfg.n_embd ÷ cfg.n_head; half = hd ÷ 2
    inv = [Float32(cfg.rope_base^(-(i - 1) / half)) for i in 1:half]
    pos = Float32[0; 0:L - 1]
    ang = inv .* pos'                                      # (half, 1 + L)
    return cos.(ang), sin.(ang)
end

# additive mask (Sk, Sq): the prefix position (query 1) sees only itself, everything else sees everything
function _level_mask(L::Int)
    m = zeros(Float32, 1 + L, 1 + L)
    m[2:end, 1] .= -1f9
    return m
end

function _build_level(gen::NCMGenerator, k::Int, B::Int)
    cfg = gen.cfg; L = cfg.levels[k + 1]; C = cfg.n_embd
    g = Graph(); reg = WeightRegistry()
    x_in = Luminal.tensor(g, [cfg.embed_dim, L, B])
    prefix = Luminal.tensor(g, [cfg.embed_dim, 1, B])
    cos_t = Luminal.tensor(g, [C ÷ cfg.n_head ÷ 2, 1 + L])
    sin_t = Luminal.tensor(g, [C ÷ cfg.n_head ÷ 2, 1 + L])
    mask = Luminal.tensor(g, [1 + L, 1 + L])
    wte = _linear(cfg.embed_dim, C, g, reg, "transformer.wte")
    wle = Luminal.tensor(g, [length(cfg.levels) - 1, C]); register_weight!(reg, "transformer.wle.weight", wle)
    blocks = [NCMBlock(cfg, g, reg, "transformer.blocks.$(i - 1)") for i in 1:cfg.n_layer]
    ln_f = _layernorm(C, g, reg, "transformer.ln_f"; epsilon=cfg.eps)
    proj = _linear(C, cfg.embed_dim, g, reg, "lm_heads.$(k - 1).proj"; bias=false)
    head = _linear(cfg.embed_dim, cfg.vocab, g, reg, "head.$k")

    lvl = Luminal.permute(Luminal.slice_along(wle, 1, k - 1, k), [2, 1])           # (C, 1)
    x = wte(x_in) + Luminal.expand(Luminal.expand(lvl, 2, L), 3, B)
    h = Luminal.concat_along(wte(prefix), x, 2)                                    # (C, 1 + L, B)
    for b in blocks
        h = b(h, cos_t, sin_t, mask, cfg.n_head)
    end
    out = head(proj(Luminal.slice_along(ln_f(h), 2, 1, 1 + L)))                    # (vocab, L, B)
    load_weights!(g, reg, gen.weights; device=gen.device)
    exec = compile(g; device=gen.device, retain=[out.id])
    return NCMLevelGraph(exec, x_in.id, prefix.id, cos_t.id, sin_t.id, mask.id, out.id, L, B)
end

"""
    level_logits(gen, k, x, prefix) -> (vocab, L, B)

The logits of level `k` (1-based over the generated levels, so level `k + 1` of the tokenizer) for the level's input
vectors `x` (embed_dim, L, B) and the prefix token's embedding `prefix` (embed_dim, 1, B).
"""
function level_logits(gen::NCMGenerator, k::Int, x::AbstractArray{Float32,3}, prefix::AbstractArray{Float32,3})
    B = size(x, 3)
    lg = get!(() -> _build_level(gen, k, B), gen.graphs, (k, B))
    cos_t, sin_t = _rope_tables(gen.cfg, lg.L)
    ins = Dict{Int,Any}(lg.x_id => x, lg.prefix_id => prefix, lg.cos_id => cos_t, lg.sin_id => sin_t, lg.mask_id => _level_mask(lg.L))
    return Array(lg.exec(ins; device=gen.device)[lg.out_id])
end


# ---------------------------------------------------------------------------------------------------------------
# The tokenizer side, on the host in Float32 (384 channels x 128 positions: a few MFLOP per level). Activations are
# (channels, positions) matrices, convolution weights keep PyTorch's index order (out, in, k) (ConvTranspose: in, out, k).

# ConvTranspose1d with kernel == stride: out[:, (l-1)k + j] = W[:, :, j]' x[:, l] + b
function _convT(x::Matrix{Float32}, W::Array{Float32,3}, b::Vector{Float32})
    k = size(W, 3); out = Matrix{Float32}(undef, size(W, 2), size(x, 2) * k)
    for j in 1:k
        out[:, j:k:end] = transpose(@view W[:, :, j]) * x .+ b
    end
    return out
end

# Conv1d, kernel 3, padding 1
function _conv3(h::Matrix{Float32}, W::Array{Float32,3}, b::Vector{Float32})
    y = (@view W[:, :, 2]) * h .+ b
    y[:, 2:end] .+= (@view W[:, :, 1]) * (@view h[:, 1:end - 1])
    y[:, 1:end - 1] .+= (@view W[:, :, 3]) * (@view h[:, 2:end])
    return y
end

# F.interpolate(mode="linear", align_corners=False)
function _interp_linear(q::Matrix{Float32}, Lout::Int)
    Lin = size(q, 2); out = Matrix{Float32}(undef, size(q, 1), Lout); scale = Lin / Lout
    for i in 0:Lout - 1
        src = max((i + 0.5) * scale - 0.5, 0.0); i0 = floor(Int, src); i1 = min(i0 + 1, Lin - 1); t = Float32(src - i0)
        out[:, i + 1] = (1f0 - t) .* q[:, i0 + 1] .+ t .* q[:, i1 + 1]
    end
    return out
end

# F.interpolate(mode="area") == adaptive average pooling
function _area(z::Matrix{Float32}, Lt::Int)
    L = size(z, 2); out = Matrix{Float32}(undef, size(z, 1), Lt)
    for i in 0:Lt - 1
        s = (i * L) ÷ Lt; e = cld((i + 1) * L, Lt)
        out[:, i + 1] = vec(sum(@view z[:, s + 1:e]; dims=2)) ./ Float32(e - s)
    end
    return out
end

# the codes of tokenizer level `k0` (0-based; `codes` is (L_k0,) 1-based indices into the codebook) as a (D, full) summand
function _level_vectors(gen::NCMGenerator, k0::Int, codes::AbstractVector{Int})
    cfg = gen.cfg; full = cfg.levels[end]; nlev = length(cfg.levels)
    q = Matrix{Float32}(transpose(gen.embeddings[k0 + 1][codes, :]))                  # (D, L_k0)
    if k0 != nlev - 1
        if k0 == 0                                                                     # learned_up_l1: transposed-conv cascade
            for i in 0:3
                q = _convT(q, gen.tk["vq.learned_up_l1.$i.weight"], gen.tk["vq.learned_up_l1.$i.bias"])
            end
        else
            q = _interp_linear(q, full)
        end
    end
    ticks = range(1 / (2nlev), 1 - 1 / (2nlev); length=nlev)                           # PhiNonShared picks the nearest tick
    i = argmin(abs.(ticks .- k0 / max(nlev - 1, 1))) - 1
    r = cfg.resi_ratio
    return q .* (1f0 - r) .+ _conv3(q, gen.tk["vq.quant_resi.$i.weight"], gen.tk["vq.quant_resi.$i.bias"]) .* r
end

# Exact GELU needs erf; Base has none. Taylor series up to |x| = 3.5, then the saturated value.
function _erf(x::Float64)
    a = abs(x)
    a > 5 && return sign(x)
    s = a; term = a; x2 = a * a
    for n in 1:80
        term *= -x2 / n
        t = term / (2n + 1)
        s += t
        abs(t) < 1e-17 && break
    end
    return sign(x) * 2 / sqrt(pi) * s
end
_gelu(x::Float32) = Float32(0.5 * x * (1 + _erf(x / sqrt(2.0))))

function _layernorm_cols(x::Matrix{Float32}, w, b, eps::Float32)
    mu = sum(x; dims=1) ./ size(x, 1)
    v = sum(abs2, x .- mu; dims=1) ./ size(x, 1)
    y = (x .- mu) ./ sqrt.(v .+ eps) .* w
    return b === nothing ? y : y .+ b
end

# The tokenizer's decoder (one transformer block, causal, no positions) and output layer: z (D, L) -> BPE ids (L,)
function _decode_z(gen::NCMGenerator, z::Matrix{Float32})
    t = gen.tk; D, L = size(z); p = "decoder.middle_block."; eps = gen.cfg.eps
    nh = 6; hd = D ÷ nh
    h = _layernorm_cols(z, t[p * "ln_1.weight"], t[p * "ln_1.bias"], eps)
    qkv = t[p * "attn.c_attn.weight"] * h .+ t[p * "attn.c_attn.bias"]
    rms(x, w) = x ./ sqrt.(sum(abs2, x; dims=1) ./ hd .+ eps) .* w
    out = Matrix{Float32}(undef, D, L)
    for hh in 1:nh
        rows(o) = (o * D + (hh - 1) * hd + 1):(o * D + hh * hd)
        q = rms(qkv[rows(0), :], t[p * "attn.q_norm.weight"])
        k = rms(qkv[rows(1), :], t[p * "attn.k_norm.weight"])
        v = qkv[rows(2), :]
        sc = transpose(k) * q ./ sqrt(Float32(hd))                                     # (key, query)
        for qi in 1:L, ki in qi + 1:L
            sc[ki, qi] = -Inf32
        end
        sc .-= maximum(sc; dims=1); sc = exp.(sc); sc ./= sum(sc; dims=1)
        out[(hh - 1) * hd + 1:hh * hd, :] = v * sc
    end
    x = z .+ t[p * "attn.c_proj.weight"] * out .+ t[p * "attn.c_proj.bias"]
    m = _gelu.(t[p * "mlp.c_fc.weight"] * _layernorm_cols(x, t[p * "ln_2.weight"], t[p * "ln_2.bias"], eps) .+ t[p * "mlp.c_fc.bias"])
    x = x .+ t[p * "mlp.c_proj.weight"] * m .+ t[p * "mlp.c_proj.bias"]
    x = _layernorm_cols(x, t["decoder.ln_f.weight"], nothing, eps)
    logits = t["decoder.output_layer.weight"] * x                                      # (vocab, L)
    return [argmax(@view logits[:, i]) - 1 for i in 1:L]                               # 0-based BPE ids
end

"""
    decode_codes(gen, codes) -> Matrix{Int}

The BPE ids (0-based, `levels[end]` x batch) of a sample: the codes of all levels (a vector of `(L_k, B)` matrices, as
`generate_codes` returns) are summed into the final residual vector `z_hat` and run through the tokenizer's decoder.
"""
function decode_codes(gen::NCMGenerator, codes::Vector{<:AbstractMatrix{<:Integer}})
    D = gen.cfg.embed_dim; full = gen.cfg.levels[end]; B = size(codes[1], 2)
    ids = Matrix{Int}(undef, full, B)
    for b in 1:B
        z = zeros(Float32, D, full)
        for k0 in 0:length(codes) - 1
            z .+= _level_vectors(gen, k0, Int.(codes[k0 + 1][:, b]) .+ 1)
        end
        ids[:, b] = _decode_z(gen, z)
    end
    return ids
end

"""
    detokenize(gen, ids; eos=0, pad=1) -> String

Text of BPE ids, cut at the first end-of-text token.
"""
function detokenize(gen::NCMGenerator, ids::AbstractVector{<:Integer}; eos::Int=0, pad::Int=1)
    isempty(gen.vocab) && error("no bpe_vocab.bin next to the checkpoint (run examples/ncm_convert.py)")
    n = something(findfirst(==(eos), ids), length(ids) + 1) - 1
    bytes = UInt8[]
    for t in ids[1:n]
        t == pad || append!(bytes, gen.vocab[t + 1])
    end
    return String(bytes)       # invalid UTF-8 prints as replacement characters, like errors="replace"
end

# top-k / top-p / temperature sampling of one logit vector (the repo's `_sample_level_raw`)
function _sample(lg::AbstractVector{Float32}, temperature::Real, top_k::Int, top_p::Real, rng)
    temperature == 0 && return argmax(lg)
    x = Float32.(lg) ./ Float32(temperature)
    if top_k > 0 && top_k < length(x)
        kth = partialsort(x, top_k; rev=true)
        x[x .< kth] .= -Inf32
    end
    order = sortperm(x; rev=true)
    p = exp.(x[order] .- x[order[1]]); p ./= sum(p)
    if top_p < 1
        c = cumsum(p) .- p
        p[c .> top_p] .= 0
    end
    r = rand(rng) * sum(p); acc = 0f0
    for i in eachindex(p)
        acc += p[i]
        acc >= r && p[i] > 0 && return order[i]
    end
    return order[findlast(>(0), p)]
end

"""
    generate_codes(gen; batch=1, temperature=1, top_k=0, top_p=1, rng, prefix_token=0) -> Vector{Matrix{Int}}

Sample the codes of every level: level 1 from the unigram table, each next level from the generator given the running
`z_hat` (the repo's `NextConceptTransformer.generate`). `prefix_token` is the BPE id fed as the prefix (end-of-text, 0).
Codes are 0-based; entry `k` is `(levels[k], batch)`.
"""
function generate_codes(gen::NCMGenerator; batch::Int=1, temperature::Real=1.0, top_k::Int=0, top_p::Real=1.0,
                        rng=Random.default_rng(), prefix_token::Int=0)
    cfg = gen.cfg; D = cfg.embed_dim; full = cfg.levels[end]; nlev = length(cfg.levels)
    probs = gen.level0_unigram ./ sum(gen.level0_unigram)
    cum = cumsum(probs)
    L0 = cfg.levels[1]
    codes = Matrix{Int}[ [min(searchsortedfirst(cum, rand(rng) * cum[end]), length(cum)) - 1 for _ in 1:L0, _ in 1:batch] ]
    wte = gen.tk["encoder.wte.weight"]
    prefix = Base.reshape(repeat(wte[prefix_token + 1, :], 1, batch), D, 1, batch)
    z_hat = zeros(Float32, D, full, batch)
    for k0 in 0:nlev - 2
        x = Array{Float32,3}(undef, D, cfg.levels[k0 + 2], batch)
        for b in 1:batch
            z_hat[:, :, b] .+= _level_vectors(gen, k0, codes[k0 + 1][:, b] .+ 1)
            x[:, :, b] = _area(z_hat[:, :, b], cfg.levels[k0 + 2])
        end
        lg = level_logits(gen, k0 + 1, x, prefix)                                   # (vocab, L, B)
        push!(codes, [_sample(@view(lg[:, l, b]), temperature, top_k, top_p, rng) - 1
                      for l in 1:size(lg, 2), b in 1:batch])
    end
    return codes
end

"""
    generate_text(gen; batch=1, kwargs...) -> Vector{String}

Sample `batch` texts (`generate_codes`, `decode_codes`, `detokenize`).
"""
function generate_text(gen::NCMGenerator; batch::Int=1, kwargs...)
    ids = decode_codes(gen, generate_codes(gen; batch, kwargs...))
    return [detokenize(gen, ids[:, b]) for b in 1:batch]
end

end # module NCM
