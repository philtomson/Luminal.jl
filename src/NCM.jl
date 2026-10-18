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

export NCMConfig, NCMGenerator, level_logits

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
                        Dict{Tuple{Int,Int}, NCMLevelGraph}())
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

end # module NCM
