# The DFlash2 block drafter (z-lab/Inco AI, GGUF arch "dflash"): one forward pass proposes a whole block of draft tokens. See docs/dflash2_port_plan.md.
#
#   context  : the target's residual stream after layers `target_layers` for the last WIN positions, concatenated (5 x 5120) -> `fc` -> `hidden_norm`
#              -> each draft layer's K (k_norm, RoPE at the absolute position) and V, kept in a ring of WIN rows per layer
#   block    : [seed, MASK x (B-1)] embedded with the target's embedding table, at positions pos .. pos + B - 1
#   layer    : x = x + conv1(wo(attention(conv0(norm(x)))));  x = x + conv1(down(silu(gate(c)) * up(c))), c = conv0(norm(x))
#              (conv0/conv1: the two phases of a dynamic causal convolution over the block rows, coefficients from `*_conv_proj` of the normed input)
#   proposal : the target's lm_head on rows 1..B-1, then the candidate selector over each row's top 16 logits, chained from the seed
#
# Activations are (features, rows) arrays and every matrix is a Luminal weight (Q4Weight / Q6Weight from the file's own blocks), multiplied with
# `batch_matmul!`; the small elementwise steps are array broadcasts and two kernels (the dynamic conv and the block attention).

using KernelAbstractions

const _DF_B = 8                      # block rows: the seed and 7 mask rows
const _DF_WIN = 2048                 # context rows kept (the draft's sliding window)
const _DF_NL = 5
const _DF_H = 5120
const _DF_NH = 32
const _DF_NKV = 8
const _DF_HD = 128
const _DF_KS = 2                     # conv taps
const _DF_GS = 16                    # channels per conv group
const _DF_EPS = 1f-6
const _DF_THETA = 1f7

mutable struct DFlash2
    session::LlamaSession
    device::Any
    fc::Any
    hnorm::Any
    onorm::Any
    layers::Vector{Any}
    selh::Any
    pred::Matrix{Float32}            # (256, vocab) selector predecessor codebook, host
    succ::Matrix{Float32}
    embed::Matrix{Float16}           # the target's token embedding (hidden, vocab), host
    lm_head::Any                     # the target's output matrix
    mask_id::Int
    target_layers::Vector{Int}       # 0-based layers whose output stream is the context
    kring::Vector{Any}               # per layer (WIN, HD, NKV): keys contiguous
    # (the V ring is (HD, WIN, NKV))
    vring::Vector{Any}
    nrows::Int                       # context rows appended so far: rows 0 .. nrows - 1 are in the ring (slot = row % WIN)
    cs_cache::Dict{Int,Any}
    work::Any                        # the graph-captured fast path's buffers and graphs (see `DFWork`)
    prof::Dict{Symbol,Float64}       # section => seconds, filled while `DF_PROFILE[]` is on (the sections synchronize the device)
end
const DF_PROFILE = Ref(false)
function _df_tick!(d::DFlash2, name::Symbol, t0)
    DF_PROFILE[] || return time()
    AMDGPU.synchronize()
    t = time(); d.prof[name] = get(d.prof, name, 0.0) + (t - t0)
    return t
end

_df_vec(f, name, dev) = Luminal.to_device(vec(gguf_tensor(f, name)), dev)
function _df_mat(f, name, dev)
    w = native_weight(f, name, dev)
    w === nothing && error("DFlash2: no native layout for $name")
    return w
end

"""
    DFlash2(session, draft_path, target_path) -> DFlash2

Load the block drafter (the file's Q4_K/Q6_K matrices as they are) next to a target `session`; `target_path` is the target's GGUF (token embedding).
"""
function DFlash2(s::LlamaSession, draft_path::String, target_path::String)
    dev = s.device
    f = read_gguf(draft_path)
    f.metadata["general.architecture"] == "dflash" || error("$draft_path is not a DFlash GGUF")
    layers = map(0:_DF_NL-1) do i
        b = "blk.$i"
        (an = _df_vec(f, "$b.attn_norm.weight", dev), fn = _df_vec(f, "$b.ffn_norm.weight", dev),
         qn = _df_vec(f, "$b.attn_q_norm.weight", dev), kn = _df_vec(f, "$b.attn_k_norm.weight", dev),
         wq = _df_mat(f, "$b.attn_q.weight", dev), wk = _df_mat(f, "$b.attn_k.weight", dev), wv = _df_mat(f, "$b.attn_v.weight", dev),
         wo = _df_mat(f, "$b.attn_output.weight", dev),
         acp = _df_mat(f, "$b.attn_conv_proj.weight", dev), acb = Luminal.to_device(gguf_tensor(f, "$b.attn_conv_base"), dev),
         fcp = _df_mat(f, "$b.ffn_conv_proj.weight", dev), fcb = Luminal.to_device(gguf_tensor(f, "$b.ffn_conv_base"), dev),
         wg = _df_mat(f, "$b.ffn_gate.weight", dev), wu = _df_mat(f, "$b.ffn_up.weight", dev), wd = _df_mat(f, "$b.ffn_down.weight", dev))
    end
    tl = Int.(f.metadata["dflash.target_layers"])
    length(tl) == 5 || error("expected 5 target layers")
    tf = read_gguf(target_path)
    embed = Float16.(gguf_tensor(tf, "token_embd.weight"))
    kring = [Luminal.to_device(zeros(Float32, _DF_WIN, _DF_HD, _DF_NKV), dev) for _ in 1:_DF_NL]
    vring = [Luminal.to_device(zeros(Float32, _DF_HD, _DF_WIN, _DF_NKV), dev) for _ in 1:_DF_NL]
    lm = get(s.weights, "lm_head.weight", nothing)
    lm === nothing && error("the session has no lm_head.weight")
    return DFlash2(s, dev, _df_mat(f, "fc.weight", dev), _df_vec(f, "enc.output_norm.weight", dev), _df_vec(f, "output_norm.weight", dev), collect(Any, layers),
                   _df_mat(f, "selector_hidden.weight", dev), gguf_tensor(f, "selector_predecessor.weight"), gguf_tensor(f, "selector_successor.weight"),
                   embed, lm, Int(f.metadata["tokenizer.ggml.mask_token_id"]), tl, kring, vring, 0, Dict{Int,Any}(), nothing, Dict{Symbol,Float64}())
end

# ── small pieces ──────────────────────────────────────────────────────────────

_df_rms(x, w) = x .* (1f0 ./ sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ _DF_EPS)) .* w            # x (features, rows), per column
_df_headnorm(x, w) = x .* (1f0 ./ sqrt.(sum(abs2, x; dims = 1) ./ _DF_HD .+ _DF_EPS)) .* Base.reshape(w, _DF_HD, 1, 1)   # x (HD, heads, rows)
_df_silu(x) = x ./ (1f0 .+ exp.(-x))
_df_mm(W, X) = (C = similar(X, Float32, size(W, 1), size(X, 2)); Luminal.batch_matmul!(C, W, X); C)

# rotate-half RoPE (all HD dims) of x (HD, heads, rows) at `pos[r]`
@kernel function _df_rope_kernel!(out, @Const(x), @Const(pos), nh, theta)
    i = @index(Global, Linear)
    d = (i - 1) % _DF_HD
    r = (i - 1) ÷ (_DF_HD * nh)
    half = _DF_HD ÷ 2
    j = d % half
    ang = pos[r + 1] * theta^(-2f0 * j / _DF_HD)
    c = cos(ang); s = sin(ang)
    @inbounds out[i] = d < half ? x[i] * c - x[i + half] * s : x[i] * c + x[i - half] * s
end
function _df_rope(x, pos)
    out = similar(x)
    _df_rope_kernel!(KernelAbstractions.get_backend(x), 256)(out, x, pos, size(x, 2), _DF_THETA; ndrange = length(x))
    return out
end

# out[c, r] = (base[c, 0, phase] + dyn[phase * KS * G + g(c), r]) * x[c, r] + (r > 1) * (base[c, 1, phase] + dyn[... + G + g(c), r]) * x[c, r - 1] (+ res)
@kernel function _df_conv_kernel!(out, @Const(x), @Const(base), @Const(dyn), @Const(res), phase, ::Val{RES}) where {RES}
    i = @index(Global, Linear)
    c = (i - 1) % _DF_H + 1
    r = (i - 1) ÷ _DF_H + 1
    ng = _DF_H ÷ _DF_GS
    g = (c - 1) ÷ _DF_GS + 1
    o = phase * _DF_KS * ng
    @inbounds begin
        acc = (base[c, 1, phase + 1] + dyn[o + g, r]) * x[c, r]
        r > 1 && (acc += (base[c, 2, phase + 1] + dyn[o + ng + g, r]) * x[c, r - 1])
        out[c, r] = RES ? acc + res[c, r] : acc
    end
end
function _df_conv(x, base, dyn, phase; res = nothing)
    out = similar(x)
    _df_conv_kernel!(KernelAbstractions.get_backend(x), 256)(out, x, base, dyn, res === nothing ? x : res, phase, Val(res !== nothing); ndrange = length(x))
    return out
end

# Attention of the block rows over the context ring and the block itself, not causal. q (HD, NH, R); K ring (WIN, HD, NKV), V ring (HD, WIN, NKV), the first `nctx` slots
# valid; block keys/values kb, vb (HD, NKV, R). One workgroup of HD threads per (head, row).
@kernel function _df_attn_kernel!(out, @Const(q), @Const(kr), @Const(vr), @Const(kb), @Const(vb), @Const(meta), nrows, scale)
    t = @index(Local, Linear)
    grp = @index(Group, Linear) - 1
    h = grp % _DF_NH + 1
    r = grp ÷ _DF_NH + 1
    g = (h - 1) ÷ (_DF_NH ÷ _DF_NKV) + 1
    qs = @localmem Float32 (_DF_HD,)
    sc = @localmem Float32 (_DF_WIN + _DF_B,)
    red = @localmem Float32 (_DF_HD,)
    @inbounds qs[t] = q[t, h, r]
    @synchronize
    nctx = Int(meta[1])
    nk = nctx + nrows
    m = -Inf32
    j = t
    while j <= nk
        s = 0f0
        if j <= nctx
            for d in 1:_DF_HD
                s += qs[d] * kr[j, d, g]
            end
        else
            for d in 1:_DF_HD
                s += qs[d] * kb[d, g, j - nctx]
            end
        end
        s *= scale
        @inbounds sc[j] = s
        m = max(m, s)
        j += _DF_HD
    end
    @inbounds red[t] = m
    @synchronize
    stride = _DF_HD ÷ 2
    while stride > 0
        if t <= stride
            @inbounds red[t] = max(red[t], red[t + stride])
        end
        @synchronize
        stride ÷= 2
    end
    @inbounds mx = red[1]
    @synchronize
    ssum = 0f0
    j = t
    while j <= nk
        @inbounds e = exp(sc[j] - mx)
        @inbounds sc[j] = e
        ssum += e
        j += _DF_HD
    end
    @inbounds red[t] = ssum
    @synchronize
    stride = _DF_HD ÷ 2
    while stride > 0
        if t <= stride
            @inbounds red[t] += red[t + stride]
        end
        @synchronize
        stride ÷= 2
    end
    @inbounds den = red[1]
    acc = 0f0
    @inbounds for j in 1:nctx
        acc += sc[j] * vr[t, j, g]
    end
    @inbounds for j in 1:nrows
        acc += sc[nctx + j] * vb[t, g, j]
    end
    @inbounds out[t, h, r] = acc / den
end

# ── the context ring ──────────────────────────────────────────────────────────

"""
    dflash_append!(d, hs, pos0)

Append the context rows for the tokens at absolute positions pos0 .. pos0 + m - 1: `hs` are the target's residual streams after `d.target_layers`
(a vector of five (hidden, m) or (hidden, m, 1) arrays). Rows must arrive in order (`pos0 == d.nrows`).
"""
function dflash_append!(d::DFlash2, hs, pos0::Int)
    pos0 == d.nrows || error("DFlash2 context rows must be appended in order (got $pos0, have $(d.nrows))")
    m = size(hs[1], 2)
    X = similar(hs[1], Float32, 5 * _DF_H, m)
    for (j, h) in enumerate(hs)
        copyto!(view(X, (j - 1) * _DF_H + 1:j * _DF_H, :), Base.reshape(h, _DF_H, m))
    end
    P = _df_rms(_df_mm(d.fc, X), Base.reshape(d.hnorm, :, 1))                       # (H, m)
    pos = Luminal.to_device(Float32.(pos0:pos0+m-1), d.device)
    slots = Luminal.to_device(Int32.([mod(p, _DF_WIN) + 1 for p in pos0:pos0+m-1]), d.device)
    for (li, l) in enumerate(d.layers)
        k = Base.reshape(_df_mm(l.wk, P), _DF_HD, _DF_NKV, m)
        k = _df_rope(_df_headnorm(k, l.kn), pos)
        v = Base.reshape(_df_mm(l.wv, P), _DF_HD, _DF_NKV, m)
        _df_store!(d.kring[li], k, slots, true); _df_store!(d.vring[li], v, slots, false)
    end
    d.nrows = pos0 + m
    return d
end
@kernel function _df_store_kernel!(ring, @Const(x), @Const(slots), ::Val{KEYS_FIRST}) where {KEYS_FIRST}
    i = @index(Global, Linear)
    d = (i - 1) % _DF_HD + 1
    g = ((i - 1) ÷ _DF_HD) % _DF_NKV + 1
    r = (i - 1) ÷ (_DF_HD * _DF_NKV) + 1
    @inbounds KEYS_FIRST ? (ring[slots[r], d, g] = x[d, g, r]) : (ring[d, slots[r], g] = x[d, g, r])
end
_df_store!(ring, x, slots, keys_first::Bool) = _df_store_kernel!(KernelAbstractions.get_backend(ring), 256)(ring, x, slots, Val(keys_first); ndrange = length(x))

# the 16 largest entries of v (indices, in no particular order), in one pass
function _df_top16(v::AbstractVector{Float32})
    ids = collect(1:16); vals = Float32[v[i] for i in 1:16]
    thr, at = findmin(vals)
    @inbounds for i in 17:length(v)
        x = v[i]
        if x > thr
            vals[at] = x; ids[at] = i
            thr, at = findmin(vals)
        end
    end
    return ids
end

# ── the block pass and the selector ───────────────────────────────────────────

"""
    dflash_draft!(d, seed, pos; nprop=7) -> Vector{Int}

The draft tokens (0-based ids, `nprop` <= 7) for the positions pos + 1 ... after `seed`, the token at position `pos` (not yet in the context:
the ring holds rows 0 .. pos - 1).
"""
function dflash_draft_ref!(d::DFlash2, seed::Int, pos::Int; nprop::Int = _DF_B - 1)
    pos == d.nrows || error("DFlash2: seed position $pos but the context has $(d.nrows) rows")
    B = _DF_B
    dev = d.device
    tt = time()
    emb = Float32.(d.embed[:, [seed + 1; fill(d.mask_id + 1, B - 1)]])
    X = Luminal.to_device(emb, dev)                                                            # (H, B)
    qpos = Luminal.to_device(Float32.(pos:pos+B-1), dev)
    nctx = min(pos, _DF_WIN)
    scale = 1f0 / sqrt(Float32(_DF_HD))
    tt = _df_tick!(d, :embed, tt)
    for (li, l) in enumerate(d.layers)
        xn = _df_rms(X, Base.reshape(l.an, :, 1))
        dyn = _df_mm(l.acp, xn)                                                        # (1280, B)
        c = _df_conv(xn, l.acb, dyn, 0)
        tt = _df_tick!(d, :attn_in, tt)
        q = _df_rope(_df_headnorm(Base.reshape(_df_mm(l.wq, c), _DF_HD, _DF_NH, B), l.qn), qpos)
        k = _df_rope(_df_headnorm(Base.reshape(_df_mm(l.wk, c), _DF_HD, _DF_NKV, B), l.kn), qpos)
        v = Base.reshape(_df_mm(l.wv, c), _DF_HD, _DF_NKV, B)
        tt = _df_tick!(d, :qkv, tt)
        att = similar(q)
        _df_attn_kernel!(KernelAbstractions.get_backend(q), _DF_HD)(att, q, d.kring[li], d.vring[li], k, v, Luminal.to_device(Int32[nctx], d.device), B, scale; ndrange = _DF_HD * _DF_NH * B)
        tt = _df_tick!(d, :attn_kernel, tt)
        ao = _df_mm(l.wo, Base.reshape(att, _DF_HD * _DF_NH, B))
        X = _df_conv(ao, l.acb, dyn, 1; res = X)
        tt = _df_tick!(d, :attn_out, tt)
        xn = _df_rms(X, Base.reshape(l.fn, :, 1))
        dyn = _df_mm(l.fcp, xn)
        c = _df_conv(xn, l.fcb, dyn, 0)
        gu = _df_silu(_df_mm(l.wg, c)) .* _df_mm(l.wu, c)
        tt = _df_tick!(d, :ffn_gate_up, tt)
        ff = _df_mm(l.wd, gu)
        X = _df_conv(ff, l.fcb, dyn, 1; res = X)
        tt = _df_tick!(d, :ffn_down, tt)
    end
    Hd = _df_rms(X, Base.reshape(d.onorm, :, 1))[:, 2:B]                               # rows 1 .. B - 1, (H, 7)
    Hd = copy(Hd)
    tt = _df_tick!(d, :final_norm, tt)
    lg = _df_mm(d.lm_head, Hd)
    tt = _df_tick!(d, :lm_head, tt)
    logits = Array(lg)                                                                 # (vocab, 7)
    proj = Array(_df_mm(d.selh, Hd))                                                   # (256, 7)
    tt = _df_tick!(d, :download, tt)
    prev = seed
    out = Int[]
    for r in 1:nprop
        top = _df_top16(view(logits, :, r))
        p = view(d.pred, :, prev + 1) .* view(proj, :, r)
        best = 0; bs = -Inf32
        for j in top
            s = logits[j, r] + sum(p .* view(d.succ, :, j))
            s > bs && (bs = s; best = j)
        end
        prev = best - 1
        push!(out, prev)
    end
    DF_PROFILE[] && (d.prof[:selector] = get(d.prof, :selector, 0.0) + (time() - tt))
    return out
end


# ── the graph-captured fast path ──────────────────────────────────────────────
# The same arithmetic as `dflash_draft_ref!` / `dflash_append!`, as a fixed sequence of kernels on persistent buffers whose per-call inputs (the block
# embeddings, the positions, the context length) sit in device arrays: it is captured once in a HIP graph and replayed, which takes the launch and allocation
# overhead of ~300 small operations out of every cycle.

mutable struct DFWork
    x0; qpos; meta; apos                         # inputs: block embeddings (H, B), the block's positions, [nctx, m, pos0], the appended rows' positions
    X; Xo; xn; dyn; c; qraw; kraw; vraw; q; k; att; ao; g; u; ff; Hd; logits; proj      # the block pass
    xin; P; kp; vp; kn                           # the ring append (8 columns; the first m are stored)
    hlogits::Matrix{Float32}; hproj::Matrix{Float32}
    graphs::Dict{Symbol,Any}
end

@kernel function _df_rmsw_kernel!(out, @Const(x), @Const(w), ::Val{H}) where {H}
    t = @index(Local, Linear)
    col = @index(Group, Linear)
    red = @localmem Float32 (256,)
    ss = 0f0
    i = t
    while i <= H
        @inbounds v = x[i, col]
        ss += v * v
        i += 256
    end
    @inbounds red[t] = ss
    @synchronize
    stride = 128
    while stride > 0
        if t <= stride
            @inbounds red[t] += red[t + stride]
        end
        @synchronize
        stride ÷= 2
    end
    @inbounds inv = 1f0 / sqrt(red[1] / H + _DF_EPS)
    i = t
    while i <= H
        @inbounds out[i, col] = x[i, col] * inv * w[i]
        i += 256
    end
end
_df_rmsw!(out, x, w) = (_df_rmsw_kernel!(KernelAbstractions.get_backend(x), 256)(out, x, w, Val(size(x, 1)); ndrange = 256 * size(x, 2)); out)

# per (head, row): RMSNorm over the head's 128 values (weight w), then rotate-half RoPE at pos[row]; x and out are (HD, heads, rows)
@kernel function _df_normrope_kernel!(out, @Const(x), @Const(w), @Const(pos), nh, theta)
    t = @index(Local, Linear)
    grp = @index(Group, Linear) - 1
    h = grp % nh + 1
    r = grp ÷ nh + 1
    red = @localmem Float32 (_DF_HD,)
    nv = @localmem Float32 (_DF_HD,)
    @inbounds v = x[t, h, r]
    @inbounds red[t] = v * v
    @synchronize
    stride = _DF_HD ÷ 2
    while stride > 0
        if t <= stride
            @inbounds red[t] += red[t + stride]
        end
        @synchronize
        stride ÷= 2
    end
    @inbounds inv = 1f0 / sqrt(red[1] / _DF_HD + _DF_EPS)
    @inbounds nv[t] = v * inv * w[t]
    @synchronize
    half = _DF_HD ÷ 2
    j = (t - 1) % half
    ang = pos[r] * theta^(-2f0 * j / _DF_HD)
    c = cos(ang); s = sin(ang)
    @inbounds out[t, h, r] = t <= half ? nv[t] * c - nv[t + half] * s : nv[t] * c + nv[t - half] * s
end
_df_normrope!(out, x, w, pos, nh) = (_df_normrope_kernel!(KernelAbstractions.get_backend(x), _DF_HD)(out, x, w, pos, nh, _DF_THETA; ndrange = _DF_HD * nh * size(x, 3)); out)

# the conv with its result written to `out` (and, with `res`, added to it)
@kernel function _df_conv2_kernel!(out, @Const(x), @Const(base), @Const(dyn), @Const(res), phase, ::Val{RES}) where {RES}
    i = @index(Global, Linear)
    c = (i - 1) % _DF_H + 1
    r = (i - 1) ÷ _DF_H + 1
    ng = _DF_H ÷ _DF_GS
    g = (c - 1) ÷ _DF_GS + 1
    o = phase * _DF_KS * ng
    @inbounds begin
        acc = (base[c, 1, phase + 1] + dyn[o + g, r]) * x[c, r]
        r > 1 && (acc += (base[c, 2, phase + 1] + dyn[o + ng + g, r]) * x[c, r - 1])
        out[c, r] = RES ? acc + res[c, r] : acc
    end
end
function _df_conv2!(out, x, base, dyn, phase, res)
    _df_conv2_kernel!(KernelAbstractions.get_backend(x), 256)(out, x, base, dyn, res === nothing ? x : res, phase, Val(res !== nothing); ndrange = length(x))
    return out
end

# ring[slot(row)] = x[:, :, r] for the first meta[2] columns; the row's position is meta[3] + r - 1
@kernel function _df_store2_kernel!(ring, @Const(x), @Const(meta), ::Val{KEYS_FIRST}) where {KEYS_FIRST}
    i = @index(Global, Linear)
    d = (i - 1) % _DF_HD + 1
    g = ((i - 1) ÷ _DF_HD) % _DF_NKV + 1
    r = (i - 1) ÷ (_DF_HD * _DF_NKV) + 1
    if r <= meta[2]
        slot = (meta[3] + r - 1) % _DF_WIN + 1
        @inbounds KEYS_FIRST ? (ring[slot, d, g] = x[d, g, r]) : (ring[d, slot, g] = x[d, g, r])
    end
end

function _df_work(d::DFlash2)
    w = d.work
    w === nothing || return w
    dev = d.device; B = _DF_B
    z(dims...) = Luminal.to_device(zeros(Float32, dims...), dev)
    V = size(d.embed, 2)
    d.work = DFWork(z(_DF_H, B), z(B), Luminal.to_device(zeros(Int32, 4), dev), z(B),
                    z(_DF_H, B), z(_DF_H, B), z(_DF_H, B), z(_DF_H ÷ 4, B), z(_DF_H, B), z(_DF_NH * _DF_HD, B), z(_DF_NKV * _DF_HD, B), z(_DF_NKV * _DF_HD, B),
                    z(_DF_HD, _DF_NH, B), z(_DF_HD, _DF_NKV, B), z(_DF_HD, _DF_NH, B), z(_DF_H, B),
                    z(17408, B), z(17408, B), z(_DF_H, B), z(_DF_H, B - 1), z(V, B - 1), z(256, B - 1),
                    z(5 * _DF_H, B), z(_DF_H, B), z(_DF_NKV * _DF_HD, B), z(_DF_NKV * _DF_HD, B), z(_DF_HD, _DF_NKV, B),
                    zeros(Float32, V, B - 1), zeros(Float32, 256, B - 1), Dict{Symbol,Any}())
    return d.work
end

_df_mm!(C, W, X) = (Luminal.batch_matmul!(C, W, X); C)

function _df_draft_body!(d::DFlash2, w::DFWork)
    B = _DF_B
    scale = 1f0 / sqrt(Float32(_DF_HD))
    w.X .= w.x0      # (a broadcast: copyto! on device arrays is hipMemcpyWithStream in current AMDGPU.jl, which cannot be captured)
    for (li, l) in enumerate(d.layers)
        _df_rmsw!(w.xn, w.X, l.an)
        _df_mm!(w.dyn, l.acp, w.xn)
        _df_conv2!(w.c, w.xn, l.acb, w.dyn, 0, nothing)
        _df_mm!(w.qraw, l.wq, w.c); _df_mm!(w.kraw, l.wk, w.c); _df_mm!(w.vraw, l.wv, w.c)
        _df_normrope!(w.q, Base.reshape(w.qraw, _DF_HD, _DF_NH, B), l.qn, w.qpos, _DF_NH)
        _df_normrope!(w.k, Base.reshape(w.kraw, _DF_HD, _DF_NKV, B), l.kn, w.qpos, _DF_NKV)
        _df_attn_kernel!(KernelAbstractions.get_backend(w.q), _DF_HD)(w.att, w.q, d.kring[li], d.vring[li], w.k, Base.reshape(w.vraw, _DF_HD, _DF_NKV, B), w.meta, B, scale;
                                                                      ndrange = _DF_HD * _DF_NH * B)
        _df_mm!(w.ao, l.wo, Base.reshape(w.att, _DF_HD * _DF_NH, B))
        _df_conv2!(w.X, w.ao, l.acb, w.dyn, 1, w.X)
        _df_rmsw!(w.xn, w.X, l.fn)
        _df_mm!(w.dyn, l.fcp, w.xn)
        _df_conv2!(w.c, w.xn, l.fcb, w.dyn, 0, nothing)
        _df_mm!(w.g, l.wg, w.c); _df_mm!(w.u, l.wu, w.c)
        @. w.g = w.g / (1f0 + exp(-w.g)) * w.u
        _df_mm!(w.ff, l.wd, w.g)
        _df_conv2!(w.X, w.ff, l.fcb, w.dyn, 1, w.X)
    end
    _df_rmsw!(w.Xo, w.X, d.onorm)
    w.Hd .= view(w.Xo, :, 2:B)
    _df_mm!(w.logits, d.lm_head, w.Hd)
    _df_mm!(w.proj, d.selh, w.Hd)
    return nothing
end

function _df_append_body!(d::DFlash2, w::DFWork)
    B = _DF_B
    _df_mm!(w.xn, d.fc, w.xin)
    _df_rmsw!(w.P, w.xn, d.hnorm)
    for (li, l) in enumerate(d.layers)
        _df_mm!(w.kp, l.wk, w.P); _df_mm!(w.vp, l.wv, w.P)
        _df_normrope!(w.kn, Base.reshape(w.kp, _DF_HD, _DF_NKV, B), l.kn, w.apos, _DF_NKV)
        n = _DF_HD * _DF_NKV * B
        _df_store2_kernel!(KernelAbstractions.get_backend(w.kn), 256)(d.kring[li], w.kn, w.meta, Val(true); ndrange = n)
        _df_store2_kernel!(KernelAbstractions.get_backend(w.kn), 256)(d.vring[li], Base.reshape(w.vp, _DF_HD, _DF_NKV, B), w.meta, Val(false); ndrange = n)
    end
    return nothing
end

# run `body!` as a captured graph: the first call per key runs it directly (compiling kernels, filling the library's workspaces), the second records it
function _df_run!(w::DFWork, key::Symbol, body!)
    g = get(w.graphs, key, nothing)
    if g === nothing
        if haskey(w.graphs, Symbol(key, :_warm))
            g = Luminal.Device._capture_hip_graph(body!)
            w.graphs[key] = g
            Luminal.Device._launch_hip_graph(g)
        else
            w.graphs[Symbol(key, :_warm)] = true
            body!()
        end
    else
        Luminal.Device._launch_hip_graph(g)
    end
    return nothing
end

"""
    dflash_draft!(d, seed, pos; nprop=7) -> Vector{Int}

`dflash_draft_ref!` through the captured graph.
"""
function dflash_draft!(d::DFlash2, seed::Int, pos::Int; nprop::Int = _DF_B - 1)
    pos == d.nrows || error("DFlash2: seed position $pos but the context has $(d.nrows) rows")
    B = _DF_B
    w = _df_work(d)
    copyto!(w.x0, Float32.(d.embed[:, [seed + 1; fill(d.mask_id + 1, B - 1)]]))
    copyto!(w.qpos, Float32.(pos:pos+B-1))
    copyto!(w.meta, Int32[min(pos, _DF_WIN), 0, 0, 0])
    _df_run!(w, :draft, () -> _df_draft_body!(d, w))
    copyto!(w.hlogits, w.logits); copyto!(w.hproj, w.proj)
    logits = w.hlogits; proj = w.hproj
    prev = seed
    out = Int[]
    for r in 1:nprop
        top = _df_top16(view(logits, :, r))
        p = view(d.pred, :, prev + 1) .* view(proj, :, r)
        best = 0; bs = -Inf32
        for j in top
            s = logits[j, r] + sum(p .* view(d.succ, :, j))
            s > bs && (bs = s; best = j)
        end
        prev = best - 1
        push!(out, prev)
    end
    return out
end

"""
    dflash_append_rows!(d, hs, pos0)

`dflash_append!` for 1 to 8 rows through the captured graph; `hs` are the five (hidden, m, ...) layer outputs (columns 1:m are used).
"""
function dflash_append_rows!(d::DFlash2, hs, pos0::Int)
    pos0 == d.nrows || error("DFlash2 context rows must be appended in order (got $pos0, have $(d.nrows))")
    m = size(hs[1], 2)
    1 <= m <= _DF_B || error("dflash_append_rows! takes 1 to $(_DF_B) rows")
    w = _df_work(d)
    for (j, h) in enumerate(hs)
        copyto!(view(w.xin, (j - 1) * _DF_H + 1:j * _DF_H, 1:m), Base.reshape(h, _DF_H, m))
    end
    copyto!(w.apos, Float32.(pos0:pos0+_DF_B-1))
    copyto!(w.meta, Int32[0, m, pos0, 0])
    _df_run!(w, :append, () -> _df_append_body!(d, w))
    d.nrows = pos0 + m
    return d
end
