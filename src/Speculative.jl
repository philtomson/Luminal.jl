# Speculative decoding with the model's own MTP head (Qwen3.5 / Qwen3.6): the head drafts one token,
# and the base model verifies it together with the token before it in a single 2-token step, so the
# weights are read once for two positions. Greedy: the output is the same as plain greedy decoding
# (up to the numerical difference between a 2-column and a 1-column matmul).
#
# One cycle (P = base positions fed so far, y = the last emitted token, not yet fed; d = the draft for
# the token after y; the MTP cache holds positions 0..P-1):
#   verify [y, d] at P, P + 1  ->  a = argmax(after y), b = argmax(after d), hidden states H_P, H_P+1
#   a == d : emit a, b; P += 2; MTP gets (a, H_P) at P, then (b, H_P+1) at P + 1, whose logits draft next
#   a != d : emit a;    P += 1; recurrent states roll back to their snapshots after y; MTP gets (a, H_P) at P
# The MTP layer pairs each token with the base hidden state *before* it, like the (x[i+1], H[i]) pairs
# its prompt prefill uses.

"""
    MTPDrafter(session)

The MTP head of `session` (its weight dict must hold the `mtp.*` weights, see `load_qwen35_mtp`) with its
own KV cache and compiled graphs. Create one per session; `generate_ids_speculative` drives it.
"""
mutable struct MTPDrafter
    session::LlamaSession
    k::Any                    # (head_dim, max_seq, kv_heads, 1) MTP K cache
    v::Any
    pos::Int                  # next MTP position
    steps::Dict{Int,Any}      # pairs per step => compiled draft step, built on first use
    prefill::Dict{Int,Any}    # padded length => compiled prefill
    draft_vocab::Int          # 0 = full vocabulary, else the head scores only token ids 0..draft_vocab-1
end

function MTPDrafter(s::LlamaSession; draft_vocab::Int=0)
    haskey(s.weights, "mtp.eh_proj.weight") || error("the session's weights have no MTP head (merge load_qwen35_mtp's result)")
    c = s.model.config
    shape = (c.head_dim, s.max_seq, c.n_kv_heads, 1)
    MTPDrafter(s, Luminal.zero_tensor(s.device, s.kv_dtype, shape...), Luminal.zero_tensor(s.device, s.kv_dtype, shape...),
               0, Dict{Int,Any}(), Dict{Int,Any}(), draft_vocab)
end

function _weight_dtype_for(s::LlamaSession, reg)
    wd = s.decode_weights
    wd isa Type && return wd
    by_id = Dict{Int, Type}(id => wd(name) for (name, id) in reg.mapping)
    return id -> get(by_id, id, Float32)
end

# The first n output rows (vocabulary entries) of a matmul weight, as the same kind of weight.
_first_rows(w::AbstractMatrix, n::Int) = w[1:n, :]
function _first_rows(w::Luminal.Q4Weight, n::Int)
    p = w.p[:, 1:n]
    v = Base.reshape(reinterpret(NTuple{16, UInt8}, vec(p)), size(p, 1) ÷ 16, n)
    Luminal.Q4Weight(p, v, w.scale[:, 1:n], w.mn === nothing ? nothing : w.mn[:, 1:n], w.group)
end
function _first_rows(w::Luminal.QuantWeight, n::Int)
    q = w.q[:, 1:n]
    v = Base.reshape(reinterpret(NTuple{16, Int8}, vec(q)), size(q, 1) ÷ 16, n)
    Luminal.QuantWeight(q, v, w.scale[:, 1:n], w.group)
end

# The draft step takes TWO (token, hidden) pairs at consecutive positions and returns both greedy tokens
# (device-side argmax): after an accepted draft both pairs are real; otherwise the second is a dummy whose
# K/V slot the next step overwrites (attention is causal, so it cannot affect the first).
function _mtp_step_graph(d::MTPDrafter, np::Int=2)
    haskey(d.steps, np) && return d.steps[np]
    s = d.session; c = s.model.config
    g = Base.invokelatest(Graph); reg = WeightRegistry()
    m = Qwen35MTP(g, reg; c..., draft_vocab=d.draft_vocab)
    tok = Luminal.tensor(g, [np, 1]); hid = Luminal.tensor(g, [c.hidden, np, 1]); pos = Luminal.tensor(g, [1])
    pk = Luminal.tensor(g, [c.head_dim, s.max_seq, c.n_kv_heads, 1]; dtype=s.kv_dtype)
    pv = Luminal.tensor(g, [c.head_dim, s.max_seq, c.n_kv_heads, 1]; dtype=s.kv_dtype)
    logits, nk, nv, xpre = mtp_decode_step_hidden(m, tok, hid, 0, pos, pk, pv)
    am = Luminal.add_op!(g, Luminal.ArgMax(), [(logits.id, 0, logits.shape)], Luminal.ShapeTracker([np, 1]))
    W = s.weights
    if d.draft_vocab > 0
        W = copy(W)
        W["mtp.draft_head.weight"] = _first_rows(W["lm_head.weight"], d.draft_vocab)
    end
    load_weights!(g, reg, W; device=s.device)
    retain = [logits.id, am.id, nk.id, nv.id, tok.id, hid.id, pos.id, pk.id, pv.id, xpre.id]
    @info "Compiling MTP draft step ($np pairs)..."
    exec = compile(g; device=s.device, retain=retain, free_intermediates=false,
                   weight_dtype=_weight_dtype_for(s, reg), capture=s.device isa Luminal.AMDDevice)
    d.steps[np] = (exec=exec, tok=tok.id, hid=hid.id, pos=pos.id, pk=pk.id, pv=pv.id, logits=logits.id, argmax=am.id, xpre=xpre.id)
    return d.steps[np]
end

function _mtp_prefill_graph(d::MTPDrafter, slen::Int)
    get!(d.prefill, slen) do
        s = d.session; c = s.model.config
        g = Base.invokelatest(Graph); reg = WeightRegistry()
        m = Qwen35MTP(g, reg; c...)
        tok = Luminal.tensor(g, [slen, 1]); hid = Luminal.tensor(g, [c.hidden, slen, 1])
        k, v = mtp_prefill(m, tok, hid)
        load_weights!(g, reg, s.weights; device=s.device)
        @info "Compiling MTP prefill ($slen)..."
        exec = compile(g; device=s.device, retain=[k.id, v.id], weight_dtype=_weight_dtype_for(s, reg))
        (exec=exec, tok=tok.id, hid=hid.id, k=k.id, v=v.id)
    end
end

# Fill the MTP cache for a prompt of n tokens from the base prefill's hidden states H (Hidden, slen >= n, 1):
# the pairs (x[i+1], H[i]) for i = 0..n-2.
function _mtp_prime!(d::MTPDrafter, prompt::Vector{Int}, H)
    n = length(prompt)
    d.pos = 0
    n >= 2 || return
    slen = _prefill_bucket(n - 1)
    gr = _mtp_prefill_graph(d, slen)
    toks = zeros(Float32, slen, 1); toks[1:n-1, 1] .= prompt[2:n]
    res = gr.exec(Dict{Int,Any}(gr.tok => toks, gr.hid => H[:, 1:slen, :]); device=d.session.device)
    view(d.k, :, 1:n-1, :, 1) .= view(res[gr.k], :, 1:n-1, :, 1)
    view(d.v, :, 1:n-1, :, 1) .= view(res[gr.v], :, 1:n-1, :, 1)
    d.pos = n - 1
end

# One draft step for the pairs (tokens[1], hidden[:, 1]) at the MTP cache's next position and
# (tokens[2], hidden[:, 2]) at the one after; `advance` = how many of them are real (the cache position
# moves by that). Returns the two greedy tokens (0-based, host).
function _mtp_step!(d::MTPDrafter, tokens::Vector{Int}, hidden, advance::Int; with_hidden::Bool=false)
    np = length(tokens)
    gr = _mtp_step_graph(d, np)
    res = gr.exec(Dict{Int,Any}(gr.tok => Float32.(Base.reshape(tokens, np, 1)), gr.hid => hidden, gr.pos => Float32[d.pos],
                                gr.pk => d.k, gr.pv => d.v); device=d.session.device)
    d.pos += advance
    am = Int.(vec(Array(res[gr.argmax])))
    return with_hidden ? (am, res[gr.xpre]) : am
end

"""
    generate_ids_speculative(session, drafter, prompt; max_new_tokens=200, stats=nothing, draft_hook=nothing)
        -> Vector{Int}

Greedy generation for one prompt with MTP speculative decoding (0-indexed token ids, like `generate_ids`).
`stats`, a `Dict`, receives `:cycles`, `:accepted` (drafts the base model confirmed) and `:emitted`.
`draft_hook(n_emitted, draft) -> token` may replace the drafted token (tests use it to force accepts
and rejects).
"""
function generate_ids_speculative(s::LlamaSession, d::MTPDrafter, prompt::Vector{Int}; max_new_tokens::Int=200,
                                  stats=nothing, draft_hook=nothing, ndraft::Int=1)
    ndraft in (1, 2) || error("ndraft must be 1 or 2")
    d.session === s || error("the drafter belongs to another session")
    pf, res, dc, plens, logits = _prefill_into_cache!(s, [prompt])
    pf.hidden_id == 0 && error("this model's prefill does not return its hidden states")
    cache = dc.cache
    n = length(prompt)
    eos_ids = s.tokenizer === nothing ? Int[] : s.tokenizer.eos_ids
    y = argmax(view(logits, :, pf.last_only ? 1 : n, 1)) - 1
    out = [y]
    (y in eos_ids || max_new_tokens <= 1) && return out
    H = res[pf.hidden_id]                                         # (Hidden, slen, 1)
    _mtp_prime!(d, prompt, H)
    hook(draft, extra=0) = draft_hook === nothing ? draft : draft_hook(length(out) + extra, draft)
    h1 = H[:, n:n, :]
    ndraft == 2 && return _spec_loop2!(s, d, y, out, h1, cache, eos_ids, hook, max_new_tokens, stats)
    draft = hook(_mtp_step!(d, [y, y], cat(h1, h1; dims=2), 1)[1])
    vg = _verify_graph(s, 1, 2)
    cycles = accepted = 0
    tm = zeros(3)                                                 # seconds: verify (incl. the readback), MTP, rollback
    while length(out) < max_new_tokens
        cache.positions[1] + 2 <= s.max_seq || break
        t0 = time_ns()
        _, hid, am = _verify_step!(s, vg, [y; draft;;])
        t1 = time_ns(); tm[1] += (t1 - t0) / 1e9
        a = am[1]; cycles += 1
        if a == draft
            accepted += 1
            b = am[2]
            push!(out, a, b)
            cache.positions[1] += 2
            y = b
            (a in eos_ids || b in eos_ids) && break
            draft = hook(_mtp_step!(d, [a, b], hid, 2)[2])
            tm[2] += (time_ns() - t1) / 1e9
        else
            push!(out, a)
            cache.positions[1] += 1
            _rollback_states!(s, vg)
            y = a
            a in eos_ids && break
            t2 = time_ns(); tm[3] += (t2 - t1) / 1e9
            draft = hook(_mtp_step!(d, [a, a], hid, 1)[1])
            tm[2] += (time_ns() - t2) / 1e9
        end
    end
    # cut at the first end-of-sequence token and at max_new_tokens
    k = findfirst(t -> t in eos_ids, out)
    k === nothing || (out = out[1:k])
    length(out) > max_new_tokens && (out = out[1:max_new_tokens])
    if stats !== nothing
        stats[:cycles] = cycles; stats[:accepted] = accepted; stats[:emitted] = length(out)
        for (k, name) in enumerate((:ms_verify, :ms_mtp, :ms_rollback))
            stats[name] = round(Int, 1000 * tm[k])
        end
    end
    return out
end


# Two drafts per cycle: the base model verifies [y, d1, d2] in one 3-token step (3 columns cost about what 2 do, see
# `_q4d_kernel!`) and the MTP head drafts d1 from the last accepted token and d2 from d1 and its own output state.
#   a = argmax after y, b = after d1, c = after d2:  a != d1: emit a;  b != d2: emit a, b;  else emit a, b, c.
# The recurrent states roll back to the snapshot after y (slot 1) or after d1 (slot 2), or stay.
# The MTP pairs are (emitted[i], hidden of the position before it), padded with dummies to 3.
function _spec_loop2!(s::LlamaSession, d::MTPDrafter, y::Int, out::Vector{Int}, h1, cache, eos_ids, hook, max_new_tokens, stats)
    draft_pair!(toks, hid, nreal) = begin
        am, xp = _mtp_step!(d, toks, hid, nreal; with_hidden=true)
        d1 = hook(am[nreal], 0)
        d2 = hook(_mtp_step!(d, [d1], xp[:, nreal:nreal, :], 0)[1], 1)
        (d1, d2)
    end
    d1, d2 = draft_pair!([y, y, y], cat(h1, h1, h1; dims=2), 1)
    vg = _verify_graph(s, 1, 3)
    cycles = acc1 = acc2 = 0
    tm = zeros(3)
    while length(out) < max_new_tokens
        cache.positions[1] + 3 <= s.max_seq || break
        t0 = time_ns()
        _, hid, am = _verify_step!(s, vg, [y; d1; d2;;])
        t1 = time_ns(); tm[1] += (t1 - t0) / 1e9
        cycles += 1
        a, b, c = am[1], am[2], am[3]
        if a != d1
            emitted = [a]
        elseif b != d2
            acc1 += 1; emitted = [a, b]
        else
            acc1 += 1; acc2 += 1; emitted = [a, b, c]
        end
        nreal = length(emitted)
        append!(out, emitted)
        cache.positions[1] += nreal
        nreal < 3 && _rollback_states!(s, vg; slot=nreal)
        y = emitted[end]
        any(e -> e in eos_ids, emitted) && break
        t2 = time_ns(); tm[3] += (t2 - t1) / 1e9
        d1, d2 = draft_pair!([emitted; fill(y, 3 - nreal)], hid, nreal)
        tm[2] += (time_ns() - t2) / 1e9
    end
    k = findfirst(t -> t in eos_ids, out)
    k === nothing || (out = out[1:k])
    length(out) > max_new_tokens && (out = out[1:max_new_tokens])
    if stats !== nothing
        stats[:cycles] = cycles; stats[:accepted] = acc1; stats[:accepted2] = acc2; stats[:emitted] = length(out)
        for (k, name) in enumerate((:ms_verify, :ms_mtp, :ms_rollback))
            stats[name] = round(Int, 1000 * tm[k])
        end
    end
    return out
end

"""
    generate_speculative(session, drafter, prompt::String; max_new_tokens=200) -> String

Text in, text out: `generate` for one prompt with MTP speculative decoding (needs the session's tokenizer).
"""
function generate_speculative(s::LlamaSession, d::MTPDrafter, prompt::String; max_new_tokens::Int=200, kwargs...)
    ids = LlamaTokenization.encode(s.tokenizer, prompt; bos=true)
    return LlamaTokenization.decode(s.tokenizer, generate_ids_speculative(s, d, ids; max_new_tokens, kwargs...))
end
