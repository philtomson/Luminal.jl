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
    step::Any                 # compiled single draft step, built on first use
    prefill::Dict{Int,Any}    # padded length => compiled prefill
end

function MTPDrafter(s::LlamaSession)
    haskey(s.weights, "mtp.eh_proj.weight") || error("the session's weights have no MTP head (merge load_qwen35_mtp's result)")
    c = s.model.config
    shape = (c.head_dim, s.max_seq, c.n_kv_heads, 1)
    MTPDrafter(s, Luminal.zero_tensor(s.device, s.kv_dtype, shape...), Luminal.zero_tensor(s.device, s.kv_dtype, shape...),
               0, nothing, Dict{Int,Any}())
end

function _weight_dtype_for(s::LlamaSession, reg)
    wd = s.decode_weights
    wd isa Type && return wd
    by_id = Dict{Int, Type}(id => wd(name) for (name, id) in reg.mapping)
    return id -> get(by_id, id, Float32)
end

function _mtp_step_graph(d::MTPDrafter)
    d.step === nothing || return d.step
    s = d.session; c = s.model.config
    g = Base.invokelatest(Graph); reg = WeightRegistry()
    m = Qwen35MTP(g, reg; c...)
    tok = Luminal.tensor(g, [1, 1]); hid = Luminal.tensor(g, [c.hidden, 1, 1]); pos = Luminal.tensor(g, [1])
    pk = Luminal.tensor(g, [c.head_dim, s.max_seq, c.n_kv_heads, 1]; dtype=s.kv_dtype)
    pv = Luminal.tensor(g, [c.head_dim, s.max_seq, c.n_kv_heads, 1]; dtype=s.kv_dtype)
    logits, nk, nv = mtp_decode_step(m, tok, hid, 0, pos, pk, pv)
    load_weights!(g, reg, s.weights; device=s.device)
    retain = [logits.id, nk.id, nv.id, tok.id, hid.id, pos.id, pk.id, pv.id]
    @info "Compiling MTP draft step..."
    exec = compile(g; device=s.device, retain=retain, free_intermediates=false,
                   weight_dtype=_weight_dtype_for(s, reg), capture=s.device isa Luminal.AMDDevice)
    d.step = (exec=exec, tok=tok.id, hid=hid.id, pos=pos.id, pk=pk.id, pv=pv.id, logits=logits.id)
    return d.step
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

# One draft step at the MTP cache's next position for (token, hidden (Hidden, 1, 1)); returns the logits
# (vocab, 1, 1) on the device and advances the position.
function _mtp_step!(d::MTPDrafter, token::Int, hidden)
    gr = _mtp_step_graph(d)
    res = gr.exec(Dict{Int,Any}(gr.tok => Float32[token;;], gr.hid => hidden, gr.pos => Float32[d.pos],
                                gr.pk => d.k, gr.pv => d.v); device=d.session.device)
    d.pos += 1
    return res[gr.logits]
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
                                  stats=nothing, draft_hook=nothing)
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
    argmax_tok(l) = argmax(Array{Float32}(l)[:, 1, 1]) - 1
    hook(draft) = draft_hook === nothing ? draft : draft_hook(length(out), draft)
    draft = hook(argmax_tok(_mtp_step!(d, y, H[:, n:n, :])))
    vg = _verify_graph(s, 1, 2)
    cycles = accepted = 0
    while length(out) < max_new_tokens
        cache.positions[1] + 2 <= s.max_seq || break
        lg, hid = _verify_step!(s, vg, [y; draft;;])
        lgh = Array{Float32}(lg)                                  # (vocab, 2, 1)
        a = argmax(view(lgh, :, 1, 1)) - 1
        cycles += 1
        if a == draft
            accepted += 1
            b = argmax(view(lgh, :, 2, 1)) - 1
            push!(out, a, b)
            cache.positions[1] += 2
            _mtp_step!(d, a, hid[:, 1:1, :])                       # K/V of the pair (a, H_P) only
            y = b
            (a in eos_ids || b in eos_ids) && break
            draft = hook(argmax_tok(_mtp_step!(d, b, hid[:, 2:2, :])))
        else
            push!(out, a)
            cache.positions[1] += 1
            _rollback_states!(s, vg)
            y = a
            a in eos_ids && break
            draft = hook(argmax_tok(_mtp_step!(d, a, hid[:, 1:1, :])))
        end
    end
    # cut at the first end-of-sequence token and at max_new_tokens
    k = findfirst(t -> t in eos_ids, out)
    k === nothing || (out = out[1:k])
    length(out) > max_new_tokens && (out = out[1:max_new_tokens])
    if stats !== nothing
        stats[:cycles] = cycles; stats[:accepted] = accepted; stats[:emitted] = length(out)
    end
    return out
end

