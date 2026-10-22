# Speculative decoding with the DFlash2 block drafter (src/DFlash.jl): each cycle drafts a block of up to 7 tokens in one pass, the target
# verifies [y, drafts...] in one batched step (`_verify_graph`, with the five context layers extracted), the longest accepted prefix is emitted
# (greedy, or rejection sampling when `sampler` is given: the selector's draft is a point mass, as for MTP), the recurrent states roll back to the
# last accepted token and the accepted rows go into the drafter's context ring.

# Conditional acceptance of draft j given 1..j-1 matched, measured on the greedy texts of benchmarks/dflash2_accept.jl; the running estimates start here.
const _DF_Q0 = (0.78, 0.80, 0.65, 0.68, 0.71, 0.87, 0.77)
# ms: a verify step with K = 2..8 rows of the native 27B (K <= 8 columns per pass of the dot2 GEMV), and one draft pass.
const _DF_COST_VERIFY = (106.5, 110.0, 119.0, 124.0, 128.0, 133.0, 138.0)
const _DF_COST_DRAFT = 18.0

# The number of drafts (1..7) with the best expected tokens per millisecond from the acceptance estimates q and the cycle costs `cost[nd]` (ms).
function _best_nprop(q, cost)
    best, bn = -1.0, 1
    e = 1.0; p = 1.0
    for nd in 1:7
        p *= q[nd]; e += p
        r = e / cost[nd]
        r > best && ((best, bn) = (r, nd))
    end
    return bn
end
# The costs start from the table above and follow the measured cycle times: an exponential average per block length, and the ratio of measured to
# tabulated cost carried over to the lengths not tried yet (the context length moves all of them together).
mutable struct DFCost
    cost::Vector{Float64}          # per nd, a whole cycle
    seen::Vector{Bool}
    scale::Float64
end
DFCost() = DFCost([_DF_COST_VERIFY[nd] + _DF_COST_DRAFT for nd in 1:7], fill(false, 7), 1.0)
function _update!(c::DFCost, nd::Int, ms::Float64)
    base = _DF_COST_VERIFY[nd] + _DF_COST_DRAFT
    c.scale += 0.2 * (ms / base - c.scale)
    c.cost[nd] = c.seen[nd] ? c.cost[nd] + 0.2 * (ms - c.cost[nd]) : ms
    c.seen[nd] = true
    for j in 1:7
        c.seen[j] || (c.cost[j] = (_DF_COST_VERIFY[j] + _DF_COST_DRAFT) * c.scale)
    end
    return c
end

"""
    generate_ids_dflash(session, drafter::DFlash2, prompt; max_new_tokens=200, sampler=nothing, on_token=nothing, prefix=nothing,
                        prefix_at=[], stats=nothing, nprop=0) -> Vector{Int}

`generate_ids_speculative` with the block drafter (0-based token ids; needs the Float16 or Int8 KV cache and the chunked prefill). `nprop` fixes the drafts per
cycle (1..7); 0 picks them from the recent acceptance. With a `prefix` cache the generation also leaves the state checkpoint of its end (see `_save_end_ckpt!`).
"""
function generate_ids_dflash(s::LlamaSession, df::DFlash2, prompt::Vector{Int}; max_new_tokens::Int=200, stats=nothing, sampler=nothing, on_token=nothing,
                             prefix=nothing, prefix_at=Int[], on_progress=nothing, nprop::Int=0)
    df.session === s || error("the drafter belongs to another session")
    s.kv_dtype in (Float16, Int8) && _can_chunk(s) || error("the block drafter needs the Float16 or Int8 KV cache and the chunked prefill")
    nprop in 0:7 || error("nprop must be 0 (adaptive) or 1..7")
    n = length(prompt)
    t_pf = time()
    logits, Hall, off = _prefill_chunked!(s, prompt; prefix, prefix_at, need_mtp=false, on_progress, dflash=df)
    PREFILL_TIMES[:prefill_total] = time() - t_pf
    dc = _decode_graph(s, 1); cache = dc.cache
    stats === nothing || (stats[:prefill_reused] = off == 0 ? 0 : off + 1)
    eos_ids = s.tokenizer === nothing ? Int[] : s.tokenizer.eos_ids
    sampler isa Sampler && sampler.temperature <= 0 && (sampler = nothing)
    y = sample_token(sampler, view(logits, :, 1, 1))
    out = [y]
    on_token !== nothing && on_token([y]) && return out
    (y in eos_ids || max_new_tokens <= 1) && return out
    layers = Tuple(df.target_layers)
    q = collect(_DF_Q0)
    costs = DFCost()
    cycles = 0
    acc = zeros(Int, 7); sel = zeros(Int, 7)
    tm = zeros(3)                                                     # seconds: verify (with the readback), drafting, rollback
    tm_append = zeros(1)                                              # ... of which the ring append (measured with DF_PROFILE)
    trace = stats === nothing ? nothing : Tuple{Int,Float64,Float64,Float64,Int}[]       # per cycle: drafts, verify, draft, rollback + ring (ms), tokens emitted
    hid_last = nothing; nreal_last = 0
    while length(out) < max_new_tokens
        pos = cache.positions[1]
        nd = nprop > 0 ? nprop : _best_nprop(q, costs.cost)
        K = nd + 1
        pos + K <= s.max_seq || break
        df.nrows == pos || error("the drafter's context has $(df.nrows) rows but the sequence is at $pos")
        DF_PROFILE[] && AMDGPU.synchronize()
        t0 = time_ns()
        ds = dflash_draft!(df, y, pos; nprop = nd)
        DF_PROFILE[] && AMDGPU.synchronize()
        t1 = time_ns(); tm[2] += (t1 - t0) / 1e9
        vg = _verify_graph(s, 1, K; extract = layers)
        res = _verify_run!(s, vg, [y; ds;;])
        lg = res[vg.idg.logits_id]; hid = res[vg.idg.hidden_id]
        cycles += 1; sel[nd] += 1
        if sampler === nothing
            am = Int.(Array(res[vg.idg.argmax_id]))
            t2 = time_ns(); tm[1] += (t2 - t1) / 1e9
            m = 0
            while m < nd && am[m + 1] == ds[m + 1]
                m += 1
            end
            emitted = am[1:m + 1]
        else
            m, emitted = _sampled_verify(sampler, lg, ds)
            t2 = time_ns(); tm[1] += (t2 - t1) / 1e9
        end
        acc[1:m] .+= 1
        for j in 1:min(m + 1, nd)                                     # draft j was tested iff 1..j-1 matched
            q[j] += 0.1 * ((j <= m) - q[j])
        end
        nreal = m + 1
        append!(out, emitted)
        DF_PROFILE[] && AMDGPU.synchronize()
        t3 = time_ns()
        dflash_append_rows!(df, [res[eid][:, 1:nreal, 1] for eid in vg.idg.extract_ids], pos)
        DF_PROFILE[] && AMDGPU.synchronize()
        t4 = time_ns(); tm_append[1] += (t4 - t3) / 1e9
        cache.positions[1] += nreal
        nreal < K && _rollback_states!(s, vg; slot = nreal)
        DF_PROFILE[] && AMDGPU.synchronize()
        tm[3] += (time_ns() - t2) / 1e9
        _update!(costs, nd, (time_ns() - t0) / 1e6)
        trace === nothing || push!(trace, (nd, (t2 - t1) / 1e6, (t1 - t0) / 1e6, (time_ns() - t2) / 1e6, nreal))
        y = emitted[end]
        hid_last = hid; nreal_last = nreal
        ((on_token !== nothing && on_token(emitted)) || any(e -> e in eos_ids, emitted)) && break
    end
    if prefix !== nothing && hid_last !== nothing && get(ENV, "LUMINAL_END_CKPT", "1") != "0"
        P = _save_end_ckpt!(s, nothing, prefix, prompt, out, cache, hid_last, nreal_last, 0, y, nothing)
        P > 0 && (prefix.live_df = df.nrows)
        stats === nothing || (stats[:end_ckpt] = P)
    end
    k = findfirst(t -> t in eos_ids, out)
    k === nothing || (out = out[1:k])
    length(out) > max_new_tokens && (out = out[1:max_new_tokens])
    if stats !== nothing
        stats[:cycles] = cycles; stats[:accepted] = acc[1]; stats[:emitted] = length(out)
        stats[:acc_by_depth] = copy(acc); stats[:cycles_nd] = copy(sel); stats[:ms_append] = round(Int, 1000 * tm_append[1]); stats[:trace] = trace
        for (k, name) in enumerate((:ms_verify, :ms_mtp, :ms_rollback))
            stats[name] = round(Int, 1000 * tm[k])
        end
    end
    return out
end
