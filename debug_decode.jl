# debug_decode.jl — decode-vs-prefill equivalence check for TinyLlama.
#
# Checks, in order:
#   A. Causal prefix consistency: K/V from prefill(prompt[1:end-1]) must equal the
#      first plen-1 positions of K/V from prefill(prompt).  (Validates the causal mask.)
#   B. One decode step at pos = plen-1 fed prompt[end], with the cache filled from
#      prefill(prompt[1:end-1]), must write K/V slot `plen` equal to prefill(prompt)'s
#      K/V at position plen — reported per layer, so the first diverging layer is visible.
#   C. The decode step's logits must equal prefill(prompt)'s last-position logits.
#
# Usage: julia --project=. debug_decode.jl [model_dir] [cpu|gpu] [f32|f16]
#   f16 stores matmul weights as Float16 (compile(...; weight_dtype=Float16))

using Luminal
using Luminal.NN
using Luminal.LlamaTokenization
using Printf

relerr(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), 1f-6)

function prefill(model, ids, weights_dict, device, rope_base, wdtype)
    g = Graph(); reg = WeightRegistry()
    m = Luminal.Decoding._rebuild_model_like(model, g, reg; rope_base=rope_base)
    n = length(ids)
    inp = Luminal.tensor(g, [n, 1])
    out, kvs = m(inp, 0; return_kv=true)
    retain = vcat(out.id, [k.id for (k, _) in kvs], [v.id for (_, v) in kvs])
    load_weights!(g, reg, weights_dict; device=device)
    ex = compile(g; device=device, retain=retain, weight_dtype=wdtype)
    res = ex(Dict{Int,Any}(inp.id => Float32.(Base.reshape(ids, n, 1))); device=device)
    logits = Array{Float32}(res[out.id])
    ks = [Array{Float32}(res[k.id]) for (k, _) in kvs]
    vs = [Array{Float32}(res[v.id]) for (_, v) in kvs]
    return logits, ks, vs
end

function main()
    model_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "tinyllama_chat")
    device = (length(ARGS) >= 2 && ARGS[2] == "cpu") ? CPUDevice() : get_device()
    wdtype = (length(ARGS) >= 3 && ARGS[3] == "f16") ? Float16 : Float32
    println("Matmul weights: ", wdtype)
    rope_base = 10000f0
    max_seq = 256
    println("Device: ", device)

    tok = LlamaTokenizer(model_dir)
    model = NN.Llama(Graph(), WeightRegistry(); vocab_size=32000, hidden=2048, n_layers=22,
                     n_heads=32, n_kv_heads=4, intermediate=5632, rope_base=rope_base)
    ids = LlamaTokenization.encode(tok, "Once upon a time"; bos=true)
    plen = length(ids)
    println("Prompt ids: ", ids)

    weights_dict = load_weights_to_dict(model_dir; device=device)

    t = time()
    logits_full, k_full, v_full = prefill(model, ids, weights_dict, device, rope_base, wdtype)
    @printf("full prefill: %.1fs\n", time() - t)
    _, k_short, v_short = prefill(model, ids[1:end-1], weights_dict, device, rope_base, wdtype)
    GC.gc(); Luminal.reclaim!(device)

    println("\n== A. causal prefix consistency (prefill[1:end-1] vs prefill[1:plen-1]) ==")
    for i in eachindex(k_full)
        ek = relerr(k_short[i], k_full[i][:, 1:plen-1, :, :])
        ev = relerr(v_short[i], v_full[i][:, 1:plen-1, :, :])
        (i <= 3 || i == length(k_full) || max(ek, ev) > 1e-3) &&
            @printf("layer %2d  K relerr=%.2e  V relerr=%.2e\n", i, ek, ev)
    end

    # ── Decode one step at pos = plen-1 ──────────────────────────────────────
    first_attn = model.layers[1].attention
    cache = LlamaKVCacheState(length(model.layers), first_attn.n_kv_heads, first_attn.head_dim;
                              batch=1, max_seq=max_seq, device=device)
    for i in eachindex(k_short)
        view(cache.self_cache[i][1], :, 1:plen-1, :, :) .= Luminal.to_device(k_short[i], device)
        view(cache.self_cache[i][2], :, 1:plen-1, :, :) .= Luminal.to_device(v_short[i], device)
    end
    cache.step_pos = plen - 1

    dg = Graph(); dreg = WeightRegistry()
    dm = Luminal.Decoding._rebuild_model_like(model, dg, dreg; rope_base=rope_base)
    idg = build_llama_decode_step!(dm, dg, Luminal.Sym{Int}(:pos); max_seq=max_seq, rope_base=rope_base)
    load_weights!(dg, dreg, weights_dict; device=device)
    retain = vcat(idg.logits_id, idg.new_self_k_ids, idg.new_self_v_ids,
                  idg.token_input_id, idg.pos_input_id, idg.self_k_ids, idg.self_v_ids)
    ex = compile(dg; device=device, retain=retain, free_intermediates=false, weight_dtype=wdtype)
    t = time()
    logits_dec = Array{Float32}(llama_decode_step!(ex, idg, cache, ids[end];
                                                   sym_vals=Dict(:pos => plen - 1), device=device))
    @printf("decode step: %.1fs\n", time() - t)

    println("\n== B. decode K/V slot at position plen vs full prefill ==")
    for i in eachindex(k_full)
        nk, nv = Array(cache.self_cache[i][1]), Array(cache.self_cache[i][2])
        ek = relerr(nk[:, plen, :, :], k_full[i][:, plen, :, :])
        ev = relerr(nv[:, plen, :, :], v_full[i][:, plen, :, :])
        # untouched slots must be preserved by the scatter
        ep = relerr(nk[:, 1:plen-1, :, :], k_short[i])
        ez = maximum(abs.(nk[:, plen+1:end, :, :]))
        @printf("layer %2d  K relerr=%.2e  V relerr=%.2e  prefix-preserved relerr=%.2e  tail max=%.2e\n",
                i, ek, ev, ep, ez)
    end

    println("\n== C. logits ==")
    lf = logits_full[:, plen, 1]
    ld = vec(logits_dec)
    @printf("relerr=%.2e  argmax full=%d  argmax decode=%d\n", relerr(ld, lf), argmax(lf) - 1, argmax(ld) - 1)
    println("full   top5: ", partialsortperm(lf, 1:5, rev=true) .- 1)
    println("decode top5: ", partialsortperm(ld, 1:5, rev=true) .- 1)
    println("full   top1 token: ", repr(LlamaTokenization.decode(tok, [argmax(lf) - 1])))

    println("\n== D. steady-state decode timing ==")
    token = argmax(ld) - 1
    times = Float64[]
    for _ in 1:20
        pos = cache.step_pos
        t = time()
        l = llama_decode_step!(ex, idg, cache, token; sym_vals=Dict(:pos => pos), device=device)
        token = argmax(vec(Array{Float32}(l))) - 1
        push!(times, time() - t)
    end
    steady = sort(times[3:end])[div(end + 1, 2)]
    @printf("median step (excluding first 2): %.1f ms  (%.2f tok/s)\n", 1000 * steady, 1 / steady)
end

main()
