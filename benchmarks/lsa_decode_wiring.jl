# lsa_decode_wiring.jl -- Latent Sparse Attention in the model's decode loop, on the first N layers of a Qwen3.8-27B GGUF.
#
# A real prompt is prefilled and decoded through `LlamaSession` with LSA enabled on every full-attention layer that has a map, and checked:
#   A  within the budget (the context is shorter than the list) LSA must reproduce dense decode - the logits and the tokens.
#   B  the index keys, from the prefill hook and the graph's per-token writes, against a CPU recomputation from the K cache.
#   C  the buckets - built by the prefill hook and, for the block the decode crossed, by the post-step hook - against a CPU rebuild.
#   D  with a budget far under the context, the device's selection list for the last token against the CPU's from the same tables.
# Then the tokens a small budget generates, next to dense's.
#
# Usage: julia -t auto,1 --project=. benchmarks/lsa_decode_wiring.jl <file.gguf> <maps.safetensors> [--layers=4] [--prompt=1900] [--new=200]
#   (uses the GPU - stop the model server first; layers=4 holds the first full-attention layer, decoder layer 3)
using Luminal, Luminal.NN, Printf, Statistics, LinearAlgebra, AMDGPU
using Luminal: LSAConfig, lsa_load_maps, LatentSparse
using Luminal.Decoding: lsa_enable!, lsa_disable!

status(args...) = (println(args...); flush(stdout))
flag(name, default) = (i = findfirst(a -> startswith(a, "--$name="), ARGS); i === nothing ? default : split(ARGS[i], "=", limit = 2)[2])

function main()
    args = filter(a -> !startswith(a, "--"), ARGS)
    path, maps_path = args[1], args[2]
    nl, n_prompt, n_new = parse(Int, flag("layers", "4")), parse(Int, flag("prompt", "1900")), parse(Int, flag("new", "200"))
    f = read_gguf(path); cfg = qwen35_config(f)
    cfg = merge(cfg, (n_layers = nl, layer_types = cfg.layer_types[1:nl]))
    tok = LlamaTokenizer(f)
    repo = get(ENV, "LSA_REPO", joinpath(homedir(), "devel/AI/latent-sparse-attention"))
    text = join(read.(filter(isfile, vcat([joinpath(repo, "README.md")], sort(readdir(joinpath(repo, "docs/guide"); join = true)))), String), "\n\n")
    prompt = encode(tok, text)[1:n_prompt]
    T = Luminal.Int4
    W = load_qwen35_gguf(path, cfg; quantize = _ -> T)
    s = LlamaSession(Qwen35(Luminal.Graph(), nothing; cfg...), tok, path; max_seq = 4096, weights = W, decode_weights = T, kv_dtype = Float16)
    maps = lsa_load_maps(maps_path)
    full_layers = [i - 1 for (i, l) in enumerate(s.model.layers) if !NN._is_state_layer(l)]
    status("layers $nl, full-attention layers $full_layers, prompt $n_prompt tokens, $n_new new")

    generate(budget_lsa) = begin
        logits = Any[]
        out = generate_ids(s, [prompt]; max_new_tokens = n_new, step_logits = logits)
        return out[1], [vec(x) for x in logits]
    end

    status("\n-- dense")
    dense_tokens, dense_logits = generate(nothing)

    status("\n-- A  LSA with the design's budget (top_k 4096) on a context under it")
    lsa_enable!(s, LSAConfig(maps = maps))
    t, l = generate(nothing)
    st = Luminal.LSA_SESSIONS[s]
    same_tokens = t == dense_tokens
    worst = maximum(norm(a - b) / norm(b) for (a, b) in zip(l, dense_logits))
    status(@sprintf("   tokens identical to dense: %s, worst relative logit difference over %d steps: %.2e", same_tokens, length(l), worst))

    status("\n-- B, C  the index state against a CPU recomputation")
    cache = s.decode[1].cache
    total = cache.positions[1]
    layer = first(full_layers)
    ls = st.layers[layer]
    wq, wk = maps[layer]
    pk = cache.self_cache[layer + 1][1]
    K = Float32.(Array(view(pk, :, 1:total, :, 1)))                                   # (D, total, KVH)
    q_dummy = zeros(Float32, size(K, 1), total, 24, 1)                                # only the keys are compared
    _, ref_keys = LatentSparse.index_vectors(q_dummy, Base.reshape(K, size(K, 1), total, size(K, 3), 1), wq, wk, cfg.rotary_dim)
    got_keys = permutedims(Array(view(ls.index_keys, :, 1:total)))
    status(@sprintf("   index keys, %d positions: worst absolute difference %.2e", total, maximum(abs.(got_keys .- ref_keys))))
    blocks = total ÷ 512
    table = LatentSparse.build_buckets(got_keys[1:blocks*512, :], 4, 8, 16, 512)
    got_members = permutedims(Array(view(ls.members, :, 1:blocks*64)))
    got_centroids = Array(view(ls.centroids, 1:blocks*64, :))
    status(@sprintf("   buckets, %d closed blocks (%d from the prefill hook, %d from the decode hook): members identical %s, centroids within %.1e",
                    blocks, n_prompt ÷ 512, blocks - n_prompt ÷ 512, got_members == table.members, maximum(abs.(got_centroids .- table.centroids))))

    status("\n-- D  a budget far under the context (top_k 192, top_p 48): the device's list against the CPU's")
    small = LSAConfig(maps = maps, top_k = 192, top_p = 48)
    lsa_enable!(s, small)
    small_tokens, _ = generate(nothing)
    st = Luminal.LSA_SESSIONS[s]; ls = st.layers[layer]
    cache = s.decode[1].cache
    p = cache.positions[1] - 1                                                         # the last token's position
    qidx = Array(ls.qidx)
    cent, members = Array(ls.centroids), permutedims(Array(ls.members))
    nb = p ÷ 512 * 64
    cpu_table = LatentSparse.BucketTable(cent, members, Int32.(sum(members .>= 0; dims = 2)[:]), 64, 512)
    scores = cent[1:nb, :] * qidx
    want = LatentSparse.select_from_scores(scores, p, cpu_table, 48, 192, 4, 512)
    n = Int(Array(ls.lastpos)[1]) + 1
    got = Int32.(Array(ls.ids)[1:n, 1])
    status(@sprintf("   position %d, %d buckets visible: %d ids, identical to the CPU's %s, same key set %s", p, nb, n, got == want, sort(got) == sort(want)))
    status(@sprintf("   tokens agreeing with dense, first mismatch at step: %s", something(findfirst(i -> small_tokens[i] != dense_tokens[i], 1:min(length(small_tokens), length(dense_tokens))), "none")))
    status("   dense: ", decode(tok, dense_tokens[1:min(40, end)]))
    status("   small: ", decode(tok, small_tokens[1:min(40, end)]))
end

main()
