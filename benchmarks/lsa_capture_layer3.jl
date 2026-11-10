# Capture layer 3's q, k, v (the first LSA layer of Qwen3.8-27B) on real text, through Luminal on the GPU.
#
#   julia -t auto,1 --project=. benchmarks/lsa_capture_layer3.jl <file.gguf> <n_tokens> <out.jls> [--min-avail=20]
#
# Uses the GPU (stop the model server first) and about 15 GB at 16k tokens. The text is the docs of a latent-sparse-attention
# checkout, LSA_REPO (default ~/devel/AI/latent-sparse-attention). The GGUF must be a type Luminal reads (Q4_K, Q5_K, Q6_K, Q8_0, ...).
#
# Output - a Serialization file for lsa_eval_layer3.jl and lsa_gather_attention.jl.
#
# Loads only the first 4 layers (3 Gated DeltaNet + the first full-attention layer). q, k, v are what the attention op receives -
# after the per-head q/k RMSNorm and RoPE on the first 64 dims, the output gate already split off q. Float16 weights.
using Luminal, Luminal.NN, Serialization, Printf, AMDGPU

meminfo_kb() = (d = Dict{String,Int}();
                for l in eachline("/proc/meminfo")
                    m = match(r"^(\w+):\s+(\d+)", l); m === nothing || (d[m[1]] = parse(Int, m[2]))
                end; d)
avail_gb() = meminfo_kb()["MemAvailable"] / 1024^2
status(args...) = (println(args...); flush(stdout))

function start_watchdog(limit_gb)
    Threads.@spawn :interactive while true
        if avail_gb() < limit_gb
            status(@sprintf("WATCHDOG: MemAvailable %.1f GB under %d GB, exiting", avail_gb(), limit_gb))
            ccall(:_exit, Cvoid, (Cint,), 3)
        end
        sleep(0.02)
    end
end

function main()
    args = filter(a -> !startswith(a, "--"), ARGS)
    flag(name, default) = (i = findfirst(a -> startswith(a, "--$name="), ARGS); i === nothing ? default : split(ARGS[i], "=", limit=2)[2])
    path, N, out_path = args[1], parse(Int, args[2]), args[3]
    start_watchdog(parse(Int, flag("min-avail", "20")))
    status(@sprintf("start: MemAvailable %.1f GB", avail_gb()))

    f = read_gguf(path)
    cfg = qwen35_config(f)
    cfg = merge(cfg, (n_layers = 4, layer_types = cfg.layer_types[1:4]))
    cfg.layer_types[4] == "full_attention" || error("layer 3 is not full attention: $(cfg.layer_types)")
    tok = LlamaTokenizer(f)
    status(@sprintf("%s: head_dim %d, rotary_dim %d, heads %d/%d, rope_base %g", f.metadata["general.name"], cfg.head_dim, cfg.rotary_dim,
                    cfg.n_heads, cfg.n_kv_heads, cfg.rope_base))

    # Real text - the repo's own docs, tokenized by the model's tokenizer
    repo = get(ENV, "LSA_REPO", joinpath(homedir(), "devel/AI/latent-sparse-attention"))
    files = vcat([joinpath(repo, "README.md")], sort(readdir(joinpath(repo, "docs/guide"); join=true)), [joinpath(repo, "docs/reference/notation.md")])
    text = join(read.(filter(isfile, files), String), "\n\n")
    ids = encode(tok, text)
    length(ids) >= N || error("only $(length(ids)) tokens of text, need $N")
    ids = ids[1:N]
    status("text: $(length(ids)) tokens kept of $(length(encode(tok, text)))")

    t = @elapsed W = load_qwen35_gguf(path, cfg; quantize = _ -> Float16)
    status(@sprintf("weights loaded in %.1f s, MemAvailable %.1f GB", t, avail_gb()))

    g = Luminal.Graph(); reg = Luminal.WeightRegistry()
    m = Qwen35(g, reg; cfg...)
    input = Luminal.tensor(g, [N, 1])
    lens = Luminal.tensor(g, [1])
    x = m.embedding(input)
    state_ids = Tuple{Int,Int,Tuple,Tuple}[]
    q = k = v = nothing
    for layer in m.layers
        h = layer.input_norm(x)
        if layer.kind === :linear
            cs, rs = NN.deltanet_state_shapes(layer.mixer, 1)
            conv = Luminal.tensor(g, cs); rec = Luminal.tensor(g, rs)
            a = layer.mixer(h, conv, rec, lens; fresh = true)
            push!(state_ids, (conv.id, rec.id, Tuple(cs), Tuple(rs)))
            x = x + a
            x = x + layer.feed_forward(layer.post_attention_norm(x))
        else
            # SelfAttention's own front half, up to the tensors the attention op receives
            sa = layer.mixer
            queries, _gate = NN._query_and_gate(sa, sa.q_proj(h), N, 1)
            queries = Luminal.permute(queries, [1, 3, 2, 4])
            keys = Luminal.permute(Luminal.reshape(sa.k_proj(h), [sa.head_dim, sa.n_kv_heads, N, 1]), [1, 3, 2, 4])
            values = Luminal.permute(Luminal.reshape(sa.v_proj(h), [sa.head_dim, sa.n_kv_heads, N, 1]), [1, 3, 2, 4])
            queries = NN._qk_norm(sa.q_norm, queries); keys = NN._qk_norm(sa.k_norm, keys)
            base = NN._rope_base(sa, m.rope_base)
            q = NN._rope(sa, queries, 0, base); k = NN._rope(sa, keys, 0, base); v = values
            break
        end
    end
    Luminal.load_weights!(g, reg, W; device = Luminal.get_device())
    status(@sprintf("graph built, MemAvailable %.1f GB; compiling", avail_gb()))
    exec = Luminal.compile(g; device = Luminal.get_device(), retain = [q.id, k.id, v.id])
    pins = Dict{Int,Any}(input.id => Base.reshape(Float32.(ids), N, 1), lens.id => Float32[N])
    for (cid, rid, cs, rs) in state_ids
        pins[cid] = zeros(Float32, cs...); pins[rid] = zeros(Float32, rs...)
    end
    t = @elapsed res = exec(pins; device = Luminal.get_device())
    status(@sprintf("ran in %.1f s, MemAvailable %.1f GB", t, avail_gb()))
    qa, ka, va = Array{Float32}(res[q.id]), Array{Float32}(res[k.id]), Array{Float32}(res[v.id])
    status("q $(size(qa)) k $(size(ka)) v $(size(va)); finite: ", all(isfinite, qa) && all(isfinite, ka) && all(isfinite, va))
    serialize(out_path, Dict("q" => qa, "k" => ka, "v" => va, "ids" => ids, "scale" => Float32(1 / sqrt(cfg.head_dim)),
                             "rotary_dim" => cfg.rotary_dim, "weights" => "Float16 from $(basename(path))"))
    status("saved $out_path")
end

main()
