# AWQ-style activation-aware weight scaling for low-bit weights (part of module Decoding).
#
# Rounding weights to 4 bits treats every input channel alike, but a few channels
# carry much larger activations, and the rounding error on their weights dominates
# the output error. AWQ multiplies each weight's input channel j by s_j before
# quantizing, and divides the layer's input by s_j instead: the operation that
# produces that input absorbs 1/s_j. In Float32 the model is unchanged; only what
# quantization loses changes. For a Llama block:
#   q/k/v     input: input_layernorm output      -> its RMSNorm weight absorbs 1/s
#   gate/up   input: post_attention_layernorm    -> that RMSNorm weight absorbs 1/s
#   down      input: silu(gate) * up             -> up_proj's output rows absorb 1/s
#   o         input: attention output            -> v_proj's output rows absorb 1/s;
#             query heads sharing a KV head (GQA) must share scales
#   lm_head   input: final norm output           -> model.norm's weight absorbs 1/s
# s = stat^alpha, stat = mean |x| per channel on calibration text; alpha is searched
# per group of weights sharing an input, minimizing || (W - Q(W s) / s) X ||^2 over
# sampled calibration activations X, with each weight's own storage format.

using Serialization
using Random

_awq_storage(T) = T === Luminal.Int4 ? Luminal.Q4Weight : T === Int8 ? Luminal.QuantWeight :
                  T === Float16 ? Luminal.HalfWeight : nothing

_free_q(q) = foreach(f -> Luminal._free_now!(getfield(q, f)), fieldnames(typeof(q)))

# AWQ groups of a Llama/Phi-3-style model: (key, member weights, absorber, kind)
function _awq_groups(n_layers::Int)
    gs = Tuple{String, Vector{String}, String, Symbol}[]
    for i in 0:n_layers-1
        p = "model.layers.$i"
        push!(gs, ("$p.qkv", ["$p.self_attn.q_proj.weight", "$p.self_attn.k_proj.weight", "$p.self_attn.v_proj.weight"],
                   "$p.input_layernorm.weight", :vector))
        push!(gs, ("$p.gate_up", ["$p.mlp.gate_proj.weight", "$p.mlp.up_proj.weight"],
                   "$p.post_attention_layernorm.weight", :vector))
        push!(gs, ("$p.down", ["$p.mlp.down_proj.weight"], "$p.mlp.up_proj.weight", :rows))
        push!(gs, ("$p.o", ["$p.self_attn.o_proj.weight"], "$p.self_attn.v_proj.weight", :rows_gqa))
    end
    push!(gs, ("lm_head", ["lm_head.weight"], "model.norm.weight", :vector))
    return gs
end

"""
    awq_scales(model, tokenizer, model_dir; text, policy=weight_preset(:int4),
               chunk=512, nchunks=16, samples=1024, grid=0:0.1:1, device=nothing,
               weights=nothing, cache=true, cache_key="") -> Dict{String, Vector{Float32}}

Calibrate AWQ input-channel scales for `model` (a `Llama`/`Phi3` template) on
`text` (use text you will not evaluate on). `policy(name) -> Type` gives each
weight's storage format (a preset name is also accepted); the scale search
minimizes that format's output error. Apply the result with `awq_apply!`, or
pass it to `LlamaSession(...; awq=scales)`. With `cache`, results are stored
under `~/.cache/Luminal.jl/awq`, keyed by the checkpoint, text, settings and
`cache_key` (name the policy there if it is a custom function).
"""
function awq_scales(model, tok, model_dir::String; text::String, policy=weight_preset(:int4),
                    chunk::Int=512, nchunks::Int=16, samples::Int=1024, grid=0:0.1:1,
                    device=nothing, weights=nothing, cache::Bool=true, cache_key::AbstractString="")
    policy isa Symbol && (cache_key = string(policy, cache_key); policy = weight_preset(policy))
    dev = device === nothing ? get_device() : device
    files = sort(filter(f -> endswith(f, ".safetensors"), readdir(model_dir; join=true)))
    key = hash((abspath(model_dir), [(basename(f), filesize(f), mtime(f)) for f in files],
                hash(text), chunk, nchunks, samples, collect(grid), cache_key))
    path = joinpath(homedir(), ".cache", "Luminal.jl", "awq", string(key, base=16) * ".jls")
    if cache && isfile(path)
        @info "AWQ scales from cache" path
        return deserialize(path)
    end

    W = weights === nothing ? load_weights_to_dict(model_dir; device=dev) : weights
    n_layers = length(model.layers)
    attn = model.layers[1].attention
    D, H, KVH = attn.head_dim, attn.n_heads, attn.n_kv_heads

    # 1. Calibration pass (Float32): each linear layer's input, per channel mean |x|
    #    and a sample of token columns
    ids = encode(tok, text[1:min(end, prevind(text, min(end, 8 * chunk * nchunks)))]; bos=false)
    length(ids) >= nchunks * (chunk - 1) || error("calibration text too short")
    chunks = [vcat(tok.bos_id, ids[(i - 1) * (chunk - 1) + 1 : i * (chunk - 1)]) for i in 1:nchunks]
    g = Graph(); reg = WeightRegistry()
    m = _rebuild_model_like(model, g, reg)
    inp = Luminal.tensor(g, [chunk, 1]); out = m(inp, 0)
    load_weights!(g, reg, W; device=dev)
    name_of = Dict(id => n for (n, id) in reg.mapping)
    xnode = Dict{String, Int}()                        # weight name => its input node
    for node in g.nodes
        node.op isa Luminal.MatMul || continue
        wid = node.inputs[1][1]
        haskey(name_of, wid) && (xnode[name_of[wid]] = node.inputs[2][1])
    end
    xs = unique(values(xnode))
    ex = compile(g; device=dev, retain=vcat(out.id, xs))
    absum = Dict{Int, Vector{Float64}}(); sample = Dict{Int, Vector{Matrix{Float32}}}()
    rng = MersenneTwister(0)
    per_chunk = cld(samples, nchunks)
    for c in chunks
        res = ex(Dict{Int,Any}(inp.id => Float32.(Base.reshape(c, chunk, 1))); device=dev)
        cols = randperm(rng, chunk - 1)[1:per_chunk] .+ 1  # skip the BOS position
        for x in xs
            a = Base.reshape(Array{Float32}(res[x]), :, chunk)
            absum[x] = get(absum, x, zeros(Float64, size(a, 1))) .+ vec(sum(abs, a; dims=2))
            push!(get!(sample, x, Matrix{Float32}[]), a[:, cols])
        end
    end
    Luminal.release!(ex); Luminal.reclaim!(dev)

    # 2. Per group: search alpha
    scales = Dict{String, Vector{Float32}}()
    for (key, members, absorber, kind) in _awq_groups(n_layers)
        all(haskey(xnode, mb) for mb in members) || continue
        x = xnode[members[1]]
        stat = Float32.(absum[x] ./ (nchunks * chunk))
        if kind === :rows_gqa                          # tie scales across heads sharing a KV head
            st = Base.reshape(stat, D, H ÷ KVH, KVH)
            stat = vec(repeat(sum(st; dims=2) ./ (H ÷ KVH), 1, H ÷ KVH, 1))
        end
        stat = max.(stat, 1f-5)
        X = Luminal.to_device(reduce(hcat, sample[x]), dev)
        fmts = [(mb, _awq_storage(policy(mb))) for mb in members]
        filter!(f -> f[2] !== nothing, fmts)
        isempty(fmts) && continue
        best_err, best_s = Inf, ones(Float32, length(stat))
        for α in grid
            s = stat .^ Float32(α); s ./= sqrt(maximum(s) * minimum(s))
            sd = Luminal.to_device(Base.reshape(s, 1, :), dev)
            err = 0.0
            for (mb, S) in fmts
                Wm = W[mb]
                Ws = Wm .* sd
                q = S(Ws); dq = Luminal.dequantize(q)
                diff = Wm .- dq ./ sd
                E = diff * X
                err += sum(abs2, E)
                # every full-size temporary freed now: left to the GC they pile up
                # (8B: 224 matrices x 11 alphas)
                foreach(Luminal._free_now!, (Ws, dq, diff, E)); _free_q(q)
            end
            Luminal.reclaim!(dev)
            if err < best_err
                best_err, best_s = err, s
            end
        end
        scales[key] = best_s
        Luminal._free_now!(X); Luminal.reclaim!(dev)
    end
    if cache
        mkpath(dirname(path)); serialize(path, scales)
    end
    return scales
end

"""
    awq_apply!(weights, scales, model) -> weights

Fold AWQ scales (from `awq_scales`) into a weight dict (as from
`load_weights_to_dict`), in place: each group's weights get their input columns
multiplied by s, and the operation producing their input absorbs 1/s. The
Float32 model computes the same function (up to rounding); quantizing the result
loses less. Apply before any graph is compiled from these weights.
"""
function awq_apply!(W::AbstractDict, scales::AbstractDict, model)
    attn = model.layers[1].attention
    D, H, KVH = attn.head_dim, attn.n_heads, attn.n_kv_heads
    for (key, members, absorber, kind) in _awq_groups(length(model.layers))
        haskey(scales, key) || continue
        s = scales[key]
        dev_s(v) = Luminal.to_device(v, Luminal.get_device())
        for mb in members
            W[mb] .*= Luminal.to_device(Base.reshape(s, 1, :), _device_of(W[mb]))
        end
        A = W[absorber]
        if kind === :vector
            A ./= Luminal.to_device(s, _device_of(A))
        elseif kind === :rows
            A ./= Luminal.to_device(s, _device_of(A))            # output row j scaled by 1/s_j
        else                                                      # :rows_gqa: one scale per KV channel
            sv = vec(Base.reshape(s, D, H ÷ KVH, KVH)[:, 1, :])
            A ./= Luminal.to_device(sv, _device_of(A))
        end
    end
    return W
end

_device_of(a) = a isa AMDGPU.ROCArray ? Luminal.AMDDevice() : Luminal.CPUDevice()

export awq_scales, awq_apply!
