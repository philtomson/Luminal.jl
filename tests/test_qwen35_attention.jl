# Qwen3.5 / Qwen3.6 gated full attention (output gate, partial RoPE, (1 + w)
# QK-norm) against transformers, on layer 3 of the tiny model in
# tests/data/qwen35_tiny: a batch-2 prefill, then 3 cached decode steps through
# DecodeAttention at positions 7, 8, 9. Compiled, CPU and GPU.
using Test
using Luminal
using Luminal.NN
using JSON3

const REF = joinpath(@__DIR__, "data", "qwen35_tiny")
const IDX = JSON3.read(read(joinpath(REF, "index.json"), String))
refarr(name) = Base.reshape(collect(reinterpret(Float32, read(joinpath(REF, "$name.f32")))),
                            reverse(Int.(IDX.arrays[Symbol(name)]))...)
const TC = let c = JSON3.read(read(joinpath(REF, "config.json"), String)); haskey(c, :text_config) ? c[:text_config] : c end
const RP = TC[:rope_parameters]
const PFX = "model.language_model.layers.3.self_attn"
const KW = (n_heads=TC[:num_attention_heads], n_kv_heads=TC[:num_key_value_heads], head_dim=TC[:head_dim],
            rotary_dim=round(Int, TC[:head_dim] * RP[:partial_rotary_factor]), rope_theta=Float32(RP[:rope_theta]),
            epsilon=Float32(TC[:rms_norm_eps]))
const MAXS = 16
relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

devices = Any[CPUDevice()]
Luminal.get_device() isa CPUDevice || push!(devices, Luminal.get_device())
W = load_weights_to_dict(REF; device=CPUDevice())

@testset "Qwen3.5 gated attention vs transformers ($(nameof(typeof(dev))))" for dev in devices
    H = TC[:hidden_size]; D = KW.head_dim; KVH = KW.n_kv_heads; B = 2
    @test KW.rotary_dim == D ÷ 4                          # partial: a quarter of each head
    x0 = refarr("at_in_0"); S = size(x0, 2)

    # prefill, returning K/V for the cache
    g = Graph(); reg = WeightRegistry()
    sa = qwen35_attention(H, g, reg, PFX; KW...)
    x = Luminal.tensor(g, [H, S, B])
    y, k, v = sa(x, 0; return_kv=true)
    load_weights!(g, reg, W; device=dev)
    r = compile(g; device=dev, retain=[y.id, k.id, v.id])(Dict{Int,Any}(x.id => x0); device=dev)
    @test relerr(Array(r[y.id]), refarr("at_out_0")) < 1e-4

    # cache (D, max_seq, KVH, B) holding the prompt's K/V; decode continues it
    kc = Luminal.zero_tensor(dev, Float32, D, MAXS, KVH, B); vc = Luminal.zero_tensor(dev, Float32, D, MAXS, KVH, B)
    kc[:, 1:S, :, :] .= r[k.id]; vc[:, 1:S, :, :] .= r[v.id]
    g = Graph(); reg = WeightRegistry()
    sa = qwen35_attention(H, g, reg, PFX; KW...)
    xd = Luminal.tensor(g, [H, 1, B]); pos = Luminal.tensor(g, [B])
    pk = Luminal.tensor(g, [D, MAXS, KVH, B]); pv = Luminal.tensor(g, [D, MAXS, KVH, B])
    yd, _, _ = llama_self_attn_cached(sa, xd, 0, pos, pk, pv; rope_base=KW.rope_theta)
    load_weights!(g, reg, W; device=dev)
    ex = compile(g; device=dev, retain=[yd.id], free_intermediates=false)
    for step in 1:3
        ins = Dict{Int,Any}(xd.id => refarr("at_in_$step"), pos.id => fill(Float32(S + step - 1), B), pk.id => kc, pv.id => vc)
        @test relerr(Array(ex(ins; device=dev)[yd.id]), refarr("at_out_$step")) < 1e-4
    end
end
