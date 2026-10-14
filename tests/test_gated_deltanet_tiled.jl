# Gated DeltaNet with `out_tiled`: the out-projection holds the GGUF file's (tiled) head order, the gated norm writes its heads there.
# Same layer, reference and tolerance as test_gated_deltanet.jl; the HF out_proj columns are permuted to the tiled order.
using Test
using Luminal
using Luminal.NN
using JSON3

const REF = joinpath(@__DIR__, "data", "qwen35_tiny")
const IDX = JSON3.read(read(joinpath(REF, "index.json"), String))
refarr(name) = Base.reshape(collect(reinterpret(Float32, read(joinpath(REF, "$name.f32")))), reverse(Int.(IDX.arrays[Symbol(name)]))...)
const CFG = JSON3.read(read(joinpath(REF, "config.json"), String))
const TC = haskey(CFG, :text_config) ? CFG[:text_config] : CFG
const PFX = "model.language_model.layers.0.linear_attn"
dn_kw = (nk=TC[:linear_num_key_heads], nv=TC[:linear_num_value_heads], dk=TC[:linear_key_head_dim],
         dv=TC[:linear_value_head_dim], K=TC[:linear_conv_kernel_dim], epsilon=Float32(TC[:rms_norm_eps]))

@testset "Gated DeltaNet with a tiled out-projection ($(nameof(typeof(dev))))" for dev in (Luminal.get_device(),)
    nk, nv, dv = dn_kw.nk, dn_kw.nv, dn_kw.dv
    @test nv > nk && nv % nk == 0
    G = nv ÷ nk
    # tiled column of HF column o (0-based): head h = k G + g goes to slot g nk + k
    pv = [((g * nk + k) * dv + d) for k in 0:nk-1 for g in 0:G-1 for d in 0:dv-1]
    W = load_weights_to_dict(REF; device=CPUDevice())
    key = "$PFX.out_proj.weight"
    Wh = Array(W[key]); Wt = similar(Wh)
    Wt[:, pv .+ 1] = Wh                                         # Wt[:, pv[o]] = Wh[:, o]
    W[key] = Wt
    g = Graph(); reg = WeightRegistry()
    dn = GatedDeltaNet(TC[:hidden_size], g, reg, PFX; dn_kw..., out_tiled=true)
    B = 2
    cs, rs = deltanet_state_shapes(dn, B)
    x0 = refarr("dn_in_0"); S = size(x0, 2)
    x = Luminal.tensor(g, [TC[:hidden_size], S, B])
    conv = Luminal.tensor(g, cs); rec = Luminal.tensor(g, rs); lens = Luminal.tensor(g, [B])
    y = dn(x, conv, rec, lens; fresh=true)
    load_weights!(g, reg, W; device=dev)
    ins = Dict{Int,Any}(x.id => Luminal.to_device(x0, dev), conv.id => Luminal.zero_tensor(dev, Float32, cs...),
                        rec.id => Luminal.zero_tensor(dev, Float32, rs...), lens.id => fill(Float32(S), B))
    out = Array(compile(g; device=dev, retain=[y.id], free_intermediates=false)(ins; device=dev)[y.id])
    ref = refarr("dn_out_0")
    @test maximum(abs.(out .- ref)) / maximum(abs.(ref)) < 1e-4
end
