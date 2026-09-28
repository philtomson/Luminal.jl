# Gated DeltaNet (Qwen3.5 / Qwen3.6 linear attention) against transformers, on
# the tiny random model in tests/data/qwen35_tiny (see examples/qwen35_reference.py):
# layer 0's output, convolution state and recurrent state after a batch-2
# prefill from empty states and after each of 3 decode steps continuing from
# them. Interpreter and compile(), CPU and GPU.
using Test
using Luminal
using Luminal.NN
using JSON3

const REF = joinpath(@__DIR__, "data", "qwen35_tiny")
const IDX = JSON3.read(read(joinpath(REF, "index.json"), String))
# a C-order (a, b, ...) array read as the column-major (..., b, a) array
refarr(name) = Base.reshape(collect(reinterpret(Float32, read(joinpath(REF, "$name.f32")))),
                            reverse(Int.(IDX.arrays[Symbol(name)]))...)

devices = Any[CPUDevice()]
Luminal.get_device() isa CPUDevice || push!(devices, Luminal.get_device())

const CFG = JSON3.read(read(joinpath(REF, "config.json"), String))
const TC = haskey(CFG, :text_config) ? CFG[:text_config] : CFG
const PFX = "model.language_model.layers.0.linear_attn"
dn_kw = (nk=TC[:linear_num_key_heads], nv=TC[:linear_num_value_heads], dk=TC[:linear_key_head_dim],
         dv=TC[:linear_value_head_dim], K=TC[:linear_conv_kernel_dim], epsilon=Float32(TC[:rms_norm_eps]))
relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

# one graph for prefill (S tokens, fresh) or decode (1 token)
function dn_graph(S, B, fresh)
    g = Graph(); reg = WeightRegistry()
    dn = GatedDeltaNet(TC[:hidden_size], g, reg, PFX; dn_kw...)
    cs, rs = deltanet_state_shapes(dn, B)
    x = Luminal.tensor(g, [TC[:hidden_size], S, B])
    conv = Luminal.tensor(g, cs); rec = Luminal.tensor(g, rs)
    y = dn(x, conv, rec; fresh=fresh)
    return g, reg, x, conv, rec, y
end

W = load_weights_to_dict(REF; device=CPUDevice())

@testset "Gated DeltaNet vs transformers ($(nameof(typeof(dev))), $mode)" for dev in devices, mode in (:interpreter, :compiled)
    mode === :interpreter && !(dev isa CPUDevice) && continue
    B = 2
    x0 = refarr("dn_in_0")                                   # (hidden, S, B)
    S = size(x0, 2)
    # states live on the device across calls, updated in place
    gp, regp, xp, convp, recp, yp = dn_graph(S, B, true)
    cs, rs = deltanet_state_shapes(GatedDeltaNet(TC[:hidden_size], Graph(), nothing; dn_kw...), B)
    conv_buf = Luminal.zero_tensor(dev, Float32, cs...); rec_buf = Luminal.zero_tensor(dev, Float32, rs...)
    run(g, reg, x, conv, rec, y, xv) = begin
        load_weights!(g, reg, W; device=dev)
        ins = Dict{Int,Any}(x.id => Luminal.to_device(xv, dev), conv.id => conv_buf, rec.id => rec_buf)
        mode === :interpreter ? execute(g, y.id, ins, CPUDevice()) :
            Array(compile(g; device=dev, retain=[y.id], free_intermediates=false)(ins; device=dev)[y.id])
    end
    # HF conv state (B, C, K) reads as (K, C, B); ours is (C, K, B). HF recurrent
    # state (B, H, dk, dv) reads as (dv, dk, H, B), which is ours.
    check(step, y) = begin
        @test relerr(y, refarr("dn_out_$step")) < 1e-4
        @test relerr(Array(conv_buf), permutedims(refarr("conv_$step"), (2, 1, 3))) < 1e-5
        @test relerr(Array(rec_buf), refarr("rec_$step")) < 1e-4
    end
    fill!(conv_buf, 7f0); fill!(rec_buf, 7f0)                 # fresh must ignore the buffers' contents
    check(0, run(gp, regp, xp, convp, recp, yp, x0))
    gd = dn_graph(1, B, false)
    for step in 1:3
        check(step, run(gd..., refarr("dn_in_$step")))
    end
end
