# Loading weights straight into reduced precision (load_weights_to_dict's
# `quantize`, LlamaSession's `weight_storage`), on the tiny Qwen3.5 model: the
# matmul weights become Q4Weight / QuantWeight / HalfWeight as they are read
# (the embedding table, and weights whose input dim doesn't fit the format,
# stay Float32), and a session on them -- prefill through the dequantize-then-
# GEMM path, decode through the GEMV kernels -- gives the same logits and tokens
# as a Float32 session holding exactly their dequantized values. CPU and GPU.
using Test
using Luminal
using Luminal.NN

const REF = joinpath(@__DIR__, "data", "qwen35_tiny")
relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))
devices = Any[CPUDevice()]
Luminal.get_device() isa CPUDevice || push!(devices, Luminal.get_device())

prompts = [[3, 14, 15, 9, 2, 6, 30], [11, 40, 7, 1, 25]]

@testset "quantized loading ($(nameof(typeof(dev))), $T)" for dev in devices, T in (Luminal.Int4, Int8, Float16)
    S = Luminal.storage_type(T)
    W = load_weights_to_dict(REF; device=dev, quantize=_ -> T)
    L0 = "model.language_model.layers"
    @test W["$L0.3.self_attn.q_proj.weight"] isa S                 # (128, 32): fits every format
    @test W["lm_head.weight"] isa S
    @test W["model.language_model.embed_tokens.weight"] isa AbstractArray{Float32}   # read by Gather
    # linear_attn.out_proj is (32, 48): 48 isn't a multiple of 32 (int4) or 16 (int8)
    @test W["$L0.0.linear_attn.out_proj.weight"] isa (T === Float16 ? S : AbstractArray{Float32})

    # the same values in Float32
    W32 = Dict{String,Any}(k => (v isa S ? Luminal.dequantize(v) : v) for (k, v) in W)
    model = model_template(REF)
    run(ws) = begin
        s = LlamaSession(model, nothing, REF; max_seq=32, device=dev, weights=ws,
                         decode_weights=T === Float16 ? Float16 : Float32)
        last = Vector{Float32}[]; steps = Matrix{Float32}[]
        gen = generate_ids(s, prompts; max_new_tokens=5, last_logits=last, step_logits=steps)
        gen, last, steps
    end
    gq, lq, sq = run(W)
    g32, l32, s32 = run(W32)
    @test all(relerr(lq[b], l32[b]) < 1e-4 for b in 1:2)                 # prefill: dequantize + GEMM
    @test all(relerr(sq[k], s32[k]) < 1e-4 for k in eachindex(s32))      # decode: GEMV kernels
    @test gq == g32
end

@testset "weight_storage session" begin
    dev = devices[end]
    s = LlamaSession(model_template(REF), nothing, REF; max_seq=32, device=dev, weight_storage=Luminal.Int4)
    @test s.weights["lm_head.weight"] isa Luminal.Q4Weight && s.decode_weights === Luminal.Int4
    @test length(only(generate_ids(s, [prompts[1]]; max_new_tokens=4))) == 4
end
