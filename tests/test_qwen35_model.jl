# The whole Qwen3.5 / Qwen3.6 model (3 Gated DeltaNet layers + 1 gated attention
# layer) against transformers, on the tiny model in tests/data/qwen35_tiny,
# through LlamaSession: a batch-2 prefill (padded to a bucket, the recurrent
# layers stopping at each prompt's length) and 3 greedy decode steps; then
# batches of prompts with different lengths against each prompt alone. CPU and GPU.
using Test
using Luminal
using Luminal.NN
using JSON3

const REF = joinpath(@__DIR__, "data", "qwen35_tiny")
const IDX = JSON3.read(read(joinpath(REF, "index.json"), String))
refarr(name) = Base.reshape(collect(reinterpret(Float32, read(joinpath(REF, "$name.f32")))),
                            reverse(Int.(IDX.arrays[Symbol(name)]))...)
relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

devices = Any[CPUDevice()]
Luminal.get_device() isa CPUDevice || push!(devices, Luminal.get_device())

@testset "Qwen3.5 model vs transformers ($(nameof(typeof(dev))))" for dev in devices
    model = model_template(REF)
    @test model isa Qwen35 && [l.kind for l in model.layers] == [:linear, :linear, :linear, :full]
    s = LlamaSession(model, nothing, REF; max_seq=32, decode_weights=Float32, device=dev)
    ids = Int.(refarr("ids"))                                   # (S, B)
    prompts = [ids[:, b] for b in 1:size(ids, 2)]
    last = Vector{Float32}[]; steps = Matrix{Float32}[]
    gen = generate_ids(s, prompts; max_new_tokens=4, last_logits=last, step_logits=steps)
    ref0 = refarr("logits_0")                                   # (V, S, B)
    for b in 1:2
        @test relerr(last[b], ref0[:, end, b]) < 1e-4
    end
    for step in 1:3
        @test relerr(steps[step], refarr("logits_$step")[:, 1, :]) < 1e-4
    end
    toks = [Int.(t) for t in IDX.decode_tokens]                 # tokens fed at steps 1..3
    for b in 1:2
        @test gen[b][1:3] == [toks[k][b] for k in 1:3]
        @test gen[b][4] == argmax(refarr("logits_3")[:, 1, b]) - 1
    end

    # different prompt lengths in one batch (padding, lens) == each alone; and a
    # second call on the same session (the cache and states are reused)
    ps = [ids[1:4, 1], ids[:, 2], ids[2:6, 1]]
    batched = generate_ids(s, ps; max_new_tokens=6)
    single = [only(generate_ids(s, [p]; max_new_tokens=6)) for p in ps]
    @test batched == single
end
