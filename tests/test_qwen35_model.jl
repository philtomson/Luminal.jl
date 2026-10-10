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

# A Float16 K/V cache (kv_dtype=Float16; the recurrent states stay Float32) gives the same
# tokens and logits to within Float16 rounding. The compiled decode graph must declare the
# cache inputs as Float16, or the executor feeds the kernel a Float32 copy and the new K/V
# never reach the cache (the logits then drift away after the first decode step).
@testset "Float16 KV cache ($(nameof(typeof(dev))))" for dev in devices
    model = model_template(REF)
    ids = Int.(refarr("ids")); prompts = [ids[:, b] for b in 1:size(ids, 2)]
    function run(kv)
        s = LlamaSession(model, nothing, REF; max_seq=32, decode_weights=Float32, device=dev, kv_dtype=kv)
        steps = Matrix{Float32}[]
        gen = generate_ids(s, prompts; max_new_tokens=8, step_logits=steps)
        gen, steps, s
    end
    g32, st32, _ = run(Float32)
    g16, st16, s16 = run(Float16)
    cache = only(values(s16.decode)).cache.self_cache
    @test [eltype(c[1]) for c in cache] == [Float32, Float32, Float32, Float16]   # states stay Float32
    @test g16 == g32
    @test all(relerr(st16[k], st32[k]) < 1e-3 for k in eachindex(st32))
end

# A 2-token verify step (speculative decoding) against two single-token steps: the same logits for
# both tokens, the same K/V and recurrent states afterwards, and the snapshot buffers hold each
# recurrent layer's state after the first token only.
@testset "multi-token verify step ($(nameof(typeof(dev))))" for dev in devices
    model = model_template(REF)
    ids = Int.(refarr("ids"))
    prompt = ids[1:5, 1]
    function prefilled()
        s = LlamaSession(model, nothing, REF; max_seq=32, decode_weights=Float32, device=dev)
        generate_ids(s, [prompt]; max_new_tokens=1)            # prefill (+ the first token's logits)
        dc = Luminal.Decoding._decode_graph(s, 1)
        s, dc
    end
    # two single-token steps
    sA, dcA = prefilled()
    toks = [7, 11]
    lA = []
    states1 = nothing
    for (n, t) in enumerate(toks)
        l = Luminal.NN.llama_decode_step!(dcA.exec, dcA.idg, dcA.cache, [t]; device=dev)
        push!(lA, Array(l)[:, 1, 1])
        if n == 1
            states1 = [(Array(copy(c[1])), Array(copy(c[2]))) for (c, lay) in zip(dcA.cache.self_cache, sA.model.layers) if Luminal.NN._is_state_layer(lay)]
        end
    end
    # one 2-token verify step from the same prefill
    sB, dcB = prefilled()
    vg = Luminal.Decoding._verify_graph(sB, 1, 2)
    lg, hid, am = Luminal.Decoding._verify_step!(sB, vg, Base.reshape(toks, 2, 1))
    lgh = Array(lg)
    @test relerr(lgh[:, 1, 1], lA[1]) < 1e-4
    @test relerr(lgh[:, 2, 1], lA[2]) < 1e-4
    @test size(Array(hid)) == (sB.model.config.hidden, 2, 1)
    @test vec(am) == [argmax(lgh[:, 1, 1]) - 1, argmax(lgh[:, 2, 1]) - 1]      # the device-side greedy picks
    for (cA, cB, lay) in zip(dcA.cache.self_cache, dcB.cache.self_cache, sA.model.layers)
        Luminal.NN._is_state_layer(lay) && continue
        @test relerr(Array(cB[1])[:, 6:7, :, :], Array(cA[1])[:, 6:7, :, :]) < 1e-4    # K of the 2 new positions
    end
    j = 0
    for (cA, cB, lay) in zip(dcA.cache.self_cache, dcB.cache.self_cache, sA.model.layers)
        Luminal.NN._is_state_layer(lay) || continue
        j += 1
        @test relerr(Array(cB[2]), Array(cA[2])) < 1e-4                                  # state after both tokens
        @test relerr(Array(vg.snaps[j][2]), states1[j][2]) < 1e-4                        # snapshot = state after token 1
        @test relerr(Array(vg.snaps[j][1]), states1[j][1]) < 1e-4
    end
end
