# MTP speculative decoding (src/Speculative.jl) on the tiny Qwen3.5 model with a *random* MTP head:
# whatever the head drafts, greedy output must equal plain greedy decoding. Wrong drafts exercise the
# rollback of the recurrent states; forcing the draft to the true next token exercises the accept path.
using Test
using Luminal
using Luminal.NN
using JSON3

const REF = joinpath(@__DIR__, "data", "qwen35_tiny")
const IDX = JSON3.read(read(joinpath(REF, "index.json"), String))
refarr(name) = Base.reshape(collect(reinterpret(Float32, read(joinpath(REF, "$name.f32")))),
                            reverse(Int.(IDX.arrays[Symbol(name)]))...)

devices = Any[CPUDevice()]
Luminal.get_device() isa CPUDevice || push!(devices, Luminal.get_device())

# random `mtp.*` weights shaped as the head's registered tensors
function random_mtp_weights(model, dev)
    g = Luminal.Graph(); reg = Luminal.NN.WeightRegistry()
    Luminal.NN.Qwen35MTP(g, reg; model.config...)
    W = Dict{String,Any}()
    for (name, id) in reg.mapping
        startswith(name, "mtp.") || continue
        dims = Tuple(Int.(Luminal.realized_dims(g.shapes[id])))
        # norm weights are stored as (w - 1): keep them small
        scale = occursin("norm", name) ? 0.1f0 : 0.3f0
        W[name] = Luminal.to_device(scale .* randn(Float32, dims...), dev)
    end
    return W
end

@testset "MTP speculative decoding == greedy ($(nameof(typeof(dev))))" for dev in devices
    model = model_template(REF)
    ids = Int.(refarr("ids"))
    prompt = ids[1:6, 1]
    s = LlamaSession(model, nothing, REF; max_seq=48, decode_weights=Float32, device=dev)
    merge!(s.weights, random_mtp_weights(model, dev))
    plain = only(generate_ids(s, [prompt]; max_new_tokens=14))
    @test length(plain) == 14
    d = MTPDrafter(s)

    # 1. the random head's own drafts (mostly wrong): rollback on nearly every cycle
    st = Dict{Symbol,Int}()
    spec = generate_ids_speculative(s, d, prompt; max_new_tokens=14, stats=st)
    @test spec == plain
    @test st[:emitted] == 14

    # 2. oracle drafts: every draft is the true next token, so every cycle accepts (2 tokens per cycle)
    st = Dict{Symbol,Int}()
    oracle(n, draft) = n + 1 <= length(plain) ? plain[n + 1] : draft
    spec = generate_ids_speculative(s, d, prompt; max_new_tokens=14, stats=st, draft_hook=oracle)
    @test spec == plain
    @test st[:accepted] == st[:cycles] && st[:cycles] >= 6

    # 3. a mix: right, wrong, right, wrong... (accepts and rollbacks interleaved)
    st = Dict{Symbol,Int}()
    flip = Ref(0)
    mixed(n, draft) = (flip[] += 1; isodd(flip[]) && n + 1 <= length(plain) ? plain[n + 1] : mod(draft + 1, model.config.vocab_size))
    spec = generate_ids_speculative(s, d, prompt; max_new_tokens=14, stats=st, draft_hook=mixed)
    @test spec == plain
    @test 0 < st[:accepted] < st[:cycles]

    # 4. a different prompt, and the same drafter again (its cache restarts with each prompt)
    prompt2 = ids[2:5, 2]
    plain2 = only(generate_ids(s, [prompt2]; max_new_tokens=10))
    @test generate_ids_speculative(s, d, prompt2; max_new_tokens=10) == plain2
    @test generate_ids_speculative(s, d, prompt2; max_new_tokens=10, draft_hook=(n, dr) -> n + 1 <= length(plain2) ? plain2[n + 1] : dr) == plain2

    # 6. two drafts per cycle (a 3-token verify step with two state snapshots): same output for any drafts
    d3 = MTPDrafter(s)
    st = Dict{Symbol,Int}()
    @test generate_ids_speculative(s, d3, prompt; max_new_tokens=14, stats=st, ndraft=2) == plain
    @test st[:emitted] == 14
    st = Dict{Symbol,Int}()
    @test generate_ids_speculative(s, d3, prompt; max_new_tokens=14, stats=st, ndraft=2, draft_hook=oracle) == plain
    @test st[:accepted2] >= st[:cycles] - 1 && st[:cycles] >= 4   # all 3 tokens per cycle (bar the last, past the end of `plain`)
    flip2 = Ref(0)
    mixed2(n, draft) = (flip2[] += 1; flip2[] % 3 != 0 && n + 1 <= length(plain) ? plain[n + 1] : mod(draft + 1, model.config.vocab_size))
    st = Dict{Symbol,Int}()
    @test generate_ids_speculative(s, d3, prompt; max_new_tokens=14, stats=st, ndraft=2, draft_hook=mixed2) == plain
    @test 0 < st[:accepted] && st[:accepted2] < st[:cycles]          # all three outcomes: reject, 2 tokens, 3 tokens
    @test generate_ids_speculative(s, d3, prompt2; max_new_tokens=10, ndraft=2) == plain2

    # 5. a restricted draft vocabulary only changes which tokens get drafted, never the output
    d2 = MTPDrafter(s; draft_vocab=model.config.vocab_size ÷ 2)
    st = Dict{Symbol,Int}()
    @test generate_ids_speculative(s, d2, prompt; max_new_tokens=14, stats=st) == plain
end
