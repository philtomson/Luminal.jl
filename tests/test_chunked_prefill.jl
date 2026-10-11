# Chunked prefill (src/Decoding.jl `_prefill_chunked!`): a prompt longer than the chunk goes through
# multi-token steps against the cache; the logits of the first token and the generation must match the
# one-shot prefill (up to the Float16 cache's rounding). GPU only (Float16 cache).
using Test
using Luminal
using Luminal.NN

if Luminal.get_device() isa CPUDevice
    @info "no GPU: skipping the chunked prefill test"
else
    REF = joinpath(@__DIR__, "data", "qwen35_tiny")
    @testset "chunked prefill == one-shot prefill" begin
        model = model_template(REF)
        s = LlamaSession(model, nothing, REF; max_seq=160, decode_weights=Float32, device=Luminal.get_device(), kv_dtype=Float16)
        V = model.config.vocab_size
        run(prompt, chunk) = withenv("LUMINAL_PREFILL_CHUNK" => string(chunk)) do
            ll = Vector{Float32}[]; sl = Matrix{Float32}[]
            g = only(generate_ids(s, [prompt]; max_new_tokens=5, last_logits=ll, step_logits=sl))
            (g, only(ll), sl)
        end
        relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))
        for (n, chunk) in ((50, 16), (50, 20), (50, 64), (32, 16), (17, 16))
            prompt = mod.(7 .* collect(1:n) .+ 3, V)
            g0, l0, s0 = run(prompt, 0)
            g1, l1, s1 = run(prompt, chunk)
            @test relerr(l1, l0) < 1e-2                         # prompt logits
            @test g1[1] == g0[1]
            @test all(relerr(vec(a), vec(b)) < 1e-2 for (a, b) in zip(s1, s0))   # the decode steps that follow
        end
    end
end
