# Validate a Llama checkpoint against Hugging Face transformers:
#   python3 examples/llama_reference.py <model_dir> <ref_dir>
#   julia --project=. examples/llama_validate.jl <model_dir> <ref_dir> [f32,f16,int8]
# For each decode weight type: prefill logits at every prompt position (first
# type only), greedy continuations against HF's, batched == single-prompt
# generation, and an estimate of decode time per token.
using Luminal, Luminal.NN, JSON3, Printf
dir, ref = ARGS[1], ARGS[2]
wtypes = Dict("f32" => Float32, "f16" => Float16, "int8" => Int8)
modes = split(get(ARGS, 3, "f32,f16,int8"), ",")
idx = JSON3.read(read(joinpath(ref, "index.json"), String))
tok = LlamaTokenizer(dir)
cfg = llama_config(dir)
model = Llama(Graph(), nothing; cfg...)
dev = get_device()
relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

for (mi, mode) in enumerate(modes)
    t = @elapsed s = LlamaSession(model, tok, dir; max_seq=512, decode_weights=wtypes[mode], device=dev)
    @printf "\n== %s decode weights (session %.1f s)\n" mode t
    for (i, p) in enumerate(idx.prompts)
        ids = Int.(p.ids)
        @assert Luminal.encode(tok, chat_prompt(tok, String(p.prompt)); bos=true) == ids "prompt tokens differ"
        if mi == 1
            # Prefill logits at every prompt position against HF (Float32 GEMM)
            pf = Luminal.Decoding._prefill_graph(s, Luminal.Decoding._prefill_bucket(length(ids)), 1)
            x = zeros(Float32, Luminal.Decoding._prefill_bucket(length(ids)), 1); x[1:length(ids)] .= ids
            lg = Array(pf.exec(Dict{Int,Any}(pf.input_id => x); device=dev)[pf.out_id])[:, 1:length(ids), 1]
            hf = Base.reshape(collect(reinterpret(Float32, read(joinpath(ref, "logits_$(i-1).f32")))), reverse(Int.(p.logits_shape))...)
            @printf "prompt %d: prefill logits relerr %.2e, argmax agree %d/%d\n" i relerr(lg, hf) count(argmax(lg, dims=1) .== argmax(hf, dims=1)) length(ids)
        end
        t = @elapsed out = generate(s, chat_prompt(tok, String(p.prompt)); max_new_tokens=length(p.generated))
        hf_text = String(p.text)
        println("  ", out == hf_text ? "SAME as HF" : "DIFFERS from HF", " (", round(t, digits=2), " s): ", repr(out))
        out == hf_text || println("  HF: ", repr(hf_text))
    end
    prompts = [chat_prompt(tok, String(p.prompt)) for p in idx.prompts]
    batched = generate(s, prompts; max_new_tokens=40)
    single = [generate(s, p; max_new_tokens=40) for p in prompts]
    println("batched == single: ", batched == single)
    # decode speed: 64 new tokens for one prompt, minus prefill (measured with 1 token)
    t1 = @elapsed generate(s, prompts[1]; max_new_tokens=1)
    tn = minimum(@elapsed(generate(s, "Count from one to one hundred: one, two, three,"; max_new_tokens=64)) for _ in 1:2)
    t1b = @elapsed generate(s, "Count from one to one hundred: one, two, three,"; max_new_tokens=1)
    @printf "decode ≈ %.1f ms/token (64 tokens %.2f s, prefill %.2f s)\n" (tn - t1b) / 63 * 1000 tn t1b
    s = nothing; GC.gc()
end
