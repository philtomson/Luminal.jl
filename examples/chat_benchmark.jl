# Chat latency benchmark in the shape of upstream Luminal's llm_chat benchmark
# (examples/llm_chat/validation/benchmark-128-*): batch 1, a chat prompt of exactly
# `input` tokens including the chat template (a story request padded with " quiet"),
# exactly `output` generated tokens (end-of-sequence ignored), greedy, one full
# warm-up request, then the median of `reps` requests.
#
#   julia --project=. examples/chat_benchmark.jl <model_dir> [f32|f16|int8|int4|int4_mixed|int4_mixed_plus] [input] [output] [reps]
#
# TTFT = time to the first generated token (prefill + one greedy pick; upstream's
# also includes one decode step). TPOT = (time for `output` tokens - TTFT) /
# (output - 1). Load = LlamaSession creation (reading weights); cold = the first
# request, which also compiles the graphs.
using Luminal, Luminal.NN, Printf, Statistics

dir = ARGS[1]
wd = let a = get(ARGS, 2, "f32")
    get(Dict("f32" => Float32, "f16" => Float16, "int8" => Int8, "int4" => Luminal.Int4), a, nothing) |>
        t -> t === nothing ? Symbol(a) : t          # else a preset name, e.g. int4_mixed
end
n_in = parse(Int, get(ARGS, 3, "128"))
n_out = parse(Int, get(ARGS, 4, "128"))
reps = parse(Int, get(ARGS, 5, "3"))

tok = LlamaTokenizer(dir)
empty!(tok.eos_ids)                      # generate exactly n_out tokens, as upstream does
cfg = llama_config(dir)
model = Llama(Graph(), nothing; cfg...)

# A story request padded to exactly n_in tokens (BOS and chat template included).
function padded_prompt()
    for k in 0:4n_in
        p = chat_prompt(tok, "Tell me a short story about a lighthouse keeper." * " quiet"^k)
        n = length(Luminal.encode(tok, p; bos=true))
        n == n_in && return p
        n > n_in && error("cannot pad the prompt to exactly $n_in tokens")
    end
end
prompt = padded_prompt()

t_load = @elapsed s = LlamaSession(model, tok, dir; max_seq=n_in + n_out + 8, decode_weights=wd)
t_cold = @elapsed generate(s, prompt; max_new_tokens=n_out)
ttft = [@elapsed(generate(s, prompt; max_new_tokens=1)) for _ in 1:reps]
full = [@elapsed(generate(s, prompt; max_new_tokens=n_out)) for _ in 1:reps]
tpot = (median(full) - median(ttft)) / (n_out - 1)

@printf("%s, %s decode weights, %d in / %d out, batch 1, %s\n", basename(abspath(dir)), wd, n_in, n_out, get_device())
@printf("load %.1f s   cold first request %.1f s (includes compilation)\n", t_load, t_cold)
@printf("TTFT %.1f ms (%.1f-%.1f)   TPOT %.2f ms/token (%.1f tok/s)   full request %.2f s\n",
        1e3median(ttft), 1e3minimum(ttft), 1e3maximum(ttft), 1e3tpot, 1 / tpot, median(full))
