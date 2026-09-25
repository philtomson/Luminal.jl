# Phi-3 architecture demo with random weights: builds a scaled-down Phi-3
# (grouped-query attention, RoPE), shows the Hugging Face weight mapping, and runs
# a compiled prefill. Real checkpoints load with `load_weights!(graph, reg, dir)`.
#
#   julia --project=. examples/phi3.jl [cpu|gpu]
using Luminal
using Luminal.NN
using Printf

device = "cpu" in ARGS ? CPUDevice() : get_device()
graph = Graph(); reg = WeightRegistry()
phi3 = Phi3(graph, reg; vocab_size=32064, hidden=768, n_layers=4, n_heads=12, n_kv_heads=4,
            intermediate=2048)
println("Registry maps $(length(reg.mapping)) Hugging Face keys, e.g. ",
        "model.layers.0.self_attn.q_proj.weight => node ",
        reg.mapping["model.layers.0.self_attn.q_proj.weight"])

seq_len = 16
input = tensor(graph, [seq_len, 1])            # (tokens, batch)
logits = phi3(input, 0)                        # (vocab, tokens, batch)
weights = Dict{String,Any}(k => 0.02f0 .* randn(Float32, Tuple(Luminal.realized_dims(graph.shapes[id]))...)
                           for (k, id) in reg.mapping)
load_weights!(graph, reg, weights; device=device)
exec = compile(graph; device=device, retain=[logits.id])

ids = Float32.(rand(0:32063, seq_len, 1))
exec(Dict{Int,Any}(input.id => ids); device=device)          # warm up
t = @elapsed out = Array(exec(Dict{Int,Any}(input.id => ids); device=device)[logits.id])
@printf("prefill of %d tokens: %.2f ms, logits %s\n", seq_len, t * 1000, size(out))
