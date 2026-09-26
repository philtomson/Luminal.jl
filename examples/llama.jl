# Llama decode benchmark with random weights.
#
# Builds a small Llama, compiles the position-independent single-token decode
# graph (HIP-graph captured on AMD GPUs) and times it against a device-resident
# KV cache. For real checkpoints and text, see examples/llama_chat.jl.
#
#   julia --project=. examples/llama.jl [cpu|gpu]
using Luminal
using Luminal.NN
using Printf

device = "cpu" in ARGS ? CPUDevice() : get_device()
cfg = (vocab_size=32000, hidden=512, n_layers=4, n_heads=8, n_kv_heads=4, intermediate=1408)
max_seq = 256
println("Llama $(cfg) on $(device)")

graph = Graph(); reg = WeightRegistry()
llama = Llama(graph, reg; cfg..., rope_base=10000f0)
weights = Dict{String,Any}(k => 0.02f0 .* randn(Float32, Tuple(Luminal.realized_dims(graph.shapes[id]))...)
                           for (k, id) in reg.mapping)
idg = build_llama_decode_step!(llama, graph, Luminal.Sym{Int}(:pos); max_seq=max_seq, rope_base=10000f0)
load_weights!(graph, reg, weights; device=device)

retain = vcat(idg.logits_id, idg.new_self_k_ids, idg.new_self_v_ids, idg.token_input_id,
              idg.pos_input_id, idg.self_k_ids, idg.self_v_ids)
t = @elapsed exec = compile(graph; device=device, retain=retain, free_intermediates=false,
                            capture=device isa Luminal.AMDDevice)
@printf("compiled in %.1f s\n", t)

head_dim = cfg.hidden ÷ cfg.n_heads
cache = LlamaKVCacheState(cfg.n_layers, cfg.n_kv_heads, head_dim; max_seq=max_seq, device=device)
token = 1
times = Float64[]
for step in 1:100
    global token
    dt = @elapsed begin
        logits = llama_decode_step!(exec, idg, cache, token;
                                    sym_vals=Dict(:pos => cache.step_pos), device=device)
        token = argmax(view(Array{Float32}(logits), :, 1, 1)) - 1
    end
    push!(times, dt)
end
med = sort(times[3:end])[length(times[3:end]) ÷ 2] * 1000
@printf("median decode step: %.2f ms (%.0f tok/s)\n", med, 1000 / med)
