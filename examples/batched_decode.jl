# Batched decode throughput on a TinyLlama checkpoint: time one decode step for
# several batch sizes and report per-step latency and aggregate tokens/s.
#
#   julia --project=. examples/batched_decode.jl path/to/TinyLlama [f16|int8|f32] [1,2,4,8]
using Luminal
using Luminal.NN
using Printf

model_dir = ARGS[1]
wdtype = get(Dict("f16" => Float16, "int8" => Int8, "f32" => Float32), get(ARGS, 2, "int8"), Int8)
batches = parse.(Int, split(get(ARGS, 3, "1,2,4,8"), ","))
device = get_device()
cfg = (vocab_size=32000, hidden=2048, n_layers=22, n_heads=32, n_kv_heads=4, intermediate=5632)
max_seq, ctx = 512, 128        # every sequence decodes at ~position 128

weights = load_weights_to_dict(model_dir; device=device)
println("TinyLlama, $(wdtype) weights, $(device)")
@printf("%6s %12s %12s\n", "batch", "ms/step", "tok/s")
for B in batches
    g = Graph(); reg = WeightRegistry()
    m = Llama(g, reg; cfg..., rope_base=10000f0)
    idg = build_llama_decode_step!(m, g, 0; max_seq=max_seq, batch=B, rope_base=10000f0)
    load_weights!(g, reg, weights; device=device)
    retain = vcat(idg.logits_id, idg.new_self_k_ids, idg.new_self_v_ids, idg.token_input_id,
                  idg.pos_input_id, idg.self_k_ids, idg.self_v_ids)
    exec = compile(g; device=device, retain=retain, free_intermediates=false,
                   weight_dtype=wdtype, capture=device isa Luminal.AMDDevice)
    cache = LlamaKVCacheState(cfg.n_layers, cfg.n_kv_heads, cfg.hidden ÷ cfg.n_heads;
                              batch=B, max_seq=max_seq, device=device)
    cache.positions .= ctx .+ (0:B-1)          # different positions per sequence
    tokens = collect(100:100+B-1)
    times = Float64[]
    for i in 1:40
        t = @elapsed begin
            logits = llama_decode_step!(exec, idg, cache, tokens; device=device)
            host = Array{Float32}(logits)
            tokens = [argmax(view(host, :, 1, b)) - 1 for b in 1:B]
        end
        i > 5 && push!(times, t)
    end
    ms = sort(times)[length(times) ÷ 2] * 1000
    @printf("%6d %12.2f %12.1f\n", B, ms, B * 1000 / ms)
end
