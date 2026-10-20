# Compares the logits files written by logits_trace.jl: julia benchmarks/logits_trace_compare.jl reference.bin other.bin ...
using Serialization, Printf
rel(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))
ref = deserialize(ARGS[1])["hand"]
for f in ARGS
    r = deserialize(f)
    println("== ", f)
    for k in ("hand", "hand2", "hand+swiglu", "prim", "prim+C1")
        haskey(r, k) && @printf("  %-12s vs reference (cpu hand): %.2e\n", k, rel(r[k], ref))
    end
    @printf("  hand vs hand2 (same graph twice): %.2e\n", rel(r["hand"], r["hand2"]))
end
