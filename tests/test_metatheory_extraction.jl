#!/usr/bin/env julia
# Debug extraction - why does it pick Luminal_ReLU instead of Luminal_Max?

using Pkg
Pkg.develop(path="/tmp/Metatheory.jl")

using Metatheory
using Metatheory.EGraphs

println("=" ^ 80)
println("Extraction Debug")
println("=" ^ 80)

# Define functions  
Luminal_ReLU(x) = :(Luminal_ReLU($x))
Luminal_Max(x, y) = :(Luminal_Max($x, $y))

# Create and saturate
theory = @theory begin
    Luminal_ReLU(~x) --> Luminal_Max(~x, 0)
end

eg = EGraph(:(Luminal_ReLU(a)))
println("Before saturation:")
println(eg)

saturate!(eg, theory)
println("\nAfter saturation:")
println(eg)

# Manual inspection of e-classes
println("\n--- E-Class Contents ---")
for (id, eclass) in eg.classes
    println("Class $id ($(length(eclass.nodes)) nodes):")
    for (i, node) in enumerate(eclass.nodes)
        expr = Metatheory.EGraphs.to_expr(eg, node)
        println("  Node $i: $expr (size: $(length(string(expr))))")
    end
end

# Try different cost functions
println("\n--- Extraction with Different Cost Functions ---")

using Metatheory.EGraphs: astsize, astsize_inv

result_astsize = extract!(eg, astsize)
println("astsize: ", result_astsize)

eg2 = EGraph(:(Luminal_ReLU(a)))
saturate!(eg2, theory)
result_inv = extract!(eg2, astsize_inv)
println("astsize_inv: ", result_inv)

# Custom cost function that prefers Luminal_Max
function prefer_max(n, g, costs)
    if !Metatheory.VecExprModule.v_isexpr(n)
        return 1.0
    end
    
    op_hash = Metatheory.VecExprModule.v_head(n)
    op = Metatheory.EGraphs.get_constant(g, op_hash)
    
    if op == :Luminal_Max
        return 1.0  # Prefer Luminal_Max
    elseif op == :Luminal_ReLU
        return 100.0  # Discourage Luminal_ReLU
    else
        return 1.0 + sum(costs)
    end
end

eg3 = EGraph(:(Luminal_ReLU(a)))
saturate!(eg3, theory)
result_custom = extract!(eg3, prefer_max)
println("prefer_max: ", result_custom)

println("\n" * "=" ^ 80)
