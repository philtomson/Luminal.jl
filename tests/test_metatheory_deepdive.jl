#!/usr/bin/env julia
# Deep dive: Compare e-graph representation of custom vs built-in functions

using Pkg
Pkg.activate("Julia")

using Metatheory
using Metatheory.EGraphs
using Metatheory.VecExprModule

println("=" ^ 80)
println("Deep Dive: E-Graph Representation")
println("=" ^ 80)

# Define custom function
Luminal_ReLU(x) = :(Luminal_ReLU($x))

# Test 1: Inspect e-graph for sin(a)
println("\n--- Built-in Function: sin(a) ---")
eg_sin = EGraph(:(sin(a)))
println("E-Graph: ", eg_sin)
println("\nConstants dict:")
for (k, v) in eg_sin.constants
    println("  hash=$k => $v (type: $(typeof(v)))")
end

# Test 2: Inspect e-graph for Luminal_ReLU(a)
println("\n--- Custom Function: Luminal_ReLU(a) ---")
eg_custom = EGraph(:(Luminal_ReLU(a)))
println("E-Graph: ", eg_custom)
println("\nConstants dict:")
for (k, v) in eg_custom.constants
    println("  hash=$k => $v (type: $(typeof(v)))")
end

# Test 3: Inspect e-graph for +(a, b)
println("\n--- Built-in Operator: +(a, b) ---")
eg_plus = EGraph(:(a + b))
println("E-Graph: ", eg_plus)
println("\nConstants dict:")
for (k, v) in eg_plus.constants
    println("  hash=$k => $v (type: $(typeof(v)))")
end

# Test 4: Look at the VecExpr structures
println("\n--- VecExpr Comparison ---")
function inspect_vecexpr(eg, label)
    println("\n$label:")
    for (id, eclass) in eg.classes
        for (i, node) in enumerate(eclass.nodes)
            head_val = v_head(node)
            head_constant = get(eg.constants, head_val, "NOT FOUND")
            println("  Node $i:")
            println("    v_flags: $(v_flags(node)) (isexpr=$(v_isexpr(node)), iscall=$(v_iscall(node)))")
            println("    v_head: $head_val => $head_constant")
            println("    v_signature: $(v_signature(node))")
            println("    v_arity: $(v_arity(node))")
        end
    end
end

inspect_vecexpr(eg_sin, "sin(a)")
inspect_vecexpr(eg_custom, "Luminal_ReLU(a)")
inspect_vecexpr(eg_plus, "a + b")

println("\n" * "=" ^ 80)
