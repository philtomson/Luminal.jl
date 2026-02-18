#!/usr/bin/env julia
# Ultimate trace: print generated matching code

using Pkg
Pkg.develop(path="/home/phil/build/Metatheory.jl")

using Metatheory
using Metatheory.Patterns

println("=" ^ 80)
println("Inspecting Generated Matcher Code")
println("=" ^ 80)

# Define functions
Luminal_ReLU(x) = :(Luminal_ReLU($x))

# Create rule
println("\n--- Creating Rule for Luminal_ReLU(~x) --> Luminal_Max(~x, 0) ---")
rule = @rule Luminal_ReLU(~x) --> Luminal_Max(~x, 0)

println("\nRule created:")
println("  LHS pattern head_hash: ", rule.left.head_hash)
println("  LHS pattern name_hash: ", rule.left.name_hash)
println("  LHS pattern v_head: ", Metatheory.VecExprModule.v_head(rule.left.n))

# Show the compiled matcher (this will be huge!)
println("\n--- Examining E-Matcher Function ---")
println("Type of ematcher_left!: ", typeof(rule.ematcher_left!))

# Try to match against an e-graph manually
using Metatheory.EGraphs

eg = EGraph(:(Luminal_ReLU(a)))
println("\n--- E-Graph ---")
for (id, eclass) in eg.classes
    println("Class $id:")
    for node in eclass.nodes
        println("  Node: v_head=$(Metatheory.VecExprModule.v_head(node)), v_sig=$(Metatheory.VecExprModule.v_signature(node))")
    end
end

println("\n--- Constants in E-Graph ---")
for (h, v) in eg.constants
    println("  $h => $v")
end

# Now let me check the signature computation
println("\n--- Signature Computation ---")
println("Pattern signature: ", Metatheory.VecExprModule.v_signature(rule.left.n))

# Compute what the e-graph node signature should be
using Metatheory: maybe_quote_operation
qop = maybe_quote_operation(Luminal_ReLU)
println("maybe_quote_operation(Luminal_ReLU): ", qop, " (type: $(typeof(qop)))")
sig = hash(qop, hash(1))  # arity = 1
println("Expected signature: hash(:$qop, hash(1)) = ", sig)

println("\n" * "=" ^ 80)
