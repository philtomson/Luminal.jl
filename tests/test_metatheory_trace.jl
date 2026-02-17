#!/usr/bin/env julia
# Trace the matching process to find where it fails

using Pkg
Pkg.develop(path="/tmp/Metatheory.jl")

using Metatheory
using Metatheory.EGraphs
using Metatheory.Patterns

println("=" ^ 80)
println("Tracing Pattern Matching Failure")
println("=" ^ 80)

# Define custom function
Luminal_ReLU(x) = :(Luminal_ReLU($x))
Luminal_Max(x, y) = :(Luminal_Max($x, $y))

# Create the pattern
pvars = Symbol[]
slots = Symbol[]
pat = Metatheory.Syntax.makepattern(:(Luminal_ReLU(~x)), pvars, slots, Main)

println("\n--- Pattern Details ---")
println("Pattern: ", pat)
println("head_hash: ", pat.head_hash)
println("name_hash: ", pat.name_hash)

# Create e-graph
eg = EGraph(:(Luminal_ReLU(a)))

println("\n--- E-Graph Constants ---")
for (h, v) in eg.constants
    println("  $h => $v")
end

# Check has_constant for both hashes
println("\n--- Constant Check ---")
using Metatheory.EGraphs: has_constant
println("has_constant(eg, pat.head_hash): ", has_constant(eg, pat.head_hash))
println("has_constant(eg, pat.name_hash): ", has_constant(eg, pat.name_hash))

# The issue might be in check_constant_exprs!
# Let's manually trace what it does

println("\n--- Manual Trace of check_constant_exprs! ---")
println("pat.type: ", pat.type)
println("PAT_EXPR: ", Metatheory.Patterns.PAT_EXPR)
println("Is PAT_EXPR: ", pat.type === Metatheory.Patterns.PAT_EXPR)

if pat.type === Metatheory.Patterns.PAT_EXPR
    println("\nChecking head...")
    println("  pat.head isa Pat: ", pat.head isa Pat)
    println("  pat.head: ", pat.head)
    
    if !(pat.head isa Pat)
        println("  Should check: has_constant(g, pat.head_hash) || has_constant(g, pat.name_hash)")
        println("    pat.head_hash $(pat.head_hash): ", has_constant(eg, pat.head_hash))
        println("    pat.name_hash $(pat.name_hash): ", has_constant(eg, pat.name_hash))
        println("    Result: ", has_constant(eg, pat.head_hash) || has_constant(eg, pat.name_hash))
    end
end

println("\n" * "=" ^ 80)
