#!/usr/bin/env julia
# Final investigation: Compare pattern structures

using Pkg
Pkg.activate("Julia")

using Metatheory
using Metatheory.Patterns

println("=" ^ 80)
println("Pattern Structure Comparison")
println("=" ^ 80)

# Define custom function
Luminal_ReLU(x) = :(Luminal_ReLU($x))
Luminal_Max(x, y) = :(Luminal_Max($x, $y))

# Create patterns manually
pvars = Symbol[]
slots = Symbol[]

println("\n--- Creating Patterns ---")

# Built-in pattern: sin(~x)
pat_sin = Metatheory.Syntax.makepattern(:(sin(~x)), pvars, slots, Main)
println("\nPattern for sin(~x):")
println("  Type: $(typeof(pat_sin))")
println("  Head: ", operation(pat_sin))
println("  head_hash: ", pat_sin.head_hash)
println("  name_hash: ", pat_sin.name_hash) 
println("  isground: ", pat_sin.isground)

# Reset pvars
pvars = Symbol[]

# Custom pattern: Luminal_ReLU(~x)
pat_relu = Metatheory.Syntax.makepattern(:(Luminal_ReLU(~x)), pvars, slots, Main)
println("\nPattern for Luminal_ReLU(~x):")
println("  Type: $(typeof(pat_relu))")
println("  Head: ", operation(pat_relu))
println("  head_hash: ", pat_relu.head_hash)
println("  name_hash: ", pat_relu.name_hash)
println("  isground: ", pat_relu.isground)

# Check if head_hash matches what's in e-graph
println("\n--- Comparing with E-Graph ---")

using Metatheory.EGraphs

eg_sin = EGraph(:(sin(a)))
sin_hash_in_eg = hash(:sin)
println("\nsin:")
println("  Pattern head_hash: ", pat_sin.head_hash)
println("  E-Graph symbol hash: ", sin_hash_in_eg)
println("  Match: ", pat_sin.head_hash == sin_hash_in_eg)

eg_relu = EGraph(:(Luminal_ReLU(a)))
relu_hash_in_eg = hash(:Luminal_ReLU)
println("\nLuminal_ReLU:")
println("  Pattern head_hash: ", pat_relu.head_hash)
println("  E-Graph symbol hash: ", relu_hash_in_eg)
println("  Match: ", pat_relu.head_hash == relu_hash_in_eg)

# Check what constants are actually in the e-graphs
println("\n--- Actual E-Graph Constants ---")
println("\nsin e-graph:")
for (h, val) in eg_sin.constants
    println("  $h => $val")
end

println("\nLuminal_ReLU e-graph:")
for (h, val) in eg_relu.constants
    println("  $h => $val")
end

println("\n" * "=" ^ 80)
