#!/usr/bin/env julia
# Debug: Why do custom functions still fail after the fix?

using Pkg
Pkg.develop(path="/tmp/Metatheory.jl")

using Metatheory
using Metatheory.EGraphs
using Metatheory.Patterns
using Metatheory.VecExprModule

println("=" ^ 80)
println("Debug: Custom vs Built-in After Fix")
println("=" ^ 80)

# Define custom function
Luminal_ReLU(x) = :(Luminal_ReLU($x))

# Create patterns
pvars = Symbol[]
slots = Symbol[]

#  Pattern for sin(~x)
pat_sin = Metatheory.Syntax.makepattern(:(sin(~x)), pvars, slots, Main)

pvars = Symbol[]
# Pattern for Luminal_ReLU(~x)
pat_relu = Metatheory.Syntax.makepattern(:(Luminal_ReLU(~x)), pvars, slots, Main)

println("\n--- Pattern head hashes (after fix) ---")
println("sin pattern:")
println("  head: ", pat_sin.head, " (type: $(typeof(pat_sin.head)))")
println("  name: ", pat_sin.name, " (type: $(typeof(pat_sin.name)))")
println("  head_hash: ", pat_sin.head_hash)
println("  name_hash: ", pat_sin.name_hash)
println("  v_head(n): ", v_head(pat_sin.n))

println("\nLuminal_ReLU pattern:")
println("  head: ", pat_relu.head, " (type: $(typeof(pat_relu.head)))")
println("  name: ", pat_relu.name, " (type: $(typeof(pat_relu.name)))")
println("   head_hash: ", pat_relu.head_hash)
println("  name_hash: ", pat_relu.name_hash)
println("  v_head(n): ", v_head(pat_relu.n))

# Check e-graphs
eg_sin = EGraph(:(sin(a)))
eg_relu = EGraph(:(Luminal_ReLU(a)))

println("\n--- E-Graph constants ---")
println("sin e-graph:")
for (h, v) in eg_sin.constants
    println("  $h => $v (type: $(typeof(v)))")
end

println("\nLuminal_ReLU e-graph:")
for (h, v) in eg_relu.constants
    println("  $h => $v (type: $(typeof(v)))")
end

# Check what makes the pattern
println("\n--- Check isdefined ---")
println("isdefined(Main, :sin): ", isdefined(Main, :sin))
println("isdefined(Base, :sin): ", isdefined(Base, :sin))
println("isdefined(Main, :Luminal_ReLU): ", isdefined(Main, :Luminal_ReLU))

println("\n--- Check function retrieval ---")
println("typeof(sin): ", typeof(sin))
println("typeof(Luminal_ReLU): ", typeof(Luminal_ReLU))
println("nameof(sin): ", nameof(sin))
println("nameof(Luminal_ReLU): ", nameof(Luminal_ReLU))

# Check maybe_quote_operation
using Metatheory: maybe_quote_operation
println("\n--- maybe_quote_operation ---")
println("maybe_quote_operation(sin): ", maybe_quote_operation(sin), " (type: $(typeof(maybe_quote_operation(sin))))")
println("maybe_quote_operation(Luminal_ReLU): ", maybe_quote_operation(Luminal_ReLU), " (type: $(typeof(maybe_quote_operation(Luminal_ReLU))))")

println("\n" * "=" ^ 80)
