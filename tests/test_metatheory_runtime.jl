#!/usr/bin/env julia
# Runtime debug: Instrument the e-matching to see where it fails

using Pkg
Pkg.develop(path="/tmp/Metatheory.jl")

using Metatheory
using Metatheory.EGraphs
using Metatheory.Patterns

println("=" ^ 80)
println("Runtime Debug: Custom vs Built-in Matching")
println("=" ^ 80)

# Define functions
Luminal_ReLU(x) = :(Luminal_ReLU($x))
Luminal_Max(x, y) = :(Luminal_Max($x, $y))

# Test both in same session
println("\n--- Test 1: Built-in sin (SHOULD WORK) ---")
theory_sin = @theory begin
    sin(~x) --> cos(~x)
end

eg_sin = EGraph(:(sin(a)))
println("Before saturation: ", eg_sin)
saturate!(eg_sin, theory_sin)
println("After saturation:  ", eg_sin)
result_sin = extract!(eg_sin, astsize)
println("Result: ", result_sin)
println(result_sin == :(cos(a)) ? "✅ SUCCESS" : "❌ FAILED")

println("\n--- Test 2: Custom Luminal_ReLU (FAILS) ---")
theory_relu = @theory begin
    Luminal_ReLU(~x) --> Luminal_Max(~x, 0)
end

eg_relu = EGraph(:(Luminal_ReLU(a)))
println("Before saturation: ", eg_relu)
saturate!(eg_relu, theory_relu)
println("After saturation:  ", eg_relu)
result_relu = extract!(eg_relu, astsize)
println("Result: ", result_relu)
println(result_relu == :(Luminal_Max(a, 0)) ? "✅ SUCCESS" : "❌ FAILED")

# Now let's check if the rules themselves are different
println("\n--- Comparing Rules ---")
println("sin rule:")
println("  LHS: ", theory_sin[1].left)
println("  LHS head: ", theory_sin[1].left.head)
println("  LHS head type: ", typeof(theory_sin[1].left.head))

println("\nLuminal_ReLU rule:")
println("  LHS: ", theory_relu[1].left)
println("  LHS head: ", theory_relu[1].left.head)
println("  LHS head type: ", typeof(theory_relu[1].left.head))

println("\n" * "=" ^ 80)
