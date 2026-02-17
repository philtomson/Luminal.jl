#!/usr/bin/env julia
# Force complete reload of Metatheory.jl

# Remove old version
using Pkg
Pkg.rm("Metatheory")

# Add the fixed version
Pkg.develop(path="/tmp/Metatheory.jl")
Pkg.build("Metatheory")

# Now test with fresh session
using Metatheory
using Metatheory.EGraphs

println("=" ^ 80)
println("Testing with Fresh Metatheory.jl Load")
println("=" ^ 80)

# Define custom functions
Luminal_ReLU(x) = :(Luminal_ReLU($x))
Luminal_Max(x, y) = :(Luminal_Max($x, $y))

# Test 1: Custom function
println("\n--- Test 1: Custom Luminal_ReLU ---")
theory1 = @theory begin
    Luminal_ReLU(~x) --> Luminal_Max(~x, 0)
end

expr1 = :(Luminal_ReLU(a))
println("Input: ", expr1)
eg1 = EGraph(expr1)
saturate!(eg1, theory1)
result1 = extract!(eg1, astsize)
println("Output: ", result1)
println(result1 == :(Luminal_Max(a, 0)) ? "✅ SUCCESS!" : "❌ FAILED")

# Test 2: Built-in
println("\n--- Test 2: Built-in sin ---")
theory2 = @theory begin
    sin(~x) --> cos(~x)
end

expr2 = :(sin(a))
println("Input: ", expr2)
eg2 = EGraph(expr2)
saturate!(eg2, theory2)
result2 = extract!(eg2, astsize)
println("Output: ", result2)
println(result2 == :(cos(a)) ? "✅ SUCCESS!" : "❌ FAILED")

println("\n" * "=" ^ 80)
