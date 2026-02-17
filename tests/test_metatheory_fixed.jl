#!/usr/bin/env julia
# Test the fix!

using Pkg

# Use the modified Metatheory.jl
# Pkg.develop(path="/tmp/Metatheory.jl")
# Pkg.instantiate()

using Metatheory
using Metatheory.EGraphs
using Metatheory.VecExprModule: VecExpr

println("=" ^ 80)
println("Testing Metatheory.jl Fix")
println("=" ^ 80)

# Define custom functions
Luminal_ReLU(x) = :(Luminal_ReLU($x))
Luminal_Max(x, y) = :(Luminal_Max($x, $y))

# Test 1: Custom function with -->
println("\n--- Test 1: Custom Luminal_ReLU with --> ---")
theory1 = @theory begin
    Luminal_ReLU(~x) --> Luminal_Max(~x, 0)
end

expr1 = :(Luminal_ReLU(a))
println("Input: ", expr1)
eg1 = EGraph(expr1)
saturate!(eg1, theory1)

# Custom cost function to prefer Max over ReLU
# Custom cost function to prefer Max over ReLU
# Signature must be (n, data, costs)
function cost_function(n::VecExpr, data, costs::Vector{Float64})
    cost = 1.0 + sum(costs)
    # Penalize ReLU to force rewrite to Max
    # Compare hash of head directly
    if v_head(n) == hash(:Luminal_ReLU)
        cost += 100.0
    end
    return cost
end

result1 = extract!(eg1, cost_function)
println("Output: ", result1)
println(result1 == :(Luminal_Max(a, 0)) ? "✅ SUCCESS!" : "❌ FAILED")

# Test 2: Built-in sin with -->
println("\n--- Test 2: Built-in sin with --> ---")
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

# Test 3: Algebraic simplification
println("\n--- Test 3: Algebraic +(~x, 0) --> ~x ---")
theory3 = @theory begin
    +(~x, 0) --> ~x
end

expr3 = :(a + 0)
println("Input: ", expr3)
eg3 = EGraph(expr3)
saturate!(eg3, theory3)
result3 = extract!(eg3, astsize)
println("Output: ", result3)
println(result3 == :a ? "✅ SUCCESS!" : "❌ FAILED")

# Test 4: Custom with ==
println("\n--- Test 4: Custom Luminal_ReLU with == ---")
theory4 = @theory begin
    Luminal_ReLU(~x) == Luminal_Max(~x, 0)
end

expr4 = :(Luminal_ReLU(a))
println("Input: ", expr4)
eg4 = EGraph(expr4)
saturate!(eg4, theory4)
result4 = extract!(eg4, cost_function)
println("Output: ", result4)
println(result4 == :(Luminal_Max(a, 0)) ? "✅ SUCCESS!" : "❌ FAILED")

println("\n" * "=" ^ 80)
println("Summary: If all tests pass, the fix works!")
println("=" ^ 80)
