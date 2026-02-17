#!/usr/bin/env julia
# Test Metatheory.jl v3.0 (ale/3.0 branch) with custom unary operators
#
# This test verifies if Metatheory.jl v3.0 correctly matches and rewrites
# custom unary operators like Luminal_ReLU.

using Pkg

println("=" ^ 80)
println("Testing Metatheory.jl v3.0 with Custom Operators")
println("=" ^ 80)

# Check if Metatheory is installed
if !haskey(Pkg.project().dependencies, "Metatheory")
    println("\n❌ Metatheory.jl not installed.")
    Pkg.add(url="https://github.com/philtomson/Metatheory.jl", rev="fix-custom-operators")
else
    println("\n✓ Metatheory.jl already installed")
end

println("\nLoading Metatheory.jl...")
using Metatheory
using Metatheory.EGraphs

println("✓ Metatheory.jl loaded successfully\n")

# Define custom operators as simple Julia functions
Luminal_ReLU(x) = :(Luminal_ReLU($x))
Luminal_Add(x, y) = :(Luminal_Add($x, $y))
Luminal_Max(x, y) = :(Luminal_Max($x, $y))

# Helper to check if two expressions are in the same e-class
function are_equivalent(eg, expr1, expr2)
    id1 = addexpr!(eg, expr1)
    id2 = addexpr!(eg, expr2)
    return find(eg, id1) == find(eg, id2)
end

println("Test 1: Custom Unary Operator (Luminal_ReLU)")
println("-" ^ 60)

theory_unary = @theory begin
    # Identity rule: ReLU(ReLU(x)) --> ReLU(x)
    Luminal_ReLU(Luminal_ReLU(~x)) --> Luminal_ReLU(~x)
    
    # Canonicalization: ReLU(x) --> Max(x, 0)
    Luminal_ReLU(~x) --> Luminal_Max(~x, 0)
end

expr_unary = Luminal_ReLU(:x)
println("Input:  ", expr_unary)

eg_unary = EGraph(expr_unary)
saturate!(eg_unary, theory_unary)

expected_unary = Luminal_Max(:x, 0)
if are_equivalent(eg_unary, expr_unary, expected_unary)
    println("✅ SUCCESS: Custom unary operator matched and rewritten!")
    println("   ReLU(x) ≡ Max(x, 0)")
else
    println("❌ FAILED: ReLU(x) is NOT equivalent to Max(x, 0)")
end

println("\n" * ("=" ^ 60))
println("\nTest 2: Custom Binary Operator (Luminal_Add)")
println("-" ^ 60)

theory_binary = @theory begin
    # Commutativity: Add(x, y) --> Add(y, x)
    Luminal_Add(~x, ~y) --> Luminal_Add(~y, ~x)
    
    # Identity: Add(x, 0) --> x
    Luminal_Add(~x, 0) --> ~x
end

expr_binary = Luminal_Add(:a, 0)
println("Input:  ", expr_binary)

eg_binary = EGraph(expr_binary)
saturate!(eg_binary, theory_binary)

if are_equivalent(eg_binary, expr_binary, :a)
    println("✅ SUCCESS: Binary operator matched and rewritten!")
    println("   Add(a, 0) ≡ a")
else
    println("❌ FAILED: Add(a, 0) is NOT equivalent to :a")
end

println("\n" * ("=" ^ 60))
println("\nTest 3: Nested Custom Operators")
println("-" ^ 60)

expr_nested = Luminal_ReLU(Luminal_Add(:x, 0))
println("Input:  ", expr_nested)

theory_combined = @theory begin
    Luminal_ReLU(~x) --> Luminal_Max(~x, 0)
    Luminal_Add(~x, 0) --> ~x
end

eg_nested = EGraph(expr_nested)
saturate!(eg_nested, theory_combined)

expected_nested = Luminal_Max(:x, 0)
if are_equivalent(eg_nested, expr_nested, expected_nested)
    println("✅ SUCCESS: Nested custom operators handled correctly!")
    println("   ReLU(Add(x, 0)) ≡ Max(x, 0)")
else
    println("❌ FAILED: ReLU(Add(x, 0)) is NOT equivalent to Max(x, 0)")
end

println("\n" * "=" ^ 80)
println("Summary")
println("=" ^ 80)
println("\nIf all tests passed:")
println("  ✅ Metatheory.jl correctly supports custom operators")
println("  ✅ Search-based compilation is feasible")
println("\n" * "=" ^ 80)
