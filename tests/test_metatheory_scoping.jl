#!/usr/bin/env julia
# Test hypothesis: Module scoping affects custom operator matching

using Pkg
Pkg.activate("Julia")

using Metatheory
using Metatheory.EGraphs

println("=" ^ 80)
println("Testing Module Scoping Hypothesis")
println("=" ^ 80)

# Test 1: Define in Main before @theory
println("\n--- Test 1: Function defined in Main before @theory ---")
Luminal_ReLU_1(x) = :(Luminal_ReLU_1($x))
Luminal_Max_1(x, y) = :(Luminal_Max_1($x, $y))

theory1 = @theory begin
    Luminal_ReLU_1(~x) => Luminal_Max_1(~x, 0)
end

expr1 = :(Luminal_ReLU_1(a))
println("Input: ", expr1)
eg1 = EGraph(expr1)
saturate!(eg1, theory1)
result1 = extract!(eg1, astsize)
println("Output: ", result1)
println("Match: ", result1 == :(Luminal_Max_1(a, 0)) ? "✅ SUCCESS" : "❌ FAILED")

# Test 2: Export the functions
println("\n--- Test 2: With export ---")
export Luminal_ReLU_2, Luminal_Max_2
Luminal_ReLU_2(x) = :(Luminal_ReLU_2($x))
Luminal_Max_2(x, y) = :(Luminal_Max_2($x, $y))

theory2 = @theory begin
    Luminal_ReLU_2(~x) => Luminal_Max_2(~x, 0)
end

expr2 = :(Luminal_ReLU_2(a))
println("Input: ", expr2)
eg2 = EGraph(expr2)
saturate!(eg2, theory2)
result2 = extract!(eg2, astsize)
println("Output: ", result2)
println("Match: ", result2 == :(Luminal_Max_2(a, 0)) ? "✅ SUCCESS" : "❌ FAILED")

# Test 3: Use built-in function for comparison
println("\n--- Test 3: Built-in function (sin) ---")
theory3 = @theory begin
    sin(~x) => cos(~x)
end

expr3 = :(sin(a))
println("Input: ", expr3)
eg3 = EGraph(expr3)
saturate!(eg3, theory3)
result3 = extract!(eg3, astsize)
println("Output: ", result3)
println("Match: ", result3 == :(cos(a)) ? "✅ SUCCESS" : "❌ FAILED")

# Test 4: Inspect pattern structure
println("\n--- Test 4: Pattern Inspection ---")
using Metatheory.Patterns

# Create pattern manually
pvars = Symbol[]
slots = Symbol[]
lhs1 = Metatheory.Syntax.makepattern(:(Luminal_ReLU_1(~x)), pvars, slots, Main)
lhs_sin = Metatheory.Syntax.makepattern(:(sin(~x)), pvars, slots, Main)

println("\nCustom function pattern:")
println("  Pattern: ", lhs1)
println("  Type: ", typeof(lhs1))
println("  Head: ", Patterns.operation(lhs1))
println("  IsGround: ", lhs1.isground)

println("\nBuilt-in function pattern:")
println("  Pattern: ", lhs_sin)
println("  Type: ", typeof(lhs_sin))
println("  Head: ", Patterns.operation(lhs_sin))
println("  IsGround: ", lhs_sin.isground)

# Test 5: Check if isdefined helps
println("\n--- Test 5: Check isdefined_nested ---")
println("isdefined(Main, :Luminal_ReLU_1): ", isdefined(Main, :Luminal_ReLU_1))
println("isdefined(Main, :sin): ", isdefined(Main, :sin))
println("isdefined(Base, :sin): ", isdefined(Base, :sin))

println("\n" * "=" ^ 80)
println("End of Tests")
println("=" ^ 80)
