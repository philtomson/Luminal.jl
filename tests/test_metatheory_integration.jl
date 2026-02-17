#!/usr/bin/env julia
# Test that fixed Metatheory.jl works with Luminal project

using Pkg
Pkg.activate(".")

println("=" ^ 80)
println("Metatheory.jl Integration Test")
println("=" ^ 80)

# Test 1: Can we load Metatheory?
println("\n--- Test 1: Loading Metatheory.jl ---")
try
    using Metatheory
    using Metatheory.EGraphs
    println("✅ Metatheory.jl loaded successfully")
catch e
    println("❌ Failed to load Metatheory.jl: $e")
    exit(1)
end

# Test 2: Custom operators work
println("\n--- Test 2: Custom Operators ---")
TestOp(x) = :(TestOp($x))
TestTransform(x) = :(TestTransform($x))

theory = @theory begin
    TestOp(~x) --> TestTransform(~x)
end

eg = EGraph(:(TestOp(a)))
saturate!(eg, theory)
result = extract!(eg, astsize_inv)

if result == :(TestTransform(a))
    println("✅ Custom operators work correctly")
    println("   Input:  TestOp(a)")
    println("   Output: $result")
else
    println("❌ Custom operators failed")
    println("   Expected: TestTransform(a)")
    println("   Got: $result")
    exit(1)
end

# Test 3: Built-in operators still work
println("\n--- Test 3: Built-in Operators ---")
theory2 = @theory begin
    sin(~x) --> cos(~x)
end

eg2 = EGraph(:(sin(a)))
saturate!(eg2, theory2)
result2 = extract!(eg2, astsize)

if result2 == :(cos(a))
    println("✅ Built-in operators work correctly")
else
    println("❌ Built-in operators failed")
    exit(1)
end

println("\n" * "=" ^ 80)
println("All tests passed! ✅")
println("Metatheory.jl is ready for Luminal integration")
println("=" ^ 80)
