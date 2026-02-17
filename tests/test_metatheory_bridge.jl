#!/usr/bin/env julia
# Test MetatheoryBridge conversion

# Add the src directory to the Julia load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Test
using Luminal
using Metatheory
using Metatheory.EGraphs

println("=" ^ 80)
println("Metatheory Bridge Test")
println("=" ^ 80)

@testset "Graph to Metatheory Conversion" begin
    # 1. Create a simple graph
    g = Luminal.Graph()
    a = Luminal.tensor(g, [32, 32])
    b = Luminal.tensor(g, [32, 32])
    c = a + b
    d = Luminal.relu(c)
    
    println("Graph created successfully")
    
    # 2. Convert to Metatheory expression
    using Luminal.MetatheoryBridge
    using Luminal.MetatheoryOps
    
    expr = to_metatheory_expr(d)
    println("Converted Expression: $expr")
    
    # Expected structure: LuminalReLU(LuminalAdd(Input, Input))
    @test expr.head == :call
    @test expr.args[1] == :LuminalReLU
    
    inner = expr.args[2]
    @test inner.head == :call
    @test inner.args[1] == :LuminalAdd
    
    # 3. Test matching
    # Define a rule relevant to this structure
    theory = @theory begin
        LuminalReLU(LuminalAdd(~x, ~y)) --> LuminalFusedAddReLU(~x, ~y)
    end
    
    eg = EGraph(expr)
    saturate!(eg, theory)
    
    # Just verify the graph saturation didn't crash
    @test true
end

@testset "Constant Handling" begin
    g = Luminal.Graph()
    a = Luminal.tensor(g, [10])
    b = a * 2.0
    
    using Luminal.MetatheoryBridge
    
    expr = to_metatheory_expr(b)
    println("Constant Expression: $expr")
    
    @test expr.head == :call
    @test expr.args[1] == :LuminalMul
    
    # Check constant arg
    const_arg = expr.args[3]
    @test const_arg.head == :call
    @test const_arg.args[1] == :LuminalConstant
    @test const_arg.args[2] == 2.0
end
