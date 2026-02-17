#!/usr/bin/env julia
# Test MetatheoryCost

# Add the src directory to the Julia load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Test
using Luminal
using Luminal.MetatheoryOps
using Luminal.MetatheoryCost
using Metatheory
using Metatheory.EGraphs

println("=" ^ 80)
println("Metatheory Cost Function Test")
println("=" ^ 80)

@testset "Cost Function Preferences" begin
    # 1. Compare Add + ReLU vs FusedAddReLU
    
    # Expr 1: ReLU(Add(x, y))
    expr1 = :(LuminalReLU(LuminalAdd(x, y)))
    eg1 = EGraph(expr1)
    
    # Calculate cost manually or via extraction (if we had the class id)
    # Ideally we'd use extract! but we need to saturate with identity rules to make them equivalent first
    # Or just construct the EGraph with both and check which one is picked
    
    # Let's add both to the SAME e-graph and merge them
    id1 = addexpr!(eg1, expr1)
    expr2 = :(LuminalFusedAddReLU(x, y))
    id2 = addexpr!(eg1, expr2)
    
    # Merge them to say they are equivalent
    union!(eg1, id1, id2)
    rebuild!(eg1)
    
    # Extract best
    best_expr = extract!(eg1, luminal_cost)
    println("Add+ReLU vs FusedAddReLU -> Best: $best_expr")
    
    # Should pick Fused
    @test best_expr.args[1] == :LuminalFusedAddReLU
    
    # 2. Compare Reshape vs Standard Op
    # Reshape should be very cheap
    
    expr3 = :(LuminalReshape(x, s))
    expr4 = :(LuminalAdd(x, x)) # Dummy comparison
    
    eg2 = EGraph()
    id3 = addexpr!(eg2, expr3)
    id4 = addexpr!(eg2, expr4)
    union!(eg2, id3, id4)
    rebuild!(eg2)
    
    best_expr2 = extract!(eg2, luminal_cost, id3)
    println("Reshape vs Add -> Best: $best_expr2")
    
    @test best_expr2.args[1] == :LuminalReshape
end
