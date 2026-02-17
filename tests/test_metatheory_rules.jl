#!/usr/bin/env julia
# Test MetatheoryRules

# Add the src directory to the Julia load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Test
using Luminal
using Luminal.MetatheoryOps
using Luminal.MetatheoryRules
using Metatheory
using Metatheory.EGraphs

println("=" ^ 80)
println("Metatheory Rules Test")
println("=" ^ 80)

@testset "Optimizer Rules" begin
    # 1. Algebraic Simplification: x + 0 -> x
    @testset "Algebraic" begin
        expr = :(LuminalAdd(x, LuminalZero()))
        eg = EGraph(expr)
        saturate!(eg, luminal_theory)
        # Check if 'x' is in the same class
        # We need to add 'x' to graph to check equivalence
        x_id = addexpr!(eg, :x)
        original_id = addexpr!(eg, expr)
        
        @test in_same_class(eg, original_id, x_id)
        println("Algebraic x+0 -> x : Passed")
    end
    
    # 2. Fusion: ReLU(Add(x, y)) -> FusedAddReLU(x, y)
    @testset "Fusion" begin
        expr = :(LuminalReLU(LuminalAdd(a, b)))
        eg = EGraph(expr)
        saturate!(eg, luminal_theory)
        
        fused_expr = :(LuminalFusedAddReLU(a, b))
        fused_id = addexpr!(eg, fused_expr)
        original_id = addexpr!(eg, expr)
        
        @test in_same_class(eg, original_id, fused_id)
        println("Fusion ReLU(Add) -> FusedAddReLU : Passed")
    end
    
    # 3. Constant Folding
    @testset "Constant Folding" begin
        # 1 + 2 -> 3
        expr = :(LuminalAdd(LuminalConstant(1), LuminalConstant(2)))
        eg = EGraph(expr)
        saturate!(eg, luminal_theory)
        
        target = :(LuminalConstant(3))
        target_id = addexpr!(eg, target)
        original_id = addexpr!(eg, expr)
        
        @test in_same_class(eg, original_id, target_id)
        println("Constant Folding 1+2 -> 3 : Passed")
    end
    
    # 4. Reshape Elimination
    @testset "Reshape Elimination" begin
        # Reshape(Reshape(x, s1), s2) -> Reshape(x, s2)
        expr = :(LuminalReshape(LuminalReshape(data, shape1), shape2))
        eg = EGraph(expr)
        saturate!(eg, luminal_theory)
        
        target = :(LuminalReshape(data, shape2))
        target_id = addexpr!(eg, target)
        original_id = addexpr!(eg, expr)
        
        @test in_same_class(eg, original_id, target_id)
        println("Reshape(Reshape) -> Reshape : Passed")
    end
end
