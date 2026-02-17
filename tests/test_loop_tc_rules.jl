#!/usr/bin/env julia

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Luminal
using Test
using SymbolicUtils
using SymbolicUtils: operation, arguments, iscall

@testset "Loop and TensorCore Rules" begin
    @testset "Loop Fusion" begin
        g = Luminal.Graph()
        inp = Luminal.tensor(g, [100])
        # LoopOut
        lo = Luminal.add_op!(g, Luminal.LoopOut("i", 100, 1), [(inp.id, 0)], ShapeTracker([1]))
        # LoopIn
        li = Luminal.add_op!(g, Luminal.LoopIn("i", 100, 1), [(lo.id, 0)], ShapeTracker([100]))
        
        optimized = Luminal.optimize_symbolic(g, li)
        println("Loop Fusion optimized to: ", optimized)
        
        # Should fuse back to the input tensor
        @test optimized isa SymbolicUtils.BasicSymbolic
        @test !iscall(optimized)
    end

    @testset "TCMatmul Pattern Matching" begin
        g = Luminal.Graph()
        a = Luminal.tensor(g, [16, 16])
        b = Luminal.tensor(g, [16, 16])
        acc = Luminal.tensor(g, [16, 16])
        
        # Manually construct the nested loop structure that TCMatmul rule expects
        # (luminal_loop_in(luminal_loop_in(~a, ~m, ~mr, ~aks), ~k, ~kr, ~akst) * ...) + ~acc
        
        # A inputs
        a_m = Luminal.add_op!(g, Luminal.LoopIn("m", 16, 16), [(a.id, 0)], ShapeTracker([16]))
        a_k = Luminal.add_op!(g, Luminal.LoopIn("k", 16, 1), [(a_m.id, 0)], ShapeTracker([1]))
        
        # B inputs
        b_n = Luminal.add_op!(g, Luminal.LoopIn("n", 16, 1), [(b.id, 0)], ShapeTracker([16]))
        b_k = Luminal.add_op!(g, Luminal.LoopIn("k", 16, 16), [(b_n.id, 0)], ShapeTracker([1]))
        
        # Product and sum
        prod = a_k * b_k
        res = prod + acc
        
        # LoopOuts
        out_k = Luminal.add_op!(g, Luminal.LoopOut("k", 16, 1), [(res.id, 0)], ShapeTracker([16, 16]))
        out_n = Luminal.add_op!(g, Luminal.LoopOut("n", 16, 1), [(out_k.id, 0)], ShapeTracker([16, 16]))
        out_m = Luminal.add_op!(g, Luminal.LoopOut("m", 16, 16), [(out_n.id, 0)], ShapeTracker([16, 16]))
        
        println("Starting TCMatmul compilation...")
        optimized = nothing
        try
            optimized = Luminal.optimize_symbolic(g, out_m)
        catch e
            println("Error during compilation: ", e)
            rethrow(e)
        end
        println("Matmul sequence optimized to: ", optimized)
        
        @test optimized !== nothing
        @test iscall(optimized)
        @test string(operation(optimized)) == "luminal_tc_matmul"
    end
end

println("Loop and TensorCore rules tests passed!")
