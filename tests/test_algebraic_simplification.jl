#!/usr/bin/env julia

# Add the src directory to the Julia load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Luminal
using Test
using SymbolicUtils
using SymbolicUtils: Sym, iscall

@testset "Algebraic Simplification" begin
    @testset "(a * 1) + 0 => a" begin
        # 1. Build the computation graph: (a * 1) + 0
        graph = Luminal.Graph()

        # Input tensor 'a'
        a = Luminal.tensor(graph, [2, 2])

        # Constant 1 (as a graph node)
        const_1_shape = Luminal.ShapeTracker([1])
        const_1 = Luminal.add_op!(graph, Luminal.Constant(1), Tuple{Int,Int}[], const_1_shape)

        # Constant 0 (as a graph node)
        const_0_shape = Luminal.ShapeTracker([1])
        const_0 = Luminal.add_op!(graph, Luminal.Constant(0), Tuple{Int,Int}[], const_0_shape)

        # Build the expression: (a * 1) + 0
        mul_node = a * const_1
        add_node = mul_node + const_0

        # 2. Compile (optimize) the graph
        optimized_expr = Luminal.compile(graph, add_node)

        # 3. Debug prints
        println("Original expression: (a * 1) + 0")
        println("Optimized expression: ", optimized_expr)
        println("Type: ", typeof(optimized_expr))

        # 4. Verify the result simplifies to just the input variable 'a'
        # SymbolicUtils automatically simplifies x*1 => x and x+0 => x
        # So the result should just be the symbolic variable for 'a'
        @test optimized_expr isa SymbolicUtils.BasicSymbolic
        @test !iscall(optimized_expr)  # It's a plain Sym, not a compound expression
        @test nameof(optimized_expr) == Symbol("InputTensor", a.id)

        println("Result is the input variable: ", nameof(optimized_expr))
    end
end

println("Algebraic simplification tests passed!")
