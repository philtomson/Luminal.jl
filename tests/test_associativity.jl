#!/usr/bin/env julia

# Add the src directory to the Julia load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Luminal
using Test
using SymbolicUtils
using SymbolicUtils: Sym, iscall, operation, arguments

@testset "Associativity" begin
    @testset "(a + 2) + 3 => a + 5" begin
        # 1. Build the computation graph: (a + 2) + 3
        graph = Luminal.Graph()

        a = Luminal.tensor(graph, [2, 2])

        # Constant 2
        const_shape = Luminal.ShapeTracker([1])
        const_2 = Luminal.add_op!(graph, Luminal.Constant(2), Tuple{Int,Int}[], const_shape)

        # Constant 3
        const_3 = Luminal.add_op!(graph, Luminal.Constant(3), Tuple{Int,Int}[], const_shape)

        # Build: (a + 2) + 3
        add_1 = a + const_2
        add_2 = add_1 + const_3

        # 2. Compile
        optimized_expr = Luminal.compile(graph, add_2)

        # 3. Debug
        println("Original: (a + 2) + 3")
        println("Optimized: ", optimized_expr)
        println("Type: ", typeof(optimized_expr))

        # 4. Verify: should be a + 5 (associativity + constant folding)
        @test iscall(optimized_expr)
        @test operation(optimized_expr) === (+)
        args = arguments(optimized_expr)

        # SymbolicUtils normalizes Add as a flat sum with sorted args
        # The result should contain the symbolic variable and the folded constant 5
        println("Args: ", args)

        # Find the symbolic variable and the constant in the arguments
        sym_args = filter(a -> a isa SymbolicUtils.BasicSymbolic, args)
        num_args = filter(a -> a isa Number, args)

        @test length(sym_args) == 1
        @test nameof(sym_args[1]) == Symbol("InputTensor", a.id)
        @test length(num_args) == 1
        @test num_args[1] == 5
    end
end

println("Associativity tests passed!")
