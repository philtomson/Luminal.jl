#!/usr/bin/env julia

# Add the src directory to the Julia load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Luminal
using Test
using SymbolicUtils
using SymbolicUtils: Sym, operation, arguments, iscall

@testset "Compiler Optimizations" begin
    @testset "ReLU to Max Canonicalization" begin
        # 1. Build a graph with a ReLU operation
        g = Luminal.Graph()
        inp = Luminal.tensor(g, [2, 2])
        output_node = Luminal.relu(inp)
        const_shape = Luminal.ShapeTracker([1])
        const_0 = Luminal.add_op!(g, Luminal.Constant(0), Tuple{Int,Int}[], const_shape)

        # 2. Compile (optimize) the graph using SymbolicUtils.jl rewrite rules
        optimized_expr = Luminal.compile(g, output_node)

        # 3. Debug prints
        println("Optimized Expression: ", optimized_expr)
        println("Type: ", typeof(optimized_expr))

        # 4. Verify the optimized expression is max(InputTensor1, 0)
        # The result should be a symbolic Term with operation `max`
        @test iscall(optimized_expr)
        @test operation(optimized_expr) === max
        args = arguments(optimized_expr)
        @test length(args) == 2

        # First argument should be the input tensor symbolic variable
        input_sym = args[1]
        @test input_sym isa SymbolicUtils.BasicSymbolic
        @test nameof(input_sym) == Symbol("InputTensor", inp.id)

        # Second argument should be 0
        @test args[2] == 0

        println("Optimized Expression Args: ", args)
        println("  Arg 1: ", args[1], " (Type: ", typeof(args[1]), ")")
        println("  Arg 2: ", args[2], " (Type: ", typeof(args[2]), ")")
    end
end

println("Optimizer tests passed!")