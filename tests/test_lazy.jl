#!/usr/bin/env julia

# Add the src directory to the Julia load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Luminal
using Test
using SymbolicUtils
using SymbolicUtils: Sym, iscall, operation, arguments, term

@testset "Lazy Graph Evaluation" begin
    @testset "Deferred computation — no eager evaluation" begin
        # Verify that building a graph doesn't execute anything,
        # and that the compile step produces symbolic expressions (not concrete values).

        g = Luminal.Graph()
        a = Luminal.tensor(g, [2, 2])
        b = Luminal.tensor(g, [2, 2])
        c = a + b

        # Graph should have 3 nodes: a, b, c=a+b
        @test length(g.nodes) == 3

        # Compiling should produce a symbolic expression, not a number
        optimized = Luminal.optimize_symbolic(g, c)

        println("Lazy result: ", optimized)
        println("Type: ", typeof(optimized))

        @test optimized isa SymbolicUtils.BasicSymbolic
        @test iscall(optimized)
        @test operation(optimized) === (+)
    end

    @testset "Chained operations stay symbolic" begin
        # A longer chain: relu((a + b) * c) should remain fully symbolic

        g = Luminal.Graph()
        a = Luminal.tensor(g, [4])
        b = Luminal.tensor(g, [4])
        c = Luminal.tensor(g, [4])

        sum_ab = a + b
        prod = sum_ab * c
        result = Luminal.relu(prod)

        # Graph should have 6 nodes: a, b, c, a+b, (a+b)*c, relu(...)
        @test length(g.nodes) == 6

        # Compile and verify structure
        optimized = Luminal.optimize_symbolic(g, result)

        println("Chained lazy result: ", optimized)
        println("Type: ", typeof(optimized))

        # After ReLU rule, relu(...) becomes max(..., 0)
        @test iscall(optimized)
        @test operation(optimized) === max
        args = arguments(optimized)
        @test length(args) == 2
        @test args[2] == 0

        # The first argument should be the multiplication expression
        inner = args[1]
        @test iscall(inner)
        @test operation(inner) === (*)
    end
end

println("Lazy evaluation tests passed!")
