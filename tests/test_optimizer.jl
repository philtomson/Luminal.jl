#!/usr/bin/env julia

# Add the src directory to the Julia load path
push!(LOAD_PATH, "../src")

using Luminal
using Test

@testset "Compiler Optimizations" begin
    @testset "ReLU to Max Canonicalization" begin
        # 1. Build a graph with a ReLU operation
        g = Luminal.Graph()
        inp = Luminal.tensor(g, [2, 2])
        output_node = Luminal.relu(inp)

        # 2. Compile the graph
        # The compile function in Compiler.jl will print the original and optimized terms
        optimized_expr = Luminal.compile(g, output_node.id)

        # 3. Verify the output expression
        # We expect ReLU(inp) to be rewritten to Max(inp, 0)
        # The `to_term` function will convert the input tensor to a symbol, e.g., :Input
        # Let's find the input node symbol
        input_node_op = g.nodes[inp.id].op
        input_symbol = Symbol(last(split(string(typeof(input_node_op)), '.')))

        expected_expr = :(Luminal_Max(Luminal_Function("InputTensor"), 0))

        @test optimized_expr == expected_expr
    end
end

println("Optimizer tests passed!")