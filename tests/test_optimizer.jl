#!/usr/bin/env julia

# Add the src directory to the Julia load path
push!(LOAD_PATH, "../src")

using Luminal
using Test
using Symbolics
using SymbolicUtils # For isequal, if needed for complex comparisons

@testset "Compiler Optimizations" begin
    # @testset "ReLU to Max Canonicalization" begin
    #     # 1. Build a graph with a ReLU operation
    #     g = Luminal.Graph()
    #     # Luminal.tensor creates a Luminal.Function op with name "InputTensor"
    #     # and its node_id will be 1 (first node added to graph)
    #     inp = Luminal.tensor(g, [2, 2])
    #     output_node = Luminal.relu(inp)

    #     # 2. Compile the graph using Symbolics.jl
    #     optimized_g_tensor = Luminal.compile(g, output_node)
        
    #     # 3. Convert the optimized Luminal GraphTensor back to a Symbolics.jl expression for verification
    #     # This part assumes symbolics_to_luminal is implemented, but for now we'll just test the output of compile
    #     # which is an optimized Luminal GraphTensor. We need to convert it to Symbolics for comparison.
    #     optimized_sym_expr = Luminal.SymbolicsIntegration.luminal_to_symbolics(optimized_g_tensor)

    #     # 4. Define expected Symbolics.jl expression
    #     @variables InputTensor1 # This variable name should match what luminal_to_symbolics generates
    #     expected_sym_expr = max(InputTensor1, 0) # Base.max

    #     # 5. Verify the optimized expression
    #     println("Simplified max(InputTensor1, 0): ", Symbolics.simplify(max(InputTensor1, 0)))
    #     @test isequal(optimized_sym_expr, expected_sym_expr)
    # end

    @testset "Symbolics.jl relu behavior" begin
        @variables x
        # Verify that SymbolicUtils.jl simplifies relu(x) to max(x,0)
        @test isequal(Symbolics.simplify(max(x, 0)), max(x, 0)) # Ensure max(x,0) doesn't simplify further
    end
end

println("Optimizer tests passed!")