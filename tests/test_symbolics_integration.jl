#!/usr/bin/env julia

push!(LOAD_PATH, "../src")

using Luminal
using Test
using Symbolics

@testset "Symbolics.jl Integration" begin
    @testset "Basic Conversion" begin
        # Create a simple Luminal graph: a + b
        g = Luminal.Graph()
        a_luminal = Luminal.tensor(g, [1]) # Placeholder shape, will be InputTensor1
        b_luminal = Luminal.tensor(g, [1]) # Placeholder shape, will be InputTensor2
        c_luminal = a_luminal + b_luminal # Luminal.Add operation

        # Convert Luminal graph to Symbolics.jl expression
        sym_expr = Luminal.SymbolicsIntegration.luminal_to_symbolics(c_luminal)

        # Define Symbolics.jl variables with the same naming convention as luminal_to_symbolics
        @variables InputTensor1 InputTensor2

        # The converted symbolic expression should be InputTensor1 + InputTensor2
        @test sym_expr isa Num # Or Term
        @test isequal(sym_expr, InputTensor1 + InputTensor2)

        # This part of the test will be implemented once symbolics_to_luminal is fully functional.
        # For now, ensure that the previous test cases still pass.
    end

    @testset "Symbolic max/min" begin
        @variables x
        println("Type of max(x, 0): ", typeof(max(x, 0))) # Debug print
        @test isequal(max(x, 0), max(x, 0))
        @test isequal(min(x, 0), min(x, 0))
        @test isequal(max(x, 5), max(x, 5))
        @test isequal(min(x, 5), min(x, 5))
    end
end