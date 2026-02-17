#!/usr/bin/env julia

# Add the src directory to the Julia load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Luminal
using Test
using SymbolicUtils
using SymbolicUtils: Sym, iscall, operation, arguments, term

@testset "SymbolicUtils Integration" begin
    @testset "Basic graph → symbolic conversion (a + b)" begin
        g = Luminal.Graph()
        a = Luminal.tensor(g, [1])
        b = Luminal.tensor(g, [1])
        c = a + b

        # Convert via SymbolicIntegration
        sym_expr = Luminal.SymbolicIntegration.luminal_to_symbolic(c)

        println("a + b symbolic: ", sym_expr)
        println("Type: ", typeof(sym_expr))

        # Should be a symbolic Add expression
        @test sym_expr isa SymbolicUtils.BasicSymbolic
        @test iscall(sym_expr)
        @test operation(sym_expr) === (+)

        # Arguments should be two symbolic variables
        args = arguments(sym_expr)
        sym_args = filter(a -> a isa SymbolicUtils.BasicSymbolic, args)
        @test length(sym_args) == 2
        @test nameof(sym_args[1]) == Symbol("InputTensor", a.id)
        @test nameof(sym_args[2]) == Symbol("InputTensor", b.id)
    end

    @testset "Symbolic max preserves structure" begin
        # Verify that term(max, ...) creates a proper symbolic term
        @syms x::Real
        max_expr = term(max, x, 0; type=Real)

        println("max(x, 0): ", max_expr)
        println("Type: ", typeof(max_expr))

        @test iscall(max_expr)
        @test operation(max_expr) === max
        @test arguments(max_expr)[1] === x
        @test arguments(max_expr)[2] == 0
    end

    @testset "Symbolic min preserves structure" begin
        @syms x::Real
        min_expr = term(min, x, 0; type=Real)

        println("min(x, 0): ", min_expr)
        println("Type: ", typeof(min_expr))

        @test iscall(min_expr)
        @test operation(min_expr) === min
        @test arguments(min_expr)[1] === x
        @test arguments(min_expr)[2] == 0
    end

    @testset "Mul conversion preserves structure (a * b)" begin
        g = Luminal.Graph()
        a = Luminal.tensor(g, [2])
        b = Luminal.tensor(g, [2])
        c = a * b

        sym_expr = Luminal.SymbolicIntegration.luminal_to_symbolic(c)

        println("a * b symbolic: ", sym_expr)

        @test sym_expr isa SymbolicUtils.BasicSymbolic
        @test iscall(sym_expr)
        @test operation(sym_expr) === (*)
    end
end

println("SymbolicUtils integration tests passed!")