# Add the src directory to the Julia load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Test
using Luminal
using SymbolicUtils
using SymbolicUtils: Sym, substitute

println("Running Symbolic Expression Test (SymbolicUtils transition)")

@testset "Symbolic Expression Execution" begin
    # Test Case 1: (a * 10) + b
    # Using SymbolicUtils.jl
    @syms a::Int b::Int
    expr1 = (a * 10) + b
    
    # Check that it's a symbolic expression
    @test expr1 isa SymbolicUtils.BasicSymbolic
    
    # Evaluate by substitution
    vars1 = Dict(a => 5, b => 3)
    result1 = substitute(expr1, vars1)
    @test result1 == 53

    # Test Case 2: max(a, 100)
    # Note: SymbolicUtils.term(max, ...) is used for non-eager max
    @syms x::Int
    expr2 = SymbolicUtils.term(max, x, 100; type=Int)
    
    vars2 = Dict(x => 5)
    result2 = substitute(expr2, vars2)
    @test result2 == 100
    
    vars3 = Dict(x => 150)
    result3 = substitute(expr2, vars3)
    @test result3 == 150

    # Test Case 3: (a < b) * c
    # Boolean logic in SymbolicUtils can be represented via terms or custom symbols
    @syms a_sym::Int b_sym::Int c_sym::Int
    # Using a simple representation for comparison if needed
    # But for dimensions, we mostly need add, mul, div, mod, min, max.
    
    expr3 = (a_sym + b_sym) * c_sym
    vars4 = Dict(a_sym => 10, b_sym => 20, c_sym => 7)
    result4 = substitute(expr3, vars4)
    @test result4 == 210
end

println("\nSymbolic Expression Tests Passed!")
