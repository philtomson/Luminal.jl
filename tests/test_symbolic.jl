# Add the src directory to the Julia load path
push!(LOAD_PATH, "../src")

using Test
using Luminal
using Luminal.Symbolic: Expression, Var, Num, MulOp, AddOp, MaxOp, LtOp, execute

println("Running Symbolic Execution Test")

@testset "Symbolic Expression Execution" begin
    # Test Case 1: (a * 10) + b
    # RPN: a, 10, *, b, +
    expr1 = Expression([
        Var('a'),
        Num(10),
        MulOp(),
        Var('b'),
        AddOp()
    ])
    vars1 = Dict('a' => 5, 'b' => 3)
    result1 = execute(expr1, vars1)
    @test result1 == 53

    # Test Case 2: max(a, 100)
    # RPN: a, 100, max
    expr2 = Expression([
        Var('a'),
        Num(100),
        MaxOp()
    ])
    vars2 = Dict('a' => 5)
    result2 = execute(expr2, vars2)
    @test result2 == 100

    # Test Case 3: (a < b) * c
    # RPN: a, b, <, c, *
    expr3 = Expression([
        Var('a'),
        Var('b'),
        LtOp(),
        Var('c'),
        MulOp()
    ])
    vars3 = Dict('a' => 10, 'b' => 20, 'c' => 7)
    result3 = execute(expr3, vars3)
    @test result3 == 7 # (10 < 20) is 1, 1 * 7 = 7
end

println("\nSymbolic Execution Tests Passed!")

