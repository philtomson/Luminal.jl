#!/usr/bin/env julia

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Luminal
using Test
using SymbolicUtils
using SymbolicUtils: operation, arguments, iscall

@testset "Inverse and Identity Rules" begin
    @testset "log2(exp2(x)) -> x" begin
        g = Luminal.Graph()
        inp = Luminal.tensor(g, [1])
        # exp2
        e = Luminal.add_op!(g, Luminal.Exp2(), [(inp.id, 0)], inp.shape)
        # log2
        l = Luminal.add_op!(g, Luminal.Log2(), [(e.id, 0)], inp.shape)
        
        optimized = Luminal.optimize_symbolic(g, l)
        println("log2(exp2(x)) optimized to: ", optimized)
        
        # Should simplify back to the input tensor
        @test optimized isa SymbolicUtils.BasicSymbolic
        @test !iscall(optimized)
        @test nameof(optimized) == Symbol("InputTensor", inp.id)
    end

    @testset "max(x, x) -> x" begin
        g = Luminal.Graph()
        inp = Luminal.tensor(g, [1])
        # max(inp, inp)
        m = Luminal.add_op!(g, Luminal.Max(), [(inp.id, 0), (inp.id, 0)], inp.shape)
        
        optimized = Luminal.optimize_symbolic(g, m)
        println("max(x, x) optimized to: ", optimized)
        
        @test optimized isa SymbolicUtils.BasicSymbolic
        @test !iscall(optimized)
    end

    @testset "sqrt(x) * sqrt(x) -> x" begin
        g = Luminal.Graph()
        inp = Luminal.tensor(g, [1])
        s1 = Luminal.add_op!(g, Luminal.Sqrt(), [(inp.id, 0)], inp.shape)
        s2 = Luminal.add_op!(g, Luminal.Sqrt(), [(inp.id, 0)], inp.shape)
        prod = s1 * s2
        
        optimized = Luminal.optimize_symbolic(g, prod)
        println("sqrt(x) * sqrt(x) optimized to: ", optimized)
        
        @test optimized isa SymbolicUtils.BasicSymbolic
        @test !iscall(optimized)
        @test nameof(optimized) == Symbol("InputTensor", inp.id)
    end
end

@testset "Identity and Recip Rules" begin
    @testset "luminal_recip cleanup" begin
        g = Luminal.Graph()
        inp = Luminal.tensor(g, [1])
        r = Luminal.add_op!(g, Luminal.Recip(), [(inp.id, 0)], inp.shape)
        
        optimized = Luminal.optimize_symbolic(g, r)
        println("recip(x) optimized to: ", optimized)
        
        @test iscall(optimized)
        @test operation(optimized) === (/)
        @test arguments(optimized)[1] == 1.0
    end
end

@testset "Operator Fusion Rules" begin
    @testset "FusedMulAdd (a * b + c)" begin
        g = Luminal.Graph()
        a = Luminal.tensor(g, [1])
        b = Luminal.tensor(g, [1])
        c = Luminal.tensor(g, [1])
        
        prod = a * b
        res = prod + c
        
        optimized = Luminal.optimize_symbolic(g, res)
        println("a * b + c optimized to: ", optimized)
        
        @test iscall(optimized)
        @test string(operation(optimized)) == "luminal_fused_mul_add"
        @test length(arguments(optimized)) == 3
    end

    @testset "FusedAddReLU (max(a + b, 0))" begin
        g = Luminal.Graph()
        a = Luminal.tensor(g, [1])
        b = Luminal.tensor(g, [1])
        
        sum_ab = a + b
        res = Luminal.relu(sum_ab)
        
        optimized = Luminal.optimize_symbolic(g, res)
        println("relu(a + b) optimized to: ", optimized)
        
        @test iscall(optimized)
        @test string(operation(optimized)) == "luminal_fused_add_relu"
        @test length(arguments(optimized)) == 2
    end
end

println("New compiler rules tests passed!")
