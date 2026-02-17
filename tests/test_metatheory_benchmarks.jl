#!/usr/bin/env julia
# Benchmarking Metatheory Optimization

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Test
using BenchmarkTools
using Luminal
using Luminal.MetatheoryOptimizer

println("=" ^ 80)
println("Metatheory Optimization Benchmarks")
println("=" ^ 80)

function benchmark_optimization(create_graph_fn, test_name)
    println("\n=== $test_name ===")
    
    # Create graphs
    g = Luminal.Graph()
    t_original = create_graph_fn(g)
    
    # 1. Manual Optimization (Baseline)
    # Note: compile(gt) usually invokes manual optimization within Luminal
    # We measure time to compile.
    println("Running Manual Optimization...")
    # Warmup
    compile(t_original.graph_ref)
    time_manual = @belapsed compile($t_original.graph_ref)
    
    # 2. Metatheory Optimization
    println("Running Search-Based Optimization...")
    # Warmup
    compile_with_search(t_original)
    time_search = @belapsed compile_with_search($t_original)
    
    # Results
    println("Manual Compile Time:   $(round(time_manual * 1000, digits=2)) ms")
    println("Search Compile Time:   $(round(time_search * 1000, digits=2)) ms")
    println("Slowdown Factor:       $(round(time_search / time_manual, digits=2))x")
    
    # Verify Structure (Optional, just check if fused op appears)
    opt_graph = compile_with_search(t_original)
    node = opt_graph.graph_ref.nodes[opt_graph.id]
    println("Resulting Op: $(typeof(node.op))")
end

@testset "Benchmarks" begin
    # Test 1: Simple Add + ReLU
    benchmark_optimization("Add + ReLU Fusion") do g
        a = Luminal.tensor(g, [128, 128])
        b = Luminal.tensor(g, [128, 128])
        c = a + b
        d = Luminal.relu(c)
        return d
    end
    
    # Test 2: MatMul + Bias (GEMM) - If implemented
    # benchmark_optimization("GEMM Fusion") do g
    #     x = Luminal.tensor(g, [128, 128])
    #     w = Luminal.tensor(g, [128, 128])
    #     b = Luminal.tensor(g, [128, 128])
    #     y = (x * w) + b # MatMul + Add
    #     return y
    # end
    
    # Test 3: Mul + Add (FMA)
    benchmark_optimization("Mul + Add Fusion") do g
        x = Luminal.tensor(g, [128, 128])
        y = Luminal.tensor(g, [128, 128])
        z = Luminal.tensor(g, [128, 128])
        res = (x * y) + z
        return res
    end
end
