#!/usr/bin/env julia
# Test MetatheoryOptimizer

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Test
using Luminal
using Luminal.MetatheoryOptimizer

println("=" ^ 80)
println("Metatheory Optimizer Test")
println("=" ^ 80)

@testset "Optimizer Integration" begin
    # Test 1: Optimize a simple expression
    # ReLU(Add(x, y)) -> FusedAddReLU(x, y)
    
    g = Luminal.Graph()
    a = Luminal.tensor(g, [32, 32])
    b = Luminal.tensor(g, [32, 32])
    c = a + b
    d = Luminal.relu(c)
    
    println("Original Graph created.")
    
    # Optimize
    d_optimized = compile_with_search(d)
    println("Optimization complete.")
    
    # Check if we got a different graph/tensor
    @test d_optimized.graph_ref !== g # Should be a new graph
    
    # Check the operation structure of the result
    # We need to inspect the new graph
    new_node = d_optimized.graph_ref.nodes[d_optimized.id]
    println("Optimized Op: $(typeof(new_node.op))")
    
    # We expect FusedAddReLU op if we had defined it in Ops.jl!
    # Wait, MetatheoryOps defined `LuminalFusedAddReLU`.
    # But `from_metatheory_expr` maps back to `GraphTensor`.
    # Does `Ops.jl` have `FusedAddReLU`?
    # I verified `Ops.jl` earlier, it has `FusedAddReLU`.
    
    # But `MetatheoryBridge.from_metatheory_expr` needs to implement mapping fo `LuminalFusedAddReLU`.
    # I implemented `from_metatheory_expr` as a placeholder in Step 5255!
    # It only handled `LuminalAdd` and `LuminalMul`.
    
    # I need to implement `from_metatheory_expr` fully!
end
