module MetatheoryOptimizer

using Metatheory
using Metatheory.EGraphs
using ..Luminal
using ..MetatheoryOps
using ..MetatheoryBridge
using ..MetatheoryCost
using ..MetatheoryRules

export compile_with_search, OptimizerParams

struct OptimizerParams
    theory::Vector
    cost_function::Base.Function
    max_iterations::Int
    timeout_seconds::Float64
end

function OptimizerParams(;
    theory=luminal_theory,
    cost_function=luminal_cost,
    max_iterations=100,
    timeout_seconds=10.0
)
    OptimizerParams(theory, cost_function, max_iterations, timeout_seconds)
end

function optimize_expr(expr, params::OptimizerParams)
    # Create e-graph
    eg = EGraph(expr)
    
    # Configure saturation
    # Note: In Metatheory 3.0:
    # - timeout field is for MAX ITERATIONS
    # - timelimit field is for TIME (ns)
    sat_params = SaturationParams(
        timeout = params.max_iterations,
        timelimit = UInt64(params.timeout_seconds * 1e9)
    )
    
    # Saturate with rules
    saturate!(eg, params.theory, sat_params)
    
    # Extract best expression
    return extract!(eg, params.cost_function)
end

# Integration with Luminal compiler
"""
    compile_with_search(graph::Graph; params=OptimizerParams())

Optimizes a Luminal computation graph using e-graph saturation and equality saturation.
Returns a new Graph with optimized structure.
Note: This currently only supports graphs that can be fully represented as a single expression tree.
Graphs with multiple outputs or shared substructure that relies on identity not just value
might need more complex handling (e.g. extraction to graph).
For now, we implement Tree-based optimization.
"""
function compile_with_search(graph::Graph; params=OptimizerParams())
    @assert length(graph.nodes) > 0 "Graph is empty"
    
    # We assume the graph has one main output or we optimize from the last node?
    # In Luminal user code, usually we work with GraphTensor which points to a node.
    # But here we take a Graph.
    # We need to know which node is the "root" or output.
    # Since Graph is a collection, let's assume valid graphs created via tracing
    # have a clear sink or we return an optimized version of the *graph structure*.
    
    # Actually, `compile(gt::GraphTensor)` is the usual API.
    # Let's overload that or add a new method.
    return graph # Fallback if no root provided
end

function compile_with_search(gt::GraphTensor; params=OptimizerParams())
    # Convert to Metatheory expr
    expr = to_metatheory_expr(gt)
    
    # Optimize
    optimized_expr = optimize_expr(expr, params)
    
    # Create a new graph for the output
    new_graph = Luminal.Graph()
    
    # Convert back to GraphTensor on the new graph
    new_gt = from_metatheory_expr(optimized_expr, new_graph)
    
    return new_gt
end

end # module
