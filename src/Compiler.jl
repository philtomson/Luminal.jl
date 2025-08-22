

# 3. Define the main compile function.
using Symbolics # For @variables and simplify
using Luminal.SymbolicsIntegration # For luminal_to_symbolics and symbolics_to_luminal

function compile(graph::Luminal.Graph, output_g_tensor::Luminal.GraphTensor)
    # output_g_tensor is already the GraphTensor representing the output

    # Convert the Luminal graph to a Symbolics.jl expression
    sym_expr = Luminal.SymbolicsIntegration.luminal_to_symbolics(output_g_tensor)
    println("Original Symbolics Term: ", sym_expr)

    # Apply Symbolics.jl's simplification
    optimized_sym_expr = Symbolics.simplify(sym_expr)
    println("Optimized Symbolics Term: ", optimized_sym_expr)

    # Convert the optimized Symbolics.jl expression back to a Luminal graph
    optimized_g_tensor = Luminal.SymbolicsIntegration.symbolics_to_luminal(optimized_sym_expr)

    # Return the optimized Luminal GraphTensor
    return optimized_g_tensor
end