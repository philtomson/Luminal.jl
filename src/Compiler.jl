
# Compiler.jl — Graph optimization using SymbolicUtils.jl rewrite rules

using SymbolicUtils
using SymbolicUtils: @rule, term, operation, arguments, Sym, BasicSymbolic, iscall
using SymbolicUtils.Rewriters: Postwalk, Fixpoint, Chain, PassThrough
using Luminal.MetatheoryIntegration # For luminal_to_symbolic and luminal_relu

# Define the ReLU canonicalization rule: luminal_relu(x) => max(x, 0)
const RELU_RULE = @rule luminal_relu(~x) => term(max, ~x, 0; type=Real)

"""
    optimize(expr)

Apply optimization rules to a symbolic expression.
Uses a manual fixpoint loop to avoid deep type nesting that causes
stack overflow in Julia 1.12's type inferencer.
"""
function optimize(expr)
    rewriter = Postwalk(PassThrough(Chain([RELU_RULE])))
    prev = nothing
    result = expr
    while !isequal(prev, result)
        prev = result
        result = rewriter(result)
    end
    return result
end

"""
    compile(graph::Luminal.Graph, output_g_tensor::Luminal.GraphTensor)

Optimize a Luminal graph by converting it to a SymbolicUtils expression,
applying rewrite rules, and returning the optimized symbolic expression.
"""
function compile(graph::Luminal.Graph, output_g_tensor::Luminal.GraphTensor)
    # Convert the Luminal graph to a SymbolicUtils symbolic expression
    sym_expr = Luminal.MetatheoryIntegration.luminal_to_symbolic(output_g_tensor)

    # Apply the optimization rules
    optimized = optimize(sym_expr)

    return optimized
end