using Metatheory
import Luminal: Luminal_Log2, Luminal_Exp2, Luminal_Sin, Luminal_Sqrt, Luminal_Recip, Luminal_Contiguous, Luminal_ReLU, Luminal_Add, Luminal_Mul, Luminal_Mod, Luminal_LessThan, Luminal_Max, Luminal_SumReduce, Luminal_MaxReduce, Luminal_Permute, Luminal_Expand, Luminal_Reshape, Luminal_MatMul, Luminal_Constant, Luminal_Function, Luminal_GraphBreak

# 1. Define a representation of our graph that Metatheory can understand.
# We will convert our Graph object into nested expressions.

function to_term(graph::Graph, node_id::Int)
    node = graph.nodes[node_id]
    op = node.op

    # Recursively convert input nodes to terms
    input_terms = [to_term(graph, inp[1]) for inp in node.inputs]

    # Construct the term for the current operation
    op_symbol = Symbol(last(split(string(typeof(op)), '.')))
    if op isa Constant
        return Expr(:call, :Luminal_Constant, op.value) # Directly create Expr
    # Add specific cases for ops with their own data
    elseif op isa Reshape
        return Expr(:call, :Luminal_Reshape, input_terms..., op.shape)
    elseif op isa Permute
        return Expr(:call, :Luminal_Permute, input_terms..., op.dims)
    elseif op isa Expand
        return Expr(:call, :Luminal_Expand, input_terms..., op.dim, op.size)
    elseif op isa ReLU
        return Expr(:call, :Luminal_ReLU, input_terms...)
    elseif op isa Add
        return Expr(:call, :Luminal_Add, input_terms...)
    elseif op isa Mul
        return Expr(:call, :Luminal_Mul, input_terms...)
    elseif op isa Max
        return Expr(:call, :Luminal_Max, input_terms...)
    elseif op isa Function
        return Expr(:call, :Luminal_Function, op.name)
    elseif isempty(input_terms)
        # Fallback for load nodes or other ops with no inputs
        # This branch should now only be hit for ops like GraphBreak that truly have no inputs
        return Expr(:call, Symbol("Luminal_$(op_symbol)")) # Directly create Expr
    else
        # Generic case for other ops
        return Expr(:call, Symbol("Luminal_$(op_symbol)"), input_terms...) # Directly create Expr
    end
end

# 2. Define a simple rewrite rule.
# This rule simplifies multiplication by 1.
const simplification_rules = @theory begin
    # Commutativity
    Luminal_Add(~a, ~b) --> Luminal_Add(~b, ~a)
    Luminal_Mul(~a, ~b) --> Luminal_Mul(~b, ~a)

    # Associativity
    Luminal_Add(Luminal_Add(~a, ~b), ~c) --> Luminal_Add(~a, Luminal_Add(~b, ~c))
    Luminal_Mul(Luminal_Mul(~a, ~b), ~c) --> Luminal_Mul(~a, Luminal_Mul(~b, ~c))

    # Constant Folding
    Luminal_Add(~a::Number, ~b::Number) => ~a + ~b
    Luminal_Mul(~a::Number, ~b::Number) => ~a * ~b

    # Algebraic Simplification
    Luminal_Add(~a, 0) => ~a
    Luminal_Mul(~a, 1) => ~a
    Luminal_Mul(~a, 0) => 0

    # ReLU Canonicalization
    Luminal_ReLU(~a) --> Luminal_Max(~a, 0)
end

# 3. Define the main compile function.
function compile(graph::Graph, output_id::Int)
    # Convert the graph to a Metatheory-compatible expression
    term = to_term(graph, output_id)
    println("Original Term: ", term)

    # Create a Theory from our rules
    theory = simplification_rules

    # Run the equality saturation process
    g = EGraph(term)
    saturate!(g, theory)

    # Extract the best expression
    best_expr = extract!(g, astsize)
    println("Optimized Term: ", best_expr)

    # (Future step: convert the optimized expression back to a graph)
    return best_expr
end