module SymbolicIntegration

using ..Luminal # This makes Luminal itself available
using SymbolicUtils
using SymbolicUtils: Sym, Term, BasicSymbolic, FnType, iscall
import SymbolicUtils: symtype, promote_symtype, operation, arguments

# Define a custom symbolic function for ReLU.
# This creates a callable symbolic variable: luminal_relu(x) produces a Term node.
@syms luminal_relu(x::Real)::Real
@syms luminal_recip(x::Real)::Real
@syms luminal_fused_mul_add(a, b, c)::Real
@syms luminal_fused_add_relu(a, b)::Real
@syms luminal_swap_loops(x, inner, outer)::Real
@syms luminal_loop_in(x, name::String, range, stride)::Real
@syms luminal_loop_out(x, name::String, range, stride)::Real
@syms luminal_tc_matmul(a, b, aks, bks, ars, brs, crs, kl)::Real

# Export them so Compiler.jl and tests can use it
export luminal_relu, luminal_recip, luminal_fused_mul_add, luminal_fused_add_relu, luminal_swap_loops, luminal_loop_in, luminal_loop_out, luminal_tc_matmul

"""
    luminal_to_symbolic(g_tensor::Luminal.GraphTensor)

Convert a Luminal GraphTensor to a SymbolicUtils symbolic expression.
"""
function luminal_to_symbolic(g_tensor::Luminal.GraphTensor)
    # Cache of symbolic variables for input tensors (by node_id)
    sym_cache = Dict{Int, Any}()
    return _node_to_symbolic(g_tensor.graph_ref, g_tensor.id, sym_cache)
end

function _node_to_symbolic(graph::Luminal.Graph, node_id::Int, sym_cache::Dict{Int, Any})
    if haskey(sym_cache, node_id)
        return sym_cache[node_id]
    end

    node = graph.nodes[node_id]
    op = node.op

    if op isa Luminal.Constant
        result = op.value
    elseif op isa Luminal.Function
        # Create a named symbolic variable for each input tensor
        name = Symbol(op.name, node_id)
        result = Sym{Real}(name)
    else
        # Recursively convert input nodes
        input_syms = [_node_to_symbolic(graph, input_id, sym_cache) for (input_id, _) in node.inputs]

        if op isa Luminal.Add
            result = input_syms[1] + input_syms[2]
        elseif op isa Luminal.Mul
            result = input_syms[1] * input_syms[2]
        elseif op isa Luminal.Max
            # Use SymbolicUtils.term to create a max Term node without eager evaluation
            result = term(max, input_syms...; type=Real)
        elseif op isa Luminal.ReLU
            # Use our custom symbolic function luminal_relu
            result = luminal_relu(input_syms[1])
        elseif op isa Luminal.Recip
            result = luminal_recip(input_syms[1])
        elseif op isa Luminal.LoopIn
            result = luminal_loop_in(input_syms[1], op.name, op.range, op.stride)
        elseif op isa Luminal.LoopOut
            result = luminal_loop_out(input_syms[1], op.name, op.range, op.stride)
        elseif op isa Luminal.TCMatmul
            result = luminal_tc_matmul(input_syms[1], input_syms[2], op.a_k_stride, op.b_k_stride, op.a_row_size, op.b_row_size, op.c_row_size, op.k_loops)
        elseif op isa Luminal.Log2
            result = term(log2, input_syms[1]; type=Real)
        elseif op isa Luminal.Exp2
            result = term(exp2, input_syms[1]; type=Real)
        elseif op isa Luminal.Sin
            result = sin(input_syms[1])
        elseif op isa Luminal.Sqrt
            result = sqrt(input_syms[1])
        elseif op isa Luminal.Recip
            result = term(/, 1.0, input_syms[1]; type=Real)
        else
            error("Unsupported Luminal Op for symbolic conversion: $(typeof(op))")
        end
    end

    sym_cache[node_id] = result
    return result
end

export luminal_to_symbolic

end # module SymbolicIntegration