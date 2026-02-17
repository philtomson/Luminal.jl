module MetatheoryBridge

import ..Graph
import ..GraphTensor
import ..ShapeTracker
import ..tensor
import ..constant
import ..add_op!
import ..broadcast_dims
import ..realized_dims
import ..LessThan
import ..Add
import ..Mul
import ..Max
import ..Mod
import ..Sin
import ..Cos
import ..Log2
import ..Exp2
import ..Recip
import ..ReLU
import ..MatMul
import ..Constant
import ..Op
import ..FusedAddReLU
import ..FusedMulAdd
import ..Function as LuminalFunction

using ..MetatheoryOps
using Metatheory
using Metatheory.EGraphs

export to_metatheory_expr, from_metatheory_expr

"""
    to_metatheory_expr(gt::GraphTensor)

Converts a Luminal GraphTensor into a Metatheory.jl expression suitable for 
pattern matching and optimization.
"""
function to_metatheory_expr(gt::GraphTensor)
    # Get the node corresponding to this tensor
    node = gt.graph_ref.nodes[gt.id]
    op = node.op
    
    args = [to_metatheory_expr(GraphTensor(input_id, input_shape, gt.graph_ref)) 
            for (input_id, output_idx, input_shape) in node.inputs]

    # Special handling for InputTensor to include ID
    if op isa LuminalFunction && op.name == "InputTensor"
        return MetatheoryOps.LuminalSimpleInput(gt.id)
    end
            
    return convert_op_to_expr(op, args)
end

function convert_op_to_expr(op::LessThan, args)
    return MetatheoryOps.LuminalLessThan(args[1], args[2])
end

# Dispatch for specific operations
function convert_op_to_expr(op::Add, args)
    return MetatheoryOps.LuminalAdd(args[1], args[2])
end

function convert_op_to_expr(op::Mul, args)
    return MetatheoryOps.LuminalMul(args[1], args[2])
end

function convert_op_to_expr(op::Max, args)
    return MetatheoryOps.LuminalMax(args[1], args[2])
end

function convert_op_to_expr(op::Mod, args)
    return :(MetatheoryOps.LuminalMod($(args[1]), $(args[2]))) 
end

function convert_op_to_expr(op::Sin, args)
    return MetatheoryOps.LuminalSin(args[1]) 
end

function convert_op_to_expr(op::Cos, args)
    return MetatheoryOps.LuminalCos(args[1]) 
end

function convert_op_to_expr(op::Log2, args)
    return MetatheoryOps.LuminalLog2(args[1])
end

function convert_op_to_expr(op::Exp2, args)
    return MetatheoryOps.LuminalExp2(args[1])
end

function convert_op_to_expr(op::Recip, args)
    return MetatheoryOps.LuminalRecip(args[1])
end

function convert_op_to_expr(op::ReLU, args)
    return MetatheoryOps.LuminalReLU(args[1])
end

function convert_op_to_expr(op::MatMul, args)
    return MetatheoryOps.LuminalMatMul(args[1], args[2])
end

function convert_op_to_expr(op::Constant, args)
    return MetatheoryOps.LuminalConstant(op.value)
end

function convert_op_to_expr(op::LuminalFunction, args)
    return :(LuminalFunction($(op.name), $(args...)))
end

# Fallback
function convert_op_to_expr(op::Op, args)
    error("Conversion not implemented for op: $(typeof(op))")
end


"""
    from_metatheory_expr(expr, graph::Graph)

Converts a Metatheory expression back into a GraphTensor, adding necessary 
nodes to the graph.
"""
function from_metatheory_expr(expr::Expr, graph::Graph)
    # This is trickier because we need to parse the expression structure
    # and reconstruct the graph operations.
    
    head = expr.head
    if head == :call
        op_symbol = expr.args[1]
        args = expr.args[2:end]
        
        # Maps Metatheory op symbols back to Luminal Ops and calls add_op!
        # This will need a mapping registry or switch statement
        
        # Example:
        if op_symbol == :LuminalAdd
            t1 = from_metatheory_expr(args[1], graph)
            t2 = from_metatheory_expr(args[2], graph)
            return t1 + t2
        elseif op_symbol == :LuminalMul
            t1 = from_metatheory_expr(args[1], graph)
            t2 = from_metatheory_expr(args[2], graph)
            return t1 * t2
        elseif op_symbol == :LuminalSub
            t1 = from_metatheory_expr(args[1], graph)
            t2 = from_metatheory_expr(args[2], graph)
            return t1 - t2
        elseif op_symbol == :LuminalDiv
            t1 = from_metatheory_expr(args[1], graph)
            t2 = from_metatheory_expr(args[2], graph)
            return t1 / t2
        elseif op_symbol == :LuminalMax
            t1 = from_metatheory_expr(args[1], graph)
            t2 = from_metatheory_expr(args[2], graph)
            return max(t1, t2)
        
        # Fused Ops
        elseif op_symbol == :LuminalFusedAddReLU
            t1 = from_metatheory_expr(args[1], graph)
            t2 = from_metatheory_expr(args[2], graph)
            # FusedAddReLU is not a high-level op usually, so use add_op!
            # We need to broadcast shapes manually or rely on broadcast_dims
            # Assume t1 and t2 are compatible
            inputs = [(t1.id, 0, t1.shape), (t2.id, 0, t2.shape)]
            # Output shape? Same as Add
            out_dims = broadcast_dims(realized_dims(t1.shape), realized_dims(t2.shape))
            output_shape = ShapeTracker(out_dims)
            return add_op!(graph, FusedAddReLU(), inputs, output_shape)

        elseif op_symbol == :LuminalReLU
            t1 = from_metatheory_expr(args[1], graph)
            return relu(t1)
            
        elseif op_symbol == :LuminalConstant
            return from_metatheory_expr(args[1], graph)
            
        elseif op_symbol == :LuminalMulAdd
            t1 = from_metatheory_expr(args[1], graph)
            t2 = from_metatheory_expr(args[2], graph)
            t3 = from_metatheory_expr(args[3], graph)
            # FusedMulAdd(x, y, z) = x * y + z
            # Inputs: x, y, z
            inputs = [(t1.id, 0, t1.shape), (t2.id, 0, t2.shape), (t3.id, 0, t3.shape)]
            # Output shape logic?
            # Start with broadcast(t1, t2)
            temp_dims = broadcast_dims(realized_dims(t1.shape), realized_dims(t2.shape))
            out_dims = broadcast_dims(temp_dims, realized_dims(t3.shape))
            output_shape = ShapeTracker(out_dims)
            return add_op!(graph, FusedMulAdd(), inputs, output_shape)
            
        elseif op_symbol == :LuminalReshape
            t1 = from_metatheory_expr(args[1], graph)
            # args[2] is shape.
            # Convert args[2] to vector of ints.
            # Since args[2] comes from Metatheory expression, it might be a Expr(:vect, ...) or Tuple
            # But converting back is hard if we don't have the value.
            # If constant folding didn't reduce it to a value, we are stuck.
            # But usually shape is constant.
            # Let's assume t1 has the data.
            # Can we infer shape?
            # Placeholder: JUST RETURN t1 (Identity for Reshape if we can't handle it)
            # This allows optimizer to work even if Reshape is ignored for now.
            return t1 
        
        elseif op_symbol == :LuminalSimpleInput
            # args[1] is the ID.
            # But we can't reuse the ID from the OLD graph directly unless we map it.
            # We should probably have a mapping from old ID to new tensor if optimizing.
            # But here we are creating a NEW graph.
            # So we create a new InputTensor?
            # Or we are reconstructing?
            # If we don't have the original data, we just create a placeholder input?
            # This is tricky. compile_with_search(graph) implies we transform the graph.
            # If we recreate inputs, they are new inputs.
            # Use `tensor(graph, ...)`?
            return tensor(graph, Int[1]) # Dummy shape? 
        end
        
        # Fallback for others
        println("Warning: Unknown op during reconstruction: $op_symbol")
        return nothing
    end
    return nothing
end

function from_metatheory_expr(val::Number, graph::Graph)
    return constant(graph, val)
end

end # module
