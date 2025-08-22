module SymbolicsIntegration

using Symbolics
using SymbolicUtils
using ..Luminal # This makes Luminal itself available

# Function to convert a Luminal GraphTensor to a Symbolics.jl expression
function luminal_to_symbolics(g_tensor::Luminal.GraphTensor)
    # Recursive helper function to convert a Luminal node to a Symbolics.jl expression
    function _luminal_node_to_symbolics(graph::Luminal.Graph, node_id::Int, visited_nodes::Dict{Int,Any})
        if haskey(visited_nodes, node_id)
            return visited_nodes[node_id]
        end

        node = graph.nodes[node_id]
        op = node.op
        
        # Base cases: Constant and Function (input) ops
        if op isa Luminal.Constant
            sym_expr = Num(op.value)
        elseif op isa Luminal.Function # This is our input tensor
            # Create a unique symbolic variable for each input tensor
            sym_expr = Symbolics.variable(Symbol(op.name, node_id))
        else # Recursive case for other ops
            input_sym_exprs = [_luminal_node_to_symbolics(graph, input_id, visited_nodes) for (input_id, _) in node.inputs]

            # Map Luminal Ops to Symbolics.jl operations (functions)
            sym_op = nothing
            if op isa Luminal.Add
                sym_op = Base.:+
            elseif op isa Luminal.Mul
                sym_op = Base.:*
            elseif op isa Luminal.Max
                sym_op = Base.max
            elseif op isa Luminal.ReLU
                sym_expr = Base.max(input_sym_exprs[1], 0) # Convert ReLU to max(x, 0)
            # Add more mappings for other Ops as needed
            else
                error("Unsupported Luminal Op for Symbolics.jl conversion: $(typeof(op))")
            end
            
            # If sym_expr was not assigned in a special case (like ReLU), apply the mapped operator
            if !@isdefined(sym_expr) || sym_expr === nothing # Check if sym_expr was already assigned in a special case
                sym_expr = sym_op(input_sym_exprs...)
            end
        end
        
        visited_nodes[node_id] = sym_expr
        return sym_expr
    end

    # Start the recursive conversion from the graph tensor's node
    return _luminal_node_to_symbolics(g_tensor.graph_ref, g_tensor.id, Dict{Int,Any}())
end

# Function to convert a Symbolics.jl expression back to a Luminal GraphTensor
function symbolics_to_luminal(sym_expr)
    g = Luminal.Graph()
    visited_exprs = Dict{Any,Luminal.GraphTensor}() # Map Symbolics.jl expr to Luminal.GraphTensor

    function _symbolics_to_luminal_node(current_sym_expr, graph::Luminal.Graph, visited_exprs::Dict{Any,Luminal.GraphTensor})
        if haskey(visited_exprs, current_sym_expr)
            return visited_exprs[current_sym_expr]
        end

        local luminal_op::Luminal.Op
        local inputs::Vector{Tuple{Int,Int}}
        local output_shape::Luminal.ShapeTracker # Placeholder for now

        if SymbolicUtils.isterm(current_sym_expr) # Check for operations first (SymbolicUtils.Term)
            op_func = SymbolicUtils.operation(current_sym_expr)
            args = SymbolicUtils.arguments(current_sym_expr)
            
            inputs = Vector{Tuple{Int,Int}}()
            for arg in args
                # Recursively convert arguments
                input_g_tensor = _symbolics_to_luminal_node(arg, graph, visited_exprs)
                push!(inputs, (input_g_tensor.id, 0)) # Assuming output index is always 0 for now
            end

            # Map Symbolics.jl operations back to Luminal Ops
            if op_func == Base.:+
                luminal_op = Luminal.Add()
            elseif op_func == Base.:*
                luminal_op = Luminal.Mul()
            elseif op_func == Base.max
                luminal_op = Luminal.Max() # Max is used for ReLU canonicalization
            # Add more mappings as needed
            else
                error("Unsupported Symbolics.jl operation for Luminal conversion: $(op_func)")
            end
            
            # For now, assume scalar output shape
            output_shape = Luminal.ShapeTracker([Luminal.Symbolic.Expression([Luminal.Symbolic.Num(1)])])

        elseif current_sym_expr isa Symbolics.Num # Symbolic variable or constant (Num)
            if SymbolicUtils.issym(current_sym_expr) # If it's a symbolic variable
                name_str = string(SymbolicUtils.operation(current_sym_expr))
                luminal_op = Luminal.Function(name_str)
                inputs = Vector{Tuple{Int,Int}}()
                output_shape = Luminal.ShapeTracker([Luminal.Symbolic.Expression([Luminal.Symbolic.Num(1)])])
            else # If it's a symbolic constant (e.g., Num(5))
                luminal_op = Luminal.Constant(SymbolicUtils.value(current_sym_expr))
                inputs = Vector{Tuple{Int,Int}}()
                output_shape = Luminal.ShapeTracker([Luminal.Symbolic.Expression([Luminal.Symbolic.Num(1)])])
            end
        elseif current_sym_expr isa Number # Literal constant (e.g., 0, 1.0)
            luminal_op = Luminal.Constant(current_sym_expr)
            inputs = Vector{Tuple{Int,Int}}()
            output_shape = Luminal.ShapeTracker([Luminal.Symbolic.Expression([Luminal.Symbolic.Num(1)])])
        else
            error("Unsupported Symbolics.jl expression type for Luminal conversion: $(typeof(current_sym_expr))")
        end
        
        # Add the Luminal Op to the graph and create a GraphTensor
        luminal_g_tensor = Luminal.add_op!(graph, luminal_op, inputs, output_shape)
        visited_exprs[current_sym_expr] = luminal_g_tensor
        return luminal_g_tensor
    end

    return _symbolics_to_luminal_node(sym_expr, g, visited_exprs)
end

export luminal_to_symbolics, symbolics_to_luminal

end # module SymbolicsIntegration