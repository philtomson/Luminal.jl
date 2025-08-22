# Implements a simple interpreter to execute a computation graph.

function execute(graph::Graph, output_id::Int, initial_inputs::Dict)
    # Results of each node are stored here
    results = Dict{Int, Any}()

    # Populate initial inputs
    for (id, value) in initial_inputs
        results[id] = value
    end

    # Iterate through nodes in topological order (our current construction order)
    for (node_id, node) in enumerate(graph.nodes)
        # Get the operation and input node IDs
        op = node.op
        input_node_ids = [inp[1] for inp in node.inputs]

        # Fetch the computed values of the inputs
        input_values = [results[id] for id in input_node_ids]

        # Perform the computation based on the op type
        current_result = nothing
        if op isa Add
            current_result = input_values[1] + input_values[2]
        elseif op isa Mul
            current_result = input_values[1] * input_values[2]
        elseif op isa Log2
            current_result = log2.(input_values[1])
        elseif op isa Exp2
            current_result = exp2.(input_values[1])
        elseif op isa Sin
            current_result = sin.(input_values[1])
        elseif op isa Sqrt
            current_result = sqrt.(input_values[1])
        elseif op isa Reshape
            current_result = reshape(input_values[1], op.shape...)
        elseif op isa Permute
            current_result = permutedims(input_values[1], op.dims)
        elseif op isa Expand
            # This is a simplified expand, assuming it adds a new dimension and repeats
            new_shape = [size(input_values[1])...]
            insert!(new_shape, op.dim, 1)
            reshaped = reshape(input_values[1], new_shape...)
            # Construct repeats tuple for the new dimension
            repeats = ones(Int, length(new_shape))
            repeats[op.dim] = op.size
            current_result = repeat(reshaped, outer=repeats)
        elseif op isa MatMul
            current_result = input_values[1] * input_values[2]
        elseif op isa SumReduce
            current_result = sum(input_values[1], dims=op.dim)
        elseif op isa MaxReduce
            current_result = maximum(input_values[1], dims=op.dim)
        elseif op isa Constant
            current_result = op.value
        elseif op isa Function
            # This is an input tensor, its value should already be in the results dict
            current_result = results[node_id]
        # NOTE: More ops will be added here as we port them.
        else
            error("Execution for op ", typeof(op), " not implemented yet.")
        end
        
        results[node_id] = current_result
    end

    # Return the final result for the requested output node
    return results[output_id]
end
