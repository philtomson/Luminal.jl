# Autograd Implementation for Luminal.jl

"""
    gradients(loss::GraphTensor, params::Vector{GraphTensor})

Compute gradients of the `loss` tensor with respect to each tensor in `params`.
Returns a dictionary mapping parameter node IDs to their gradient `GraphTensor`s.
"""
function gradients(loss::GraphTensor, params::Vector{GraphTensor})
    graph = loss.graph_ref
    
    # Map from node_id -> gradient GraphTensor
    grads = Dict{Int, GraphTensor}()
    
    # Seed the loss gradient with 1.0 (assuming scalar loss)
    # We should handle different shapes if loss is not a scalar, 
    # but typically loss is a scalar.
    grads[loss.id] = constant(graph, 1.0f0)
    if length(realized_dims(loss.shape)) > 0
         # If loss is not a scalar, we might need to expand the seeding gradient
         # OR just broad-multiply it. Luminal's constant is scalar.
         grads[loss.id] = expand_to(grads[loss.id], realized_dims(loss.shape))
    end

    # Sort nodes in reverse topological order
    # Since node IDs are strictly increasing during construction, 
    # we can just iterate backwards from loss.id to 1.
    for node_id in loss.id:-1:1
        haskey(grads, node_id) || continue
        
        node = graph.nodes[node_id]
        grad_out = grads[node_id]
        
        # Dispatch to VJP rules based on the Op type
        vjp_rules(node.op, node_id, node, grad_out, grads)
    end
    
    # Return only the gradients for the requested params
    return Dict(p.id => grads[p.id] for p in params if haskey(grads, p.id))
end

"""
    backward(loss::GraphTensor)

Convenience function to compute gradients for all tensors marked as `trainable` in the graph.
"""
function backward(loss::GraphTensor)
    graph = loss.graph_ref
    params = [GraphTensor(id, graph.shapes[id], graph) for id in graph.trainable]
    return gradients(loss, params)
end

function mark_trainable!(t::GraphTensor)
    push!(t.graph_ref.trainable, t.id)
end

# --- VJP Rules ---

function accumulate_grad!(grads, node_id, grad_val)
    if haskey(grads, node_id)
        grads[node_id] = grads[node_id] + grad_val
    else
        grads[node_id] = grad_val
    end
end

"""
    unbroadcast(grad::GraphTensor, target_shape::Vector{Luminal.DimType})

Sum out dimensions that were broadcasted to reach the current gradient shape.
"""
function unbroadcast(grad::GraphTensor, target_shape::Vector{Luminal.DimType})
    res = grad
    
    # 1. Reduce rank until matches target
    while length(realized_dims(res.shape)) > length(target_shape)
        res = sum(res, 1)
    end
    
    # 2. Sum out broadcasted dimensions
    # We must iterate backwards to keep indices stable if we reduce rank,
    # OR better: use fixed indices if we can keep the rank.
    # Luminal's sum reduces rank. So we need to be careful.
    
    # Let's find which dimensions in the CURRENT res shape need to be summed
    # to match the 1s in target_shape.
    curr_shape = realized_dims(res.shape)
    is_one(x) = (x isa Number && x == 1)
    
    # Pre-pad target_shape with 1s to match curr_shape length if necessary
    # (though step 1 should have handled this)
    padded_target = [ones(Int, max(0, length(curr_shape) - length(target_shape)))..., target_shape...]
    
    # Iterate backwards so summing doesn't shift the indices we haven't processed yet
    for i in length(curr_shape):-1:1
        if is_one(padded_target[i]) && !is_one(curr_shape[i])
            res = sum(res, i)
        end
    end
    
    # Final reshape to ensure exact match with target_shape
    if realized_dims(res.shape) != target_shape
        # Convert target_shape to Vector{Int} for reshape
        target_int = [x isa Int ? x : Luminal.eval_dim(x) for x in target_shape]
        res = reshape(res, target_int)
    end
    
    return res
end

function expand_to(t::GraphTensor, dims::Vector{Luminal.DimType})
    res = t
    curr_dims = realized_dims(res.shape)
    for (i, d) in enumerate(dims)
        is_one(x) = (x isa Number && x == 1)
        if length(curr_dims) < i || is_one(curr_dims[i])
             if !is_one(d)
                 # Expand requires concrete Int for size currently in HighLevelOps
                 # We might need to handle symbolic expansion if d is symbolic
                 if d isa Int
                     res = expand(res, i, d)
                 else
                     # For now, let's hope we only expand to concrete sizes in AD
                     # or we need to update expand() to support symbolic size
                     error("Symbolic expansion in autograd not yet supported: $d")
                 end
             end
        end
    end
    return res
end

# Default rule: do nothing
vjp_rules(op::Op, node_id::Int, node::Node, grad_out::GraphTensor, grads::Dict) = nothing

# --- Arithmetic Ops ---

function vjp_rules(op::Add, node_id::Int, node::Node, grad_out::GraphTensor, grads::Dict)
    # d(a+b)/da = 1, d(a+b)/db = 1
    # grad_a = grad_out * 1, grad_b = grad_out * 1
    input_a_id = node.inputs[1][1]
    input_b_id = node.inputs[2][1]
    
    accumulate_grad!(grads, input_a_id, unbroadcast(grad_out, realized_dims(node.inputs[1][3])))
    accumulate_grad!(grads, input_b_id, unbroadcast(grad_out, realized_dims(node.inputs[2][3])))
end

function vjp_rules(op::Mul, node_id::Int, node::Node, grad_out::GraphTensor, grads::Dict)
    # d(a*b)/da = b, d(a*b)/db = a
    input_a_id = node.inputs[1][1]
    input_b_id = node.inputs[2][1]
    
    # We need GraphTensors for inputs to multiply them
    graph = grad_out.graph_ref
    a = GraphTensor(input_a_id, node.inputs[1][3], graph)
    b = GraphTensor(input_b_id, node.inputs[2][3], graph)
    
    accumulate_grad!(grads, input_a_id, unbroadcast(grad_out * b, realized_dims(a.shape)))
    accumulate_grad!(grads, input_b_id, unbroadcast(grad_out * a, realized_dims(b.shape)))
end

function vjp_rules(op::Recip, node_id::Int, node::Node, grad_out::GraphTensor, grads::Dict)
    # d(1/x)/dx = -1/x^2
    input_id = node.inputs[1][1]
    graph = grad_out.graph_ref
    x = GraphTensor(input_id, node.inputs[1][3], graph)
    
    grad_x = grad_out * (-1.0f0 / (x * x))
    accumulate_grad!(grads, input_id, grad_x)
end

# --- Unary Ops ---

function vjp_rules(op::Log2, node_id::Int, node::Node, grad_out::GraphTensor, grads::Dict)
    # d(log2(x))/dx = 1 / (x * ln(2))
    input_id = node.inputs[1][1]
    graph = grad_out.graph_ref
    x = GraphTensor(input_id, node.inputs[1][3], graph)
    
    grad_x = grad_out / (x * Float32(log(2.0)))
    accumulate_grad!(grads, input_id, grad_x)
end

function vjp_rules(op::Exp2, node_id::Int, node::Node, grad_out::GraphTensor, grads::Dict)
    # d(2^x)/dx = 2^x * ln(2)
    input_id = node.inputs[1][1]
    graph = grad_out.graph_ref
    x = GraphTensor(input_id, node.inputs[1][3], graph)
    
    # The output of this node IS exp2(x), so we can reuse it
    out = GraphTensor(node_id, graph.shapes[node_id], graph)
    grad_x = grad_out * out * Float32(log(2.0))
    accumulate_grad!(grads, input_id, grad_x)
end

function vjp_rules(op::Sin, node_id::Int, node::Node, grad_out::GraphTensor, grads::Dict)
    # d(sin(x))/dx = cos(x)
    input_id = node.inputs[1][1]
    graph = grad_out.graph_ref
    x = GraphTensor(input_id, node.inputs[1][3], graph)
    
    grad_x = grad_out * Base.cos(x)
    accumulate_grad!(grads, input_id, grad_x)
end

function vjp_rules(op::Sqrt, node_id::Int, node::Node, grad_out::GraphTensor, grads::Dict)
    # d(sqrt(x))/dx = 0.5 / sqrt(x)
    input_id = node.inputs[1][1]
    graph = grad_out.graph_ref
    
    out = GraphTensor(node_id, graph.shapes[node_id], graph)
    grad_x = grad_out * (0.5f0 / out)
    accumulate_grad!(grads, input_id, grad_x)
end

function vjp_rules(op::ReLU, node_id::Int, node::Node, grad_out::GraphTensor, grads::Dict)
    # d(relu(x))/dx = 1 if x > 0 else 0
    input_id = node.inputs[1][1]
    graph = grad_out.graph_ref
    x = GraphTensor(input_id, node.inputs[1][3], graph)
    
    grad_x = grad_out * (x > 0.0f0)
    accumulate_grad!(grads, input_id, grad_x)
end

# --- Reduction Ops ---

function vjp_rules(op::SumReduce, node_id::Int, node::Node, grad_out::GraphTensor, grads::Dict)
    # d(sum(x, dim))/dx = expand(grad_out, dim, x.shape[dim])
    input_id = node.inputs[1][1]
    graph = grad_out.graph_ref
    x_shape = realized_dims(node.inputs[1][3])
    
    grad_x = expand(grad_out, op.dim, x_shape[op.dim])
    accumulate_grad!(grads, input_id, grad_x)
end

# --- Movement Ops ---

function vjp_rules(op::Permute, node_id::Int, node::Node, grad_out::GraphTensor, grads::Dict)
    # gradient of permute is inverse permute
    input_id = node.inputs[1][1]
    
    # Calculate inverse permutation
    inv_dims = zeros(Int, length(op.dims))
    for (i, d) in enumerate(op.dims)
        inv_dims[d] = i
    end
    
    grad_x = permute(grad_out, inv_dims)
    accumulate_grad!(grads, input_id, grad_x)
end

function vjp_rules(op::Reshape, node_id::Int, node::Node, grad_out::GraphTensor, grads::Dict)
    input_id = node.inputs[1][1]
    input_shape = realized_dims(node.inputs[1][3])
    
    grad_x = reshape(grad_out, input_shape)
    accumulate_grad!(grads, input_id, grad_x)
end

function vjp_rules(op::Expand, node_id::Int, node::Node, grad_out::GraphTensor, grads::Dict)
    # gradient of expand is sum over the expanded dimension
    input_id = node.inputs[1][1]
    grad_x = sum(grad_out, op.dim)
    accumulate_grad!(grads, input_id, grad_x)
end

function vjp_rules(op::Slice, node_id::Int, node::Node, grad_out::GraphTensor, grads::Dict)
    # gradient of slice is pad
    input_id = node.inputs[1][1]
    input_shape = realized_dims(node.inputs[1][3])
    
    padding = []
    for (i, (start, stop)) in enumerate(op.ranges)
        # stop is exclusive in Rust? In Luminal.jl slice_along uses start, stop
        # Let's check HighLevelOps logic
        # pad! adds left, right padding
        left = start
        # realized_dims[i] is the total size
        # the sliced size is (stop - start)
        # so right padding is (input_shape[i] - stop)
        # Assuming stop is NOT typemax(Int) here because it's a realized Op
        actual_stop = stop == typemax(Int) ? input_shape[i] : stop
        right = input_shape[i] - actual_stop
        push!(padding, (left, right))
    end
    
    grad_x = pad(grad_out, convert(Vector{Tuple{Int, Int}}, padding))
    accumulate_grad!(grads, input_id, grad_x)
end

function vjp_rules(op::Pad, node_id::Int, node::Node, grad_out::GraphTensor, grads::Dict)
    # gradient of pad is slice
    input_id = node.inputs[1][1]
    input_shape = realized_dims(node.inputs[1][3])
    
    slices = []
    for (i, (left, right)) in enumerate(op.padding)
        push!(slices, (left, left + input_shape[i]))
    end
    
    grad_x = slice(grad_out, convert(Vector{Tuple{Int, Int}}, slices))
    accumulate_grad!(grads, input_id, grad_x)
end

# --- MatMul Op ---

function vjp_rules(op::MatMul, node_id::Int, node::Node, grad_out::GraphTensor, grads::Dict)
    # d(A*B)/dA = grad_out * B^T
    # d(A*B)/dB = A^T * grad_out
    input_a_id = node.inputs[1][1]
    input_b_id = node.inputs[2][1]
    graph = grad_out.graph_ref
    a = GraphTensor(input_a_id, node.inputs[1][3], graph)
    b = GraphTensor(input_b_id, node.inputs[2][3], graph)
    
    # rank of matmul is >= 2
    rank_a = length(realized_dims(a.shape))
    rank_b = length(realized_dims(b.shape))
    
    perm_a = [collect(1:rank_a-2)..., rank_a, rank_a-1]
    perm_b = [collect(1:rank_b-2)..., rank_b, rank_b-1]
    
    grad_a = matmul(grad_out, permute(b, perm_b))
    grad_b = matmul(permute(a, perm_a), grad_out)
    
    accumulate_grad!(grads, input_a_id, unbroadcast(grad_a, realized_dims(a.shape)))
    accumulate_grad!(grads, input_b_id, unbroadcast(grad_b, realized_dims(b.shape)))
end
