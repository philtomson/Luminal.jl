import Base: +, *, log2, exp2, sin, sqrt

# Binary Ops
# ----------

"""
    +(a::GraphTensor, b::GraphTensor)

Overload the addition operator to build the computation graph.
"""
function Base.:+(a::GraphTensor, b::GraphTensor)
    @assert a.graph_ref === b.graph_ref "Tensors must be from the same graph"
    # Broadcasting logic will be handled by the ShapeTracker in the future.
    # For now, we assume shapes are compatible and the output shape is the same as input `a`.
    inputs = [(a.id, 0), (b.id, 0)]
    return add_op!(a.graph_ref, Add(), inputs, a.shape)
end

"""
    *(a::GraphTensor, b::GraphTensor)

Overload the multiplication operator to build the computation graph.
"""
function Base.:*(a::GraphTensor, b::GraphTensor)
    @assert a.graph_ref === b.graph_ref "Tensors must be from the same graph"
    inputs = [(a.id, 0), (b.id, 0)]
    return add_op!(a.graph_ref, Mul(), inputs, a.shape)
end

function max(a::GraphTensor, b::GraphTensor)
    @assert a.graph_ref === b.graph_ref "Tensors must be from the same graph"
    inputs = [(a.id, 0), (b.id, 0)]
    return add_op!(a.graph_ref, Max(), inputs, a.shape)
end

# Unary Ops
# ---------

"""
    log2(a::GraphTensor)

Overload the base-2 logarithm function.
"""
function Base.log2(a::GraphTensor)
    inputs = [(a.id, 0)]
    return add_op!(a.graph_ref, Log2(), inputs, a.shape)
end

"""
    exp2(a::GraphTensor)

Overload the base-2 exponent function.
"""
function Base.exp2(a::GraphTensor)
    inputs = [(a.id, 0)]
    return add_op!(a.graph_ref, Exp2(), inputs, a.shape)
end

"""
    sin(a::GraphTensor)

Overload the sine function.
"""
function Base.sin(a::GraphTensor)
    inputs = [(a.id, 0)]
    return add_op!(a.graph_ref, Sin(), inputs, a.shape)
end

"""
    sqrt(a::GraphTensor)

Overload the square root function.
"""
function Base.sqrt(a::GraphTensor)
    inputs = [(a.id, 0)]
    return add_op!(a.graph_ref, Sqrt(), inputs, a.shape)
end

function relu(a::GraphTensor)
    inputs = [(a.id, 0)]
    return add_op!(a.graph_ref, ReLU(), inputs, a.shape)
end

# Movement Ops
# ------------

function reshape(a::GraphTensor, new_shape_vec::Vector{Int})
    # Convert integer shape to symbolic expressions
    new_dims = [Expression([Num(d)]) for d in new_shape_vec]
    # Use the ShapeTracker's reshape method
    output_shape = reshape(a.shape, new_dims)
    inputs = [(a.id, 0)]
    return add_op!(a.graph_ref, Reshape(new_dims), inputs, output_shape)
end

function permute(a::GraphTensor, dims::Vector{Int})
    output_shape = deepcopy(a.shape)
    permute!(output_shape, dims)
    inputs = [(a.id, 0)]
    return add_op!(a.graph_ref, Permute(dims), inputs, output_shape)
end

# Note: expand is more complex and will be updated later.

# Matmul
# ------
function matmul(a::GraphTensor, b::GraphTensor)
    @assert a.graph_ref === b.graph_ref "Tensors must be from the same graph"
    
    # This is a simplified shape calculation. A full implementation would use the ShapeTracker.
    # For now, we manually construct the output shape vector.
    a_dims = [Symbolic.execute(d, Dict{Char, Int64}()) for d in a.shape.dims] # Evaluate expressions
    b_dims = [Symbolic.execute(d, Dict{Char, Int64}()) for d in b.shape.dims] # Evaluate expressions
    @assert length(a_dims) >= 2 && length(b_dims) >= 2 "Matmul inputs must be at least 2D"
    @assert a_dims[2] == b_dims[1] "Inner dimensions must match"
    if length(a_dims) > 2
        @assert a_dims[3:end] == b_dims[3:end] "Batch dimensions must match"
    end

    output_shape_vec = [a_dims[1], b_dims[2]]
    if length(a_dims) > 2
        append!(output_shape_vec, a_dims[3:end])
    end
    output_dims_expr = [Expression([Num(d)]) for d in output_shape_vec]
    output_shape = ShapeTracker(output_dims_expr)

    inputs = [(a.id, 0), (b.id, 0)]
    return add_op!(a.graph_ref, MatMul(), inputs, output_shape)
end

# Reduction Ops
# -------------

function sum(a::GraphTensor, dim::Int)
    # This is a simplified shape calculation.
    output_dims = deepcopy(a.shape.dims)
    deleteat!(output_dims, dim)
    output_shape = ShapeTracker(output_dims)
    inputs = [(a.id, 0)]
    return add_op!(a.graph_ref, SumReduce(dim), inputs, output_shape)
end

function max(a::GraphTensor, dim::Int)
    output_dims = deepcopy(a.shape.dims)
    deleteat!(output_dims, dim)
    output_shape = ShapeTracker(output_dims)
    inputs = [(a.id, 0)]
    return add_op!(a.graph_ref, MaxReduce(dim), inputs, output_shape)
end