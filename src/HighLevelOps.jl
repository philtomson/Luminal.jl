import Base: +, -, *, /, %, <, >, <=, >=, ==, !=, log2, exp2, sin, cos, exp, sqrt, max, min, abs, sum, maximum

# Binary Arithmetics with Scalar support
# -------------------------------------

function broadcast_dims(dims1, dims2)
    N1 = length(dims1)
    N2 = length(dims2)
    N = max(N1, N2)
    # Prepend 1s
    d1 = [ones(Int, N - N1)..., dims1...]
    d2 = [ones(Int, N - N2)..., dims2...]
    res = Luminal.DimType[]
    for i in 1:N
        v1 = d1[i]
        v2 = d2[i]
        
        # Helper to check for 1
        is_one(x) = (x isa Number && x == 1)
        
        if v1 == v2
            push!(res, v1)
        elseif is_one(v1)
            push!(res, v2)
        elseif is_one(v2)
            push!(res, v1)
        else
            if v1 isa Int && v2 isa Int
                 error("Dimension mismatch in broadcasting: $v1 vs $v2")
            end
            # Prefer symbolic or keep v1
            push!(res, v1)
        end
    end
    return res
end

function Base.:+(a::GraphTensor, b::GraphTensor)
    @assert a.graph_ref === b.graph_ref "Tensors must be from the same graph"
    inputs = [(a.id, 0, a.shape), (b.id, 0, b.shape)]
    out_dims = broadcast_dims(realized_dims(a.shape), realized_dims(b.shape))
    output_shape = ShapeTracker(out_dims)
    return add_op!(a.graph_ref, Add(), inputs, output_shape)
end
Base.:+(a::GraphTensor, b::Number) = a + constant(a.graph_ref, b)
Base.:+(a::Number, b::GraphTensor) = b + a

function Base.:*(a::GraphTensor, b::GraphTensor)
    @assert a.graph_ref === b.graph_ref "Tensors must be from the same graph"
    inputs = [(a.id, 0, a.shape), (b.id, 0, b.shape)]
    out_dims = broadcast_dims(realized_dims(a.shape), realized_dims(b.shape))
    output_shape = ShapeTracker(out_dims)
    return add_op!(a.graph_ref, Mul(), inputs, output_shape)
end
Base.:*(a::GraphTensor, b::Number) = a * constant(a.graph_ref, b)
Base.:*(a::Number, b::GraphTensor) = b * a

Base.:-(a::GraphTensor) = a * -1.0f0
Base.:-(a::GraphTensor, b::GraphTensor) = a + (-b)
Base.:-(a::GraphTensor, b::Number) = a + (-b)
Base.:-(a::Number, b::GraphTensor) = constant(b.graph_ref, a) - b

Base.:/(a::GraphTensor, b::GraphTensor) = a * reciprocal(b)
Base.:/(a::GraphTensor, b::Number) = a * (1.0f0 / Float32(b))
Base.:/(a::Number, b::GraphTensor) = constant(b.graph_ref, a) / b

function Base.:%(a::GraphTensor, b::GraphTensor)
    @assert a.graph_ref === b.graph_ref "Tensors must be from the same graph"
    inputs = [(a.id, 0, a.shape), (b.id, 0, b.shape)]
    out_dims = broadcast_dims(realized_dims(a.shape), realized_dims(b.shape))
    output_shape = ShapeTracker(out_dims)
    return add_op!(a.graph_ref, Mod(), inputs, output_shape)
end
Base.:%(a::GraphTensor, b::Number) = a % constant(a.graph_ref, b)

# Comparisons
# -----------

function Base.:<(a::GraphTensor, b::GraphTensor)
    @assert a.graph_ref === b.graph_ref "Tensors must be from the same graph"
    inputs = [(a.id, 0, a.shape), (b.id, 0, b.shape)]
    out_dims = broadcast_dims(realized_dims(a.shape), realized_dims(b.shape))
    output_shape = ShapeTracker(out_dims)
    return add_op!(a.graph_ref, LessThan(), inputs, output_shape)
end
Base.:<(a::GraphTensor, b::Number) = a < constant(a.graph_ref, b)
Base.:<(a::Number, b::GraphTensor) = constant(b.graph_ref, a) < b

Base.:>(a::GraphTensor, b::GraphTensor) = b < a
Base.:>(a::GraphTensor, b::Number) = a > constant(a.graph_ref, b)
Base.:>(a::Number, b::GraphTensor) = constant(b.graph_ref, a) > b

Base.:<=(a::GraphTensor, b::GraphTensor) = (a > b) * -1.0f0 + 1.0f0
Base.:<=(a::GraphTensor, b::Number) = a <= constant(a.graph_ref, b)

Base.:>=(a::GraphTensor, b::GraphTensor) = (a < b) * -1.0f0 + 1.0f0
Base.:>=(a::GraphTensor, b::Number) = a >= constant(a.graph_ref, b)

Base.:!=(a::GraphTensor, b::GraphTensor) = (a < b) + (a > b)
Base.:!=(a::GraphTensor, b::Number) = a != constant(a.graph_ref, b)

Base.:(==)(a::GraphTensor, b::GraphTensor) = (a != b) * -1.0f0 + 1.0f0
Base.:(==)(a::GraphTensor, b::Number) = a == constant(a.graph_ref, b)

# Unary Ops
# ---------

function Base.log2(a::GraphTensor)
    inputs = [(a.id, 0, a.shape)]
    return add_op!(a.graph_ref, Log2(), inputs, a.shape)
end

function Base.exp2(a::GraphTensor)
    inputs = [(a.id, 0, a.shape)]
    return add_op!(a.graph_ref, Exp2(), inputs, a.shape)
end

function Base.sin(a::GraphTensor)
    inputs = [(a.id, 0, a.shape)]
    return add_op!(a.graph_ref, Sin(), inputs, a.shape)
end

function Base.cos(a::GraphTensor)
    return sin(Float32(pi/2) - a)
end

function Base.exp(a::GraphTensor)
    return exp2(a * (1.0f0 / log(2.0f0)))
end

function Base.sqrt(a::GraphTensor)
    inputs = [(a.id, 0, a.shape)]
    return add_op!(a.graph_ref, Sqrt(), inputs, a.shape)
end

function reciprocal(a::GraphTensor)
    inputs = [(a.id, 0, a.shape)]
    return add_op!(a.graph_ref, Recip(), inputs, a.shape)
end

function relu(a::GraphTensor)
    inputs = [(a.id, 0, a.shape)]
    return add_op!(a.graph_ref, ReLU(), inputs, a.shape)
end

function Base.abs(a::GraphTensor)
    return relu(a) + relu(-a)
end

# Activations
# -----------

function sigmoid(a::GraphTensor)
    return 1.0f0 / (1.0f0 + exp2(-a * (1.0f0 / log(2.0f0))))
end

function swish(a::GraphTensor)
    return a * sigmoid(a)
end
const silu = swish

function gelu(a::GraphTensor)
    return a * 0.5f0 * (1.0f0 + tanh(0.7978845608f0 * a * (1.0f0 + 0.044715f0 * a * a)))
end

function Base.tanh(a::GraphTensor)
    return sigmoid(a * 2.0f0) * 2.0f0 - 1.0f0
end

# Movement Ops
# ------------

function reshape(a::GraphTensor, new_shape_vec::Vector{Int})
    # Luminal philosophy: Reshape always works on contiguous data
    a_cont = contiguous(a)
    # Calculate target dimensions
    # For concrete op, we want a clean ShapeTracker with the new dimensions
    output_shape = ShapeTracker(new_shape_vec)
    inputs = [(a_cont.id, 0, a_cont.shape)]
    return add_op!(a_cont.graph_ref, Reshape(new_shape_vec), inputs, output_shape)
end

function permute(a::GraphTensor, dims::Vector{Int})
    # Calculate resulting dimensions
    calc_shape = deepcopy(a.shape)
    permute!(calc_shape, dims)
    final_dims = realized_dims(calc_shape)
    
    output_shape = ShapeTracker(final_dims)
    inputs = [(a.id, 0, a.shape)]
    return add_op!(a.graph_ref, Permute(dims), inputs, output_shape)
end

function expand(a::GraphTensor, dim::Int, size::Int)
    calc_shape = expand(a.shape, dim, size)
    final_dims = realized_dims(calc_shape)
    output_shape = ShapeTracker(final_dims)
    inputs = [(a.id, 0, a.shape)]
    return add_op!(a.graph_ref, Expand(dim, size), inputs, output_shape)
end

function contiguous(a::GraphTensor)
    if !is_reshaped(a.shape)
        return a
    end
    # Contiguous produces a clean shape matching the current realized dims
    final_dims = realized_dims(a.shape)
    output_shape = ShapeTracker(final_dims)
    inputs = [(a.id, 0, a.shape)]
    return add_op!(a.graph_ref, Contiguous(), inputs, output_shape)
end

function pad(a::GraphTensor, padding_vec::Vector{Tuple{Int, Int}})
    calc_shape = deepcopy(a.shape)
    pad!(calc_shape, padding_vec)
    final_dims = realized_dims(calc_shape)
    output_shape = ShapeTracker(final_dims)
    inputs = [(a.id, 0, a.shape)]
    return add_op!(a.graph_ref, Pad(padding_vec), inputs, output_shape)
end

function pad_along(a::GraphTensor, axis::Int, left::Int, right::Int)
    p = fill((0, 0), length(a.shape.indexes))
    p[axis] = (left, right)
    return pad(a, p)
end

function slice(a::GraphTensor, slice_vec::Vector{Tuple{Int, Int}})
    calc_shape = deepcopy(a.shape)
    slice!(calc_shape, slice_vec)
    final_dims = realized_dims(calc_shape)
    output_shape = ShapeTracker(final_dims)
    inputs = [(a.id, 0, a.shape)]
    return add_op!(a.graph_ref, Slice(slice_vec), inputs, output_shape)
end

function slice_along(a::GraphTensor, axis::Int, start::Int, stop::Int)
    s = fill((0, typemax(Int)), length(a.shape.indexes))
    s[axis] = (start, stop)
    return slice(a, s)
end

function concat_along(a::GraphTensor, b::GraphTensor, axis::Int)
    # Pad and add
    a_padded = pad_along(a, axis, 0, realized_dims(b.shape)[axis])
    b_padded = pad_along(b, axis, realized_dims(a.shape)[axis], 0)
    return a_padded + b_padded
end

# Matmul
# ------
function matmul(a::GraphTensor, b::GraphTensor)
    @assert a.graph_ref === b.graph_ref "Tensors must be from the same graph"
    a_dims = realized_dims(a.shape)
    b_dims = realized_dims(b.shape)
    @assert length(a_dims) >= 2 && length(b_dims) >= 2 "Matmul inputs must be at least 2D"
    
    # Simple matmul shape logic: (..., M, K) * (..., K, N) -> (..., M, N)
    output_shape_vec = [a_dims[1:(end-1)]..., b_dims[end]]
    output_shape = ShapeTracker(output_shape_vec)

    inputs = [(a.id, 0, a.shape), (b.id, 0, b.shape)]
    return add_op!(a.graph_ref, MatMul(), inputs, output_shape)
end

# Reduction Ops
# -------------

function sum(a::GraphTensor, dim::Int)
    output_dims = deepcopy(realized_dims(a.shape))
    deleteat!(output_dims, dim)
    output_shape = ShapeTracker(output_dims)
    inputs = [(a.id, 0, a.shape)]
    return add_op!(a.graph_ref, SumReduce(dim), inputs, output_shape)
end

function maximum(a::GraphTensor, b::GraphTensor)
    @assert a.graph_ref === b.graph_ref "Tensors must be from the same graph"
    # (a < b) * b + (b <= a) * a
    return (a < b) * b + (b <= a) * a
end
maximum(a::GraphTensor, b::Number) = maximum(a, constant(a.graph_ref, b))

function mean(a::GraphTensor, dim::Int)
    dim_size = realized_dims(a.shape)[dim]
    return sum(a, dim) * (1.0f0 / Float32(dim_size))
end

# Normalizations
# --------------

function mean_norm(a::GraphTensor, dim::Int)
    m = mean(a, dim)
    # Expand result back to original shape for subtraction
    m_expanded = expand(m, dim, realized_dims(a.shape)[dim])
    return a - m_expanded
end

function std_norm(a::GraphTensor, dim::Int, epsilon::Float32=1f-5)
    var = mean(a * a, dim)
    inv_std = reciprocal(sqrt(var + epsilon))
    inv_std_expanded = expand(inv_std, dim, realized_dims(a.shape)[dim])
    return a * inv_std_expanded
end

function layer_norm(a::GraphTensor, dim::Int, epsilon::Float32=1f-5)
    return std_norm(mean_norm(a, dim), dim, epsilon)
end

# Other Ops
# ---------

function softmax(a::GraphTensor, dim::Int)
    m = max_reduce(a, dim)
    m_expanded = expand(m, dim, realized_dims(a.shape)[dim])
    shifted = a - m_expanded
    e = exp(shifted)
    return e / expand(sum(e, dim), dim, realized_dims(a.shape)[dim])
end

function max_reduce(a::GraphTensor, dim::Int)
    output_dims = deepcopy(realized_dims(a.shape))
    deleteat!(output_dims, dim)
    output_shape = ShapeTracker(output_dims)
    inputs = [(a.id, 0, a.shape)]
    return add_op!(a.graph_ref, MaxReduce(dim), inputs, output_shape)
end

# arange and gather
# -----------------

function arange(graph::Graph, to::Int)
    if to == 1
        return expand(constant(graph, 0.0f0), 1, 1)
    else
        one = constant(graph, 1.0f0)
        expanded = expand(one, 1, to)
        return cumsum_last_dim(expanded) - 1.0f0
    end
end

function cumsum_last_dim(a::GraphTensor)
    axis = length(a.shape.indexes)
    # Force contiguity
    a_cont = contiguous(a)
    orig_length = realized_dims(a_cont.shape)[axis]
    
    inputs = [(a_cont.id, 0, a_cont.shape)]
    output_shape = deepcopy(a_cont.shape)
    return add_op!(a.graph_ref, Function("CumSum"), inputs, output_shape)
end

function triu(graph::Graph, size::Int, diagonal::Int=0)
    h = expand(arange(graph, size), 1, size) # (size, 1) then expanded to (size, size)
    v = expand(arange(graph, size), 2, size) # (size) -> expand(2, size) -> (size, size)
    # Wait, arange(size) is (size). 
    # expand(arange(size), 1, size) -> (size, size) where 1st dim is expanded.
    # So h is (size, size) where each row is [0, 1, ..., N-1].
    # v is (size, size) where each column is [0, 1, ..., N-1].
    return h - Float32(diagonal - 1) > v
end

function gather(matrix::GraphTensor, indexes::GraphTensor)
    @assert matrix.graph_ref === indexes.graph_ref "Tensors must be from the same graph"
    m_cont = contiguous(matrix)
    idx_cont = contiguous(indexes)
    
    m_dims = realized_dims(m_cont.shape)
    dim = m_dims[2]
    idx_dims = realized_dims(idx_cont.shape)
    output_shape_vec = [idx_dims..., dim]
    
    inputs = [(m_cont.id, 0, m_cont.shape), (idx_cont.id, 0, idx_cont.shape)]
    output_shape = ShapeTracker(output_shape_vec)
    return add_op!(m_cont.graph_ref, Function("Gather"), inputs, output_shape)
end
function flash_attention(q::GraphTensor, k::GraphTensor, v::GraphTensor; scale=nothing, causal=false)
    # q, k, v shape: (Batch, Head, Seq, HeadDim)
    q_dims = realized_dims(q.shape)
    if isnothing(scale)
        scale = 1.0f0 / sqrt(Float32(q_dims[end]))
    end
    
    inputs = [(q.id, 0, q.shape), (k.id, 0, k.shape), (v.id, 0, v.shape)]
    output_shape = q.shape # Output has same shape as Q
    
    return add_op!(q.graph_ref, FlashAttentionOp(Float32(scale), causal), inputs, output_shape)
end

function unfold(a::GraphTensor, kernel_shape::Vector{Int}, stride_shape::Vector{Int}, dilation_shape::Vector{Int})
    spatial = length(kernel_shape)
    rank = length(realized_dims(a.shape))
    @assert rank > spatial "Input rank $rank must be greater than spatial dims $spatial"
    
    batch_len = rank - spatial - 1
    a_dims = realized_dims(a.shape)
    
    out_spatial = Int[]
    for i in 1:spatial
        # evaluate the dynamic/static dimension size
        s_i = typeof(a_dims[batch_len + 1 + i]) == Int ? a_dims[batch_len + 1 + i] : 1 # or throw for symbolic for now
        if typeof(a_dims[batch_len + 1 + i]) != Int
             error("Unfold currently requires concrete spatial dimensions, got $(a_dims[batch_len + 1 + i])")
        end
        
        k_i = kernel_shape[i]
        d_i = dilation_shape[i]
        st_i = stride_shape[i]
        
        o_i = (s_i - d_i * (k_i - 1) - 1) ÷ st_i + 1
        push!(out_spatial, o_i)
    end
    
    # New shape: [batch..., channels, out_spatial..., kernel_shape...]
    output_shape_vec = [a_dims[1:batch_len+1]..., out_spatial..., kernel_shape...]
    output_shape = ShapeTracker(output_shape_vec)
    
    inputs = [(a.id, 0, a.shape)]
    return add_op!(a.graph_ref, Unfold(kernel_shape, stride_shape, dilation_shape), inputs, output_shape)
end
