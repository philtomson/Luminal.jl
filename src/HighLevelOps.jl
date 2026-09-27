import Base: +, -, *, /, %, <, >, <=, >=, ==, !=, log2, exp2, sin, cos, exp, sqrt, max, min, abs, sum, maximum

# Binary Arithmetics with Scalar support
# -------------------------------------

function broadcast_dims(dims1, dims2)
    N1 = length(dims1)
    N2 = length(dims2)
    N = max(N1, N2)
    # Append 1s (Left-aligned broadcasting for Julia column-major)
    d1 = [dims1..., ones(Int, N - N1)...]
    d2 = [dims2..., ones(Int, N - N2)...]
    res = Luminal.DimType[]
    for i in 1:N
        v1 = d1[i]
        v2 = d2[i]
        
        # Helper to check for 1
        is_one(x) = (x isa Number && x == 1)
        
        if isequal(v1, v2)
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

# Tensor / tensor is one exact Div; tensor / number stays a multiply by the
# (rounded) reciprocal, the cheap form scaling paths rely on.
function Base.:/(a::GraphTensor, b::GraphTensor)
    @assert a.graph_ref === b.graph_ref "Tensors must be from the same graph"
    inputs = [(a.id, 0, a.shape), (b.id, 0, b.shape)]
    out_dims = broadcast_dims(realized_dims(a.shape), realized_dims(b.shape))
    return add_op!(a.graph_ref, Div(), inputs, ShapeTracker(out_dims))
end
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
    inputs = [(a.id, 0, a.shape)]
    return add_op!(a.graph_ref, Exp(), inputs, a.shape)
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

"""
    gather(data, coords::Vector{GraphTensor})

Coordinate-form gather: `out[c] = data[coords[1][c], .., coords[r][c]]`, one
coordinate tensor per axis of `data` (0-based, as Float32 values), all of the
same shape, which is the output's. Out-of-range coordinates read 0. (The
two-tensor `gather(weight, ids)` is the embedding row lookup.)
"""
function gather(data::GraphTensor, coords::AbstractVector{GraphTensor})
    d = _coords_dims(data, coords)
    inputs = vcat([(data.id, 0, data.shape)], [(c.id, 0, c.shape) for c in coords])
    return add_op!(data.graph_ref, GatherND(), inputs, ShapeTracker(d))
end

"""
    scatter(init, src, coords::Vector{GraphTensor}; mode=:replace)

Coordinate-form scatter: a copy of `init` with `out[coords[1][c], .., coords[r][c]]`
set to `src[c]` (`mode=:replace`) or incremented by it (`mode=:add`, atomic, so
repeated coordinates accumulate). One coordinate tensor per axis of `init`,
each of `src`'s shape; out-of-range writes are dropped. With `:replace`, which of
several writes to one element survives is unspecified on the GPU.
"""
function scatter(init::GraphTensor, src::GraphTensor, coords::AbstractVector{GraphTensor}; mode::Symbol=:replace)
    mode in (:replace, :add) || throw(ArgumentError("scatter mode must be :replace or :add, got :$mode"))
    d = _coords_dims(init, coords)
    isequal(d, realized_dims(src.shape)) || throw(DimensionMismatch("scatter: coordinates have shape $d, src $(realized_dims(src.shape))"))
    inputs = vcat([(init.id, 0, init.shape), (src.id, 0, src.shape)], [(c.id, 0, c.shape) for c in coords])
    return add_op!(init.graph_ref, ScatterND(mode), inputs, ShapeTracker(realized_dims(init.shape)))
end

function _coords_dims(data::GraphTensor, coords)
    r = length(realized_dims(data.shape))
    length(coords) == r || throw(ArgumentError("need one coordinate tensor per axis of the data ($r), got $(length(coords))"))
    all(c -> c.graph_ref === data.graph_ref, coords) || throw(ArgumentError("coordinates must be from the data's graph"))
    d = realized_dims(coords[1].shape)
    all(c -> isequal(realized_dims(c.shape), d), coords) ||
        throw(DimensionMismatch("coordinate tensors must share one shape: $([realized_dims(c.shape) for c in coords])"))
    return d
end

"""
    iota(graph, dims, f)

A tensor of shape `dims` with `out[c_1, .., c_k] = f(c_1, .., c_k)` over its
0-based coordinates (Int arguments; the result is stored as Float32). For
example `iota(g, [n], i -> i)` is `0:n-1`, and `iota(g, [n, m], (i, j) -> i * m + j)`
a row-major index. Evaluated on the host; `f` should be pure.
"""
iota(graph::Graph, dims::AbstractVector, f) =
    add_op!(graph, Iota(f), Tuple{Int, Int, ShapeTracker}[], ShapeTracker(collect(Luminal.DimType, dims)))

# Rounding, elementwise; values stay Float32 (round: half to even)
Base.floor(a::GraphTensor) = add_op!(a.graph_ref, Floor(), [(a.id, 0, a.shape)], a.shape)
Base.ceil(a::GraphTensor) = add_op!(a.graph_ref, Ceil(), [(a.id, 0, a.shape)], a.shape)
Base.round(a::GraphTensor) = add_op!(a.graph_ref, Round(), [(a.id, 0, a.shape)], a.shape)
Base.trunc(a::GraphTensor) = add_op!(a.graph_ref, Trunc(), [(a.id, 0, a.shape)], a.shape)

"""
    select(cond, a, b)

Elementwise `cond != 0 ? a : b`, broadcasting all three (`a` and `b` may be
numbers). Unlike `cond * a + (1 - cond) * b`, the branch not taken never
reaches the result, so an Inf or NaN there does not leak through.
"""
function select(c::GraphTensor, a::GraphTensor, b::GraphTensor)
    @assert c.graph_ref === a.graph_ref === b.graph_ref "Tensors must be from the same graph"
    inputs = [(c.id, 0, c.shape), (a.id, 0, a.shape), (b.id, 0, b.shape)]
    out_dims = broadcast_dims(broadcast_dims(realized_dims(c.shape), realized_dims(a.shape)), realized_dims(b.shape))
    return add_op!(c.graph_ref, Select(), inputs, ShapeTracker(out_dims))
end
select(c::GraphTensor, a::Number, b::GraphTensor) = select(c, constant(c.graph_ref, a), b)
select(c::GraphTensor, a::GraphTensor, b::Number) = select(c, a, constant(c.graph_ref, b))
select(c::GraphTensor, a::Number, b::Number) = select(c, constant(c.graph_ref, a), constant(c.graph_ref, b))

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

"""
    gelu(a; approximate=true)

GELU. `approximate=true` is the tanh approximation; `approximate=false` is the
exact `0.5x(1 + erf(x/√2))` (PyTorch's default, used by Whisper), with erf from
Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7).
"""
function gelu(a::GraphTensor; approximate::Bool=true)
    approximate && return a * 0.5f0 * (1.0f0 + tanh(0.7978845608f0 * a * (1.0f0 + 0.044715f0 * a * a)))
    # For z = |x|/√2, 1 - erf(z) = q = poly(t) exp(-z²) with t = 1/(1 + p z), so
    # gelu(x) = x(1 - q/2) for x >= 0 and x q/2 for x < 0; both are relu(x) - |x| q/2.
    ax = abs(a)
    z = ax * Float32(1 / sqrt(2))
    t = reciprocal(1.0f0 + 0.3275911f0 * z)
    poly = t * (0.254829592f0 + t * (-0.284496736f0 + t * (1.421413741f0 +
               t * (-1.453152027f0 + t * 1.061405429f0))))
    q = poly * exp(-(z * z))
    return relu(a) - ax * q * 0.5f0
end

function Base.tanh(a::GraphTensor)
    return sigmoid(a * 2.0f0) * 2.0f0 - 1.0f0
end

# Movement Ops
# ------------

function reshape(a::GraphTensor, new_shape_vec::AbstractVector)
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

function expand(a::GraphTensor, dim::Int, size::DimType)
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

function pad(a::GraphTensor, padding_vec::AbstractVector)
    calc_shape = deepcopy(a.shape)
    pad!(calc_shape, padding_vec)
    final_dims = realized_dims(calc_shape)
    output_shape = ShapeTracker(final_dims)
    inputs = [(a.id, 0, a.shape)]
    return add_op!(a.graph_ref, Pad(padding_vec), inputs, output_shape)
end

function pad_along(a::GraphTensor, axis::Int, left::DimType, right::DimType)
    p = Tuple{DimType, DimType}[(0, 0) for _ in 1:length(a.shape.indexes)]
    p[axis] = (left, right)
    return pad(a, p)
end

function slice(a::GraphTensor, slice_vec::AbstractVector)
    calc_shape = deepcopy(a.shape)
    slice!(calc_shape, slice_vec)
    final_dims = realized_dims(calc_shape)
    output_shape = ShapeTracker(final_dims)
    inputs = [(a.id, 0, a.shape)]
    return add_op!(a.graph_ref, Slice(slice_vec), inputs, output_shape)
end

function slice_along(a::GraphTensor, axis::Int, start::DimType, stop::DimType)
    s = Tuple{DimType, DimType}[(0, typemax(Int)) for _ in 1:length(a.shape.indexes)]
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
    
    # Left-aligned matmul shape logic: (M, K, ...) * (K, N, ...) -> (M, N, ...)
    batch_dims = broadcast_dims(a_dims[3:end], b_dims[3:end])
    output_shape_vec = [a_dims[1], b_dims[2], batch_dims...]
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

# Softmax over dim 1 as one fused op (see SoftmaxOp). Unlike `softmax`, it is a
# single node without autograd rules; use it where speed matters (inference).
function softmax1(a::GraphTensor)
    return add_op!(a.graph_ref, SoftmaxOp(), [(a.id, 0, a.shape)], ShapeTracker(realized_dims(a.shape)))
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

# [0, 1, ..., to-1] as a single ARange node. (Formerly cumsum(ones) - 1, which
# costs several ops and a scalar-indexed CumSum on AMD GPUs.)
function arange(graph::Graph, to::DimType)
    return add_op!(graph, Function("ARange"), Tuple{Int, Int, ShapeTracker}[], ShapeTracker([to]))
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

function triu(graph::Graph, size::DimType, diagonal::Int=0)
    # h will be row index, v will be column index
    # (size) -> expand(2, size) -> (size, size) where each row is [0, 1, ..., N-1]
    v = expand(arange(graph, size), 1, size)
    # (size) -> expand(1, size) -> (size, size) where each col is [0, 1, ..., N-1] 
    h = expand(arange(graph, size), 2, size)
    
    # In Julia column-major (B, S):
    # Dim 1 is S (rows), Dim 2 is B (columns) - actually for (S, S):
    # h = expand(arange, 2, size) -> new dim is 2 (cols). so rows vary. h[i, j] = i-1
    # v = expand(arange, 1, size) -> new dim is 1 (rows). so cols vary. v[i, j] = j-1
    # We want Upper Triangle (col > row + diag - 1)
    return v - Float32(diagonal - 1) > h
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
    # q, k, v shape: (HeadDim, Seq, Head, Batch), the layout the Llama code uses
    q_dims = realized_dims(q.shape)
    if isnothing(scale)
        scale = 1.0f0 / sqrt(Float32(q_dims[1]))
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
