# Defines the main Graph and GraphTensor data structures.

# A node in the computation graph
struct Node
    op::Op
    inputs::Vector{Tuple{Int, Int, ShapeTracker}} # (NodeID, OutputIndex, InputShape)
end

# A tensor on the graph, which is a symbolic handle to a node's output.
struct GraphTensor
    id::Int # NodeIndex in the graph
    shape::ShapeTracker
    graph_ref::Any # A reference to the parent Graph
end

# The main computation graph structure
mutable struct Graph
    nodes::Vector{Node}
    shapes::Vector{ShapeTracker} # Output shapes for each node
    dtypes::Vector{DataType}     # Output element type of each node
    tensors::Dict{Tuple{Int, Int}, Any} # (NodeID, OutputIndex) -> Tensor data
    dyn_map::Dict{Char, Int}
    no_delete::Set{Int}
    to_retrieve::Set{Int}
    trainable::Set{Int}
    cse::Dict{Any, Int}  # structural key -> node id, for common-subexpression elimination

    # Default constructor
    function Graph()
        new(Vector{Node}(), 
            Vector{ShapeTracker}(),
            Vector{DataType}(),
            Dict{Tuple{Int, Int}, Any}(), 
            Dict{Char, Int}(), 
            Set{Int}(), 
            Set{Int}(),
            Set{Int}(),
            Dict{Any, Int}())
    end
end

"""
    add_op!(graph::Graph, op::Op, inputs::Vector{Tuple{Int, Int, ShapeTracker}}, output_shape::ShapeTracker; dtype=nothing)

Add a new operation node to the graph and return a GraphTensor representing it.
Its dtype is inferred from the op and its inputs' dtypes (`output_dtype`, which
also rejects operands that don't type-check); `dtype` sets it for input tensors.
"""
function add_op!(graph::Graph, op::Op, inputs::Vector{Tuple{Int, Int, ShapeTracker}}, output_shape::ShapeTracker;
                 dtype=nothing)
    T = dtype === nothing ? output_dtype(op, DataType[graph.dtypes[id] for (id, _, _) in inputs]) : _check_dtype(dtype)
    # Common-subexpression elimination: ops are pure, so an op identical to an
    # existing node (same op, same input views, same output shape and dtype)
    # reuses it. This collapses e.g. RoPE tables that every layer rebuilds for
    # the same positions. Input tensors are distinct by definition and never merged.
    key = _cse_key(op, inputs, output_shape, T)
    if key !== nothing
        existing = get(graph.cse, key, 0)
        existing != 0 && return GraphTensor(existing, output_shape, graph)
    end
    node = Node(op, inputs)
    push!(graph.nodes, node)
    push!(graph.shapes, output_shape)
    push!(graph.dtypes, T)
    node_id = length(graph.nodes)
    key !== nothing && (graph.cse[key] = node_id)
    return GraphTensor(node_id, output_shape, graph)
end

function _cse_key(op::Op, inputs, output_shape::ShapeTracker, T)
    op isa Function && op.name == "InputTensor" && return nothing
    # (T distinguishes e.g. Constant(1) from Constant(1f0), which compare equal)
    return (typeof(op), ntuple(i -> getfield(op, i), nfields(op)),
            [(id, idx, _st_key(st)) for (id, idx, st) in inputs], _st_key(output_shape), T)
end

# Structural value of a ShapeTracker (vectors hash and compare by content).
_st_key(st::ShapeTracker) = (st.dims, st.indexes, st.fake, st.mask, st.padding)

"""
    tensor(graph::Graph, shape::Vector{Int}; dtype=Float32)

Define a new input tensor on the graph, of element type `dtype` (one of
`Luminal.DTYPES`). Values given for it are converted to `dtype`.
"""
function tensor(graph::Graph, shape::AbstractVector{<:Union{Integer, SymbolicUtils.BasicSymbolic}}; dtype::Type=Float32)
    st = ShapeTracker(shape)
    op = Function("InputTensor")
    inputs = Vector{Tuple{Int, Int, ShapeTracker}}()
    return add_op!(graph, op, inputs, st; dtype=dtype)
end

"""
    tensor(graph::Graph, data::AbstractArray)

Convenience method to create an input tensor with shape matching the provided data.
(A vector of integers or symbolic dims is a shape, not data: see the method above.)
"""
function tensor(graph::Graph, data::AbstractArray; dtype::Type=Float32)
    return tensor(graph, Int[size(data)...]; dtype=dtype)
end

# An untyped vector is a shape if every element is an integer or a symbolic dim
tensor(graph::Graph, v::AbstractVector{Any}; dtype::Type=Float32) =
    all(e -> e isa Union{Integer, SymbolicUtils.BasicSymbolic}, v) ?
        tensor(graph, DimType[e for e in v]; dtype=dtype) : tensor(graph, Int[size(v)...]; dtype=dtype)

"""
    constant(graph::Graph, value::Number, dtype=Float32)

A scalar constant of element type `dtype`, holding `value` converted exactly
(an `InexactError` if it can't be: `constant(g, 0.5, Int32)`). Float64 constants
keep full double precision (upstream's constant_f64). Arithmetic between a
tensor and a number makes the number a constant of the tensor's dtype.
"""
function constant(graph::Graph, value::Number, dtype::Type=Float32)
    st = ShapeTracker(Int[])
    op = Constant(convert(_check_dtype(dtype), value))
    inputs = Vector{Tuple{Int, Int, ShapeTracker}}()
    return add_op!(graph, op, inputs, st)
end