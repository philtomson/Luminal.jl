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
    tensors::Dict{Tuple{Int, Int}, Any} # (NodeID, OutputIndex) -> Tensor data
    dyn_map::Dict{Char, Int}
    no_delete::Set{Int}
    to_retrieve::Set{Int}

    # Default constructor
    function Graph()
        new(Vector{Node}(), 
            Vector{ShapeTracker}(),
            Dict{Tuple{Int, Int}, Any}(), 
            Dict{Char, Int}(), 
            Set{Int}(), 
            Set{Int}())
    end
end

"""
    add_op!(graph::Graph, op::Op, inputs::Vector{Tuple{Int, Int, ShapeTracker}}, output_shape::ShapeTracker)

Add a new operation node to the graph and return a GraphTensor representing it.
"""
function add_op!(graph::Graph, op::Op, inputs::Vector{Tuple{Int, Int, ShapeTracker}}, output_shape::ShapeTracker)
    node = Node(op, inputs)
    push!(graph.nodes, node)
    push!(graph.shapes, output_shape)
    node_id = length(graph.nodes)
    return GraphTensor(node_id, output_shape, graph)
end

"""
    tensor(graph::Graph, shape::Vector{Int})

Define a new input tensor on the graph.
"""
function tensor(graph::Graph, shape::Vector{Int})
    st = ShapeTracker(shape)
    op = Function("InputTensor")
    inputs = Vector{Tuple{Int, Int, ShapeTracker}}()
    return add_op!(graph, op, inputs, st)
end

"""
    tensor(graph::Graph, data::AbstractArray)

Convenience method to create an input tensor with shape matching the provided data.
"""
function tensor(graph::Graph, data::AbstractArray)
    return tensor(graph, Int[size(data)...])
end

"""
    constant(graph::Graph, value::Number)

Create a scalar constant on the graph.
"""
function constant(graph::Graph, value::Number)
    st = ShapeTracker(Int[])
    op = Constant(value)
    inputs = Vector{Tuple{Int, Int, ShapeTracker}}()
    return add_op!(graph, op, inputs, st)
end