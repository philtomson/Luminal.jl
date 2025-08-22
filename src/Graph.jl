# Defines the main Graph and GraphTensor data structures.

# A node in the computation graph
struct Node
    op::Op
    inputs::Vector{Tuple{Int, Int}} # (NodeID, OutputIndex)
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
    tensors::Dict{Tuple{Int, Int}, Any} # (NodeID, OutputIndex) -> Tensor data
    dyn_map::Dict{Char, Int}
    no_delete::Set{Int}
    to_retrieve::Set{Int}

    # Default constructor
    function Graph()
        new(Vector{Node}(), 
            Dict{Tuple{Int, Int}, Any}(), 
            Dict{Char, Int}(), 
            Set{Int}(), 
            Set{Int}())
    end
end

"""
    add_op!(graph::Graph, op::Op, inputs::Vector{Tuple{Int, Int}}, output_shape::ShapeTracker)

Add a new operation node to the graph and return a GraphTensor representing it.
"""
function add_op!(graph::Graph, op::Op, inputs::Vector{Tuple{Int, Int}}, output_shape::ShapeTracker)
    node = Node(op, inputs)
    push!(graph.nodes, node)
    node_id = length(graph.nodes)
    return GraphTensor(node_id, output_shape, graph)
end

"""
    tensor(graph::Graph, shape::Vector{Int})

Define a new input tensor on the graph.
"""
function tensor(graph::Graph, shape::Vector{Int})
    # Convert the integer shape to a vector of symbolic Expressions
    dims = [Luminal.Symbolic.Expression([Luminal.Symbolic.Num(d)]) for d in shape]
    # Create a new ShapeTracker for this input tensor
    st = ShapeTracker(dims)
    
    op = Function("InputTensor")
    inputs = Vector{Tuple{Int, Int}}()
    return add_op!(graph, op, inputs, st)
end