# Add the src directory to the Julia load path
push!(LOAD_PATH, "../src")

using Luminal

println("Running Optimizer Test: (a * 1) + 0")

# 1. Initialize a Graph
graph = Luminal.Graph()

# 2. Build the computation graph

# Input tensor 'a'
a = Luminal.tensor(graph, [2, 2])

# Constant 1
push!(graph.nodes, Luminal.Node(Luminal.Constant(1), []))
const_1 = Luminal.GraphTensor(2, [], graph)

# Constant 0
push!(graph.nodes, Luminal.Node(Luminal.Constant(0), []))
const_0 = Luminal.GraphTensor(3, [], graph)

# Build the expression
mul_node = a * const_1
add_node = mul_node + const_0

# 3. Compile the graph
optimized_term = Luminal.compile(graph, add_node.id)

# 4. Verify the result
# The expected result is the term for the input tensor 'a', which is :Function
expected_term = :Function

println("Final Optimized Term: ", optimized_term)
println("Expected Term: ", expected_term)

@assert optimized_term == expected_term "Optimizer Test Failed!"

println("\nOptimizer Test Passed!")
