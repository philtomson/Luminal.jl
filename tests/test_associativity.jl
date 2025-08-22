# Add the src directory to the Julia load path
push!(LOAD_PATH, "../src")

using Luminal

println("Running Optimizer Test: (a + 2) + 3")

# 1. Initialize a Graph
graph = Luminal.Graph()

# 2. Build the computation graph

# Input tensor 'a'
a = Luminal.tensor(graph, [2, 2])

# Constant 2
push!(graph.nodes, Luminal.Node(Luminal.Constant(2), []))
const_2 = Luminal.GraphTensor(2, [], graph)

# Constant 3
push!(graph.nodes, Luminal.Node(Luminal.Constant(3), []))
const_3 = Luminal.GraphTensor(3, [], graph)

# Build the expression: (a + 2) + 3
add_node_1 = a + const_2
add_node_2 = add_node_1 + const_3

# 3. Compile the graph
optimized_term = Luminal.compile(graph, add_node_2.id)

# 4. Verify the result
# The associative rule should change the expression to a + (2 + 3),
# and then constant folding should simplify it to a + 5.
# The term for 'a' is :Function
expected_term = :(Add(Function, 5))

println("Final Optimized Term: ", optimized_term)
println("Expected Term: ", expected_term)

@assert optimized_term == expected_term "Optimizer Test Failed!"

println("\nOptimizer Test Passed!")
