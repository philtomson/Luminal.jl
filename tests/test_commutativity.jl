# Add the src directory to the Julia load path
push!(LOAD_PATH, "../src")

using Luminal

println("Running Optimizer Test: (2 * a) * 3")

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

# Build the expression: (2 * a) * 3
mul_node_1 = const_2 * a
mul_node_2 = mul_node_1 * const_3

# 3. Compile the graph
optimized_term = Luminal.compile(graph, mul_node_2.id)

# 4. Verify the result
# The optimizer should perform the following steps:
# 1. (2 * a) * 3
# 2. (a * 2) * 3   (Commutativity)
# 3. a * (2 * 3)   (Associativity)
# 4. a * 6         (Constant Folding)
# The term for 'a' is :Function
expected_term = :(Mul(Function, 6))

println("Final Optimized Term: ", optimized_term)
println("Expected Term: ", expected_term)

@assert optimized_term == expected_term "Optimizer Test Failed!"

println("\nOptimizer Test Passed!")
