# Add the src directory to the Julia load path
push!(LOAD_PATH, "../src")

using Luminal

println("Running Test: (a * b) + c")

# 1. Initialize a Graph
graph = Luminal.Graph()

# 2. Define input tensors
a = Luminal.tensor(graph, [2, 2])
b = Luminal.tensor(graph, [2, 2])
c = Luminal.tensor(graph, [2, 2])

# 3. Build the computation graph
d = Luminal.matmul(a, b)
e = d + c

# 4. Provide concrete numerical data for the inputs
data_a = [1.0 2.0; 3.0 4.0]
data_b = [5.0 6.0; 7.0 8.0]
data_c = [1.0 1.0; 1.0 1.0]

initial_inputs = Dict(
    a.id => data_a,
    b.id => data_b,
    c.id => data_c
)

# 5. Execute the graph
result = Luminal.execute(graph, e.id, initial_inputs)

# 6. Calculate the expected result
expected = (data_a * data_b) + data_c

# 7. Verify the result
println("Graph Result:")
println(result)
println("\nExpected Result:")
println(expected)

@assert result == expected "Test Failed: The graph result does not match the expected result!"

println("\nTest Passed!")
