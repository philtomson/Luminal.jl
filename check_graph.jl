
using CUDA
device!(0) # Select first device
println("Checking CUDAGraph execution:")
# capture returns a CUDAGraph
graph = CUDA.@captured begin
    A = CuArray([1.0f0])
    B = CuArray([2.0f0])
    C = A + B
end
println("Graph type: ", typeof(graph))

# Is there a launch function?
println("Methods of CUDA.launch:")
for m in methods(CUDA.launch)
    println(m)
end

println("\nChecking if graph is callable:")
try
    graph()
    println("graph() worked!")
catch e
    println("graph() failed: ", e)
end

println("\nChecking for graph_launch or similar:")
for s in names(CUDA; all=true)
    if contains(lowercase(string(s)), "graph") && contains(lowercase(string(s)), "launch")
        println(s)
    end
end
 Done.
