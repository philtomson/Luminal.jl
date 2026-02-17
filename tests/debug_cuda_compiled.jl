
using Luminal
using Luminal: compile
using CUDA
using Test

g = Graph()
a = tensor(g, [3, 3])
b = tensor(g, [3, 3])
c = a + b

println("Compiling...")
exec_fn = compile(g)
println("Compiled.")

a_val = rand(Float32, 3, 3)
b_val = rand(Float32, 3, 3)

if CUDA.functional()
    device = CUDADevice()
    println("Using CUDADevice")
    inputs_gpu = Dict{Int, Any}(
        a.id => CuArray(a_val),
        b.id => CuArray(b_val)
    )
    
    println("Executing...")
    try
        res_gpu = exec_fn(inputs_gpu, device)[c.id]
        println("Execution successful.")
    catch e
        println("Caught error:")
        showerror(stdout, e)
        println()
        Base.show_backtrace(stdout, catch_backtrace())
    end
else
    println("No CUDA/GPU found. Skipping.")
end
