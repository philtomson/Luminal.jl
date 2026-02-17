
using Test
using Luminal
using CUDA

# Test Graph Compilation
@testset "Graph Compilation" begin
    # 1. Simple Add Graph
    g = Graph()
    a = tensor(g, [3, 3])
    b = tensor(g, [3, 3])
    c = a + b
    
    # Compile
    exec_fn = compile(g)
    
    # Prepare inputs
    a_val = rand(Float32, 3, 3)
    b_val = rand(Float32, 3, 3)
    inputs = Dict{Int, Any}(
        a.id => a_val,
        b.id => b_val
    )
    
    # Execute (device determines whether we use captured graph or not, 
    # but the BUFFERS are determined at compile time by get_device())
    device = Luminal.get_device()
    res_buf = exec_fn(inputs, device)[c.id]
    
    # Always convert to Array for comparison to avoid GPU-CPU broadcast issues
    @test Array(res_buf) ≈ (a_val + b_val)
    
    # 2. Test execution with explicit capture if on GPU
    if device isa CUDADevice
         @test haskey(exec_fn.cache, :cuda_exec)
         println("CUDA Graph captured successfully.")
    end
    
    # 3. Test more complex graph (Mul + Add + ReLU)
    g2 = Graph()
    x = tensor(g2, [2, 2])
    w = tensor(g2, [2, 2])
    bias = tensor(g2, [2, 2])
    
    y = (x * w) + bias
    z = relu(y)
    
    exec_fn2 = compile(g2)
    
    x_val = rand(Float32, 2, 2)
    w_val = rand(Float32, 2, 2)
    b_val = rand(Float32, 2, 2) .- 0.5f0
    
    inputs2 = Dict{Int, Any}(
        x.id => x_val,
        w.id => w_val,
        bias.id => b_val
    )
    
    res2_buf = exec_fn2(inputs2, device)[z.id]
    expected = max.((x_val .* w_val) .+ b_val, 0.0f0)
    
    @test Array(res2_buf) ≈ expected
end
