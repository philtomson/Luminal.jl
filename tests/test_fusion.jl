
using Test
using Luminal
using Luminal: compile
using CUDA

@testset "Operator Fusion" begin
    # 1. Simple Fusion: (a + b) * c
    g = Graph()
    a = tensor(g, [3, 3])
    b = tensor(g, [3, 3])
    c = tensor(g, [3, 3])
    
    d = (a + b) * c
    
    # a+b should be fused into d
    exec_fn = compile(g)
    
    # Check if fusion happened: 
    @test length(exec_fn.steps) == 1
    println("Fusion successful: 1 step instead of 2.")

    a_val = rand(Float32, 3, 3)
    b_val = rand(Float32, 3, 3)
    c_val = rand(Float32, 3, 3)
    
    inputs = Dict(a.id => a_val, b.id => b_val, c.id => c_val)
    device = Luminal.get_device()
    res = exec_fn(inputs, device)[d.id]
    
    @test Array(res) ≈ (a_val .+ b_val) .* c_val
    println("Results verified.")

    # 2. Complex Fusion: relu((x * w) + bias) where * is Mul
    g2 = Graph()
    x = tensor(g2, [2, 2])
    w = tensor(g2, [2, 2])
    bias = tensor(g2, [2, 2])
    
    y = (x * w) + bias  # * is Mul here
    out = relu(y)
    
    exec_fn2 = compile(g2)
    # x*w is Mul (ew), +bias is Add (ew), relu is ReLU (ew)
    # All should be fused into 1 step.
    @test length(exec_fn2.steps) == 1
    println("Fusion successful for full element-wise chain: 1 step.")

    x_val = rand(Float32, 2, 2)
    w_val = rand(Float32, 2, 2)
    b_val = rand(Float32, 2, 2) .- 0.5f0
    
    inputs2 = Dict(x.id => x_val, w.id => w_val, bias.id => b_val)
    res2 = exec_fn2(inputs2, device)[out.id]
    expected = max.((x_val .* w_val) .+ b_val, 0.0f0)
    @test Array(res2) ≈ expected
    println("ReLU chain results verified.")
end
