using Test
using Luminal

@testset "Autograd Basic Arithmetic" begin
    g = Graph()
    a = tensor(g, [1])
    b = tensor(g, [1])
    c = a * b + a
    
    mark_trainable!(a)
    mark_trainable!(b)
    
    grads = backward(c)
    
    # Define inputs
    inputs = Dict(a.id => Float32[2.0;;], b.id => Float32[3.0;;])
    
    # Execute graph for both output and gradients
    # In Luminal, gradients are just more nodes in the graph
    res = execute(g, [c.id, grads[a.id].id, grads[b.id].id], inputs, CPUDevice())
    
    @test res[c.id][1] == 8.0f0 # 2*3 + 2
    @test res[grads[a.id].id][1] == 4.0f0 # d(ab+a)/da = b + 1 = 3 + 1
    @test res[grads[b.id].id][1] == 2.0f0 # d(ab+a)/db = a = 2
end

@testset "Autograd Broadcasting" begin
    g = Graph()
    a = tensor(g, [2, 2]) # [2, 2]
    b = tensor(g, [1, 2]) # [1, 2] - will be broadcasted
    c = sum(a * b, 1) # sum over dim 1 (rows) -> [2]
    
    mark_trainable!(a)
    mark_trainable!(b)
    
    loss = sum(c, 1) # total sum -> scalar
    grads = backward(loss)
    
    a_data = Float32[1 2; 3 4]
    b_data = Float32[10 20]
    
    inputs = Dict(a.id => a_data, b.id => b_data)
    res = execute(g, [loss.id, grads[a.id].id, grads[b.id].id], inputs, CPUDevice())
    
    # Forward: a*b = [10 40; 30 80], sum = [50; 110], total_sum = 160
    @test res[loss.id][1] == 160.0f0
    
    # Gradients:
    # d(loss)/da = b (broadcasted) = [10 20; 10 20]
    @test res[grads[a.id].id] == Float32[10 20; 10 20]
    
    # d(loss)/db = sum(a, 1) = [1+3, 2+4] = [4, 6]
    @test res[grads[b.id].id] == Float32[4 6]
end

@testset "Autograd MatMul" begin
    g = Graph()
    a = tensor(g, [2, 3])
    b = tensor(g, [3, 4])
    c = matmul(a, b)
    
    mark_trainable!(a)
    mark_trainable!(b)
    
    loss = sum(c, 1) # first sum reduces 2x4 to 4
    loss = sum(loss, 1) # second sum reduces 4 to 1
    
    grads = backward(loss)
    
    a_data = randn(Float32, 2, 3)
    b_data = randn(Float32, 3, 4)
    
    inputs = Dict(a.id => a_data, b.id => b_data)
    res = execute(g, [loss.id, grads[a.id].id, grads[b.id].id], inputs, CPUDevice())
    
    # d(sum(A*B))/dA = ones(2,4) * B^T
    # d(sum(A*B))/dB = A^T * ones(2,4)
    ones_out = ones(Float32, 2, 4)
    @test isapprox(res[grads[a.id].id], ones_out * b_data', atol=1f-5)
    @test isapprox(res[grads[b.id].id], a_data' * ones_out, atol=1f-5)
end

@testset "Autograd Unary Ops" begin
    g = Graph()
    a = tensor(g, [10])
    # Use positive values for log, sqrt
    a_data = Float32.(collect(1:10))
    
    # Test multiple unary ops
    b = log2(a)
    c = exp2(b) # should be a
    d = sin(a)
    e = sqrt(a)
    
    loss = sum(b + c + d + e, 1)
    mark_trainable!(a)
    grads = backward(loss)
    
    inputs = Dict(a.id => a_data)
    res = execute(g, [loss.id, grads[a.id].id], inputs, CPUDevice())
    
    # Analytics gradients:
    # d(log2(a))/da = 1/(a*ln(2))
    # d(exp2(log2(a))) / da = 1
    # d(sin(a))/da = cos(a)
    # d(sqrt(a))/da = 0.5/sqrt(a)
    expected_grad = (1.0f0 ./ (a_data .* Float32(log(2.0)))) .+ 1.0f0 .+ cos.(a_data) .+ (0.5f0 ./ sqrt.(a_data))
    
    @test isapprox(res[grads[a.id].id], expected_grad, atol=1f-5)
end
