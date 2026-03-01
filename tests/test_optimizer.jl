using Test
using Luminal
using Luminal.Optimizer

@testset "SGD Optimizer" begin
    g = Graph()
    w = tensor(g, [1])
    # Give it some initial data
    g.tensors[(w.id, 1)] = Float32[10.0;;]
    
    mark_trainable!(w)
    
    # Loss = w^2
    loss = w * w
    grads_t = backward(loss)
    opt = SGD(lr=0.1f0)
    # One step
    # Forward/Backward
    # We need to map param.id to grad_data
    grad_ids = [grads_t[w.id].id]
    res = execute(g, [loss.id, grad_ids...], Dict(), CPUDevice())
    
    @test res[loss.id][1] == 100.0f0
    @test res[grads_t[w.id].id][1] == 20.0f0 # d(w^2)/dw = 2w = 20
    
    # Optimizer step: needs Dict(param_id => grad_data)
    actual_grads = Dict(w.id => res[grads_t[w.id].id])
    step!(opt, g, actual_grads)
    
    # New w should be w - lr * grad = 10 - 0.1 * 20 = 8.0
    @test g.tensors[(w.id, 1)][1] == 8.0f0
    
    # Step 2
    res2 = execute(g, [loss.id, grads_t[w.id].id], Dict(), CPUDevice())
    @test res2[loss.id][1] == 64.0f0
    @test res2[grads_t[w.id].id][1] == 16.0f0
    
    actual_grads2 = Dict(w.id => res2[grads_t[w.id].id])
    step!(opt, g, actual_grads2)
    # 8.0 - 0.1 * 16.0 = 8.0 - 1.6 = 6.4
    @test g.tensors[(w.id, 1)][1] == 6.4f0
end

@testset "Adam Optimizer" begin
    g = Graph()
    w = tensor(g, [1])
    g.tensors[(w.id, 1)] = Float32[1.0;;]
    mark_trainable!(w)
    
    loss = (w - 0.0f0) * (w - 0.0f0) # (w-0)^2
    grads_t = backward(loss)
    
    opt = Adam(lr=0.1f0)
    
    for i in 1:10
        res = execute(g, [loss.id, grads_t[w.id].id], Dict(), CPUDevice())
        actual_grads = Dict(w.id => res[grads_t[w.id].id])
        step!(opt, g, actual_grads)
    end
    
    # w should be closer to 0
    final_w = g.tensors[(w.id, 1)][1]
    @test final_w < 1.0f0
    @test final_w > -1.0f0 # Should not overshoot massively if lr is sane
    println("Adam final w after 10 steps: ", final_w)
end