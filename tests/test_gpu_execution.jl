using Test
using Luminal
using CUDA
using AMDGPU

dev = get_device()

@testset "GPU Graph Execution" begin
    g = Luminal.Graph()
    
    # Simple a * b + c graph
    a = Luminal.tensor(g, Float32[1.0, 2.0, 3.0])
    b = Luminal.tensor(g, Float32[4.0, 5.0, 6.0])
    c = Luminal.tensor(g, Float32[7.0, 8.0, 9.0])
    out = a * b + c
    
    # Verify CPU execution
    println("Running on CPU...")
    cpu_res = Luminal.execute(g, out.id, Dict(
        a.id => Float32[1.0, 2.0, 3.0],
        b.id => Float32[4.0, 5.0, 6.0],
        c.id => Float32[7.0, 8.0, 9.0]
    ), CPUDevice())
    @test cpu_res ≈ Float32[11.0, 18.0, 27.0]
    
    # Verify GPU execution
    if dev isa CPUDevice
        @warn "No GPU device available (or health check failed), skipping GPU execution test."
    else
        println("Running on GPU ($dev)...")
        gpu_res = Luminal.execute(g, out.id, Dict(
            a.id => Float32[1.0, 2.0, 3.0],
            b.id => Float32[4.0, 5.0, 6.0],
            c.id => Float32[7.0, 8.0, 9.0]
        ), dev)
        @test gpu_res ≈ [11.0, 18.0, 27.0]
    end
end

@testset "GPU Reductions" begin
    if ! (dev isa CPUDevice)
        g = Luminal.Graph()
        a = Luminal.tensor(g, rand(Float32, 4, 4))
        out = Luminal.sum(a, 1) # sum along first dimension
        
        data = rand(Float32, 4, 4)
        expected = dropdims(sum(data, dims=1), dims=1)
        
        res = Luminal.execute(g, out.id, Dict(a.id => data), dev)
        @test res ≈ expected
    end
end

@testset "GPU Fused Ops" begin
    if ! (dev isa CPUDevice)
        g = Luminal.Graph()
        a = Luminal.tensor(g, rand(Float32, 10))
        b = Luminal.tensor(g, rand(Float32, 10))
        c = Luminal.tensor(g, rand(Float32, 10))
        
        # Test FusedMulAdd
        fma_op = Luminal.FusedMulAdd()
        fma_node = Luminal.add_op!(g, fma_op, [(a.id, 0, a.shape), (b.id, 0, b.shape), (c.id, 0, c.shape)], Luminal.ShapeTracker([10]))
        
        data_a = rand(Float32, 10)
        data_b = rand(Float32, 10)
        data_c = rand(Float32, 10)
        expected = (data_a .* data_b) .+ data_c
        
        res = Luminal.execute(g, fma_node.id, Dict(
            a.id => data_a,
            b.id => data_b,
            c.id => data_c
        ), dev)
        @test res ≈ expected
    end
end
