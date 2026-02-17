using Test
using Luminal
using CUDA

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
    if CUDA.functional()
        println("Running on GPU (CUDA)...")
        gpu_res = Luminal.execute(g, out.id, Dict(
            a.id => Float32[1.0, 2.0, 3.0],
            b.id => Float32[4.0, 5.0, 6.0],
            c.id => Float32[7.0, 8.0, 9.0]
        ), CUDADevice())
        @test gpu_res ≈ [11.0, 18.0, 27.0]
    else
        @warn "CUDA not available, skipping GPU execution test."
    end
end

@testset "GPU Reductions" begin
    if CUDA.functional()
        g = Luminal.Graph()
        a = Luminal.tensor(g, rand(Float32, 4, 4))
        out = Luminal.sum(a, 1) # sum along first dimension
        
        data = rand(Float32, 4, 4)
        expected = sum(data, dims=1)
        
        res = Luminal.execute(g, out.id, Dict(a.id => data), CUDADevice())
        @test res ≈ expected
    end
end

@testset "GPU Fused Ops" begin
    if CUDA.functional()
        g = Luminal.Graph()
        a = Luminal.tensor(g, rand(Float32, 10))
        b = Luminal.tensor(g, rand(Float32, 10))
        c = Luminal.tensor(g, rand(Float32, 10))
        
        # Test FusedMulAdd
        fma_op = Luminal.FusedMulAdd()
        fma_node = Luminal.add_op!(g, fma_op, [(a.id, 0), (b.id, 0), (c.id, 0)], Luminal.ShapeTracker([10]))
        
        data_a = rand(Float32, 10)
        data_b = rand(Float32, 10)
        data_c = rand(Float32, 10)
        expected = (data_a .* data_b) .+ data_c
        
        res = Luminal.execute(g, fma_node.id, Dict(
            a.id => data_a,
            b.id => data_b,
            c.id => data_c
        ), CUDADevice())
        @test res ≈ expected
    end
end
