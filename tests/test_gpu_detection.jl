using Test
using Luminal
using CUDA
using AMDGPU

@testset "Device Detection" begin
    dev = get_device()
    println("Detected device: ", dev)
    
    if CUDA.functional()
        @test dev isa CUDADevice
    elseif AMDGPU.functional()
        @test dev isa AMDDevice
    else
        @test dev isa CPUDevice
    end
end

@testset "Data Transfer" begin
    dev = get_device()
    data = rand(Float32, 10, 10)
    
    # Move to device
    d_data = to_device(data, dev)
    
    if dev isa CUDADevice
        @test d_data isa CuArray
    elseif dev isa AMDDevice
        @test d_data isa ROCArray
    else
        @test d_data isa Array
    end
    
    # Move back to CPU
    h_data = from_device(d_data)
    @test h_data isa Array
    @test h_data == data
end
