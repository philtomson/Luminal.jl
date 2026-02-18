using Test
using Luminal
using CUDA
using AMDGPU

dev = get_device()
@testset "Device Detection: $dev" begin
    println("Detected device: ", dev)
    
    if CUDA.functional()
        # Note: Functional check might pass while health check fails (e.g. OOM)
        @test (dev isa CUDADevice || dev isa VulkanDevice || dev isa CPUDevice)
    elseif AMDGPU.functional()
        # Due to gfx1151 driver/OOM issues, we might fall back to Vulkan or CPU
        @test (dev isa AMDDevice || dev isa VulkanDevice || dev isa CPUDevice)
    else
        @test (dev isa VulkanDevice || dev isa CPUDevice)
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
        # VulkanDevice and CPUDevice currently use Array
        @test d_data isa Array
    end
    
    # Move back to CPU
    h_data = from_device(d_data)
    @test h_data isa Array
    @test h_data == data
end
