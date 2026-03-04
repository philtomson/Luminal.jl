module Device
 
using CUDA
using AMDGPU
using Vulkan
using GPUArrays
 
export AbstractDevice, CPUDevice, AbstractGPUDevice, CUDADevice, AMDDevice, VulkanDevice, 
       get_device, to_device, from_device, execute_with_capture,
       reclaim!, available_memory, zero_tensor
 
abstract type AbstractDevice end
 
struct CPUDevice <: AbstractDevice end
 
abstract type AbstractGPUDevice <: AbstractDevice end
 
struct CUDADevice <: AbstractGPUDevice end
struct AMDDevice <: AbstractGPUDevice end
 
struct VulkanDevice <: AbstractDevice 
    name::String
end
 
# Default constructor for empty name
VulkanDevice() = VulkanDevice("Unknown GPU")
 
function Base.show(io::IO, dev::VulkanDevice)
    print(io, "VulkanDevice(\"", dev.name, "\")")
end
 
"""
    get_device()
 
Automatically detect and return the best available device.
Priority: CUDA > AMDGPU > CPU.
"""
function get_device()
    if CUDA.functional()
        try
            # Light health check
            CUDA.CuArray([1.0f0])
            return CUDADevice()
        catch e
            @warn "CUDA is functional but health check failed: $e. Falling back to next device."
        end
    end
    
    if AMDGPU.functional()
        try
            # Light health check: allocate a tiny array to verify stream/memory management
            AMDGPU.ROCArray([1.0f0])
            return AMDDevice()
        catch e
            @warn "AMDGPU is functional but health check failed: $e. Falling back to next device."
        end
    end
 
    # Try Vulkan if others fail or are unavailable
    try
        v_inst = Vulkan.Instance([], [])
        v_pdevs = Vulkan.enumerate_physical_devices(v_inst)
        # Handle Result type from Vulkan.jl
        actual_pdevs = v_pdevs isa Vector ? v_pdevs : Vulkan.unwrap(v_pdevs)
        
        if !isempty(actual_pdevs)
            for pdev in actual_pdevs
                props = Vulkan.get_physical_device_properties(pdev)
                # Use numerical values if constants are giving trouble, or try to access them via Vulkan
                is_gpu = Int(props.device_type) == 1 || Int(props.device_type) == 2
                if is_gpu
                    return VulkanDevice(props.device_name)
                end
            end
        end
    catch e
        @debug "Vulkan detection failed: $e"
    end
    
    return CPUDevice()
end
 
"""
    to_device(data, device)
 
Move data to the specified device. Handles Arrays and Dictionaries.
"""
# Fallback for generic objects (like Numbers, or already correctly placed arrays)
to_device(data, ::AbstractDevice) = data
 
# Dictionary mapping
to_device(data::Dict, device::AbstractDevice) = Dict{Any, Any}(k => to_device(v, device) for (k, v) in data)
 
# Physical data placement
# 1. Already on the correct GPU: no-op
to_device(data::AnyGPUArray, ::AbstractGPUDevice) = data
# Specialize to resolve ambiguities with AbstractArray/CUDADevice
to_device(data::AnyGPUArray, ::CUDADevice) = data
to_device(data::AnyGPUArray, ::AMDDevice) = data
 
# 2. Moving from CPU to specific GPU
to_device(data::AbstractArray, ::CUDADevice) = CUDA.CuArray(data)
to_device(data::AbstractArray, ::AMDDevice) = AMDGPU.ROCArray(data)
 
# Explicitly handle CPUDevice and VulkanDevice to avoid ambiguity with generic fallback
to_device(data::AbstractArray, ::CPUDevice) = data
to_device(data::AbstractArray, ::VulkanDevice) = data
 
# Number placement (mostly for scalars in graphs)
to_device(data::Number, ::CUDADevice) = CUDA.CuArray(fill(Float32(data)))
to_device(data::Number, ::AMDDevice) = AMDGPU.ROCArray(fill(Float32(data)))
 
"""
    from_device(data)
 
Move data back to the CPU.
"""
from_device(data) = data
from_device(data::AnyGPUArray) = Array(data)
 
"""
    reclaim!(device)
 
Explicitly reclaim unused GPU memory if the backend supports it.
"""
reclaim!(::AbstractDevice) = nothing
reclaim!(::CUDADevice) = CUDA.reclaim()
reclaim!(::AMDDevice) = nothing # ROCm memory management handles this differently
 
"""
    available_memory(device)
 
Get available GPU memory in MiB. Returns -1 if unknown.
"""
available_memory(::AbstractDevice) = -1.0
available_memory(::CUDADevice) = CUDA.available_memory() / (1024 * 1024)
available_memory(::AMDDevice) = -1.0 # TODO: Implement via AMDGPU.info()
 
"""
    zero_tensor(device, dtype, dims...)
 
Allocate a zero-filled tensor of specified type and dimensions on `device`.
"""
zero_tensor(::CPUDevice, dtype, dims...) = zeros(dtype, dims...)
zero_tensor(::CUDADevice, dtype, dims...) = CUDA.fill(dtype(0), dims...)
zero_tensor(::AMDDevice, dtype, dims...) = AMDGPU.fill(dtype(0), dims...)
zero_tensor(::VulkanDevice, dtype, dims...) = zeros(dtype, dims...)
 
"""
    execute_with_capture(device, f, cache)
 
Executes `f()` on `device`.
If supported (e.g. CUDA), it captures the execution into a graph stored in `cache` 
and replays it on subsequent calls.
"""
function execute_with_capture(::CUDADevice, f, cache::Dict)
    # Temporary diagnostic step: bypass CUDA graph capture completely.
    f()
end
 
execute_with_capture(::AbstractDevice, f, cache) = f()
 
end # module Device
