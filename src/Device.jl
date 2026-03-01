module Device

using CUDA
using AMDGPU
using Vulkan

export AbstractDevice, CPUDevice, CUDADevice, AMDDevice, VulkanDevice, get_device, to_device, from_device, execute_with_capture

abstract type AbstractDevice end

struct CPUDevice <: AbstractDevice end
struct CUDADevice <: AbstractDevice end
struct AMDDevice <: AbstractDevice end
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
to_device(data::AbstractArray, ::CUDADevice) = CuArray(data)
to_device(data::AbstractArray, ::AMDDevice) = ROCArray(data)
# Explicitly handle CPUDevice and VulkanDevice to avoid ambiguity with generic fallback
to_device(data::AbstractArray, ::CPUDevice) = data
to_device(data::AbstractArray, ::VulkanDevice) = data

# Number placement (mostly for scalars in graphs)
to_device(data::Number, ::CUDADevice) = CuArray(fill(Float32(data)))
to_device(data::Number, ::AMDDevice) = ROCArray(fill(Float32(data)))

"""
    from_device(data)

Move data back to the CPU.
"""
from_device(data) = data
from_device(data::CuArray) = Array(data)
from_device(data::ROCArray) = Array(data)

"""
    execute_with_capture(device, f, cache)

Executes `f()` on `device`.
If supported (e.g. CUDA), it captures the execution into a graph stored in `cache` 
and replays it on subsequent calls.
"""
function execute_with_capture(::CUDADevice, f, cache::Dict)
    # Temporary diagnostic step: bypass CUDA graph capture completely.
    # We suspect CUBLAS batched_mul loops or similar operations implicitly allocate
    # transient workspaces which get freed and invalidate the captured graph.
    f()
end

execute_with_capture(::AbstractDevice, f, cache) = f()

end # module Device
