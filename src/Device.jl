module Device

using CUDA
using AMDGPU

export AbstractDevice, CPUDevice, CUDADevice, AMDDevice, get_device, to_device, from_device, execute_with_capture

abstract type AbstractDevice end

struct CPUDevice <: AbstractDevice end
struct CUDADevice <: AbstractDevice end
struct AMDDevice <: AbstractDevice end

"""
    get_device()

Automatically detect and return the best available device.
Priority: CUDA > AMDGPU > CPU.
"""
function get_device()
    if CUDA.functional()
        return CUDADevice()
    elseif AMDGPU.functional()
        return AMDDevice()
    else
        return CPUDevice()
    end
end

"""
    to_device(data, device)

Move data to the specified device. Handles Arrays and Dictionaries.
"""
to_device(data, ::CPUDevice) = data
to_device(data::AbstractArray, ::CUDADevice) = CuArray(data)
to_device(data::AbstractArray, ::AMDDevice) = ROCArray(data)
to_device(data::Dict, device::AbstractDevice) = Dict{Any, Any}(k => to_device(v, device) for (k, v) in data)
to_device(data::Dict, ::CPUDevice) = Dict{Any, Any}(k => v for (k, v) in data)
to_device(data::Number, ::CUDADevice) = CuArray(fill(Float32(data)))
to_device(data::Number, ::AMDDevice) = ROCArray(fill(Float32(data)))
to_device(data::Number, ::CPUDevice) = data
to_device(data::Number, ::AbstractDevice) = data

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
    # Check if executable graph exists in cache
    if haskey(cache, :cuda_exec)
        exec = cache[:cuda_exec]
        # Launch existing graph
        CUDA.launch(exec)
    else
        # 1. Warm up JIT! 
        # Capturing during JIT compilation often fails/invalidates.
        # Run once to ensure all kernels are compiled.
        f()
        
        # 2. Capture and instantiate new graph
        graph = CUDA.capture() do
            f()
        end
        
        exec = CUDA.instantiate(graph)
        cache[:cuda_exec] = exec
        
        # 3. Launch the graph
        CUDA.launch(exec)
    end
end

execute_with_capture(::AbstractDevice, f, cache) = f()

end # module Device
