module Device
 
using CUDA
using AMDGPU
# using Vulkan # Deferring to avoid InitError if libvulkan.so.1 is missing

using GPUArrays
 
export AbstractDevice, CPUDevice, AbstractGPUDevice, CUDADevice, AMDDevice, 
       get_device, to_device, from_device, execute_with_capture,
       reclaim!, available_memory, zero_tensor, synchronize_device
 
abstract type AbstractDevice end
 
struct CPUDevice <: AbstractDevice end
 
abstract type AbstractGPUDevice <: AbstractDevice end
 
struct CUDADevice <: AbstractGPUDevice end
struct AMDDevice <: AbstractGPUDevice end
 
# struct VulkanDevice <: AbstractDevice 
#     name::String
# end
 
# Default constructor for empty name
# VulkanDevice() = VulkanDevice("Unknown GPU")
#  
# function Base.show(io::IO, dev::VulkanDevice)
#     print(io, "VulkanDevice(\"", dev.name, "\")")
# end
 
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
    # Disabled for now to avoid InitError in environments without libvulkan
    # try
    #     v_inst = Vulkan.Instance([], [])
    #     v_pdevs = Vulkan.enumerate_physical_devices(v_inst)
    #     # ...
    # catch e
    #     @debug "Vulkan detection failed: $e"
    # end
    
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
# to_device(data::AbstractArray, ::VulkanDevice) = data
 
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
# AMDGPU.jl keeps freed buffers in HIP's stream-ordered memory pool (up to a high
# release threshold); on an APU that is system RAM the rest of the machine can't
# use. Collect garbage, then trim the pool.
reclaim!(::AMDDevice) = (GC.gc(); AMDGPU.HIP.reclaim(); nothing)
 
"""
    available_memory(device)
 
Get available GPU memory in MiB. Returns -1 if unknown.
"""
available_memory(::AbstractDevice) = -1.0
available_memory(::CUDADevice) = CUDA.available_memory() / (1024 * 1024)
# Free memory the AMD GPU can still allocate, in MB (-1 if unknown). On an APU
# (little dedicated VRAM, e.g. Strix Halo) GPU buffers live in system RAM through
# the GTT pool, so the limit is the smaller of the pool's free space and the
# kernel's MemAvailable -- exceeding it invokes the OOM killer, not an HIP error.
function available_memory(::AMDDevice)
    try
        for dev in readdir("/sys/class/drm"; join=true)
            f(n) = joinpath(dev, "device", n)
            isfile(f("mem_info_vram_total")) || continue
            rd(n) = parse(Int, strip(read(f(n), String)))
            vram_total = rd("mem_info_vram_total")
            vram_total >= 4 * 2^30 && return (vram_total - rd("mem_info_vram_used") + _pool_free_bytes()) / 2^20
            gtt_free = rd("mem_info_gtt_total") - rd("mem_info_gtt_used")
            return (min(gtt_free, _mem_available_bytes()) + _pool_free_bytes()) / 2^20
        end
    catch
    end
    return -1.0
end

# Memory HIP's pool holds for this process but has not handed out: it counts as
# used system-wide, yet our allocations reuse it (it may be too fragmented to trim).
function _pool_free_bytes()
    pool = AMDGPU.HIP.memory_pool(AMDGPU.device())
    return Int(AMDGPU.HIP.reserved_memory(pool)) - Int(AMDGPU.HIP.used_memory(pool))
end

function _mem_available_bytes()
    for line in eachline("/proc/meminfo")
        startswith(line, "MemAvailable:") && return parse(Int, split(line)[2]) * 1024
    end
    return typemax(Int)
end
 
"""
    zero_tensor(device, dtype, dims...)
 
Allocate a zero-filled tensor of specified type and dimensions on `device`.
"""
zero_tensor(::CPUDevice, dtype, dims...) = zeros(dtype, dims...)
zero_tensor(::CUDADevice, dtype, dims...) = CUDA.fill(dtype(0), dims...)
zero_tensor(::AMDDevice, dtype, dims...) = AMDGPU.fill(dtype(0), dims...)
# zero_tensor(::VulkanDevice, dtype, dims...) = zeros(dtype, dims...)
 
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

"""
    synchronize_device(device)

Block until work queued on `device` has finished (no-op on CPU).
"""
synchronize_device(::AbstractDevice) = nothing
synchronize_device(::CUDADevice) = CUDA.synchronize()
synchronize_device(::AMDDevice) = AMDGPU.synchronize()

# HIP graph capture and replay. The caller opts in by putting a `:key` in
# `cache` that identifies everything a replay would bake in (symbolic dim
# values, input buffer identities). The first run with a key executes
# normally (warm-up: buffers allocated, kernels compiled); the next run with
# the same key is captured into a HIP graph; later runs with that key replay
# the graph with a single launch instead of issuing every kernel from Julia.
# Graphs whose owner was garbage collected. They are destroyed at the next safe
# point (outside any capture), not from the finalizer: finalizers run at
# arbitrary allocation points, which may be inside another graph's capture.
const _DEAD_GRAPHS = Tuple{AMDGPU.HIP.hipGraph_t, AMDGPU.HIP.hipGraphExec_t}[]
const _DEAD_GRAPHS_LOCK = ReentrantLock()

function _destroy_dead_graphs()
    dead = lock(_DEAD_GRAPHS_LOCK) do
        d = copy(_DEAD_GRAPHS); empty!(_DEAD_GRAPHS); d
    end
    for (graph, exec) in dead
        AMDGPU.HIP.hipGraphExecDestroy(exec)
        AMDGPU.HIP.hipGraphDestroy(graph)
    end
end

mutable struct HIPGraphReplay
    graph::AMDGPU.HIP.hipGraph_t
    exec::AMDGPU.HIP.hipGraphExec_t
end

function _capture_hip_graph(f)
    HIP = AMDGPU.HIP
    s = AMDGPU.stream().stream
    # No GC inside the capture: collected GPU arrays are freed with stream-ordered
    # frees, which would be recorded into the graph and replayed. Collect first so
    # pending finalizers run now, outside the capture.
    GC.gc()
    _destroy_dead_graphs()
    AMDGPU.synchronize()
    gc_was_enabled = GC.enable(false)
    HIP.hipStreamBeginCapture(s, HIP.hipStreamCaptureModeThreadLocal)
    graph = Ref{HIP.hipGraph_t}()
    try
        f()
    catch
        try HIP.hipStreamEndCapture(s, graph) catch end
        GC.enable(gc_was_enabled)
        rethrow()
    end
    HIP.hipStreamEndCapture(s, graph)
    GC.enable(gc_was_enabled)
    exec = Ref{HIP.hipGraphExec_t}()
    HIP.hipGraphInstantiateWithFlags(exec, graph[], 0)
    r = HIPGraphReplay(graph[], exec[])
    finalizer(r) do r
        lock(_DEAD_GRAPHS_LOCK) do
            push!(_DEAD_GRAPHS, (r.graph, r.exec))
        end
    end
    return r
end

_launch_hip_graph(r::HIPGraphReplay) = AMDGPU.HIP.hipGraphLaunch(r.exec, AMDGPU.stream().stream)

function execute_with_capture(::AMDDevice, f, cache::Dict)
    key = get(cache, :key, nothing)
    key === nothing && return f()
    isempty(_DEAD_GRAPHS) || _destroy_dead_graphs()
    g = get(cache, :graph, nothing)
    if g !== nothing && isequal(cache[:graph_key], key)
        _launch_hip_graph(g)
    elseif isequal(get(cache, :warm_key, nothing), key)
        g = _capture_hip_graph(f)
        cache[:graph] = g
        cache[:graph_key] = key
        _launch_hip_graph(g)   # capturing records the work without running it
    else
        cache[:warm_key] = key
        f()
    end
    return nothing
end
 
end # module Device
