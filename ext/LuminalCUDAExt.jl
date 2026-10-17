# CUDA backend: the CUDA-specific methods of Luminal.Device and the strided-batched GEMM.
# CUDA.jl is a weak dependency so that Luminal can be installed next to an AMDGPU.jl that CUDA.jl's release does not support yet.
module LuminalCUDAExt

using Luminal
using Luminal.Device
using CUDA

Device.to_device(data::AbstractArray, ::CUDADevice) = CUDA.CuArray(data)
Device.to_device(data::Number, ::CUDADevice) = CUDA.CuArray(fill(data))
Device.reclaim!(::CUDADevice) = CUDA.reclaim()
Device.available_memory(::CUDADevice) = CUDA.available_memory() / (1024 * 1024)
Device.zero_tensor(::CUDADevice, dtype, dims...) = CUDA.fill(dtype(0), dims...)
Device.synchronize_device(::CUDADevice) = CUDA.synchronize()

Luminal._batched_gemm!(C::CuArray{T,3}, A::CuArray{T,3}, B::CuArray{T,3}) where {T<:Union{Float32,Float64}} =
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', one(T), A, B, zero(T), C)
Luminal._batched_gemm!(C::CuArray{T,3}, A::CuArray{T,3}, B::CuArray{T,3}, ta::Char, tb::Char) where {T<:Union{Float32,Float64}} =
    CUDA.CUBLAS.gemm_strided_batched!(ta, tb, one(T), A, B, zero(T), C)

function __init__()
    Device.CUDA_PROBE[] = function ()
        CUDA.functional() || return false
        try
            CUDA.CuArray([1.0f0])     # light health check
            return true
        catch e
            @warn "CUDA is functional but health check failed: $e. Falling back to next device."
            return false
        end
    end
end

end
