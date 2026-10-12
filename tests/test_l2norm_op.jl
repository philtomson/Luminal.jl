using Test
using Luminal
using AMDGPU

# L2NormOp: x * rsqrt(sum(x^2) + eps) over dim 1, in one kernel (Gated DeltaNet's q and k).
ref(x, eps) = x .* (1 ./ sqrt.(sum(abs2, x; dims = 1) .+ eps))
@testset "L2NormOp" begin
    op = Luminal.L2NormOp(1f-6)
    for dims in ((128, 16, 33, 1), (64, 5), (32, 7, 2), (200, 3))
        x = randn(Float32, dims...)
        o = zeros(Float32, dims...)
        Luminal.execute_op!(o, op, x)                       # CPU path
        @test o ≈ ref(x, 1f-6) rtol = 1e-5
        dev = get_device()
        if dev isa Luminal.AMDDevice
            xg = ROCArray(x); og = AMDGPU.zeros(Float32, dims...)
            Luminal.execute_op!(og, op, xg); AMDGPU.synchronize()
            @test Array(og) ≈ ref(x, 1f-6) rtol = 1e-5
            oh = AMDGPU.zeros(Float16, dims...)             # Float16 output
            Luminal.execute_op!(oh, op, xg); AMDGPU.synchronize()
            @test Float32.(Array(oh)) ≈ ref(x, 1f-6) rtol = 2e-3
            big = ROCArray(randn(Float32, dims[1] + 3, dims[2:end]...))   # a strided view as the input
            og2 = AMDGPU.zeros(Float32, dims...)
            Luminal.execute_op!(og2, op, view(big, 2:dims[1]+1, ntuple(_ -> :, length(dims) - 1)...)); AMDGPU.synchronize()
            @test Array(og2) ≈ ref(Array(big)[2:dims[1]+1, ntuple(_ -> :, length(dims) - 1)...], 1f-6) rtol = 1e-5
        end
    end
end
