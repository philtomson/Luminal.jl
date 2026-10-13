using Test
using Luminal
using AMDGPU

# GatedRMSNormOp: rmsnorm(x, w) * silu(z) over dim 1 in one kernel (Gated DeltaNet's output norm).
ref(x, w, z, eps) = (x .* (1 ./ sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ eps))) .* w .* (z ./ (1 .+ exp.(-z)))
@testset "GatedRMSNormOp" begin
    op = Luminal.GatedRMSNormOp(1f-6)
    for dims in ((128, 48, 33, 1), (64, 5), (32, 7, 2), (200, 3))
        x = randn(Float32, dims...); z = randn(Float32, dims...) .* 2; w = randn(Float32, dims[1])
        r = ref(x, Base.reshape(w, :, ntuple(_ -> 1, length(dims) - 1)...), z, 1f-6)
        o = zeros(Float32, dims...)
        Luminal.execute_op!(o, op, x, w, z)
        @test o ≈ r rtol = 1e-5
        if get_device() isa Luminal.AMDDevice
            og = AMDGPU.zeros(Float32, dims...)
            Luminal.execute_op!(og, op, ROCArray(x), ROCArray(w), ROCArray(z)); AMDGPU.synchronize()
            @test Array(og) ≈ r rtol = 1e-4
            oh = AMDGPU.zeros(Float16, dims...)
            Luminal.execute_op!(oh, op, ROCArray(x), ROCArray(w), ROCArray(z)); AMDGPU.synchronize()
            @test Float32.(Array(oh)) ≈ r rtol = 3e-3 atol = 1e-3
        end
    end
end
