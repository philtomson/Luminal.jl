using Test
using Luminal
using AMDGPU

# SwiGLUOp: silu(gate) * up
@testset "SwiGLUOp" begin
    op = Luminal.SwiGLUOp()
    for dims in ((17, 5), (64, 3, 2), (128, 33))
        g = randn(Float32, dims...) .* 2; u = randn(Float32, dims...)
        r = g ./ (1 .+ exp.(-g)) .* u
        o = zeros(Float32, dims...); Luminal.execute_op!(o, op, g, u)
        @test o ≈ r rtol = 1e-5
        if get_device() isa Luminal.AMDDevice
            og = AMDGPU.zeros(Float32, dims...); Luminal.execute_op!(og, op, ROCArray(g), ROCArray(u)); AMDGPU.synchronize()
            @test Array(og) ≈ r rtol = 1e-5
            oh = AMDGPU.zeros(Float16, dims...); Luminal.execute_op!(oh, op, ROCArray(g), ROCArray(u)); AMDGPU.synchronize()
            @test Float32.(Array(oh)) ≈ r rtol = 3e-3 atol = 1e-3
        end
    end
end
