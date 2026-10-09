using Test
using Luminal
using AMDGPU

# WMMA (matrix-core) GEMM for prefill on Float16 weights (src/WMMA.jl). GPU-only, and
# only on gfx11 devices.
dev = get_device()
if dev isa Luminal.AMDDevice && Luminal.wmma_supported()
    relerr(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), 1f-6)

    @testset "wmma_gemm! against Float32" begin
        for (M, K, N) in ((64, 32, 64), (128, 96, 192), (192, 64, 128), (2048, 2048, 128))
            W = randn(Float16, M, K); X = randn(Float16, K, N)
            C = AMDGPU.zeros(Float32, M, N)
            Luminal.wmma_gemm!(C, ROCArray(collect(W')), ROCArray(X), N)
            @test relerr(Array(C), Float32.(W) * Float32.(X)) < 1e-4
        end
        # N not a multiple of 64: padded columns are ignored and nothing past N is written
        M, K, N = 128, 64, 70
        W = randn(Float16, M, K); X = randn(Float16, K, 128); X[:, N+1:end] .= 0
        C = AMDGPU.zeros(Float32, M, N)
        Luminal.wmma_gemm!(C, ROCArray(collect(W')), ROCArray(X), N)
        @test relerr(Array(C), Float32.(W) * Float32.(X[:, 1:N])) < 1e-4
    end

    @testset "HalfWeight prefill matmul takes the WMMA path" begin
        for (Out, In, N) in ((128, 64, 20), (256, 96, 100), (64, 32, 17), (192, 128, 3))
            W = Float16.(randn(Float32, Out, In)); x = randn(Float32, In, N)
            hw = Luminal.HalfWeight(Luminal.to_device(Float32.(W), dev))
            out = Luminal.zero_tensor(dev, Float32, Out, N)
            Luminal.batch_matmul!(out, hw, Luminal.to_device(x, dev))
            # N > QUANT_GEMM_COLS runs on the matrix cores, which round activations to Float16;
            # fewer columns take the Float32-activation GEMV
            xr = N > Luminal.QUANT_GEMM_COLS ? Float32.(Float16.(x)) : x
            @test relerr(Array(out), Float32.(W) * xr) < 1e-4
        end
        # with a residual
        Out, In, N = 128, 64, 40
        W = Float16.(randn(Float32, Out, In)); x = randn(Float32, In, N); r = randn(Float32, Out, N)
        hw = Luminal.HalfWeight(Luminal.to_device(Float32.(W), dev))
        out = Luminal.zero_tensor(dev, Float32, Out, N)
        Luminal._half_matmul!(out, hw, Luminal.to_device(x, dev); residual = Luminal.to_device(r, dev))
        @test relerr(Array(out), Float32.(W) * Float32.(Float16.(x)) .+ r) < 1e-4
    end
end
