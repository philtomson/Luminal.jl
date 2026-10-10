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

    @testset "int8 QuantWeight prefill matmul takes the WMMA path" begin
        # reference: the exactly dequantized weight (Float32) times Float16-rounded activations;
        # the kernel also rounds the dequantized weight to Float16 (~5e-4 per element)
        for (Out, In, N, group) in ((128, 64, 20, 16), (128, 128, 100, 32), (192, 256, 70, 128),
                                    (64, 96, 17, 96), (256, 512, 130, 64), (128, 64, 3, 32))
            W = randn(Float32, Out, In); x = randn(Float32, In, N)
            qw = Luminal.QuantWeight(Luminal.to_device(W, dev); group = group)
            out = Luminal.zero_tensor(dev, Float32, Out, N)
            Luminal.batch_matmul!(out, qw, Luminal.to_device(x, dev))
            xr = N > Luminal.QUANT_GEMM_COLS ? Float32.(Float16.(x)) : x
            @test relerr(Array(out), Array(Luminal.dequantize(qw)) * xr) < 2e-3
        end
        Out, In, N = 128, 128, 40
        W = randn(Float32, Out, In); x = randn(Float32, In, N); r = randn(Float32, Out, N)
        qw = Luminal.QuantWeight(Luminal.to_device(W, dev); group = 32)
        out = Luminal.zero_tensor(dev, Float32, Out, N)
        Luminal._q8_matmul!(out, qw, Luminal.to_device(x, dev); residual = Luminal.to_device(r, dev))
        @test relerr(Array(out), Array(Luminal.dequantize(qw)) * Float32.(Float16.(x)) .+ r) < 2e-3
    end

    @testset "4-bit Q4Weight prefill matmul takes the WMMA path" begin
        for (Out, In, N, group, sym) in ((128, 64, 20, 32, true), (128, 128, 100, 32, false),
                                         (192, 256, 70, 64, true), (64, 96, 17, 96, false),
                                         (256, 512, 130, 128, false), (128, 64, 3, 32, true),
                                         # M % 128 == 0 and N >= 192: the 128x128 kernel
                                         (256, 256, 200, 32, true), (384, 512, 300, 64, false),
                                         (128, 128, 192, 32, false))
            W = randn(Float32, Out, In); x = randn(Float32, In, N)
            qw = Luminal.Q4Weight(Luminal.to_device(W, dev); group = group, symmetric = sym)
            out = Luminal.zero_tensor(dev, Float32, Out, N)
            Luminal.batch_matmul!(out, qw, Luminal.to_device(x, dev))
            xr = N > Luminal.QUANT_GEMM_COLS ? Float32.(Float16.(x)) : x
            @test relerr(Array(out), Array(Luminal.dequantize(qw)) * xr) < 2e-3
        end
        Out, In, N = 128, 128, 40
        W = randn(Float32, Out, In); x = randn(Float32, In, N); r = randn(Float32, Out, N)
        qw = Luminal.Q4Weight(Luminal.to_device(W, dev); group = 32)
        out = Luminal.zero_tensor(dev, Float32, Out, N)
        Luminal._q4_matmul!(out, qw, Luminal.to_device(x, dev); residual = Luminal.to_device(r, dev))
        @test relerr(Array(out), Array(Luminal.dequantize(qw)) * Float32.(Float16.(x)) .+ r) < 2e-3
    end
end
