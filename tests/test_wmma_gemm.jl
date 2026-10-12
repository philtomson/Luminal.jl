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
                                         (128, 128, 192, 32, false),
                                         # N >= 384: the 128x256 kernel
                                         (256, 128, 400, 32, true), (128, 256, 512, 64, false), (256, 96, 384, 32, false),
                                         # M % 16 == 0 only: a ragged last block
                                         (48, 128, 100, 32, true), (80, 256, 250, 64, false), (176, 128, 200, 32, true),
                                         # 7..16 columns: the skinny kernel (group 32, K % 256 == 0), else the GEMV
                                         (128, 256, 8, 32, true), (48, 256, 16, 32, false), (256, 512, 12, 32, false), (80, 768, 9, 32, true),
                                         (64, 96, 10, 32, true), (128, 128, 6, 32, false),
                                         (128, 256, 20, 32, true), (48, 512, 32, 32, false), (256, 256, 33, 32, true),
                                         # 2..6 columns: the multi-column GEMV (4 rows per workgroup when M % 4 == 0)
                                         (128, 128, 2, 32, true), (48, 256, 3, 32, false), (64, 96, 5, 32, true), (130, 128, 2, 32, false),
                                         (256, 512, 4, 64, true))
            W = randn(Float32, Out, In); x = randn(Float32, In, N)
            qw = Luminal.Q4Weight(Luminal.to_device(W, dev); group = group, symmetric = sym)
            out = Luminal.zero_tensor(dev, Float32, Out, N)
            Luminal.batch_matmul!(out, qw, Luminal.to_device(x, dev))
            f16 = N > Luminal.QUANT_GEMM_COLS || (Luminal.SKINNY_MIN_COLS <= N <= Luminal.SKINNY_MAX_COLS && In % 256 == 0 && group == 32 && Out % 16 == 0)
            xr = f16 ? Float32.(Float16.(x)) : x
            @test relerr(Array(out), Array(Luminal.dequantize(qw)) * xr) < 2e-3
        end
        # with a residual, through the 4-row GEMV (2 columns) and the skinny kernel (9)
        for N in (2, 9)
            Out, In = 128, 256   # (group 32, K % 256 == 0: the skinny kernel)
            W = randn(Float32, Out, In); x = randn(Float32, In, N); r = randn(Float32, Out, N)
            qw = Luminal.Q4Weight(Luminal.to_device(W, dev); group = 32)
            out = Luminal.zero_tensor(dev, Float32, Out, N)
            Luminal._q4_matmul!(out, qw, Luminal.to_device(x, dev); residual = Luminal.to_device(r, dev))
            xr = dev isa Luminal.AbstractGPUDevice && N >= Luminal.SKINNY_MIN_COLS ? Float32.(Float16.(x)) : x
            @test relerr(Array(out), Array(Luminal.dequantize(qw)) * xr .+ r) < 2e-3
        end
        Out, In, N = 128, 128, 40
        W = randn(Float32, Out, In); x = randn(Float32, In, N); r = randn(Float32, Out, N)
        qw = Luminal.Q4Weight(Luminal.to_device(W, dev); group = 32)
        out = Luminal.zero_tensor(dev, Float32, Out, N)
        Luminal._q4_matmul!(out, qw, Luminal.to_device(x, dev); residual = Luminal.to_device(r, dev))
        @test relerr(Array(out), Array(Luminal.dequantize(qw)) * Float32.(Float16.(x)) .+ r) < 2e-3
        # the residual added in the kernels' store (128x128 and 128x256 kernels, a ragged M), with Float32 and Float16 activations
        for (Out, In, N) in ((256, 256, 200), (256, 128, 400), (384, 96, 512), (176, 128, 200))
            W = randn(Float32, Out, In); x = randn(Float32, In, N); r = randn(Float32, Out, N)
            qw = Luminal.Q4Weight(Luminal.to_device(W, dev); group = 32)
            ref = Array(Luminal.dequantize(qw)) * Float32.(Float16.(x)) .+ r
            for xin in (Luminal.to_device(x, dev), Luminal.to_device(Float16.(x), dev))
                out = Luminal.zero_tensor(dev, Float32, Out, N)
                Luminal._q4_matmul!(out, qw, xin; residual = Luminal.to_device(r, dev))
                @test relerr(Array(out), ref) < 2e-3
            end
        end
    end
end
