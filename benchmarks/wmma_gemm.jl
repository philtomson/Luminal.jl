# WMMA (matrix-core) GEMM for prefill on gfx11 (RDNA3 / 3.5; kernels in src/WMMA.jl),
# versus the GEMMs Luminal used before: rocBLAS gemm_ex (Float16 weights, Float32
# accumulation and output) and sgemm (Float32 weights); for int8 weights, versus
# int8 and 4-bit weights, versus dequantizing to a Float32 matrix and sgemm
# (`_dequant_gemm!`).
#
# Usage: julia --project=. benchmarks/wmma_gemm.jl
#
# C (M, N, Float32) = A (M, K) * B (K, N). Both operands have K contiguous: A as
# At (K, M) -- Luminal's `HalfWeight.t` / `QuantWeight.q` -- and B as an activation
# matrix (K, N). Requires M % 64 == 0 and K % 32 == 0; N is arbitrary.
using Luminal, AMDGPU, LinearAlgebra

# rocBLAS gemm_ex as Luminal's `_gemm_ex_f16!`: W (M, K) column-major Float16.
function gemm_ex!(C, W, X)
    M, K = size(W); N = size(X, 2)
    RB = AMDGPU.rocBLAS
    α = Ref{Float32}(1f0); β = Ref{Float32}(0f0)
    p(A) = reinterpret(Ptr{Cvoid}, pointer(A))
    RB.rocblas_gemm_ex_64(RB.handle(), RB.rocblas_operation_none, RB.rocblas_operation_none,
        M, N, K, α, p(W), RB.rocblas_datatype_f16_r, M, p(X), RB.rocblas_datatype_f16_r, K,
        β, p(C), RB.rocblas_datatype_f32_r, M, p(C), RB.rocblas_datatype_f32_r, M,
        RB.rocblas_datatype_f32_r, RB.rocblas_gemm_algo_standard, Int32(0), UInt32(0))
    return C
end

function bench(f; n = 20)
    f(); AMDGPU.synchronize()
    AMDGPU.@elapsed(for _ in 1:n; f(); end) / n
end

function main()
    println("device: ", AMDGPU.device())
    # (Out, In, tokens): TinyLlama (2048 hidden, 5632 MLP, 32000 vocab), then larger.
    shapes = [(2048, 2048, 128), (2048, 2048, 512), (5632, 2048, 128), (5632, 2048, 512),
              (2048, 5632, 128), (2048, 5632, 512), (32000, 2048, 512),
              (4096, 4096, 512), (11008, 4096, 512), (5120, 5120, 512), (17408, 5120, 512),
              (4096, 4096, 4096)]
    tf(fl, t) = round(fl / t / 1e12, digits = 2)
    ms(t) = round(t * 1e3, digits = 3)
    println("--- Float16 weights: WMMA | gemm_ex | sgemm")
    for (M, K, N) in shapes
        W = rand(Float16, M, K); X = rand(Float16, K, N)
        dW = ROCArray(W); dWt = ROCArray(collect(W')); dX = ROCArray(X)
        dX32 = ROCArray(Float32.(X)); dW32 = ROCArray(Float32.(W))
        Cw = AMDGPU.zeros(Float32, M, N); Ce = AMDGPU.zeros(Float32, M, N); Cs = AMDGPU.zeros(Float32, M, N)
        Np = cld(N, 64) * 64
        dXp = AMDGPU.zeros(Float16, K, Np); copyto!(view(dXp, :, 1:N), dX)
        ref = Float32.(W) * Float32.(X)
        Luminal.wmma_gemm!(Cw, dWt, dXp, N); gemm_ex!(Ce, dW, dX); AMDGPU.synchronize()
        rel(C) = maximum(abs.(Array(C) .- ref)) / maximum(abs.(ref))
        tw = bench(() -> Luminal.wmma_gemm!(Cw, dWt, dXp, N))
        te = bench(() -> gemm_ex!(Ce, dW, dX))
        ts = bench(() -> mul!(Cs, dW32, dX32))
        fl = 2.0 * M * N * K
        println("Out=$M In=$K N=$N  err wmma=$(round(rel(Cw), sigdigits=2)) gemm_ex=$(round(rel(Ce), sigdigits=2))",
                " | $(ms(tw)) ms $(tf(fl, tw)) TF | $(ms(te)) ms $(tf(fl, te)) TF | $(ms(ts)) ms $(tf(fl, ts)) TF")
    end
    println("--- int8 weights (group 128): WMMA | dequantize + sgemm")
    for (M, K, N) in shapes
        K > 8192 && continue
        qw = Luminal.QuantWeight(ROCArray(randn(Float32, M, K)); group = 128)
        X = ROCArray(randn(Float32, K, N)); C1 = AMDGPU.zeros(Float32, M, N); C0 = similar(C1)
        Luminal._wmma_q8_gemm!(C1, qw, X, nothing); Luminal._dequant_gemm!(C0, qw, X, nothing); AMDGPU.synchronize()
        ref = Array(C0)
        err = maximum(abs.(Array(C1) .- ref)) / maximum(abs.(ref))
        tw = bench(() -> Luminal._wmma_q8_gemm!(C1, qw, X, nothing))
        td = bench(() -> Luminal._dequant_gemm!(C0, qw, X, nothing); n = 5)
        fl = 2.0 * M * N * K
        println("Out=$M In=$K N=$N  err vs sgemm=$(round(err, sigdigits=2)) | $(ms(tw)) ms $(tf(fl, tw)) TF | $(ms(td)) ms $(tf(fl, td)) TF")
    end
    println("--- 4-bit weights (group 32, symmetric): WMMA | dequantize + sgemm")
    for (M, K, N) in shapes
        K > 8192 && continue
        qw = Luminal.Q4Weight(ROCArray(randn(Float32, M, K)); group = 32)
        X = ROCArray(randn(Float32, K, N)); C1 = AMDGPU.zeros(Float32, M, N); C0 = similar(C1)
        Luminal._wmma_q4_gemm!(C1, qw, X, nothing); Luminal._dequant_gemm!(C0, qw, X, nothing); AMDGPU.synchronize()
        ref = Array(C0)
        err = maximum(abs.(Array(C1) .- ref)) / maximum(abs.(ref))
        tw = bench(() -> Luminal._wmma_q4_gemm!(C1, qw, X, nothing))
        td = bench(() -> Luminal._dequant_gemm!(C0, qw, X, nothing); n = 5)
        fl = 2.0 * M * N * K
        println("Out=$M In=$K N=$N  err vs sgemm=$(round(err, sigdigits=2)) | $(ms(tw)) ms $(tf(fl, tw)) TF | $(ms(td)) ms $(tf(fl, td)) TF")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
