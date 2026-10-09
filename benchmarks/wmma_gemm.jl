# WMMA (matrix-core) f16 GEMM for prefill on gfx11 (RDNA3 / 3.5), versus the GEMMs
# Luminal calls today: rocBLAS sgemm (Float32 weights) and rocBLAS gemm_ex (Float16
# weights, Float32 accumulation and output; `_gemm_ex_f16!`).
#
# Usage: julia --project=. benchmarks/wmma_gemm.jl
#
# C (M, N, Float32) = A (M, K) * B (K, N), A and B Float16. Both operands are stored
# with K contiguous: A as At (K, M) -- exactly Luminal's `HalfWeight.t` -- and B as
# (K, N), the layout of an activation matrix. So each WMMA fragment (one row of A or
# column of B, 16 k) is 32 contiguous bytes. Requires M % 64 == 0, N % 64 == 0,
# K % 32 == 0.
#
# Kernel: 64x64 block, 4 waves (wave32), each wave 32x32 = 2x2 WMMA 16x16x16 tiles,
# BK = 32. 16-byte global loads, LDS rows padded to 80 B, and the next K tile's global
# loads are issued into registers before the current tile's WMMAs.
using AMDGPU, LinearAlgebra

const V16H = NTuple{16,VecElement{Float16}}
const V8H  = NTuple{8,VecElement{Float16}}
const V8F  = NTuple{8,VecElement{Float32}}
@inline wmma(a::V16H, b::V16H, c::V8F) =
    ccall("llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16", llvmcall, V8F, (V16H, V16H, V8F), a, b, c)
@inline zacc() = ntuple(_ -> VecElement(0f0), Val(8))
@inline cat16(a::V8H, b::V8H)::V16H = (a..., b...)

const PADC = 5   # LDS row = 4 chunks of 16 B + 1 pad chunk, to spread banks

# Av, Bv: At (K, M) and B (K, N) reinterpreted as 16-byte chunks (8 halves along K).
function wmma_gemm!(C, Av, Bv, K)
    tid = workitemIdx().x - 1
    l = tid % 32; r = l % 16; h = l ÷ 16; w = tid ÷ 32
    wm = (w % 2) * 32; wn = (w ÷ 2) * 32
    bm = (workgroupIdx().y - 1) * 64; bn = (workgroupIdx().x - 1) * 64
    As = @ROCStaticLocalArray(V8H, (PADC, 64), false)
    Bs = @ROCStaticLocalArray(V8H, (PADC, 64), false)
    a00 = zacc(); a01 = zacc(); a10 = zacc(); a11 = zacc()
    c0 = tid % 4; row0 = tid ÷ 4; row1 = row0 + 32   # this thread's chunks of each K tile
    nk = K ÷ 32
    ra0 = Av[c0 + 1, bm + row0 + 1]; ra1 = Av[c0 + 1, bm + row1 + 1]
    rb0 = Bv[c0 + 1, bn + row0 + 1]; rb1 = Bv[c0 + 1, bn + row1 + 1]
    for t in 0:nk-1
        As[c0 + 1, row0 + 1] = ra0; As[c0 + 1, row1 + 1] = ra1
        Bs[c0 + 1, row0 + 1] = rb0; Bs[c0 + 1, row1 + 1] = rb1
        sync_workgroup()
        if t + 1 < nk
            kc = (t + 1) * 4 + c0 + 1
            ra0 = Av[kc, bm + row0 + 1]; ra1 = Av[kc, bm + row1 + 1]
            rb0 = Bv[kc, bn + row0 + 1]; rb1 = Bv[kc, bn + row1 + 1]
        end
        for s in 0:1
            A0 = cat16(As[2s + 1, wm + r + 1],      As[2s + 2, wm + r + 1])
            A1 = cat16(As[2s + 1, wm + 16 + r + 1], As[2s + 2, wm + 16 + r + 1])
            B0 = cat16(Bs[2s + 1, wn + r + 1],      Bs[2s + 2, wn + r + 1])
            B1 = cat16(Bs[2s + 1, wn + 16 + r + 1], Bs[2s + 2, wn + 16 + r + 1])
            a00 = wmma(A0, B0, a00); a01 = wmma(A0, B1, a01)
            a10 = wmma(A1, B0, a10); a11 = wmma(A1, B1, a11)
        end
        sync_workgroup()
    end
    m0 = bm + wm; n0 = bn + wn
    for i in 0:7   # lane l holds D[2i + l÷16, l%16] of each 16x16 tile
        C[m0+2i+h+1,    n0+r+1]    = a00[i+1].value
        C[m0+2i+h+1,    n0+16+r+1] = a01[i+1].value
        C[m0+16+2i+h+1, n0+r+1]    = a10[i+1].value
        C[m0+16+2i+h+1, n0+16+r+1] = a11[i+1].value
    end
    return
end

# C (M, N) = At' * B for ROCArrays At (K, M), B (K, N) of Float16, C of Float32.
function wmma_gemm!(C::ROCArray{Float32,2}, At::ROCArray{Float16,2}, B::ROCArray{Float16,2})
    K, M = size(At); N = size(B, 2)
    (M % 64 == 0 && N % 64 == 0 && K % 32 == 0 && size(B, 1) == K) ||
        throw(ArgumentError("wmma_gemm!: need M % 64 == 0, N % 64 == 0, K % 32 == 0"))
    Av = reinterpret(V8H, At); Bv = reinterpret(V8H, B)
    @roc groupsize=128 gridsize=(N ÷ 64, M ÷ 64) wmma_gemm!(C, Av, Bv, K)
    return C
end

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
              (4096, 4096, 512), (11008, 4096, 512), (2048, 2048, 2048), (4096, 4096, 4096)]
    for (M, K, N) in shapes
        W = rand(Float16, M, K); X = rand(Float16, K, N)
        dW = ROCArray(W); dWt = ROCArray(collect(W')); dX = ROCArray(X)
        dX32 = ROCArray(Float32.(X)); dW32 = ROCArray(Float32.(W))
        Cw = AMDGPU.zeros(Float32, M, N); Ce = AMDGPU.zeros(Float32, M, N); Cs = AMDGPU.zeros(Float32, M, N)
        ref = Float32.(W) * Float32.(X)
        wmma_gemm!(Cw, dWt, dX); gemm_ex!(Ce, dW, dX); AMDGPU.synchronize()
        rel(C) = maximum(abs.(Array(C) .- ref)) / maximum(abs.(ref))
        tw = bench(() -> wmma_gemm!(Cw, dWt, dX))
        te = bench(() -> gemm_ex!(Ce, dW, dX))
        ts = bench(() -> mul!(Cs, dW32, dX32))
        fl = 2.0 * M * N * K
        tf(t) = round(fl / t / 1e12, digits = 2)
        ms(t) = round(t * 1e3, digits = 3)
        println("Out=$M In=$K N=$N  err wmma=$(round(rel(Cw), sigdigits=2)) gemm_ex=$(round(rel(Ce), sigdigits=2))",
                " | wmma $(ms(tw)) ms $(tf(tw)) TF | gemm_ex $(ms(te)) ms $(tf(te)) TF | sgemm $(ms(ts)) ms $(tf(ts)) TF")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
