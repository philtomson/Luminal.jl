# Matrix-core (WMMA) f16 GEMM for prefill on gfx11 GPUs (RDNA3 / 3.5). See
# benchmarks/wmma_gemm.jl for the standalone version and the comparison against
# rocBLAS.
#
# C (M, N, Float32) = A (M, K) * B (K, N), A and B Float16, both stored with K
# contiguous (A as (K, M), Luminal's `HalfWeight.t`; B as an activation matrix), so
# each WMMA fragment is 32 contiguous bytes. 64x64 block, 4 waves (wave32) of 32x32
# (2x2 WMMA 16x16x16 tiles), BK = 32; 16-byte global loads, LDS rows padded to 80 B,
# and the next K tile's global loads are issued before the current tile's WMMAs.
# Requires M % 64 == 0 and K % 32 == 0; B must have a multiple of 64 columns (the
# caller pads with zeros) and C is only written for the first N columns.

using AMDGPU

const WMMA_V16H = NTuple{16,VecElement{Float16}}
const WMMA_V8H  = NTuple{8,VecElement{Float16}}
const WMMA_V8F  = NTuple{8,VecElement{Float32}}
@inline _wmma16(a::WMMA_V16H, b::WMMA_V16H, c::WMMA_V8F) =
    ccall("llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16", llvmcall, WMMA_V8F,
          (WMMA_V16H, WMMA_V16H, WMMA_V8F), a, b, c)
@inline _wmma_zero() = ntuple(_ -> VecElement(0f0), Val(8))
@inline _wmma_cat(a::WMMA_V8H, b::WMMA_V8H)::WMMA_V16H = (a..., b...)

const WMMA_PADC = 5   # LDS row = 4 chunks of 16 B + 1 pad chunk, to spread banks

# Av, Bv: A (K, M) and B (K, Npad) viewed as 16-byte chunks (8 halves along K).
function _wmma_gemm_kernel!(C, Av, Bv, K, N)
    tid = workitemIdx().x - 1
    l = tid % 32; r = l % 16; h = l ÷ 16; w = tid ÷ 32
    wm = (w % 2) * 32; wn = (w ÷ 2) * 32
    bm = (workgroupIdx().y - 1) * 64; bn = (workgroupIdx().x - 1) * 64
    As = @ROCStaticLocalArray(WMMA_V8H, (WMMA_PADC, 64), false)
    Bs = @ROCStaticLocalArray(WMMA_V8H, (WMMA_PADC, 64), false)
    a00 = _wmma_zero(); a01 = _wmma_zero(); a10 = _wmma_zero(); a11 = _wmma_zero()
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
            A0 = _wmma_cat(As[2s + 1, wm + r + 1],      As[2s + 2, wm + r + 1])
            A1 = _wmma_cat(As[2s + 1, wm + 16 + r + 1], As[2s + 2, wm + 16 + r + 1])
            B0 = _wmma_cat(Bs[2s + 1, wn + r + 1],      Bs[2s + 2, wn + r + 1])
            B1 = _wmma_cat(Bs[2s + 1, wn + 16 + r + 1], Bs[2s + 2, wn + 16 + r + 1])
            a00 = _wmma16(A0, B0, a00); a01 = _wmma16(A0, B1, a01)
            a10 = _wmma16(A1, B0, a10); a11 = _wmma16(A1, B1, a11)
        end
        sync_workgroup()
    end
    m0 = bm + wm; n0 = bn + wn
    ok0 = n0 + r < N; ok1 = n0 + 16 + r < N
    for i in 0:7   # lane l holds D[2i + l÷16, l%16] of each 16x16 tile
        if ok0
            C[m0+2i+h+1,    n0+r+1]    = a00[i+1].value
            C[m0+16+2i+h+1, n0+r+1]    = a10[i+1].value
        end
        if ok1
            C[m0+2i+h+1,    n0+16+r+1] = a01[i+1].value
            C[m0+16+2i+h+1, n0+16+r+1] = a11[i+1].value
        end
    end
    return
end

# gfx11 has the 16x16x16 f16 WMMA instruction used above (gfx12's differs).
const _WMMA_ARCH = Ref{Union{Nothing,Bool}}(nothing)
function wmma_supported()
    if _WMMA_ARCH[] === nothing
        _WMMA_ARCH[] = try
            AMDGPU.functional() && startswith(string(AMDGPU.HIP.gcn_arch(AMDGPU.device())), "gfx11")
        catch
            false
        end
    end
    return _WMMA_ARCH[]
end
# Set LUMINAL_WMMA=0 to fall back to rocBLAS (for comparisons).
wmma_enabled() = get(ENV, "LUMINAL_WMMA", "1") != "0" && wmma_supported()

# C (M, N) = At' * B for ROCArrays At (K, M) and B (K, Npad) of Float16, Npad a
# multiple of 64 (columns >= N are ignored), C (M, N) Float32.
function wmma_gemm!(C::ROCArray{Float32,2}, At::ROCArray{Float16,2}, B::ROCArray{Float16,2}, N::Int)
    K, M = size(At); Npad = size(B, 2)
    (M % 64 == 0 && Npad % 64 == 0 && K % 32 == 0 && size(B, 1) == K && N <= Npad && size(C) == (M, N)) ||
        throw(ArgumentError("wmma_gemm!: need M % 64 == 0, K % 32 == 0, padded N % 64 == 0"))
    Av = reinterpret(WMMA_V8H, At); Bv = reinterpret(WMMA_V8H, B)
    @roc groupsize=128 gridsize=(Npad ÷ 64, M ÷ 64) _wmma_gemm_kernel!(C, Av, Bv, K, N)
    return C
end
