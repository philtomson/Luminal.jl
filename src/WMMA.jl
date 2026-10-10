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

# One BK=32 tile of WMMAs for a wave's 32x32 output: As/Bs hold 4 chunks (32 halves)
# per row of the 64-row A and B tiles; (wm, wn) is the wave's offset in the block.
@inline function _wmma_tile(acc, As, Bs, wm, wn, r)
    a00, a01, a10, a11 = acc
    for s in 0:1
        A0 = _wmma_cat(As[2s + 1, wm + r + 1],      As[2s + 2, wm + r + 1])
        A1 = _wmma_cat(As[2s + 1, wm + 16 + r + 1], As[2s + 2, wm + 16 + r + 1])
        B0 = _wmma_cat(Bs[2s + 1, wn + r + 1],      Bs[2s + 2, wn + r + 1])
        B1 = _wmma_cat(Bs[2s + 1, wn + 16 + r + 1], Bs[2s + 2, wn + 16 + r + 1])
        a00 = _wmma16(A0, B0, a00); a01 = _wmma16(A0, B1, a01)
        a10 = _wmma16(A1, B0, a10); a11 = _wmma16(A1, B1, a11)
    end
    return (a00, a01, a10, a11)
end

# Write the wave's 32x32 result; columns >= N (padding) are skipped.
@inline function _wmma_store!(C, acc, m0, n0, r, h, N)
    a00, a01, a10, a11 = acc
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

@inline _wmma_zeros() = (_wmma_zero(), _wmma_zero(), _wmma_zero(), _wmma_zero())

# Float16 weights. Av, Bv: A (K, M) and B (K, Npad) viewed as 16-byte chunks (8 halves
# along K). Each thread stages 2 chunks of A and 2 of B per K tile.
function _wmma_gemm_kernel!(C, Av, Bv, K, N)
    tid = workitemIdx().x - 1
    l = tid % 32; r = l % 16; h = l ÷ 16; w = tid ÷ 32
    wm = (w % 2) * 32; wn = (w ÷ 2) * 32
    bm = (workgroupIdx().y - 1) * 64; bn = (workgroupIdx().x - 1) * 64
    As = @ROCStaticLocalArray(WMMA_V8H, (WMMA_PADC, 64), false)
    Bs = @ROCStaticLocalArray(WMMA_V8H, (WMMA_PADC, 64), false)
    acc = _wmma_zeros()
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
        acc = _wmma_tile(acc, As, Bs, wm, wn, r)
        sync_workgroup()
    end
    _wmma_store!(C, acc, bm + wm, bn + wn, r, h, N)
    return
end

# 16 int8 of one weight row times the Float32 scale of their group, as two chunks of 8
# halves. (The rounding to Float16 is far below the int8 quantization error.)
@inline _wmma_dq8(q::NTuple{16,Int8}, s::Float32) =
    (ntuple(i -> VecElement(Float16(Float32(q[i]) * s)), Val(8)),
     ntuple(i -> VecElement(Float16(Float32(q[i + 8]) * s)), Val(8)))

# int8 weights (QuantWeight): Qv (K/16, M) of 16 int8 along K, Sc (K/group, M) Float32
# scales (group a multiple of 16). Each thread stages one 16-int8 chunk of A per K tile,
# dequantized to Float16 on the way into LDS, and 2 chunks of B.
function _wmma_q8_kernel!(C, Qv, Sc, Bv, K, N, group)
    tid = workitemIdx().x - 1
    l = tid % 32; r = l % 16; h = l ÷ 16; w = tid ÷ 32
    wm = (w % 2) * 32; wn = (w ÷ 2) * 32
    bm = (workgroupIdx().y - 1) * 64; bn = (workgroupIdx().x - 1) * 64
    As = @ROCStaticLocalArray(WMMA_V8H, (WMMA_PADC, 64), false)
    Bs = @ROCStaticLocalArray(WMMA_V8H, (WMMA_PADC, 64), false)
    acc = _wmma_zeros()
    arow = tid ÷ 2; ah = tid % 2                       # A: 64 rows x 2 chunks per K tile
    c0 = tid % 4; row0 = tid ÷ 4; row1 = row0 + 32     # B as in the Float16 kernel
    nk = K ÷ 32
    qa = Qv[ah + 1, bm + arow + 1]; sa = Sc[(16ah) ÷ group + 1, bm + arow + 1]
    rb0 = Bv[c0 + 1, bn + row0 + 1]; rb1 = Bv[c0 + 1, bn + row1 + 1]
    for t in 0:nk-1
        lo, hi = _wmma_dq8(qa, sa)
        As[2ah + 1, arow + 1] = lo; As[2ah + 2, arow + 1] = hi
        Bs[c0 + 1, row0 + 1] = rb0; Bs[c0 + 1, row1 + 1] = rb1
        sync_workgroup()
        if t + 1 < nk
            kq = 2 * (t + 1) + ah                      # next 16-wide chunk along K (0-based)
            qa = Qv[kq + 1, bm + arow + 1]; sa = Sc[(16kq) ÷ group + 1, bm + arow + 1]
            kc = (t + 1) * 4 + c0 + 1
            rb0 = Bv[kc, bn + row0 + 1]; rb1 = Bv[kc, bn + row1 + 1]
        end
        acc = _wmma_tile(acc, As, Bs, wm, wn, r)
        sync_workgroup()
    end
    _wmma_store!(C, acc, bm + wm, bn + wn, r, h, N)
    return
end

# 16 4-bit codes (8 bytes: even k in the low nibble) of one weight row, dequantized as
# q * scale + min, as two chunks of 8 halves.
@inline function _wmma_nib(b::NTuple{8,UInt8}, i::Int)::Float32
    x = @inbounds b[(i + 1) >> 1]
    return Float32(isodd(i) ? (x & 0x0f) : (x >> 0x04))
end
# Packed Float16: the codes are ORed into the mantissa of 1024.0 (0x6400), 1024 is
# subtracted exactly, and one packed fma applies the scale and minimum (a single
# rounding, to Float16), two codes per instruction.
const WMMA_V2H = NTuple{2,VecElement{Float16}}
@inline _wmma_fma2(a::WMMA_V2H, b::WMMA_V2H, c::WMMA_V2H) =
    ccall("llvm.fma.v2f16", llvmcall, WMMA_V2H, (WMMA_V2H, WMMA_V2H, WMMA_V2H), a, b, c)
@inline _wmma_bitcast2(u::UInt32) = Base.llvmcall(
    ("""define <2 x half> @entry(i32 %0) alwaysinline {
        %r = bitcast i32 %0 to <2 x half>
        ret <2 x half> %r
    }""", "entry"), WMMA_V2H, Tuple{UInt32}, u)
@inline function _wmma_pair(b::UInt8, s2::WMMA_V2H, m2::WMMA_V2H)
    x = UInt32(b)
    h = _wmma_bitcast2(0x64006400 | (x & 0x0f) | ((x >> 0x04) << 0x10))
    one2 = (VecElement(Float16(1)), VecElement(Float16(1)))
    off2 = (VecElement(Float16(-1024)), VecElement(Float16(-1024)))
    return _wmma_fma2(_wmma_fma2(h, one2, off2), s2, m2)
end
@inline function _wmma_dq4(b::NTuple{8,UInt8}, s::Float32, m::Float32)
    s2 = (VecElement(Float16(s)), VecElement(Float16(s)))
    m2 = (VecElement(Float16(m)), VecElement(Float16(m)))
    p = ntuple(i -> _wmma_pair(b[i], s2, m2), Val(8))
    return (ntuple(i -> p[(i + 1) >> 1][2 - i % 2], Val(8)),
            ntuple(i -> p[4 + (i + 1) >> 1][2 - i % 2], Val(8)))
end

# 4-bit weights (Q4Weight): Pv (K/32, M) of 32 codes along K (16 bytes) = one K tile per
# chunk, Sc / Mn (K/group, M) Float16 scales and minimums (Mn = Sc and ignored when
# symmetric: min = -8 * scale). Two threads share each row's chunk and dequantize 16
# codes each into LDS.
function _wmma_q4_kernel!(C, Pv, Sc, Mn, Bv, K, N, group, ::Val{SYM}) where {SYM}
    tid = workitemIdx().x - 1
    l = tid % 32; r = l % 16; h = l ÷ 16; w = tid ÷ 32
    wm = (w % 2) * 32; wn = (w ÷ 2) * 32
    bm = (workgroupIdx().y - 1) * 64; bn = (workgroupIdx().x - 1) * 64
    As = @ROCStaticLocalArray(WMMA_V8H, (WMMA_PADC, 64), false)
    Bs = @ROCStaticLocalArray(WMMA_V8H, (WMMA_PADC, 64), false)
    acc = _wmma_zeros()
    arow = tid ÷ 2; ah = tid % 2
    c0 = tid % 4; row0 = tid ÷ 4; row1 = row0 + 32
    nk = K ÷ 32
    qa = Pv[1, bm + arow + 1]; sa = Float32(Sc[1, bm + arow + 1])
    ma = SYM ? -8f0 * sa : Float32(Mn[1, bm + arow + 1])
    rb0 = Bv[c0 + 1, bn + row0 + 1]; rb1 = Bv[c0 + 1, bn + row1 + 1]
    for t in 0:nk-1
        bytes = let q = qa   # (a closure must not capture the reassigned qa)
            ah == 0 ? ntuple(i -> q[i], Val(8)) : ntuple(i -> q[i + 8], Val(8))
        end
        lo, hi = _wmma_dq4(bytes, sa, ma)
        As[2ah + 1, arow + 1] = lo; As[2ah + 2, arow + 1] = hi
        Bs[c0 + 1, row0 + 1] = rb0; Bs[c0 + 1, row1 + 1] = rb1
        sync_workgroup()
        if t + 1 < nk
            gi = (32 * (t + 1)) ÷ group + 1
            qa = Pv[t + 2, bm + arow + 1]; sa = Float32(Sc[gi, bm + arow + 1])
            ma = SYM ? -8f0 * sa : Float32(Mn[gi, bm + arow + 1])
            kc = (t + 1) * 4 + c0 + 1
            rb0 = Bv[kc, bn + row0 + 1]; rb1 = Bv[kc, bn + row1 + 1]
        end
        acc = _wmma_tile(acc, As, Bs, wm, wn, r)
        sync_workgroup()
    end
    _wmma_store!(C, acc, bm + wm, bn + wn, r, h, N)
    return
end

# 128x128 block, 8 waves (wave = 32 rows x 64 columns = 2x4 WMMA tiles). Each dequantized
# A tile is shared by twice as many columns as in the 64x64 kernel, and a wave does 8
# WMMAs per 6 LDS fragment loads (4 per 4 before).
@inline function _wmma_tile_big(acc::NTuple{8,WMMA_V8F}, As, Bs, wm, wn, r)
    c0, c1, c2, c3, c4, c5, c6, c7 = acc   # (A tile i, B tile j) is acc[4i + j + 1]
    for s in 0:1
        A0 = _wmma_cat(As[2s + 1, wm + r + 1],      As[2s + 2, wm + r + 1])
        A1 = _wmma_cat(As[2s + 1, wm + 16 + r + 1], As[2s + 2, wm + 16 + r + 1])
        B = _wmma_cat(Bs[2s + 1, wn + r + 1], Bs[2s + 2, wn + r + 1])
        c0 = _wmma16(A0, B, c0); c4 = _wmma16(A1, B, c4)
        B = _wmma_cat(Bs[2s + 1, wn + 16 + r + 1], Bs[2s + 2, wn + 16 + r + 1])
        c1 = _wmma16(A0, B, c1); c5 = _wmma16(A1, B, c5)
        B = _wmma_cat(Bs[2s + 1, wn + 32 + r + 1], Bs[2s + 2, wn + 32 + r + 1])
        c2 = _wmma16(A0, B, c2); c6 = _wmma16(A1, B, c6)
        B = _wmma_cat(Bs[2s + 1, wn + 48 + r + 1], Bs[2s + 2, wn + 48 + r + 1])
        c3 = _wmma16(A0, B, c3); c7 = _wmma16(A1, B, c7)
    end
    return (c0, c1, c2, c3, c4, c5, c6, c7)
end

@inline function _wmma_store_big!(C, acc, m0, n0, r, h, N)
    for j in 0:3
        n0 + 16j + r < N || continue
        for i in 0:1, e in 0:7   # lane l holds D[2e + l÷16, l%16] of each 16x16 tile
            C[m0 + 16i + 2e + h + 1, n0 + 16j + r + 1] = acc[4i + j + 1][e + 1].value
        end
    end
    return
end

function _wmma_q4_big_kernel!(C, Pv, Sc, Mn, Bv, K, N, group, ::Val{SYM}) where {SYM}
    tid = workitemIdx().x - 1
    l = tid % 32; r = l % 16; h = l ÷ 16; w = tid ÷ 32
    wm = (w % 4) * 32; wn = (w ÷ 4) * 64
    bm = (workgroupIdx().y - 1) * 128; bn = (workgroupIdx().x - 1) * 128
    As = @ROCStaticLocalArray(WMMA_V8H, (WMMA_PADC, 128), false)
    Bs = @ROCStaticLocalArray(WMMA_V8H, (WMMA_PADC, 128), false)
    acc = ntuple(_ -> _wmma_zero(), Val(8))
    arow = tid ÷ 2; ah = tid % 2                       # A: 128 rows x 2 half-chunks per K tile
    c0 = tid % 4; row0 = tid ÷ 4; row1 = row0 + 64     # B: 128 columns x 4 chunks
    nk = K ÷ 32
    qa = Pv[1, bm + arow + 1]; sa = Float32(Sc[1, bm + arow + 1])
    ma = SYM ? -8f0 * sa : Float32(Mn[1, bm + arow + 1])
    rb0 = Bv[c0 + 1, bn + row0 + 1]; rb1 = Bv[c0 + 1, bn + row1 + 1]
    for t in 0:nk-1
        bytes = let q = qa
            ah == 0 ? ntuple(i -> q[i], Val(8)) : ntuple(i -> q[i + 8], Val(8))
        end
        lo, hi = _wmma_dq4(bytes, sa, ma)
        As[2ah + 1, arow + 1] = lo; As[2ah + 2, arow + 1] = hi
        Bs[c0 + 1, row0 + 1] = rb0; Bs[c0 + 1, row1 + 1] = rb1
        sync_workgroup()
        if t + 1 < nk
            gi = (32 * (t + 1)) ÷ group + 1
            qa = Pv[t + 2, bm + arow + 1]; sa = Float32(Sc[gi, bm + arow + 1])
            ma = SYM ? -8f0 * sa : Float32(Mn[gi, bm + arow + 1])
            kc = (t + 1) * 4 + c0 + 1
            rb0 = Bv[kc, bn + row0 + 1]; rb1 = Bv[kc, bn + row1 + 1]
        end
        acc = _wmma_tile_big(acc, As, Bs, wm, wn, r)
        sync_workgroup()
    end
    _wmma_store_big!(C, acc, bm + wm, bn + wn, r, h, N)
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

# C (M, N) = dequantize(Q, scale)' * B for Q (K, M) Int8, scale (K/group, M) Float32,
# B (K, Npad) Float16 (see `wmma_gemm!`); K % 32 == 0, group % 16 == 0.
function wmma_q8_gemm!(C::ROCArray{Float32,2}, Q::ROCArray{Int8,2}, scale::ROCArray{Float32,2},
                       group::Int, B::ROCArray{Float16,2}, N::Int)
    K, M = size(Q); Npad = size(B, 2)
    (M % 64 == 0 && Npad % 64 == 0 && K % 32 == 0 && group % 16 == 0 && size(B, 1) == K &&
     size(scale) == (K ÷ group, M) && N <= Npad && size(C) == (M, N)) ||
        throw(ArgumentError("wmma_q8_gemm!: need M % 64 == 0, K % 32 == 0, padded N % 64 == 0"))
    Qv = reinterpret(NTuple{16,Int8}, Q); Bv = reinterpret(WMMA_V8H, B)
    @roc groupsize=128 gridsize=(Npad ÷ 64, M ÷ 64) _wmma_q8_kernel!(C, Qv, scale, Bv, K, N, group)
    return C
end

# C (M, N) = dequantize(P, scale, mn)' * B for the 4-bit layout of `Q4Weight`: Pv (K/32, M)
# of NTuple{16,UInt8}, scale / mn (K/group, M) Float16 (`mn` = nothing when symmetric).
function wmma_q4_gemm!(C::ROCArray{Float32,2}, Pv, scale::ROCArray{Float16,2}, mn,
                       group::Int, B::ROCArray{Float16,2}, N::Int)
    K32, M = size(Pv); K = 32 * K32; Npad = size(B, 2)
    (M % 64 == 0 && Npad % 64 == 0 && group % 32 == 0 && size(B, 1) == K &&
     size(scale) == (K ÷ group, M) && N <= Npad && size(C) == (M, N)) ||
        throw(ArgumentError("wmma_q4_gemm!: need M % 64 == 0, group % 32 == 0, padded N % 64 == 0"))
    Bv = reinterpret(WMMA_V8H, B)
    args = (C, Pv, scale, mn === nothing ? scale : mn, Bv, K, N, group, Val(mn === nothing))
    if M % 128 == 0 && Npad % 128 == 0 && get(ENV, "LUMINAL_WMMA_BIG", "1") != "0"
        @roc groupsize=256 gridsize=(Npad ÷ 128, M ÷ 128) _wmma_q4_big_kernel!(args...)
    else
        @roc groupsize=128 gridsize=(Npad ÷ 64, M ÷ 64) _wmma_q4_kernel!(args...)
    end
    return C
end
