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
const WMMA_V4F  = NTuple{4,VecElement{Float32}}
@inline _wmma16(a::WMMA_V16H, b::WMMA_V16H, c::WMMA_V8F) =
    ccall("llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16", llvmcall, WMMA_V8F,
          (WMMA_V16H, WMMA_V16H, WMMA_V8F), a, b, c)
@inline _wmma_zero() = ntuple(_ -> VecElement(0f0), Val(8))
@inline _wmma_cat(a::WMMA_V8H, b::WMMA_V8H)::WMMA_V16H = (a..., b...)

const WMMA_SKINNY_WAVES = Ref(4)   # waves (= K splits) per workgroup of the skinny int4 kernel
const WMMA_Q4_WPE = Ref(0)   # waves_per_eu launch bound of the int4 kernels (0 = default); needs the AMDGPU.jl launch-bounds branch
const WMMA_Q4_PD = Ref(1)   # prefetch depth of the 128x128 int4 kernel (K tiles in flight)
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
    M = size(C, 1)   # rows past M (a ragged last block) are skipped
    for i in 0:7   # lane l holds D[2i + l÷16, l%16] of each 16x16 tile
        if ok0
            m0 + 2i + h < M && (C[m0+2i+h+1,    n0+r+1]    = a00[i+1].value)
            m0 + 16 + 2i + h < M && (C[m0+16+2i+h+1, n0+r+1]    = a10[i+1].value)
        end
        if ok1
            m0 + 2i + h < M && (C[m0+2i+h+1,    n0+16+r+1] = a01[i+1].value)
            m0 + 16 + 2i + h < M && (C[m0+16+2i+h+1, n0+16+r+1] = a11[i+1].value)
        end
    end
    return
end

@inline _wmma_zeros() = (_wmma_zero(), _wmma_zero(), _wmma_zero(), _wmma_zero())

# Float16 weights. Av, Bv: A (K, M) and B (K, Npad) viewed as 16-byte chunks (8 halves
# along K). Each thread stages 2 chunks of A and 2 of B per K tile.
function _wmma_gemm_kernel!(C, Av, Bv, K, N)
    @inbounds begin
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
    @inbounds begin
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
    @inbounds begin
    tid = workitemIdx().x - 1
    l = tid % 32; r = l % 16; h = l ÷ 16; w = tid ÷ 32
    wm = (w % 2) * 32; wn = (w ÷ 2) * 32
    bm = (workgroupIdx().y - 1) * 64; bn = (workgroupIdx().x - 1) * 64
    As = @ROCStaticLocalArray(WMMA_V8H, (WMMA_PADC, 64), false)
    Bs = @ROCStaticLocalArray(WMMA_V8H, (WMMA_PADC, 64), false)
    acc = _wmma_zeros()
    arow = tid ÷ 2; ah = tid % 2
    grow = min(bm + arow, size(C, 1) - 1)   # a ragged last block re-reads the last row
    c0 = tid % 4; row0 = tid ÷ 4; row1 = row0 + 32
    nk = K ÷ 32
    qa = Pv[1, grow + 1]; sa = Float32(Sc[1, grow + 1])
    ma = SYM ? -8f0 * sa : Float32(Mn[1, grow + 1])
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
            qa = Pv[t + 2, grow + 1]; sa = Float32(Sc[gi, grow + 1])
            ma = SYM ? -8f0 * sa : Float32(Mn[gi, grow + 1])
            kc = (t + 1) * 4 + c0 + 1
            rb0 = Bv[kc, bn + row0 + 1]; rb1 = Bv[kc, bn + row1 + 1]
        end
        acc = _wmma_tile(acc, As, Bs, wm, wn, r)
        sync_workgroup()
    end
    _wmma_store!(C, acc, bm + wm, bn + wn, r, h, N)
    return
    end
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

# Cv: C viewed as 4-float vectors along its rows, Rv: an optional residual the same way (HR = whether it is used).
# A lane of a 16x16 result tile holds D[2e + h, r] (e = 0..7): 8 rows of column r, every other row. One shuffle with
# the partner lane (r, 1 - h) regroups them so that lane h = 0 holds rows 0..7 and lane h = 1 rows 8..15, and each
# lane then writes its 8 consecutive rows as two 16-byte stores (a column of the tile is a 64-byte run) instead of
# 8 four-byte stores to 8 different cache lines: the store cost 7-15% of the GEMM time.
@inline function _wmma_store_big!(Cv, acc, m0, n0, r, h, N, Rv, ::Val{HR}) where {HR}
    for j in 0:3
        col = n0 + 16j + r
        for i in 0:1
            a = acc[4i + j + 1]
            v1, v2, v3, v4, v5, v6, v7, v8 = a[1].value, a[2].value, a[3].value, a[4].value, a[5].value, a[6].value, a[7].value, a[8].value
            lo = h == 0
            q1 = AMDGPU.Device.shfl_xor(ifelse(lo, v5, v1), Cint(16))
            q2 = AMDGPU.Device.shfl_xor(ifelse(lo, v6, v2), Cint(16))
            q3 = AMDGPU.Device.shfl_xor(ifelse(lo, v7, v3), Cint(16))
            q4 = AMDGPU.Device.shfl_xor(ifelse(lo, v8, v4), Cint(16))
            # h = 0: rows 0..7 = v1 q1 v2 q2 v3 q3 v4 q4;  h = 1: rows 8..15 = q1 v5 q2 v6 q3 v7 q4 v8
            f1 = ifelse(lo, v1, q1); f2 = ifelse(lo, q1, v5); f3 = ifelse(lo, v2, q2); f4 = ifelse(lo, q2, v6)
            f5 = ifelse(lo, v3, q3); f6 = ifelse(lo, q3, v7); f7 = ifelse(lo, v4, q4); f8 = ifelse(lo, q4, v8)
            vr = (m0 + 16i + 8h) ÷ 4 + 1        # first of the two 4-row vectors of this lane
            if col < N                          # (the shuffles above run for every lane: only the memory accesses are guarded)
                if HR
                    r1 = Rv[vr, col + 1]; r2 = Rv[vr + 1, col + 1]
                    f1 += r1[1].value; f2 += r1[2].value; f3 += r1[3].value; f4 += r1[4].value
                    f5 += r2[1].value; f6 += r2[2].value; f7 += r2[3].value; f8 += r2[4].value
                end
                Cv[vr, col + 1] = (VecElement(f1), VecElement(f2), VecElement(f3), VecElement(f4))
                Cv[vr + 1, col + 1] = (VecElement(f5), VecElement(f6), VecElement(f7), VecElement(f8))
            end
        end
    end
    return
end

# One K tile's operands for this thread: its 16 bytes of weight codes with the scale and minimum
# of their group, and its two 16-byte chunks of the activations.
@inline function _wmma_q4_load(Pv, Sc, Mn, Bv, grow, c0, bcol0, bcol1, group, ::Val{SYM}, t) where {SYM}
    gi = (32 * t) >> trailing_zeros(group) + 1
    sa = Float32(Sc[gi, grow + 1])
    return (Pv[t + 1, grow + 1], sa, SYM ? -8f0 * sa : Float32(Mn[gi, grow + 1]),
            Bv[4t + c0 + 1, bcol0 + 1], Bv[4t + c0 + 1, bcol1 + 1])
end

# PD = how many K tiles of global loads are in flight (a register queue): one tile of WMMAs
# (~0.4 us per block) is shorter than a DRAM load, so a single tile of lookahead stalls.
function _wmma_q4_big_kernel!(C, Pv, Sc, Mn, Bv, K, N, group, ::Val{SYM}, ::Val{PD}, R, hr::Val) where {SYM, PD}
    @inbounds begin
    tid = workitemIdx().x - 1
    l = tid % 32; r = l % 16; h = l ÷ 16; w = tid ÷ 32
    wm = (w % 4) * 32; wn = (w ÷ 4) * 64
    bm = (workgroupIdx().y - 1) * 128; bn = (workgroupIdx().x - 1) * 128
    As = @ROCStaticLocalArray(WMMA_V8H, (WMMA_PADC, 128), false)
    Bs = @ROCStaticLocalArray(WMMA_V8H, (WMMA_PADC, 128), false)
    acc = ntuple(_ -> _wmma_zero(), Val(8))
    arow = tid ÷ 2; ah = tid % 2                       # A: 128 rows x 2 half-chunks per K tile
    grow = min(bm + arow, 4 * size(C, 1) - 1)          # (C is the (M/4, N) vector view) a ragged last block re-reads the last row
    c0 = tid % 4; row0 = tid ÷ 4; row1 = row0 + 64     # B: 128 columns x 4 chunks
    nk = K ÷ 32
    pipe = ntuple(d -> _wmma_q4_load(Pv, Sc, Mn, Bv, grow, c0, bn + row0, bn + row1, group, Val(SYM), min(d - 1, nk - 1)), Val(PD))
    for t in 0:nk-1
        qa, sa, ma, rb0, rb1 = pipe[1]
        bytes = let q = qa
            ah == 0 ? ntuple(i -> q[i], Val(8)) : ntuple(i -> q[i + 8], Val(8))
        end
        lo, hi = _wmma_dq4(bytes, sa, ma)
        As[2ah + 1, arow + 1] = lo; As[2ah + 2, arow + 1] = hi
        Bs[c0 + 1, row0 + 1] = rb0; Bs[c0 + 1, row1 + 1] = rb1
        sync_workgroup()
        nxt = _wmma_q4_load(Pv, Sc, Mn, Bv, grow, c0, bn + row0, bn + row1, group, Val(SYM), min(t + PD, nk - 1))
        pipe = (Base.tail(pipe)..., nxt)
        acc = _wmma_tile_big(acc, As, Bs, wm, wn, r)
        sync_workgroup()
    end
    _wmma_store_big!(C, acc, bm + wm, bn + wn, r, h, N, R, hr)
    return
    end
end

# 128 x 256 block, 16 waves (each 32 x 64): the dequantized weight tile feeds 256 columns, so
# the (VALU) dequantization costs half as much per MMA as in the 128x128 kernel, and all 512
# threads share it (4 bytes = 8 codes each).
@inline function _wmma_dq4w(w::UInt32, s::Float32, m::Float32)
    s2 = (VecElement(Float16(s)), VecElement(Float16(s)))
    m2 = (VecElement(Float16(m)), VecElement(Float16(m)))
    p = ntuple(i -> _wmma_pair(UInt8((w >> UInt32(8 * (i - 1))) & 0xff), s2, m2), Val(4))
    return ntuple(i -> p[(i + 1) >> 1][2 - i % 2], Val(8))
end

@inline function _wmma_q4w_load(Pw, Sc, Mn, Bv, grow, qrt, c0, bcol0, bcol1, group, ::Val{SYM}, t) where {SYM}
    gi = (32 * t) >> trailing_zeros(group) + 1
    sa = Float32(Sc[gi, grow + 1])
    return (Pw[4t + qrt + 1, grow + 1], sa, SYM ? -8f0 * sa : Float32(Mn[gi, grow + 1]),
            Bv[4t + c0 + 1, bcol0 + 1], Bv[4t + c0 + 1, bcol1 + 1])
end

function _wmma_q4_wide_kernel!(C, Pw, Sc, Mn, Bv, K, N, group, ::Val{SYM}, ::Val{PD}, R, hr::Val) where {SYM, PD}
    @inbounds begin
    tid = workitemIdx().x - 1
    l = tid % 32; r = l % 16; h = l ÷ 16; w = tid ÷ 32
    wm = (w % 4) * 32; wn = (w ÷ 4) * 64
    bm = (workgroupIdx().y - 1) * 128; bn = (workgroupIdx().x - 1) * 256
    As = @ROCStaticLocalArray(WMMA_V8H, (WMMA_PADC, 128, 2), false)   # two stages: one barrier per K tile
    Bs = @ROCStaticLocalArray(WMMA_V8H, (WMMA_PADC, 256, 2), false)
    acc = ntuple(_ -> _wmma_zero(), Val(8))
    arow = tid ÷ 4; qrt = tid % 4                      # A: 128 rows x 4 words (8 codes) per K tile
    grow = min(bm + arow, 4 * size(C, 1) - 1)
    c0 = tid % 4; row0 = tid ÷ 4; row1 = row0 + 128    # B: 256 columns x 4 chunks, 2 per thread
    nk = K ÷ 32
    pipe = ntuple(d -> _wmma_q4w_load(Pw, Sc, Mn, Bv, grow, qrt, c0, bn + row0, bn + row1, group, Val(SYM), min(d - 1, nk - 1)), Val(PD))
    for t in 0:nk-1
        qa, sa, ma, rb0, rb1 = pipe[1]
        st = t & 1 + 1
        As[qrt + 1, arow + 1, st] = _wmma_dq4w(qa, sa, ma)
        Bs[c0 + 1, row0 + 1, st] = rb0; Bs[c0 + 1, row1 + 1, st] = rb1
        sync_workgroup()
        nxt = _wmma_q4w_load(Pw, Sc, Mn, Bv, grow, qrt, c0, bn + row0, bn + row1, group, Val(SYM), min(t + PD, nk - 1))
        pipe = (Base.tail(pipe)..., nxt)
        acc = _wmma_tile_big(acc, view(As, :, :, st), view(Bs, :, :, st), wm, wn, r)
    end
    _wmma_store_big!(C, acc, bm + wm, bn + wn, r, h, N, R, hr)
    return
    end
end

# Warp-specialized variant of the 128x256 int4 kernel: 512 + 32 NP threads. 16 consumer waves (a 32 x 64 tile each) only run the
# MMAs of tile t from one LDS stage, while NP producer waves load, dequantize and stage tile t + 1 into the other stage (PD tiles
# of global loads in flight in registers); one barrier per K tile. In the plain kernel every wave staged its share and then
# computed, all in lockstep around the barrier, so the matrix unit idled while the VALU / LDS work ran; here the two overlap
# (measured: 30 -> 32-34 TFLOP/s, interleaved best-of-40 on the 27B's shapes). The NP = 16 producers (1024 threads in all) hold
# 1/16 of a tile's rows and chunks each; 4 producer waves were slower (one wave per SIMD cannot keep the stage fed).
@inline function _ws_load(Pw, Sc, Mn, Bv, p, bm, bn, t, group, ::Val{SYM}, M4, ::Val{NP}) where {SYM, NP}
    gi = (32 * t) >> trailing_zeros(group) + 1
    qrt = p % 4; r0 = p ÷ 4
    a = ntuple(Val(16 ÷ NP)) do k
        grow = min(bm + r0 + 8NP * (k - 1), M4 - 1)
        sa = Float32(Sc[gi, grow + 1])
        (Pw[4t + qrt + 1, grow + 1], sa, SYM ? -8f0 * sa : Float32(Mn[gi, grow + 1]))
    end
    b = ntuple(k -> Bv[4t + qrt + 1, bn + r0 + 8NP * (k - 1) + 1], Val(32 ÷ NP))
    return (a, b)
end
@inline function _ws_stage(As, Bs, p, ab, st, ::Val{NP}) where {NP}
    a, b = ab
    qrt = p % 4; r0 = p ÷ 4
    ntuple(Val(16 ÷ NP)) do k
        w, sa, ma = a[k]
        As[qrt + 1, r0 + 8NP * (k - 1) + 1, st] = _wmma_dq4w(w, sa, ma)
        nothing
    end
    ntuple(Val(32 ÷ NP)) do k
        Bs[qrt + 1, r0 + 8NP * (k - 1) + 1, st] = b[k]
        nothing
    end
    return nothing
end
function _wmma_q4_ws_kernel!(C, Pw, Sc, Mn, Bv, K, N, group, ::Val{SYM}, ::Val{PD}, R, hr::Val, ::Val{NP}) where {SYM, PD, NP}
    @inbounds begin
    tid = workitemIdx().x - 1
    wv = tid ÷ 32
    l = tid % 32; r = l % 16; h = l ÷ 16
    bm = (workgroupIdx().y - 1) * 128; bn = (workgroupIdx().x - 1) * 256
    As = @ROCStaticLocalArray(WMMA_V8H, (WMMA_PADC, 128, 2), false)
    Bs = @ROCStaticLocalArray(WMMA_V8H, (WMMA_PADC, 256, 2), false)
    nk = K ÷ 32
    if wv < 16
        wm = (wv % 4) * 32; wn = (wv ÷ 4) * 64
        acc = ntuple(_ -> _wmma_zero(), Val(8))
        sync_workgroup()                                       # tile 0 is staged
        for t in 0:nk-1
            st = t & 1 + 1
            acc = _wmma_tile_big(acc, view(As, :, :, st), view(Bs, :, :, st), wm, wn, r)
            sync_workgroup()                                   # tile t + 1 is staged, tile t's stage is free
        end
        _wmma_store_big!(C, acc, bm + wm, bn + wn, r, h, N, R, hr)
    else
        p = tid - 512
        M4 = 4 * size(C, 1)
        pipe = ntuple(d -> _ws_load(Pw, Sc, Mn, Bv, p, bm, bn, min(d - 1, nk - 1), group, Val(SYM), M4, Val(NP)), Val(PD))
        _ws_stage(As, Bs, p, pipe[1], 1, Val(NP))
        pipe = (Base.tail(pipe)..., _ws_load(Pw, Sc, Mn, Bv, p, bm, bn, min(PD, nk - 1), group, Val(SYM), M4, Val(NP)))
        sync_workgroup()
        for t in 0:nk-1
            if t + 1 < nk
                _ws_stage(As, Bs, p, pipe[1], (t + 1) & 1 + 1, Val(NP))
                pipe = (Base.tail(pipe)..., _ws_load(Pw, Sc, Mn, Bv, p, bm, bn, min(t + 1 + PD, nk - 1), group, Val(SYM), M4, Val(NP)))
            end
            sync_workgroup()
        end
    end
    return
    end
end

# --- gate/up GEMM with the SwiGLU in its epilogue ---------------------------------------------------------
# H (M, N) Float16 = silu(Wg B) * (Wu B) for two 4-bit weights Wg, Wu (M, K) and one activation matrix B (K, N). The structure is
# that of `_wmma_q4_ws_kernel!` (16 consumer waves running the MMAs of tile t while 16 producer waves stage tile t + 1), with a block of
# 64 rows of H x 256 columns: each consumer wave computes a 32 x 32 piece of BOTH the gate and the up result (4 + 4 accumulator
# tiles, the same register budget and MMA count per block as the 128 x 256 kernel), so in the epilogue every lane holds the gate and
# the up value of the same elements, applies silu(g) * u and stores Float16: the two Float32 results (71 MB per layer at N = 512) and
# the elementwise kernel that read them back (460 us) are gone.
@inline function _wmma_tile_swiglu(acc::NTuple{8,WMMA_V8F}, Ag, Au, Bs, wm, wn, r)
    c0, c1, c2, c3, c4, c5, c6, c7 = acc    # gate (i, j) = c[2i + j + 1], up (i, j) = c[4 + 2i + j + 1]
    for s in 0:1
        # the two B fragments stay live; the four A fragments are read one at a time (fewer live registers than reading them all first)
        B0 = _wmma_cat(Bs[2s + 1, wn + r + 1], Bs[2s + 2, wn + r + 1]); B1 = _wmma_cat(Bs[2s + 1, wn + 16 + r + 1], Bs[2s + 2, wn + 16 + r + 1])
        A = _wmma_cat(Ag[2s + 1, wm + r + 1], Ag[2s + 2, wm + r + 1]);           c0 = _wmma16(A, B0, c0); c1 = _wmma16(A, B1, c1)
        A = _wmma_cat(Ag[2s + 1, wm + 16 + r + 1], Ag[2s + 2, wm + 16 + r + 1]); c2 = _wmma16(A, B0, c2); c3 = _wmma16(A, B1, c3)
        A = _wmma_cat(Au[2s + 1, wm + r + 1], Au[2s + 2, wm + r + 1]);           c4 = _wmma16(A, B0, c4); c5 = _wmma16(A, B1, c5)
        A = _wmma_cat(Au[2s + 1, wm + 16 + r + 1], Au[2s + 2, wm + 16 + r + 1]); c6 = _wmma16(A, B0, c6); c7 = _wmma16(A, B1, c7)
    end
    return (c0, c1, c2, c3, c4, c5, c6, c7)
end

@inline _swiglu_f(g, u) = (g / (1f0 + exp(-g))) * u

# store H (viewed as vectors of 8 Float16 rows): the same lane regrouping as `_wmma_store_big!`, applied to h = silu(g) * u
@inline function _wmma_store_swiglu!(Hv, acc, m0, n0, r, h, N)
    for j in 0:1
        col = n0 + 16j + r
        for i in 0:1
            g = acc[2i + j + 1]; u = acc[4 + 2i + j + 1]
            v1, v2, v3, v4, v5, v6, v7, v8 = ntuple(e -> _swiglu_f(g[e].value, u[e].value), Val(8))
            lo = h == 0
            q1 = AMDGPU.Device.shfl_xor(ifelse(lo, v5, v1), Cint(16))
            q2 = AMDGPU.Device.shfl_xor(ifelse(lo, v6, v2), Cint(16))
            q3 = AMDGPU.Device.shfl_xor(ifelse(lo, v7, v3), Cint(16))
            q4 = AMDGPU.Device.shfl_xor(ifelse(lo, v8, v4), Cint(16))
            f1 = ifelse(lo, v1, q1); f2 = ifelse(lo, q1, v5); f3 = ifelse(lo, v2, q2); f4 = ifelse(lo, q2, v6)
            f5 = ifelse(lo, v3, q3); f6 = ifelse(lo, q3, v7); f7 = ifelse(lo, v4, q4); f8 = ifelse(lo, q4, v8)
            if col < N
                Hv[(m0 + 16i + 8h) ÷ 8 + 1, col + 1] = (VecElement(Float16(f1)), VecElement(Float16(f2)), VecElement(Float16(f3)), VecElement(Float16(f4)),
                                                         VecElement(Float16(f5)), VecElement(Float16(f6)), VecElement(Float16(f7)), VecElement(Float16(f8)))
            end
        end
    end
    return
end

# One producer thread (512 of them): one weight word (8 codes) of the gate or the up tile and two 16-byte chunks of the activations
@inline function _sw_load(Pwg, Scg, Mng, Pwu, Scu, Mnu, Bv, p, bm, bn, t, group, ::Val{SYM}, M) where {SYM}
    gi = (32 * t) >> trailing_zeros(group) + 1
    which = p ÷ 256; rem = p % 256
    row = min(bm + rem ÷ 4, M - 1) + 1; qrt = rem % 4
    word = which == 0 ? Pwg[4t + qrt + 1, row] : Pwu[4t + qrt + 1, row]
    sa = Float32(which == 0 ? Scg[gi, row] : Scu[gi, row])
    ma = SYM ? -8f0 * sa : Float32(which == 0 ? Mng[gi, row] : Mnu[gi, row])
    c0 = p % 4; r0 = p ÷ 4
    return ((word, sa, ma), Bv[4t + c0 + 1, bn + r0 + 1], Bv[4t + c0 + 1, bn + r0 + 128 + 1])
end
@inline function _sw_stage(Ag, Au, Bs, p, ld, st)
    (word, sa, ma), b0, b1 = ld
    which = p ÷ 256; rem = p % 256
    row = rem ÷ 4 + 1; qrt = rem % 4
    frag = _wmma_dq4w(word, sa, ma)
    which == 0 ? (Ag[qrt + 1, row, st] = frag) : (Au[qrt + 1, row, st] = frag)
    c0 = p % 4; r0 = p ÷ 4
    Bs[c0 + 1, r0 + 1, st] = b0; Bs[c0 + 1, r0 + 128 + 1, st] = b1
    return nothing
end

function _wmma_q4_swiglu_kernel!(Hv, Pwg, Scg, Mng, Pwu, Scu, Mnu, Bv, K, N, group, ::Val{SYM}, ::Val{PD}) where {SYM, PD}
    @inbounds begin
    tid = workitemIdx().x - 1
    wv = tid ÷ 32
    l = tid % 32; r = l % 16; h = l ÷ 16
    bm = (workgroupIdx().y - 1) * 64; bn = (workgroupIdx().x - 1) * 256
    Ag = @ROCStaticLocalArray(WMMA_V8H, (WMMA_PADC, 64, 2), false)
    Au = @ROCStaticLocalArray(WMMA_V8H, (WMMA_PADC, 64, 2), false)
    Bs = @ROCStaticLocalArray(WMMA_V8H, (WMMA_PADC, 256, 2), false)
    nk = K ÷ 32
    if wv < 16
        wm = (wv % 2) * 32; wn = (wv ÷ 2) * 32
        acc = ntuple(_ -> _wmma_zero(), Val(8))
        sync_workgroup()                                       # tile 0 is staged
        for t in 0:nk-1
            st = t & 1 + 1
            acc = _wmma_tile_swiglu(acc, view(Ag, :, :, st), view(Au, :, :, st), view(Bs, :, :, st), wm, wn, r)
            sync_workgroup()                                   # tile t + 1 is staged, tile t's stage is free
        end
        _wmma_store_swiglu!(Hv, acc, bm + wm, bn + wn, r, h, N)
    else
        p = tid - 512
        M = 8 * size(Hv, 1)
        pipe = ntuple(d -> _sw_load(Pwg, Scg, Mng, Pwu, Scu, Mnu, Bv, p, bm, bn, min(d - 1, nk - 1), group, Val(SYM), M), Val(PD))
        _sw_stage(Ag, Au, Bs, p, pipe[1], 1)
        pipe = (Base.tail(pipe)..., _sw_load(Pwg, Scg, Mng, Pwu, Scu, Mnu, Bv, p, bm, bn, min(PD, nk - 1), group, Val(SYM), M))
        sync_workgroup()
        for t in 0:nk-1
            if t + 1 < nk
                _sw_stage(Ag, Au, Bs, p, pipe[1], (t + 1) & 1 + 1)
                pipe = (Base.tail(pipe)..., _sw_load(Pwg, Scg, Mng, Pwu, Scu, Mnu, Bv, p, bm, bn, min(t + 1 + PD, nk - 1), group, Val(SYM), M))
            end
            sync_workgroup()
        end
    end
    return
    end
end

# H (M, N) Float16 = silu(Wg B) * (Wu B); weights in the layout of `Q4Weight` (Pg / Pu: (K/32, M) of 16 bytes, scale / mn (K/group, M)),
# B (K, Npad) Float16. M % 64 == 0, padded N % 256 == 0, both weights symmetric or both with minimums.
function wmma_q4_swiglu!(H::ROCArray{Float16,2}, Pg, scale_g, mn_g, Pu, scale_u, mn_u, group::Int, B::ROCArray{Float16,2}, N::Int)
    K32, M = size(Pg); K = 32 * K32; Npad = size(B, 2)
    (M % 64 == 0 && Npad % 256 == 0 && group % 32 == 0 && size(B, 1) == K && size(Pu) == size(Pg) && N <= Npad && size(H) == (M, N) &&
     (mn_g === nothing) == (mn_u === nothing)) ||
        throw(ArgumentError("wmma_q4_swiglu!: need M % 64 == 0, padded N % 256 == 0 and two weights of the same size and kind"))
    Bv = reinterpret(WMMA_V8H, B); Hv = reinterpret(WMMA_V8H, H)
    Pwg = reinterpret(UInt32, Pg); Pwu = reinterpret(UInt32, Pu)
    if M % 128 == 0 && K % 64 == 0 && ispow2(group) && get(ENV, "LUMINAL_WMMA_G", "1") != "0"
        _launch_q4(_wmma_q4_gpair_kernel!, (Hv, Pwg, scale_g, mn_g === nothing ? scale_g : mn_g, Pwu, scale_u, mn_u === nothing ? scale_u : mn_u, Bv, K, N, group,
                                            Val(mn_g === nothing)), 1024, (Npad ÷ 256, M ÷ 128), 0)
        return H
    end
    _launch_q4(_wmma_q4_swiglu_kernel!, (Hv, Pwg, scale_g, mn_g === nothing ? scale_g : mn_g, Pwu, scale_u, mn_u === nothing ? scale_u : mn_u, Bv, K, N, group,
                                         Val(mn_g === nothing), Val(2)), 1024, (Npad ÷ 256, M ÷ 64), 0)
    return H
end

# --- the 256 x 256 int4 GEMM (the structure of gufo's HalfPrefillGemmKernel) -------------------------------------
# 32 waves (8 x 4) per block, each computing a 32 x 64 tile (8 accumulator tiles), K64 per iteration: every thread dequantizes 16 weights
# (a K16 step of one row) and copies 16 activations into ONE 64 KB LDS stage between two barriers, and the next tile's global loads are
# issued right after the first barrier, so they are in flight during the 32 MMAs of the wave. What made it as fast as gufo's own kernel
# (33-34 TFLOP/s in a standalone measurement): (1) the four K16 steps unrolled into straight-line code; (2) a fragment row's two 16-byte
# halves in separate planes, so 16 lanes read 256 contiguous bytes (a 32-byte row stride made every fragment read 2-way bank conflicted);
# (3) the row position of a K16 step rotated by 2 x the step, so 8 lanes of a store hit 8 different bank groups; (4) no integer division
# by the runtime group size in the loop (a shift). Needs M % 256 == 0, K % 64 == 0, padded N % 256 == 0 and a power-of-two group.
@inline function _g_load(Pw, Sc, Mn, Bv, grow, tok, kk, q, group, ::Val{SYM}) where {SYM}
    gi = (64 * q + 16 * kk) >> trailing_zeros(group) + 1
    sa = Float32(Sc[gi, grow + 1])
    ma = SYM ? -8f0 * sa : Float32(Mn[gi, grow + 1])
    return (Pw[8q + 2kk + 1, grow + 1], Pw[8q + 2kk + 2, grow + 1], sa, ma, Bv[8q + 2kk + 1, tok + 1], Bv[8q + 2kk + 2, tok + 1])
end
@inline function _g_ks(acc::NTuple{8,WMMA_V8F}, As, Bs, wr, wt, sl, ks)
    c0, c1, c2, c3, c4, c5, c6, c7 = acc      # (A tile i, B tile j) is acc[4i + j + 1]
    p = ((sl + 2 * ks) & 15) + 1
    a0 = _wmma_cat(As[p, 2wr + 1, ks + 1, 1], As[p, 2wr + 1, ks + 1, 2])
    a1 = _wmma_cat(As[p, 2wr + 2, ks + 1, 1], As[p, 2wr + 2, ks + 1, 2])
    b = _wmma_cat(Bs[p, 4wt + 1, ks + 1, 1], Bs[p, 4wt + 1, ks + 1, 2]); c0 = _wmma16(a0, b, c0); c4 = _wmma16(a1, b, c4)
    b = _wmma_cat(Bs[p, 4wt + 2, ks + 1, 1], Bs[p, 4wt + 2, ks + 1, 2]); c1 = _wmma16(a0, b, c1); c5 = _wmma16(a1, b, c5)
    b = _wmma_cat(Bs[p, 4wt + 3, ks + 1, 1], Bs[p, 4wt + 3, ks + 1, 2]); c2 = _wmma16(a0, b, c2); c6 = _wmma16(a1, b, c6)
    b = _wmma_cat(Bs[p, 4wt + 4, ks + 1, 1], Bs[p, 4wt + 4, ks + 1, 2]); c3 = _wmma16(a0, b, c3); c7 = _wmma16(a1, b, c7)
    return (c0, c1, c2, c3, c4, c5, c6, c7)
end
@inline function _g_decode(ld)
    w1, w2, sa, ma, b1, b2 = ld
    return (_wmma_dq4w(w1, sa, ma), _wmma_dq4w(w2, sa, ma), b1, b2)
end
function _wmma_q4_g_kernel!(C, Pw, Sc, Mn, Bv, K, N, group, ::Val{SYM}, R, hr::Val) where {SYM}
    @inbounds begin
    tid = workitemIdx().x - 1
    wave = tid ÷ 32; lane = tid % 32; sl = lane % 16; h = lane ÷ 16
    wr = wave ÷ 4; wt = wave % 4
    bm = (workgroupIdx().y - 1) * 256; bn = (workgroupIdx().x - 1) * 256
    As = @ROCStaticLocalArray(WMMA_V8H, (16, 16, 4, 2), false)     # (row in tile, row tile, k16 step, k half): 16 lanes read 256 contiguous bytes of one half
    Bs = @ROCStaticLocalArray(WMMA_V8H, (16, 16, 4, 2), false)
    acc = ntuple(_ -> _wmma_zero(), Val(8))
    row = tid ÷ 4; kk = tid % 4                                   # this thread's weight row / token and k16 step of every K64 tile
    grow = min(bm + row, 4 * size(C, 1) - 1)
    tok = bn + row
    tile = row ÷ 16; inr = row % 16
    pos = ((inr + 2 * kk) & 15) + 1                                # rotation of the row within the fragment by 2 x the k16 step
    nk = K ÷ 64
    ld = _g_load(Pw, Sc, Mn, Bv, grow, tok, kk, 0, group, Val(SYM))
    dec = _g_decode(ld)
    for q in 0:nk-1
        # commit: the already dequantized weights and the activations go to LDS
        a1, a2, b1, b2 = dec
        As[pos, tile + 1, kk + 1, 1] = a1
        As[pos, tile + 1, kk + 1, 2] = a2
        Bs[pos, tile + 1, kk + 1, 1] = b1
        Bs[pos, tile + 1, kk + 1, 2] = b2
        sync_workgroup()
        q + 1 < nk && (ld = _g_load(Pw, Sc, Mn, Bv, grow, tok, kk, q + 1, group, Val(SYM)))     # next tile's global loads, in flight during the MMAs
        acc = _g_ks(acc, As, Bs, wr, wt, sl, 0)
        acc = _g_ks(acc, As, Bs, wr, wt, sl, 1)
        acc = _g_ks(acc, As, Bs, wr, wt, sl, 2)
        dec = _g_decode(ld)                                       # (VALU work on the arrived data, in the same block as the last MMAs)
        acc = _g_ks(acc, As, Bs, wr, wt, sl, 3)
        sync_workgroup()
    end
    _wmma_store_big!(C, acc, bm + wr * 32, bn + wt * 64, sl, h, N, R, hr)
    return
    end
end

# --- the paired gate/up GEMM with the SwiGLU in the epilogue, in the structure of the 256 x 256 kernel ---------------
# H (M, N) Float16 = silu(Wg B) * (Wu B): a 256-row block holds 128 gate rows and the 128 up rows of the same output rows, interleaved in
# 16-row tiles (tile 2p = gate rows 16p..16p+15 of the block, tile 2p + 1 = the up rows), so a wave's two A tiles (2 wr, 2 wr + 1) are
# the gate and the up fragments of the same 16 output rows against the same activation fragments: in the epilogue every lane holds g
# and u of the same elements. Same requirements as the plain kernel on the paired weights (M % 128 == 0 output rows).
@inline function _gp_load(Pwg, Scg, Mng, Pwu, Scu, Mnu, Bv, grow, up, tok, kk, q, group, ::Val{SYM}) where {SYM}
    gi = (64 * q + 16 * kk) >> trailing_zeros(group) + 1
    sa = Float32(up ? Scu[gi, grow + 1] : Scg[gi, grow + 1])
    ma = SYM ? -8f0 * sa : Float32(up ? Mnu[gi, grow + 1] : Mng[gi, grow + 1])
    w1 = up ? Pwu[8q + 2kk + 1, grow + 1] : Pwg[8q + 2kk + 1, grow + 1]
    w2 = up ? Pwu[8q + 2kk + 2, grow + 1] : Pwg[8q + 2kk + 2, grow + 1]
    return (w1, w2, sa, ma, Bv[8q + 2kk + 1, tok + 1], Bv[8q + 2kk + 2, tok + 1])
end

@inline function _wmma_store_gpair!(Hv, acc, m0, n0, r, h, N)
    for j in 0:3
        col = n0 + 16j + r
        g = acc[j + 1]; u = acc[4 + j + 1]                       # (A tile 0 = gate, A tile 1 = up) x token tile j
        v1, v2, v3, v4, v5, v6, v7, v8 = ntuple(e -> _swiglu_f(g[e].value, u[e].value), Val(8))
        lo = h == 0
        q1 = AMDGPU.Device.shfl_xor(ifelse(lo, v5, v1), Cint(16))
        q2 = AMDGPU.Device.shfl_xor(ifelse(lo, v6, v2), Cint(16))
        q3 = AMDGPU.Device.shfl_xor(ifelse(lo, v7, v3), Cint(16))
        q4 = AMDGPU.Device.shfl_xor(ifelse(lo, v8, v4), Cint(16))
        f1 = ifelse(lo, v1, q1); f2 = ifelse(lo, q1, v5); f3 = ifelse(lo, v2, q2); f4 = ifelse(lo, q2, v6)
        f5 = ifelse(lo, v3, q3); f6 = ifelse(lo, q3, v7); f7 = ifelse(lo, v4, q4); f8 = ifelse(lo, q4, v8)
        if col < N
            Hv[(m0 + 8h) ÷ 8 + 1, col + 1] = (VecElement(Float16(f1)), VecElement(Float16(f2)), VecElement(Float16(f3)), VecElement(Float16(f4)),
                                               VecElement(Float16(f5)), VecElement(Float16(f6)), VecElement(Float16(f7)), VecElement(Float16(f8)))
        end
    end
    return
end

function _wmma_q4_gpair_kernel!(Hv, Pwg, Scg, Mng, Pwu, Scu, Mnu, Bv, K, N, group, ::Val{SYM}) where {SYM}
    @inbounds begin
    tid = workitemIdx().x - 1
    wave = tid ÷ 32; lane = tid % 32; sl = lane % 16; h = lane ÷ 16
    wr = wave ÷ 4; wt = wave % 4
    bm = (workgroupIdx().y - 1) * 128; bn = (workgroupIdx().x - 1) * 256          # bm: first output row of the block
    As = @ROCStaticLocalArray(WMMA_V8H, (16, 16, 4, 2), false)
    Bs = @ROCStaticLocalArray(WMMA_V8H, (16, 16, 4, 2), false)
    acc = ntuple(_ -> _wmma_zero(), Val(8))
    row = tid ÷ 4; kk = tid % 4
    tile = row ÷ 16; inr = row % 16
    up = (tile & 1) == 1                                          # odd tiles hold the up rows (a whole wave: 64 threads per tile)
    grow = min(bm + (tile >> 1) * 16 + inr, 8 * size(Hv, 1) - 1)  # the output row this weight row belongs to
    tok = bn + row
    pos = ((inr + 2 * kk) & 15) + 1
    nk = K ÷ 64
    ld = _gp_load(Pwg, Scg, Mng, Pwu, Scu, Mnu, Bv, grow, up, tok, kk, 0, group, Val(SYM))
    dec = _g_decode(ld)
    for q in 0:nk-1
        a1, a2, b1, b2 = dec
        As[pos, tile + 1, kk + 1, 1] = a1
        As[pos, tile + 1, kk + 1, 2] = a2
        Bs[pos, tile + 1, kk + 1, 1] = b1
        Bs[pos, tile + 1, kk + 1, 2] = b2
        sync_workgroup()
        q + 1 < nk && (ld = _gp_load(Pwg, Scg, Mng, Pwu, Scu, Mnu, Bv, grow, up, tok, kk, q + 1, group, Val(SYM)))
        acc = _g_ks(acc, As, Bs, wr, wt, sl, 0)
        acc = _g_ks(acc, As, Bs, wr, wt, sl, 1)
        acc = _g_ks(acc, As, Bs, wr, wt, sl, 2)
        dec = _g_decode(ld)
        acc = _g_ks(acc, As, Bs, wr, wt, sl, 3)
        sync_workgroup()
    end
    _wmma_store_gpair!(Hv, acc, bm + wr * 16, bn + wt * 64, sl, h, N)
    return
    end
end

# Skinny GEMM for N <= 16 columns (prefill of a short prompt, a few decode streams, speculative
# verification): the weights are streamed once at memory speed. No LDS and no barriers: each
# wave owns 16 weight rows, dequantizes its 16-byte chunk per K tile straight into the two WMMA
# A fragments, and takes the B fragments (16 activation columns, K contiguous) from cache.
# `PD` K tiles of loads are kept in flight in registers.
# The weights are read with contiguous accesses: for 8 K tiles at a time (128 bytes = one cache
# line of each row) the wave's 32 lanes load 4 rows x 8 chunks per instruction, park them in a
# per-wave LDS tile, and read them back one tile at a time in the WMMA layout (lane = row). Loading
# 16 bytes per lane straight from 16 different rows 2.5 KB apart ran at ~70 GB/s; this streams.
# Requires group == 32 (one Float16 scale per K tile: 8 scales = one 16-byte load per row) and
# K % 256 == 0.
@inline function _wmma_q4s_raw(Pv, row0, M, l, g)
    ntuple(j -> Pv[8g + l % 8 + 1, min(row0 + 4 * (j - 1) + l ÷ 8, M - 1) + 1], Val(4))
end

function _wmma_q4_skinny_kernel!(C, Pv, Sc8, Mn8, Bv, K, N, ::Val{SYM}, ::Val{NW}, ::Val{NT}) where {SYM, NW, NT}
    # One workgroup = NW waves = the same 16 weight rows, each wave streaming every NW-th group of
    # 8 K tiles (more independent streams per row block); the partial sums meet in LDS at the end.
    @inbounds begin
    tid = workitemIdx().x - 1
    l = tid % 32; r = l % 16; h = l ÷ 16; w = tid ÷ 32
    M = size(C, 1)
    row0 = (workgroupIdx().x - 1) * 16
    grow = min(row0 + r, M - 1)
    Ls = @ROCStaticLocalArray(NTuple{16,UInt8}, (9, 16, NW), false)   # [chunk (+1 pad), row, wave]
    Rs = @ROCStaticLocalArray(Float32, (8, 32, NW), false)
    ng = K ÷ 256
    nu = min(8, 8 * ng)   # = 8; opaque to the compiler, so the tile loop is not unrolled (it would hoist all 8 tiles' B loads: spills)
    acc0 = ntuple(_ -> _wmma_zero(), Val(NT)); acc1 = ntuple(_ -> _wmma_zero(), Val(NT))   # per 16-column tile
    g0 = min(w, ng - 1)
    raw = _wmma_q4s_raw(Pv, row0, M, l, g0)
    sc = Sc8[g0 + 1, grow + 1]; mn = SYM ? sc : Mn8[g0 + 1, grow + 1]
    for g in w:NW:ng-1
        for j in 1:4
            Ls[l % 8 + 1, 4 * (j - 1) + l ÷ 8 + 1, w + 1] = raw[j]
        end
        gn = min(g + NW, ng - 1)
        scn = Sc8[gn + 1, grow + 1]; mnn = SYM ? scn : Mn8[gn + 1, grow + 1]
        nraw = _wmma_q4s_raw(Pv, row0, M, l, gn)
        # (the tile is private to this wave: LDS accesses of a wave execute in order, so no barrier)
        for u in 0:nu-1
            t = 8g + u
            qa = Ls[u + 1, r + 1, w + 1]
            sa = Float32(sc[u + 1]); ma = SYM ? -8f0 * sa : Float32(mn[u + 1])
            lo0, hi0 = _wmma_dq4(ntuple(i -> qa[i], Val(8)), sa, ma)
            lo1, hi1 = _wmma_dq4(ntuple(i -> qa[i + 8], Val(8)), sa, ma)
            a0 = _wmma_cat(lo0, hi0); a1 = _wmma_cat(lo1, hi1)
            acc0 = let a = acc0
                ntuple(j -> _wmma16(a0, _wmma_cat(Bv[4t + 1, 16 * (j - 1) + r + 1], Bv[4t + 2, 16 * (j - 1) + r + 1]), a[j]), Val(NT))
            end
            acc1 = let a = acc1
                ntuple(j -> _wmma16(a1, _wmma_cat(Bv[4t + 3, 16 * (j - 1) + r + 1], Bv[4t + 4, 16 * (j - 1) + r + 1]), a[j]), Val(NT))
            end
        end
        raw = nraw; sc = scn; mn = mnn
    end
    for j in 1:NT
        for i in 0:7
            Rs[i + 1, l + 1, w + 1] = acc0[j][i + 1].value + acc1[j][i + 1].value
        end
        sync_workgroup()
        if w == 0 && 16 * (j - 1) + r < N
            for i in 0:7   # lane l holds D[2i + l÷16, l%16]: weight row 2i + h, activation column 16(j-1) + r
                row = row0 + 2i + h
                v = 0f0
                for k in 1:NW
                    v += Rs[i + 1, l + 1, k]
                end
                row < M && (C[row + 1, 16 * (j - 1) + r + 1] = v)
            end
        end
        sync_workgroup()
    end
    return
    end
end

# C (M, N <= 32) = dequantize(Pv, scale, mn)' * B for B (K, >= 16 * cld(N, 16)) Float16 (columns >= N ignored).
function wmma_q4_skinny!(C::ROCArray{Float32,2}, Pv, scale::ROCArray{Float16,2}, mn,
                         group::Int, B::ROCArray{Float16,2}, N::Int)
    K32, M = size(Pv); K = 32 * K32
    (M % 16 == 0 && K % 256 == 0 && group == 32 && size(B, 1) == K && size(B, 2) >= 16 * cld(N, 16) && N <= 32 &&
     size(scale) == (K ÷ group, M) && size(C) == (M, N)) ||
        throw(ArgumentError("wmma_q4_skinny!: need M % 16 == 0, K % 256 == 0, group == 32, N <= 32"))
    Bv = reinterpret(WMMA_V8H, B)
    Sc8 = reinterpret(NTuple{8,Float16}, scale)
    Mn8 = mn === nothing ? Sc8 : reinterpret(NTuple{8,Float16}, mn)
    @roc groupsize=32 * WMMA_SKINNY_WAVES[] gridsize=cld(M, 16) _wmma_q4_skinny_kernel!(
        C, Pv, Sc8, Mn8, Bv, K, N, Val(mn === nothing), Val(WMMA_SKINNY_WAVES[]), Val(N <= 16 ? 1 : 2))
    return C
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

# `wpe` > 0 bounds the registers per thread (waves per SIMD) when AMDGPU.jl has the launch-bounds options
function _launch_q4(kern, args, nthreads, grid, wpe)
    @static if :waves_per_eu in AMDGPU.COMPILER_KWARGS
        if wpe > 0
            @roc groupsize=nthreads gridsize=grid flat_work_group_size=nthreads waves_per_eu=wpe kern(args...)
            return nothing
        end
    end
    @roc groupsize=nthreads gridsize=grid kern(args...)
    return nothing
end

# C (M, N) = dequantize(P, scale, mn)' * B for the 4-bit layout of `Q4Weight`: Pv (K/32, M)
# of NTuple{16,UInt8}, scale / mn (K/group, M) Float16 (`mn` = nothing when symmetric).
function wmma_q4_gemm!(C::ROCArray{Float32,2}, Pv, scale::ROCArray{Float16,2}, mn,
                       group::Int, B::ROCArray{Float16,2}, N::Int; kind::Symbol = :auto, pd::Int = WMMA_Q4_PD[], wpe::Int = WMMA_Q4_WPE[],
                       residual = nothing)
    K32, M = size(Pv); K = 32 * K32; Npad = size(B, 2)
    (M % 16 == 0 && Npad % 64 == 0 && group % 32 == 0 && size(B, 1) == K &&
     size(scale) == (K ÷ group, M) && N <= Npad && size(C) == (M, N)) ||
        throw(ArgumentError("wmma_q4_gemm!: need M % 16 == 0, group % 32 == 0, padded N % 64 == 0"))
    Bv = reinterpret(WMMA_V8H, B)
    args = (C, Pv, scale, mn === nothing ? scale : mn, Bv, K, N, group, Val(mn === nothing))
    if kind == :auto
        kind = M % 256 == 0 && K % 64 == 0 && Npad % 256 == 0 && ispow2(group) && get(ENV, "LUMINAL_WMMA_G", "1") != "0" ? :g :
               M % 128 == 0 && Npad % 256 == 0 && get(ENV, "LUMINAL_WMMA_WIDE", "1") != "0" ?
                   (get(ENV, "LUMINAL_WMMA_WS", "1") != "0" ? :ws : :wide) :
               M % 128 == 0 && Npad % 128 == 0 && get(ENV, "LUMINAL_WMMA_BIG", "1") != "0" ? :big : :small
    end
    # `residual` (an (M, N) Float32 array) is added to the result: in the 128-row kernels' store, else after the kernel
    R = residual isa ROCArray{Float32,2} && size(residual) == (M, N) ? residual : nothing
    residual !== nothing && R === nothing && throw(ArgumentError("wmma_q4_gemm!: the residual must be a Float32 ROCArray of the output's size"))
    hr = Val(R !== nothing)
    if kind == :wide || kind == :big || kind == :ws || kind == :g
        Cv = reinterpret(WMMA_V4F, C)              # (M/4, N): the 128-row kernels store 16 bytes at a time
        Rv = R === nothing ? Cv : reinterpret(WMMA_V4F, R)
    end
    if kind == :g
        Pw = reinterpret(UInt32, Pv)
        _launch_q4(_wmma_q4_g_kernel!, (Cv, Pw, scale, mn === nothing ? scale : mn, Bv, K, N, group, Val(mn === nothing), Rv, hr),
                   1024, (Npad ÷ 256, M ÷ 256), wpe)
    elseif kind == :ws
        Pw = reinterpret(UInt32, Pv)
        _launch_q4(_wmma_q4_ws_kernel!, (Cv, Pw, scale, mn === nothing ? scale : mn, Bv, K, N, group, Val(mn === nothing), Val(2), Rv, hr, Val(16)),
                   1024, (Npad ÷ 256, M ÷ 128), wpe)
    elseif kind == :wide
        Pw = reinterpret(UInt32, Pv)   # (K/8, M): words of 8 codes
        _launch_q4(_wmma_q4_wide_kernel!, (Cv, Pw, scale, mn === nothing ? scale : mn, Bv, K, N, group, Val(mn === nothing), Val(pd), Rv, hr),
                   512, (Npad ÷ 256, M ÷ 128), wpe)
    elseif kind == :big
        _launch_q4(_wmma_q4_big_kernel!, (Cv, Pv, scale, mn === nothing ? scale : mn, Bv, K, N, group, Val(mn === nothing), Val(pd), Rv, hr),
                   256, (Npad ÷ 128, M ÷ 128), wpe)
    else
        @roc groupsize=128 gridsize=(Npad ÷ 64, cld(M, 64)) _wmma_q4_kernel!(args...)
        R === nothing || (C .+= R)
    end
    return C
end
