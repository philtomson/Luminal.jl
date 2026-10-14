# GGUF weights in the file's own values (no re-quantization).
#
# The K-quant formats are all "codes times a per-group scale plus a per-group minimum": Q4_K is
# d * sc_j * q - dmin * m_j with q in 0..15 over sub-blocks j of 32 inputs. That is `Q4Weight`'s asymmetric
# group-32 layout, so the file's bytes are repacked on the GPU into its planes (codes in `Q4Weight`'s
# nibble order, Float16 scale = d * sc_j and min = -dmin * m_j per group of 32) and every existing kernel
# reads them. The values are the file's, to Float16 rounding of d * sc_j.
#
# Row (output channel) order is the file's; `_wrows` / `_wvcat` / `_wcols` reorder a native weight the way
# the loaders reorder Float32 matrices.

# GGUF dims (in, out): output row o holds K/256 consecutive blocks.
@kernel function _repack_q4k_kernel!(p, scale, mn, @Const(raw))
    i = @index(Global, Linear)                      # group of 32 inputs: block (i-1) ÷ 8, sub-block (i-1) % 8
    g0 = i - 1
    blk = g0 >> 3; j = g0 & 7
    o = blk * 144
    d = Float32(reinterpret(Float16, UInt16(raw[o + 1]) | UInt16(raw[o + 2]) << 8))
    dmin = Float32(reinterpret(Float16, UInt16(raw[o + 3]) | UInt16(raw[o + 4]) << 8))
    sc = o + 4                                      # 12 scale bytes: 0-based k at raw[sc + k + 1]
    if j < 4
        s = raw[sc + j + 1] & 0x3f; m = raw[sc + j + 5] & 0x3f
    else
        s = (raw[sc + j + 5] & 0x0f) | ((raw[sc + j - 3] >> 0x06) << 0x04)
        m = (raw[sc + j + 5] >> 0x04) | ((raw[sc + j + 1] >> 0x06) << 0x04)
    end
    scale[i] = Float16(d * Float32(s))
    mn[i] = Float16(-dmin * Float32(m))
    q = o + 16 + (j >> 1) * 32                      # 32 bytes shared by sub-blocks j and j+1 (low / high nibbles)
    sh = UInt8(4) * UInt8(j & 1)
    for t in 0:15
        lo = (raw[q + 2t + 1] >> sh) & 0x0f
        hi = (raw[q + 2t + 2] >> sh) & 0x0f
        p[g0 * 16 + t + 1] = lo | (hi << 0x04)
    end
end

# Q5_K: 176-byte blocks, d, dmin, 12 scale bytes, 32 bytes qh (bit j of qh[l] = fifth bit of code l of sub-block j), 128 bytes qs.
@kernel function _repack_q5k_kernel!(p, meta, @Const(raw))
    i = @index(Global, Linear)
    g0 = i - 1
    blk = g0 >> 3; j = g0 & 7
    o = blk * 176
    d = Float32(reinterpret(Float16, UInt16(raw[o + 1]) | UInt16(raw[o + 2]) << 8))
    dmin = Float32(reinterpret(Float16, UInt16(raw[o + 3]) | UInt16(raw[o + 4]) << 8))
    sc = o + 4
    if j < 4
        s = raw[sc + j + 1] & 0x3f; m = raw[sc + j + 5] & 0x3f
    else
        s = (raw[sc + j + 5] & 0x0f) | ((raw[sc + j - 3] >> 0x06) << 0x04)
        m = (raw[sc + j + 5] >> 0x04) | ((raw[sc + j + 1] >> 0x06) << 0x04)
    end
    sb = UInt64(reinterpret(UInt16, Float16(d * Float32(s))))
    mb = UInt64(reinterpret(UInt16, Float16(-dmin * Float32(m))))
    q = o + 48 + (j >> 1) * 32
    qh = o + 16
    sh = UInt8(4) * UInt8(j & 1)
    jj = UInt8(j)
    w = UInt32(0)
    for t in 0:15
        lo = (raw[q + 2t + 1] >> sh) & 0x0f
        hc = (raw[q + 2t + 2] >> sh) & 0x0f
        p[g0 * 16 + t + 1] = lo | (hc << 0x04)
        bl = UInt32((raw[qh + 2t + 1] >> jj) & 0x01)
        bh = UInt32((raw[qh + 2t + 2] >> jj) & 0x01)
        w |= (bl << UInt32(4 + t)) | (bh << UInt32((20 + t) & 31))
    end
    meta[i] = sb | (mb << 16) | (UInt64(w) << 32)
end

# IQ4_XS: 136-byte blocks: d (f16), scales_h (u16), scales_l (4 bytes), 128 bytes qs; sub-block ib (32 values) has the 6-bit scale
# ls = scales_l nibble | 2 bits of scales_h << 4, step d * (ls - 32); its first 16 values are the low nibbles of 16 bytes, the next 16 the high ones.
@kernel function _repack_iq4xs_kernel!(p, scale, @Const(raw))
    i = @index(Global, Linear)
    g0 = i - 1
    blk = g0 >> 3; ib = g0 & 7
    o = blk * 136
    d = Float32(reinterpret(Float16, UInt16(raw[o + 1]) | UInt16(raw[o + 2]) << 8))
    sh = UInt32(raw[o + 3]) | UInt32(raw[o + 4]) << 8
    ls = Int((raw[o + 5 + (ib >> 1)] >> UInt8(4 * (ib & 1))) & 0x0f) | Int((sh >> UInt32(2 * ib)) & 3) << 4
    scale[i] = Float16(d * Float32(ls - 32))
    q = o + 8 + ib * 16
    for t in 0:15                                   # byte t holds codes 2t (low) and 2t + 1 (high) of the group
        c0 = 2t; c1 = 2t + 1
        v0 = c0 < 16 ? raw[q + c0 + 1] & 0x0f : raw[q + c0 - 15] >> 0x04
        v1 = c1 < 16 ? raw[q + c1 + 1] & 0x0f : raw[q + c1 - 15] >> 0x04
        p[g0 * 16 + t + 1] = v0 | (v1 << 0x04)
    end
end

# (the group size follows from the shapes: 32 inputs, or 16 for Q6_K's steps)
# IQ4_NL: 18-byte blocks of 32: d (f16), 16 bytes (first 16 values in the low nibbles, the next 16 in the high ones)
@kernel function _repack_iq4nl_kernel!(p, scale, @Const(raw))
    i = @index(Global, Linear)
    o = (i - 1) * 18
    scale[i] = Float16(Float32(reinterpret(Float16, UInt16(raw[o + 1]) | UInt16(raw[o + 2]) << 8)))
    for t in 0:15
        c0 = 2t; c1 = 2t + 1
        v0 = c0 < 16 ? raw[o + 3 + c0] & 0x0f : raw[o + 3 + c0 - 16] >> 0x04
        v1 = c1 < 16 ? raw[o + 3 + c1] & 0x0f : raw[o + 3 + c1 - 16] >> 0x04
        p[(i - 1) * 16 + t + 1] = v0 | (v1 << 0x04)
    end
end

# Q8_0: 34-byte blocks of 32: d (f16), 32 int8: a `QuantWeight` of group 32 (Float32 scale = d)
@kernel function _repack_q80_kernel!(q, scale, @Const(raw))
    i = @index(Global, Linear)
    o = (i - 1) * 34
    scale[i] = Float32(reinterpret(Float16, UInt16(raw[o + 1]) | UInt16(raw[o + 2]) << 8))
    for l in 1:32
        q[(i - 1) * 32 + l] = reinterpret(Int8, raw[o + 2 + l])
    end
end
_q8_planes(q, scale) = QuantWeight(q, Base.reshape(reinterpret(NTuple{16, Int8}, vec(q)), size(q, 1) ÷ 16, size(q, 2)), scale, 32)

_vview(p) = Base.reshape(reinterpret(NTuple{16, UInt8}, vec(p)), size(p, 1) ÷ 16, size(p, 2))
_q4_planes(p, scale, mn) = Q4Weight(p, Base.reshape(reinterpret(NTuple{16, UInt8}, vec(p)), size(p, 1) ÷ 16, size(p, 2)),
                                    scale, mn, 2 * size(p, 1) ÷ size(scale, 1))

# Q6_K: 210-byte blocks: ql (128 bytes, low nibbles), qh (64 bytes, 2 bits per code), 16 int8 scales, d (f16). Code e (0..255) of a block:
# half n = e >> 7, quarter qd = (e >> 5) & 3, l = e & 31: ql[64 n + 32 (qd & 1) + l] (high nibble when qd >= 2), qh[32 n + l] >> (2 qd), step d * scales[e >> 4].
@kernel function _repack_q6k_kernel!(p, meta, @Const(raw))
    i = @index(Global, Linear)
    g0 = i - 1                                      # group of 32 codes
    blk = g0 >> 3; j = g0 & 7
    o = blk * 210
    d = Float32(reinterpret(Float16, UInt16(raw[o + 209]) | UInt16(raw[o + 210]) << 8))
    sb0 = UInt32(reinterpret(UInt16, Float16(d * Float32(reinterpret(Int8, raw[o + 193 + ((32j) >> 4)])))))
    sb1 = UInt32(reinterpret(UInt16, Float16(d * Float32(reinterpret(Int8, raw[o + 193 + ((32j + 16) >> 4)])))))
    w0 = UInt32(0); w1 = UInt32(0)
    for t in 0:15
        v = 0x00
        for u in 0:1
            l32 = 2t + u                            # code within the group of 32
            e = 32j + l32
            n = e >> 7; qd = (e >> 5) & 3; l = e & 31
            qb = raw[o + 64n + 32 * (qd & 1) + l + 1]
            lo = qd >= 2 ? qb >> 0x04 : qb & 0x0f
            h2 = (raw[o + 128 + 32n + l + 1] >> UInt8(2qd)) & 0x03
            v |= lo << UInt8(4u)
            r = t & 7                               # byte within its K16 step
            pos = u == 0 ? 4 + 2r : (20 + 2r) & 31
            if t < 8
                w0 |= UInt32(h2) << UInt32(pos)
            else
                w1 |= UInt32(h2) << UInt32(pos)
            end
        end
        p[g0 * 16 + t + 1] = v
    end
    meta[g0 + 1] = (sb0 | (sb1 << 16), w0, w1)
end

"""
    native_weight(f::GGUFFile, name, device) -> a Q4Weight / Q5Weight / IQ4Weight / Q6Weight or nothing

Tensor `name` (a weight matrix, GGUF dims (in, out)) as a device `Q4Weight` of the file's own values, or
`nothing` when its type has no native layout yet (the caller then dequantizes it).
"""
function native_weight(f::GGUFFile, name::String, device)
    t = f.tensors[name]
    ty = ggml_type(t)
    length(t.dims) == 2 || return nothing
    K, M = t.dims
    if ty === :Q4_K
        K % 256 == 0 || return nothing
        raw = to_device(gguf_raw(f, name), device)
        G = K ÷ 32
        p = similar(raw, UInt8, K ÷ 2, M)
        scale = similar(raw, Float16, G, M); mn = similar(raw, Float16, G, M)
        _repack_q4k_kernel!(KernelAbstractions.get_backend(raw), 256)(p, scale, mn, raw; ndrange = G * M)
        _free_now!(raw)
        return _q4_planes(p, scale, mn)
    end
    if ty === :Q5_K
        K % 256 == 0 || return nothing
        raw = to_device(gguf_raw(f, name), device)
        G = K ÷ 32
        p = similar(raw, UInt8, K ÷ 2, M); meta = similar(raw, UInt64, G, M)
        _repack_q5k_kernel!(KernelAbstractions.get_backend(raw), 256)(p, meta, raw; ndrange = G * M)
        _free_now!(raw)
        return Q5Weight(p, _vview(p), meta)
    end
    if ty === :IQ4_XS
        K % 256 == 0 || return nothing
        raw = to_device(gguf_raw(f, name), device)
        G = K ÷ 32
        p = similar(raw, UInt8, K ÷ 2, M); scale = similar(raw, Float16, G, M)
        _repack_iq4xs_kernel!(KernelAbstractions.get_backend(raw), 256)(p, scale, raw; ndrange = G * M)
        _free_now!(raw)
        return IQ4Weight(_q4_planes(p, scale, nothing))
    end
    if ty === :Q8_0
        (K % 32 == 0 && K ÷ 32 <= 512) || return nothing         # (the int8 GEMV stages at most 512 scales)
        raw = to_device(gguf_raw(f, name), device)
        q = similar(raw, Int8, K, M); scale = similar(raw, Float32, K ÷ 32, M)
        _repack_q80_kernel!(KernelAbstractions.get_backend(raw), 256)(q, scale, raw; ndrange = (K ÷ 32) * M)
        _free_now!(raw)
        return _q8_planes(q, scale)
    end
    if ty === :IQ4_NL
        K % 256 == 0 || return nothing
        raw = to_device(gguf_raw(f, name), device)
        G = K ÷ 32
        p = similar(raw, UInt8, K ÷ 2, M); scale = similar(raw, Float16, G, M)
        _repack_iq4nl_kernel!(KernelAbstractions.get_backend(raw), 256)(p, scale, raw; ndrange = G * M)
        _free_now!(raw)
        return IQ4Weight(_q4_planes(p, scale, nothing))
    end
    if ty === :Q6_K
        K % 256 == 0 || return nothing
        raw = to_device(gguf_raw(f, name), device)
        p = similar(raw, UInt8, K ÷ 2, M); meta = similar(raw, NTuple{3, UInt32}, K ÷ 32, M)
        _repack_q6k_kernel!(KernelAbstractions.get_backend(raw), 256)(p, meta, raw; ndrange = (K ÷ 32) * M)
        _free_now!(raw)
        return Q6Weight(p, _vview(p), meta)
    end
    return nothing
end

# --- reordering ---------------------------------------------------------------

_wrows(w::Q4Weight, idx) = _q4_planes(w.p[:, idx], w.scale[:, idx], w.mn === nothing ? nothing : w.mn[:, idx])
function _wvcat(a::Q4Weight, b::Q4Weight)
    _q4_planes(hcat(a.p, b.p), hcat(a.scale, b.scale), a.mn === nothing ? nothing : hcat(a.mn, b.mn))
end
# Input channels reordered in units of 32: `perm` (length K) must move whole groups of 32.
function _wcols(w::Q4Weight, perm::AbstractVector{Int})
    G = size(w.scale, 1); M = size(w.scale, 2)
    gperm = (perm[1:32:end] .- 1) .÷ 32 .+ 1
    gd = to_device(gperm, _device_of(w.p))
    p3 = Base.reshape(w.p, 16, G, M)[:, gd, :]
    return _q4_planes(Base.reshape(p3, 16 * G, M), w.scale[gd, :], w.mn === nothing ? nothing : w.mn[gd, :])
end
_device_of(x) = x isa AnyGPUArray ? get_device() : CPUDevice()

export native_weight

_wrows(w::IQ4Weight, idx) = IQ4Weight(_wrows(w.q, idx))
_wvcat(a::IQ4Weight, b::IQ4Weight) = IQ4Weight(_wvcat(a.q, b.q))
_wcols(w::IQ4Weight, perm::AbstractVector{Int}) = IQ4Weight(_wcols(w.q, perm))

_wrows(w::QuantWeight, idx) = _q8_planes(w.q[:, idx], w.scale[:, idx])
_wvcat(a::QuantWeight, b::QuantWeight) = _q8_planes(hcat(a.q, b.q), hcat(a.scale, b.scale))
function _wcols(w::QuantWeight, perm::AbstractVector{Int})
    gd = to_device((perm[1:32:end] .- 1) .÷ 32 .+ 1, _device_of(w.q))
    return _q8_planes(w.q[to_device(perm, _device_of(w.q)), :], w.scale[gd, :])
end

# Q5Weight / Q6Weight: planes `p` and `meta`, reordered like the other layouts
_nat_like(w::Q5Weight, p, meta) = Q5Weight(p, _vview(p), meta)
_nat_like(w::Q6Weight, p, meta) = Q6Weight(p, _vview(p), meta)
_wrows(w::Union{Q5Weight, Q6Weight}, idx) = _nat_like(w, w.p[:, idx], w.meta[:, idx])
_wvcat(a::T, b::T) where {T <: Union{Q5Weight, Q6Weight}} = _nat_like(a, hcat(a.p, b.p), hcat(a.meta, b.meta))
function _wcols(w::Union{Q5Weight, Q6Weight}, perm::AbstractVector{Int})      # whole groups of 32
    G, M = size(w.meta)
    gd = to_device((perm[1:32:end] .- 1) .÷ 32 .+ 1, _device_of(w.meta))
    p3 = Base.reshape(w.p, 16, G, M)[:, gd, :]
    return _nat_like(w, Base.reshape(p3, 16 * G, M), w.meta[gd, :])
end
