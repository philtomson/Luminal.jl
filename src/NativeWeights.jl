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
@kernel function _repack_q5k_kernel!(p, scale, mn, hi, @Const(raw))
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
    scale[i] = Float16(d * Float32(s))
    mn[i] = Float16(-dmin * Float32(m))
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
    hi[i] = w
end

_q4_planes(p, scale, mn) = Q4Weight(p, Base.reshape(reinterpret(NTuple{16, UInt8}, vec(p)), size(scale, 1), size(scale, 2)),
                                    scale, mn, 32)

"""
    native_weight(f::GGUFFile, name, device) -> Q4Weight, Q5Weight or nothing

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
        p = similar(raw, UInt8, K ÷ 2, M)
        scale = similar(raw, Float16, G, M); mn = similar(raw, Float16, G, M); hi = similar(raw, UInt32, G, M)
        _repack_q5k_kernel!(KernelAbstractions.get_backend(raw), 256)(p, scale, mn, hi, raw; ndrange = G * M)
        _free_now!(raw)
        return Q5Weight(_q4_planes(p, scale, mn), hi)
    end
    return nothing
end

# --- reordering ---------------------------------------------------------------

_wrows(w::Q4Weight, idx) = _q4_planes(w.p[:, idx], w.scale[:, idx], w.mn[:, idx])
function _wvcat(a::Q4Weight, b::Q4Weight)
    _q4_planes(hcat(a.p, b.p), hcat(a.scale, b.scale), hcat(a.mn, b.mn))
end
# Input channels reordered in units of 32: `perm` (length K) must move whole groups of 32.
function _wcols(w::Q4Weight, perm::AbstractVector{Int})
    G = size(w.scale, 1); M = size(w.scale, 2)
    gperm = (perm[1:32:end] .- 1) .÷ 32 .+ 1
    gd = to_device(gperm, _device_of(w.p))
    p3 = Base.reshape(w.p, 16, G, M)[:, gd, :]
    return _q4_planes(Base.reshape(p3, 16 * G, M), w.scale[gd, :], w.mn[gd, :])
end
_device_of(x) = x isa AnyGPUArray ? get_device() : CPUDevice()

export native_weight

_wrows(w::Q5Weight, idx) = Q5Weight(_wrows(w.q, idx), w.hi[:, idx])
_wvcat(a::Q5Weight, b::Q5Weight) = Q5Weight(_wvcat(a.q, b.q), hcat(a.hi, b.hi))
function _wcols(w::Q5Weight, perm::AbstractVector{Int})
    gd = to_device((perm[1:32:end] .- 1) .÷ 32 .+ 1, _device_of(w.hi))
    return Q5Weight(_wcols(w.q, perm), w.hi[gd, :])
end
