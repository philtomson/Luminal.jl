# GGUF (llama.cpp's model file format): header, metadata, tensor table, and
# dequantization of the common tensor types to Float32.
#
# Layout (version 3): "GGUF", version, tensor count, metadata count; metadata
# key/value pairs; tensor infos (name, dims with ne[1] fastest, type, offset);
# then the data, aligned to `general.alignment` (32). A tensor with dims
# (ne1, ne2) holds a row-major (ne2, ne1) matrix, i.e. Hugging Face's
# (out, in) weight with `in` fastest; read column-major it is (in, out).

struct GGUFTensor
    name::String
    dims::Vector{Int}      # ne, fastest first
    type::Int              # ggml type id
    offset::Int            # from the start of the data section
end

struct GGUFFile
    path::String
    version::Int
    metadata::Dict{String, Any}
    tensors::Dict{String, GGUFTensor}
    data_start::Int
end

const GGML_TYPES = Dict(0 => :F32, 1 => :F16, 2 => :Q4_0, 3 => :Q4_1, 6 => :Q5_0, 7 => :Q5_1, 8 => :Q8_0,
                        10 => :Q2_K, 11 => :Q3_K, 12 => :Q4_K, 13 => :Q5_K, 14 => :Q6_K, 20 => :IQ4_NL, 21 => :IQ3_S, 23 => :IQ4_XS,
                        30 => :BF16)
# (values per block, bytes per block)
const GGML_BLOCKS = Dict(:F32 => (1, 4), :F16 => (1, 2), :BF16 => (1, 2), :Q4_0 => (32, 18), :Q4_1 => (32, 20),
                         :Q5_0 => (32, 22), :Q5_1 => (32, 24), :Q8_0 => (32, 34), :Q2_K => (256, 84),
                         :Q3_K => (256, 110), :Q4_K => (256, 144), :Q5_K => (256, 176), :Q6_K => (256, 210),
                         :IQ4_NL => (32, 18), :IQ3_S => (256, 110), :IQ4_XS => (256, 136))

ggml_type(t::GGUFTensor) = get(GGML_TYPES, t.type, Symbol("type", t.type))

function _gguf_string(io)
    n = read(io, UInt64)
    return String(read(io, n))
end

function _gguf_value(io, t)
    t == 0 && return read(io, UInt8)
    t == 1 && return read(io, Int8)
    t == 2 && return read(io, UInt16)
    t == 3 && return read(io, Int16)
    t == 4 && return read(io, UInt32)
    t == 5 && return read(io, Int32)
    t == 6 && return read(io, Float32)
    t == 7 && return read(io, UInt8) != 0
    t == 8 && return _gguf_string(io)
    if t == 9
        et = read(io, UInt32); n = read(io, UInt64)
        return [_gguf_value(io, et) for _ in 1:n]
    end
    t == 10 && return read(io, UInt64)
    t == 11 && return read(io, Int64)
    t == 12 && return read(io, Float64)
    error("GGUF: unknown metadata value type $t")
end

"""
    read_gguf(path) -> GGUFFile

The header of a GGUF file: its metadata (`f.metadata`, key => value) and tensor
table (`f.tensors`, name => GGUFTensor). Tensors are read with `gguf_tensor`.
"""
function read_gguf(path::String)
    open(path, "r") do io
        read(io, UInt32) == 0x46554747 || error("$path is not a GGUF file")
        version = Int(read(io, UInt32))
        version >= 2 || error("GGUF version $version is not supported")
        nt = Int(read(io, UInt64)); nkv = Int(read(io, UInt64))
        md = Dict{String, Any}()
        for _ in 1:nkv
            k = _gguf_string(io)
            md[k] = _gguf_value(io, read(io, UInt32))
        end
        ts = Dict{String, GGUFTensor}()
        for _ in 1:nt
            name = _gguf_string(io)
            nd = Int(read(io, UInt32))
            dims = [Int(read(io, UInt64)) for _ in 1:nd]
            ty = Int(read(io, UInt32)); off = Int(read(io, UInt64))
            ts[name] = GGUFTensor(name, dims, ty, off)
        end
        align = Int(get(md, "general.alignment", 32))
        start = cld(position(io), align) * align
        return GGUFFile(path, version, md, ts, start)
    end
end

# ── dequantization ────────────────────────────────────────────────────────────

_f16(b, i) = Float32(reinterpret(Float16, UInt16(b[i]) | UInt16(b[i+1]) << 8))

# (Each dequantizer runs its blocks on all of Julia's threads: start Julia with
# `-t auto` for a many-fold faster load.)
function _deq_q8_0!(out, b)
    Threads.@threads for blk in 0:length(out) ÷ 32 - 1
        o = blk * 34
        d = _f16(b, o + 1)
        @inbounds for j in 1:32
            out[blk * 32 + j] = d * reinterpret(Int8, b[o + 2 + j])
        end
    end
end

# the 6-bit scale and min of sub-block j (0-based) of a Q4_K / Q5_K block
@inline function _k_scale_min(sc, o, j)
    if j < 4
        return sc[o + j + 1] & 63, sc[o + j + 5] & 63
    else
        return (sc[o + j + 5] & 0x0f) | ((sc[o + j - 3] >> 6) << 4),
               (sc[o + j + 5] >> 4) | ((sc[o + j + 1] >> 6) << 4)
    end
end

function _deq_q4_k!(out, b)
    Threads.@threads for blk in 0:length(out) ÷ 256 - 1
        o = blk * 144
        d = _f16(b, o + 1); dmin = _f16(b, o + 3)
        so = o + 4                                     # 12 scale bytes at so+1..so+12
        qo = o + 16                                    # 128 quant bytes
        y = blk * 256
        for j in 0:2:7                                 # sub-blocks in pairs: low then high nibbles of 32 bytes
            s1, m1 = _k_scale_min(b, so, j); s2, m2 = _k_scale_min(b, so, j + 1)
            d1, mm1 = d * s1, dmin * m1; d2, mm2 = d * s2, dmin * m2
            q = qo + (j ÷ 2) * 32
            @inbounds for l in 1:32
                out[y + l] = d1 * (b[q + l] & 0x0f) - mm1
                out[y + 32 + l] = d2 * (b[q + l] >> 4) - mm2
            end
            y += 64
        end
    end
end

function _deq_q5_k!(out, b)
    Threads.@threads for blk in 0:length(out) ÷ 256 - 1
        o = blk * 176
        d = _f16(b, o + 1); dmin = _f16(b, o + 3)
        so = o + 4; ho = o + 16; qo = o + 48
        y = blk * 256
        u1, u2 = 0x01, 0x02
        for j in 0:2:7
            s1, m1 = _k_scale_min(b, so, j); s2, m2 = _k_scale_min(b, so, j + 1)
            d1, mm1 = d * s1, dmin * m1; d2, mm2 = d * s2, dmin * m2
            q = qo + (j ÷ 2) * 32
            @inbounds for l in 1:32
                h = b[ho + l]
                out[y + l] = d1 * ((b[q + l] & 0x0f) + (h & u1 != 0 ? 16 : 0)) - mm1
                out[y + 32 + l] = d2 * ((b[q + l] >> 4) + (h & u2 != 0 ? 16 : 0)) - mm2
            end
            y += 64; u1 <<= 2; u2 <<= 2
        end
    end
end

function _deq_q6_k!(out, b)
    Threads.@threads for blk in 0:length(out) ÷ 256 - 1
        o = blk * 210
        ql = o; qh = o + 128; sc = o + 192
        d = _f16(b, o + 209)
        y = blk * 256
        for n in 0:1                                   # two halves of 128 values
            qlo = ql + 64n; qho = qh + 32n; sco = sc + 8n
            @inbounds for l in 0:31
                is = l ÷ 16
                q1 = Int((b[qlo + l + 1] & 0x0f) | ((b[qho + l + 1] >> 0) & 3) << 4) - 32
                q2 = Int((b[qlo + l + 33] & 0x0f) | ((b[qho + l + 1] >> 2) & 3) << 4) - 32
                q3 = Int((b[qlo + l + 1] >> 4) | ((b[qho + l + 1] >> 4) & 3) << 4) - 32
                q4 = Int((b[qlo + l + 33] >> 4) | ((b[qho + l + 1] >> 6) & 3) << 4) - 32
                s(k) = Float32(reinterpret(Int8, b[sco + k + 1]))
                out[y + l + 1] = d * s(is) * q1
                out[y + l + 33] = d * s(is + 2) * q2
                out[y + l + 65] = d * s(is + 4) * q3
                out[y + l + 97] = d * s(is + 6) * q4
            end
            y += 128
        end
    end
end

# --- i-quants and Q3_K (ported from llama.cpp's ggml-quants.c reference dequantizers) ---
const _KVALUES_IQ4NL = Float32[-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
# iq3s_grid (ggml-common.h): 512 entries of 4 bytes (little endian) each
const _IQ3S_GRID = UInt32[
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
]

function _deq_iq4_nl!(out, b)
    Threads.@threads for blk in 0:length(out) ÷ 32 - 1
        o = blk * 18; d = _f16(b, o + 1); y = blk * 32
        @inbounds for j in 1:16
            q = b[o + 2 + j]
            out[y + j] = d * _KVALUES_IQ4NL[(q & 0x0f) + 1]
            out[y + 16 + j] = d * _KVALUES_IQ4NL[(q >> 4) + 1]
        end
    end
end

function _deq_iq4_xs!(out, b)
    Threads.@threads for blk in 0:length(out) ÷ 256 - 1
        o = blk * 136; d = _f16(b, o + 1)
        sh = UInt32(b[o + 3]) | UInt32(b[o + 4]) << 8            # scales_h
        sl = o + 4                                                # scales_l: 4 bytes at sl+1..sl+4
        qo = o + 8                                                # 128 quant bytes
        y = blk * 256
        @inbounds for ib in 0:7
            ls = Int((b[sl + ib ÷ 2 + 1] >> (4 * (ib % 2))) & 0x0f) | Int((sh >> (2 * ib)) & 3) << 4
            dl = d * (ls - 32)
            for j in 1:16
                q = b[qo + ib * 16 + j]
                out[y + ib * 32 + j] = dl * _KVALUES_IQ4NL[(q & 0x0f) + 1]
                out[y + ib * 32 + 16 + j] = dl * _KVALUES_IQ4NL[(q >> 4) + 1]
            end
        end
    end
end

function _deq_q3_k!(out, b)
    Threads.@threads for blk in 0:length(out) ÷ 256 - 1
        o = blk * 110
        d_all = _f16(b, o + 109)
        hm = o; qs = o + 32; so = o + 96                          # hmask[32], qs[64], scales[12], d
        aux = ntuple(i -> UInt32(b[so + 4i - 3]) | UInt32(b[so + 4i - 2]) << 8 | UInt32(b[so + 4i - 1]) << 16 | UInt32(b[so + 4i]) << 24, 3)
        k1 = 0x03030303; k2 = 0x0f0f0f0f
        tmp = aux[3]
        a2 = ((aux[1] >> 4) & k2) | (((tmp >> 4) & k1) << 4)
        a3 = ((aux[2] >> 4) & k2) | (((tmp >> 6) & k1) << 4)
        a0 = (aux[1] & k2) | (((tmp >> 0) & k1) << 4)
        a1 = (aux[2] & k2) | (((tmp >> 2) & k1) << 4)
        sc = (a0, a1, a2, a3)
        scale(i) = Int(reinterpret(Int8, UInt8((sc[(i ÷ 4) + 1] >> (8 * (i % 4))) & 0xff)))   # i = 0..15
        y = blk * 256; is = 0; m = UInt8(1)
        @inbounds for n in 0:128:255
            q = qs + (n ÷ 128) * 32; shift = 0
            for j in 0:3
                for half in 0:1
                    dl = d_all * (scale(is) - 32); is += 1
                    for l in 1:16
                        lo = Int((b[q + half * 16 + l] >> shift) & 3)
                        hi = (b[hm + half * 16 + l] & m) != 0 ? 0 : 4
                        out[y + l] = dl * (lo - hi)
                    end
                    y += 16
                end
                shift += 2; m <<= 1
            end
        end
    end
end

function _deq_iq3_s!(out, b)
    Threads.@threads for blk in 0:length(out) ÷ 256 - 1
        o = blk * 110
        d = _f16(b, o + 1)
        qs = o + 2; qh = o + 66; sg = o + 74; sc = o + 106        # qs[64], qh[8], signs[32], scales[4]
        y = blk * 256
        @inbounds for ib32 in 0:2:7
            sb = b[sc + ib32 ÷ 2 + 1]
            for (half, db) in ((0, d * (1 + 2 * Int(sb & 0x0f))), (1, d * (1 + 2 * Int(sb >> 4))))
                qhv = Int(b[qh + ib32 + half + 1])
                qb = qs + (ib32 + half) * 8; sb4 = sg + (ib32 + half) * 4
                for l in 0:3
                    g1 = _IQ3S_GRID[(Int(b[qb + 2l + 1]) | ((qhv << (8 - 2l)) & 256)) + 1]
                    g2 = _IQ3S_GRID[(Int(b[qb + 2l + 2]) | ((qhv << (7 - 2l)) & 256)) + 1]
                    sgn = b[sb4 + l + 1]
                    for j in 0:3
                        out[y + j + 1] = db * Float32((g1 >> (8j)) & 0xff) * ((sgn & (UInt8(1) << j)) != 0 ? -1f0 : 1f0)
                        out[y + j + 5] = db * Float32((g2 >> (8j)) & 0xff) * ((sgn & (UInt8(1) << (j + 4))) != 0 ? -1f0 : 1f0)
                    end
                    y += 8
                end
            end
        end
    end
end

"""
    gguf_raw(f::GGUFFile, name) -> Vector{UInt8}

The tensor's bytes as stored in the file (blocks of its ggml type, row after row). The pages are dropped
from the page cache once read (`LUMINAL_GGUF_KEEP_CACHE=1` keeps them).
"""
function gguf_raw(f::GGUFFile, name::String)
    t = f.tensors[name]
    ty = ggml_type(t)
    haskey(GGML_BLOCKS, ty) || error("GGUF tensor $name: type $ty is not supported")
    per, bytes = GGML_BLOCKS[ty]
    return open(f.path, "r") do io
        off = f.data_start + t.offset; len = (prod(t.dims) ÷ per) * bytes
        seek(io, off)
        data = read(io, len)
        # The model is read once: let the kernel drop these pages from the page cache at once. (A 28 GB file
        # otherwise fills the cache and the kernel swaps out other processes' idle memory to make room.
        # Only clean, unmapped pages are dropped. LUMINAL_GGUF_KEEP_CACHE=1 disables this.)
        get(ENV, "LUMINAL_GGUF_KEEP_CACHE", "0") == "1" ||
            ccall(:posix_fadvise, Cint, (Cint, Int64, Int64, Cint), Cint(reinterpret(Int32, fd(io))), off, len, 4)
        data
    end
end

"""
    gguf_tensor(f::GGUFFile, name) -> Array{Float32}

Tensor `name` dequantized to Float32, in GGUF's order read column-major: dims
`(ne1, ne2, ...)`. A weight matrix comes out as (in, out); `permutedims` it for
Hugging Face's (out, in).
"""
function gguf_tensor(f::GGUFFile, name::String)
    t = f.tensors[name]
    ty = ggml_type(t)
    haskey(GGML_BLOCKS, ty) || error("GGUF tensor $name: type $ty is not supported")
    n = prod(t.dims)
    raw = gguf_raw(f, name)
    out = Vector{Float32}(undef, n)
    if ty === :F32
        out .= reinterpret(Float32, raw)
    elseif ty === :F16
        h = reinterpret(Float16, raw)
        Threads.@threads for i in eachindex(out); @inbounds out[i] = Float32(h[i]); end
    elseif ty === :BF16
        h = reinterpret(UInt16, raw)
        Threads.@threads for i in eachindex(out); @inbounds out[i] = reinterpret(Float32, UInt32(h[i]) << 16); end
    elseif ty === :Q8_0
        _deq_q8_0!(out, raw)
    elseif ty === :Q4_K
        _deq_q4_k!(out, raw)
    elseif ty === :Q5_K
        _deq_q5_k!(out, raw)
    elseif ty === :Q6_K
        _deq_q6_k!(out, raw)
    elseif ty === :Q3_K
        _deq_q3_k!(out, raw)
    elseif ty === :IQ4_NL
        _deq_iq4_nl!(out, raw)
    elseif ty === :IQ4_XS
        _deq_iq4_xs!(out, raw)
    elseif ty === :IQ3_S
        _deq_iq3_s!(out, raw)
    else
        error("GGUF tensor $name: dequantizing $ty is not implemented")
    end
    return Base.reshape(out, t.dims...)
end

export GGUFFile, GGUFTensor, read_gguf, gguf_tensor, gguf_raw
