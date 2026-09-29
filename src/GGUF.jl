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
                        10 => :Q2_K, 11 => :Q3_K, 12 => :Q4_K, 13 => :Q5_K, 14 => :Q6_K, 30 => :BF16)
# (values per block, bytes per block)
const GGML_BLOCKS = Dict(:F32 => (1, 4), :F16 => (1, 2), :BF16 => (1, 2), :Q4_0 => (32, 18), :Q4_1 => (32, 20),
                         :Q5_0 => (32, 22), :Q5_1 => (32, 24), :Q8_0 => (32, 34), :Q2_K => (256, 84),
                         :Q3_K => (256, 110), :Q4_K => (256, 144), :Q5_K => (256, 176), :Q6_K => (256, 210))

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
    per, bytes = GGML_BLOCKS[ty]
    raw = open(f.path, "r") do io
        seek(io, f.data_start + t.offset)
        read(io, (n ÷ per) * bytes)
    end
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
    else
        error("GGUF tensor $name: dequantizing $ty is not implemented")
    end
    return Base.reshape(out, t.dims...)
end

export GGUFFile, GGUFTensor, read_gguf, gguf_tensor
