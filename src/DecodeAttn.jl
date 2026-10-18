# Split-K ("flash-decoding") attention for the decode step on gfx11: the cache is cut into chunks of
# DA_CH keys and every (chunk, KV head, sequence) is its own workgroup, so a long cache keeps the whole
# GPU streaming instead of one workgroup per query head scanning it serially. All the query heads of a
# KV group and all S new tokens of a speculative step share each K/V read (rows = S * Gq <= 24).
#
# Per workgroup (8 waves): the rows' queries are staged as Float16 in LDS; the waves compute the
# chunk's scores with WMMA (16 query rows x 16 keys per tile, D contiguous in the cache), a softmax
# per row runs in LDS (a wave per row), and the probability-weighted sum of V is a coalesced scalar pass
# (a thread per pair of head dims, over the chunk's keys, 8 keys' loads in flight). Each chunk writes
# its unnormalized output with its max and sum; `_da_merge_kernel!` combines the chunks.
# Needs a Float16 cache, D in (128, 256), no window; row r = (token s, head g) attends slots 1..pos + s.
#
# An Int8 cache (`kv_dtype = Int8`, half the Float16 cache's bytes) stores each (slot, KV head) row as D signed bytes followed by
# DA_Q8_PAD bytes whose first four are the row's Float32 scale (absmax / 127): the cache's first dimension is D + DA_Q8_PAD.
# Scores are computed on the integer K (dequantized to Float16 exactly, (1024 + (q + 128)) - 1152, in the dims order b0 b2 b1 b3 of
# each 4-byte word: the queries are staged in the same order) and scaled per key; V's scale is folded into the probabilities
# (after the softmax sum, which uses the unscaled ones).

using AMDGPU
using KernelAbstractions.Extras: @unroll

const DA_CH = 256                # keys per workgroup
const DA_MAX_TOKENS = 8            # a step of more tokens takes the prefill attention
const DA_MAX_ROWS = 24           # S * Gq limit (LDS; 4 tokens x 6 query heads per KV head)
const DA_Q8_PAD = 16             # bytes appended to an int8 cache row (the Float32 scale + padding to 16 B)
const _DA_NEG1152 = 0xe480e480   # -1152 in f16 pairs
const DA_WS = Dict{Any, Any}()
# The chunks' partial outputs (WO) are stored as Float16: at 150k keys they are 115 MB per layer for an 8-token step, written by the part kernel and read
# back by the merge kernel, so their bytes set the merge kernel's time (it runs at the DRAM rate). LUMINAL_DA_WO16=0 keeps Float32.
_da_wo_f16() = get(ENV, "LUMINAL_DA_WO16", "1") != "0"   # partial-result workspaces (a captured graph bakes in the pointers)

# helpers with their own scopes (closures in the kernel body must not capture reassigned variables)
@inline _da_fma(a::NTuple{N,Float32}, p::NTuple{N,Float32}, v::Float32) where {N} = ntuple(q -> a[q] + p[q] * v, Val(N))
@inline _da_lds(scT, base, ::Val{N}) where {N} = ntuple(q -> @inbounds(scT[base + q]), Val(N))
@inline _da_vload8(Vp, dp, k0, ngrp, kvh, b) =
    ntuple(u -> @inbounds(Vp[dp + 1, k0 + (u - 1) * ngrp + 1, kvh + 1, b + 1]), Val(8))
@inline _da_kb(::Val{false}, Kv, ks, kk, kvh, b) = _wmma_cat(@inbounds(Kv[2ks + 1, kk, kvh + 1, b + 1]), @inbounds(Kv[2ks + 2, kk, kvh + 1, b + 1]))
# 4 int8 -> 4 halves (exact): bytes biased to unsigned, ORed into the mantissa of 1024.0, 1152 subtracted; order b0 b2 b1 b3
@inline function _da_dq8(w::UInt32)
    u = w ⊻ 0x80808080
    lo = _wmma_bitcast2(_pkadd((u & 0x00ff00ff) | 0x64006400, _DA_NEG1152))
    hi = _wmma_bitcast2(_pkadd(((u >> 0x08) & 0x00ff00ff) | 0x64006400, _DA_NEG1152))
    return (lo[1], lo[2], hi[1], hi[2])
end
@inline function _da_kb(::Val{true}, Kq, ks, kk, kvh, b)
    w = @inbounds Kq[ks + 1, kk, kvh + 1, b + 1]                 # 16 int8 = one 16-wide tile of the key's row
    a = _da_dq8(w[1]); c = _da_dq8(w[2]); d = _da_dq8(w[3]); e = _da_dq8(w[4])
    return (a..., c..., d..., e...)::WMMA_V16H
end
@inline _da_v1(v::NTuple{2,Float16}) = Float32(v[1])
@inline _da_v2(v::NTuple{2,Float16}) = Float32(v[2])
@inline _da_v1(v::UInt16) = Float32(v % Int8)
@inline _da_v2(v::UInt16) = Float32((v >> 0x08) % Int8)
@inline _da_dperm(d, ::Val{false}) = d
@inline _da_dperm(d, ::Val{true}) = d ⊻ ((d & 1) ⊻ ((d >> 1) & 1)) * 3    # swaps dims 1 and 2 of each group of 4
@inline _da_ksc(::Val{false}, Ksc, D, kk, kvh, b) = 1f0
@inline _da_ksc(::Val{true}, Ksc, D, kk, kvh, b) = @inbounds Ksc[D ÷ 4 + 1, kk, kvh + 1, b + 1]
@inline _da_qa(qs, ks, row) = ntuple(j -> VecElement(@inbounds qs[16ks + j, row]), Val(16))

function _da_write_kernel!(pk, pv, kn, vn, posv, S)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x - 1
    D = size(kn, 1); KVH = size(kn, 3); B = size(kn, 4)
    i < D * S * KVH * B || return
    d = i % D; t = (i ÷ D) % S; kvh = (i ÷ (D * S)) % KVH; b = i ÷ (D * S * KVH)
    pos = unsafe_trunc(Int, @inbounds posv[length(posv) == 1 ? 1 : b + 1])
    @inbounds pk[d + 1, pos + t + 1, kvh + 1, b + 1] = kn[d + 1, t + 1, kvh + 1, b + 1]
    @inbounds pv[d + 1, pos + t + 1, kvh + 1, b + 1] = vn[d + 1, t + 1, kvh + 1, b + 1]
    return
end

# int8 cache: one wave per (token, KV head, sequence) quantizes the K and V rows (scale = absmax / 127) into slot pos + t
@inline function _da_quant_row!(dst, sdst, src, D, t, kvh, b, slot, l)
    m = 0f0
    for j in 0:D ÷ 32 - 1
        @inbounds m = max(m, abs(Float32(src[l + 32j + 1, t + 1, kvh + 1, b + 1])))
    end
    for off in (1, 2, 4, 8, 16)
        m = max(m, AMDGPU.Device.shfl_xor(m, Cint(off)))
    end
    inv = m > 0f0 ? 127f0 / m : 0f0
    for j in 0:D ÷ 32 - 1
        @inbounds x = Float32(src[l + 32j + 1, t + 1, kvh + 1, b + 1])
        @inbounds dst[l + 32j + 1, slot + 1, kvh + 1, b + 1] = unsafe_trunc(Int8, round(x * inv))
    end
    l == 0 && (@inbounds sdst[D ÷ 4 + 1, slot + 1, kvh + 1, b + 1] = m / 127f0)
    return
end
function _da_write_q8_kernel!(pk, pv, ksc, vsc, kn, vn, posv, S)
    l = workitemIdx().x - 1; wg = workgroupIdx().x - 1
    D = size(kn, 1); KVH = size(kn, 3)
    t = wg % S; kvh = (wg ÷ S) % KVH; b = wg ÷ (S * KVH)
    pos = unsafe_trunc(Int, @inbounds posv[length(posv) == 1 ? 1 : b + 1])
    _da_quant_row!(pk, ksc, kn, D, t, kvh, b, pos + t, l)
    _da_quant_row!(pv, vsc, vn, D, t, kvh, b, pos + t, l)
    return
end

# Prefill's K/V (D, slen, KVH, B) -> an int8 cache's slots 0..n-1 of sequence b (1-based)
function _kv_quant_kernel!(dst, sdst, src, b0)
    t = workgroupIdx().x - 1; kvh = workgroupIdx().y - 1
    _da_quant_row!(dst, sdst, src, size(src, 1), t, kvh, b0, t, workitemIdx().x - 1)   # (src and dst share the batch index)
    return
end
function quantize_kv!(dst::ROCArray{Int8,4}, src, n::Int, b::Int)
    D = size(src, 1)
    size(dst, 1) == D + DA_Q8_PAD || error("int8 KV cache: row length $(size(dst, 1)) != head_dim + $DA_Q8_PAD")
    sd = reinterpret(Float32, dst)
    s4 = src isa ROCArray ? src : ROCArray(src)
    @roc groupsize=32 gridsize=(n, size(src, 3)) _kv_quant_kernel!(dst, sd, s4, b - 1)
    return dst
end

# The int8 cache's first NK slots as Float16: kt (D, NKpad, KVH) and V as vt, transposed (NKpad, D, KVH) when `::Val{true}`,
# else (D, NKpad, KVH) like kt
function _kv_dequant_kernel!(kt, vt, pk, pv, ksc, vsc, NK, ::Val{TR}) where {TR}
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x - 1
    D = size(kt, 1); KVH = size(kt, 3)
    i < D * NK * KVH || return
    d = i % D; key = (i ÷ D) % NK; kvh = i ÷ (D * NK)
    @inbounds kt[d + 1, key + 1, kvh + 1, 1] = Float16(Float32(pk[d + 1, key + 1, kvh + 1, 1]) * ksc[D ÷ 4 + 1, key + 1, kvh + 1, 1])
    vv = Float16(Float32(pv[d + 1, key + 1, kvh + 1, 1]) * vsc[D ÷ 4 + 1, key + 1, kvh + 1, 1])
    TR ? (@inbounds vt[key + 1, d + 1, kvh + 1, 1] = vv) : (@inbounds vt[d + 1, key + 1, kvh + 1, 1] = vv)
    return
end

function _da_part_kernel!(WO, WM, WL, Qa, Kv, Vp, Ksc, Vsc, posv, scale2, S, t0, Gq, rep, ::Val{D}, ::Val{QR}, ::Val{Q8}) where {D, QR, Q8}
    tid = workitemIdx().x - 1
    l = tid % 32; r = l % 16; hh = l ÷ 16; w = tid ÷ 32
    # rep > 1: the query heads of a KV group are split in `rep` groups of Gq (the heads of a workgroup), so that a many-token step still fits the LDS; the
    # groups of one key chunk are adjacent workgroups, which read the same K/V (the second read comes from the cache). vg: the (KV group, head group) index.
    xi = workgroupIdx().x - 1; c = xi ÷ rep; half = xi % rep
    kvh = workgroupIdx().y - 1; vg = kvh * rep + half; b = workgroupIdx().z - 1
    pos = unsafe_trunc(Int, @inbounds posv[length(posv) == 1 ? 1 : b + 1])
    nmax = pos + S                           # slots 1..n are attended (the new tokens are already in the cache)
    k0 = c * DA_CH
    k0 < nmax || return
    qs = @ROCStaticLocalArray(Float16, (256, 32), false)
    scT = @ROCStaticLocalArray(Float32, (6144,), false)      # (key, row) = DA_CH * DA_MAX_ROWS
    # a key group's partial sums = D * QR floats. With many rows they reuse the scores' LDS (dead after the first sync of the
    # reduction), which lets a third workgroup fit on a CU; with few rows a separate array is faster (fewer concurrent streams).
    exch = QR > 8 ? scT : (@ROCStaticLocalArray(Float32, (4608,), false))
    svs = Q8 ? (@ROCStaticLocalArray(Float32, (256,), false)) : scT   # int8 cache: the chunk's V scales
    mq = @ROCStaticLocalArray(Float32, (24,), false)
    lq = @ROCStaticLocalArray(Float32, (24,), false)
    red = @ROCStaticLocalArray(Float32, (256,), false)
    # stage the queries: row = (token s, head g); rows >= QR are zero
    idx = tid
    while idx < 32 * D
        rr = idx ÷ D; d = idx % D
        @inbounds qs[_da_dperm(d, Val(Q8)) + 1, rr + 1] = rr < QR ? Float16(Qa[d + 1, rr ÷ Gq + 1 + t0, vg * Gq + rr % Gq + 1, b + 1]) : Float16(0)
        idx += 256
    end
    if Q8
        @inbounds svs[tid + 1] = k0 + tid < nmax ? Vsc[D ÷ 4 + 1, k0 + tid + 1, kvh + 1, b + 1] : 0f0
    end
    sync_workgroup()
    # scores: the wave's two tiles (tt = w, w + 8) of 16 keys, K loads issued ahead of the WMMAs
    kkA = min(k0 + w * 16 + r, nmax - 1) + 1
    kkB = min(k0 + (w + 8) * 16 + r, nmax - 1) + 1
    klA = w * 16 + r; klB = (w + 8) * 16 + r
    s2A = scale2 * _da_ksc(Val(Q8), Ksc, D, kkA, kvh, b); s2B = scale2 * _da_ksc(Val(Q8), Ksc, D, kkB, kvh, b)
    for rt in 0:(QR - 1) ÷ 16
        scA = _wmma_zero(); scB = _wmma_zero()
        for ks in 0:2:D ÷ 16 - 1
            bA0 = _da_kb(Val(Q8), Kv, ks, kkA, kvh, b); bB0 = _da_kb(Val(Q8), Kv, ks, kkB, kvh, b)
            bA1 = _da_kb(Val(Q8), Kv, ks + 1, kkA, kvh, b); bB1 = _da_kb(Val(Q8), Kv, ks + 1, kkB, kvh, b)
            qa0 = _da_qa(qs, ks, 16rt + r + 1); qa1 = _da_qa(qs, ks + 1, 16rt + r + 1)
            scA = _wmma16(qa0, bA0, scA); scB = _wmma16(qa0, bB0, scB)
            scA = _wmma16(qa1, bA1, scA); scB = _wmma16(qa1, bB1, scB)
        end
        for i in 0:7
            row = 16rt + 2i + hh
            if row < QR
                n = pos + row ÷ Gq + t0 + 1       # this row's token attends slots 1..n
                @inbounds scT[klA * QR + row + 1] = k0 + klA < n ? scA[i + 1].value * s2A : -1f30
                @inbounds scT[klB * QR + row + 1] = k0 + klB < n ? scB[i + 1].value * s2B : -1f30
            end
        end
    end
    sync_workgroup()
    # softmax of each row over the chunk. scT is (key, row) with the row fastest, so thread (row, g) = (tid % QR, tid / QR) reads consecutive
    # rows in a wave (no LDS bank conflicts) and takes every ng-th key; the ng partial maxima / sums of a row are combined through `red`.
    ng = 256 ÷ QR
    srow = tid % QR; sg = tid ÷ QR
    act = sg < ng
    m = -1f30
    if act
        kk = sg
        while kk < DA_CH
            @inbounds m = max(m, scT[kk * QR + srow + 1])
            kk += ng
        end
        @inbounds red[sg * QR + srow + 1] = m
    end
    sync_workgroup()
    M = -1f30
    if act
        for g2 in 0:ng - 1
            @inbounds M = max(M, red[g2 * QR + srow + 1])
        end
    end
    sync_workgroup()
    ssum = 0f0
    if act
        kk = sg
        while kk < DA_CH
            @inbounds sv = scT[kk * QR + srow + 1]
            e = sv <= -1f29 ? 0f0 : exp2(sv - M)
            @inbounds scT[kk * QR + srow + 1] = Q8 ? e * svs[kk + 1] : e
            ssum += e
            kk += ng
        end
        @inbounds red[sg * QR + srow + 1] = ssum
    end
    sync_workgroup()
    if tid < QR
        L = 0f0
        for g2 in 0:ng - 1
            @inbounds L += red[g2 * QR + tid + 1]
        end
        @inbounds mq[tid + 1] = M; @inbounds lq[tid + 1] = L
    end
    sync_workgroup()
    if QR > 12
        _da_vhalf!(WO, scT, Vp, tid, min(nmax - k0, DA_CH), k0, c, kvh, vg, b, Val(D), Val(QR))
    else
        # out[row, d] = sum_key p[row, key] V[d, key]: thread (dp, g) takes the key subset g, d pair dp
        ndp = D ÷ 2
        ngrp = 256 ÷ ndp
        dp = tid % ndp; g = tid ÷ ndp
        nv = min(nmax - k0, DA_CH)
        a0 = ntuple(_ -> 0f0, Val(QR)); a1 = ntuple(_ -> 0f0, Val(QR))
        kj = g
        while kj + 7 * ngrp < nv               # 8 keys at a time: their V loads are in flight together
            vs = _da_vload8(Vp, dp, k0 + kj, ngrp, kvh, b)
            @unroll for u in 1:8
                p = _da_lds(scT, (kj + (u - 1) * ngrp) * QR, Val(QR))
                a0 = _da_fma(a0, p, _da_v1(vs[u])); a1 = _da_fma(a1, p, _da_v2(vs[u]))
            end
            kj += 8 * ngrp
        end
        while kj < nv
            v = @inbounds Vp[dp + 1, k0 + kj + 1, kvh + 1, b + 1]
            p = _da_lds(scT, kj * QR, Val(QR))
            a0 = _da_fma(a0, p, _da_v1(v)); a1 = _da_fma(a1, p, _da_v2(v))
            kj += ngrp
        end
        # sum the key groups' partials into group 0
        for gg in 1:ngrp-1
            sync_workgroup()
            if g == gg
                @unroll for q in 1:QR
                    @inbounds exch[(q - 1) * D + 2dp + 1] = a0[q]
                    @inbounds exch[(q - 1) * D + 2dp + 2] = a1[q]
                end
            end
            sync_workgroup()
            if g == 0
                a0 = _da_add_exch(a0, exch, 2dp + 1, D)
                a1 = _da_add_exch(a1, exch, 2dp + 2, D)
            end
        end
        if g == 0
            @unroll for q in 1:QR
                @inbounds WO[2dp + 1, q, c + 1, vg + 1, b + 1] = a0[q]
                @inbounds WO[2dp + 2, q, c + 1, vg + 1, b + 1] = a1[q]
            end
        end
    end
    if tid < QR
        @inbounds WM[tid + 1, c + 1, vg + 1, b + 1] = mq[tid + 1]
        @inbounds WL[tid + 1, c + 1, vg + 1, b + 1] = lq[tid + 1]
    end
    return
end
# Many rows (QR > 12, D = 256): two thread groups, each takes half of the rows and all the keys (half the accumulators a thread
# and no reduction across groups; the two groups read the same V). Thread (dp, g): head dims 2dp, 2dp + 1, rows g*QH + 1..
function _da_vhalf!(WO, scT, Vp, tid, nv, k0, c, kvh, vg, b, ::Val{D}, ::Val{QR}) where {D, QR}
    QH = (QR + 1) ÷ 2
    dp = tid % (D ÷ 2); g = tid ÷ (D ÷ 2)
    r0 = g * QH
    a0 = ntuple(_ -> 0f0, Val(QH)); a1 = ntuple(_ -> 0f0, Val(QH))
    kj = 0
    while kj + 7 < nv
        vs = _da_vload8(Vp, dp, k0 + kj, 1, kvh, b)
        @unroll for u in 1:8
            p = _da_lds(scT, (kj + u - 1) * QR + r0, Val(QH))
            a0 = _da_fma(a0, p, _da_v1(vs[u])); a1 = _da_fma(a1, p, _da_v2(vs[u]))
        end
        kj += 8
    end
    while kj < nv
        v = @inbounds Vp[dp + 1, k0 + kj + 1, kvh + 1, b + 1]
        p = _da_lds(scT, kj * QR + r0, Val(QH))
        a0 = _da_fma(a0, p, _da_v1(v)); a1 = _da_fma(a1, p, _da_v2(v))
        kj += 1
    end
    @unroll for q in 1:QH
        if r0 + q <= QR
            @inbounds WO[2dp + 1, r0 + q, c + 1, vg + 1, b + 1] = a0[q]
            @inbounds WO[2dp + 2, r0 + q, c + 1, vg + 1, b + 1] = a1[q]
        end
    end
    return
end
@inline _da_add_exch(a::NTuple{N,Float32}, exch, off, D) where {N} = ntuple(q -> a[q] + @inbounds(exch[(q - 1) * D + off]), Val(N))

# one workgroup per (head, token, sequence): out[:, h, s, b] = the chunks' outputs weighted by exp2(m_c - M) / L
function _da_merge_kernel!(out, WO, WM, WL, posv, S, t0, Gq)
    tid = workitemIdx().x - 1
    h = workgroupIdx().x - 1; sb = workgroupIdx().y - 1
    s = sb % S + 1; b = sb ÷ S
    D = size(out, 1)
    pos = unsafe_trunc(Int, @inbounds posv[length(posv) == 1 ? 1 : b + 1])
    nv = cld(pos + s + t0, DA_CH)
    kvh = h ÷ Gq; row = (s - 1) * Gq + h % Gq
    fc = @ROCStaticLocalArray(Float32, (1024,), false)
    red = @ROCStaticLocalArray(Float32, (8,), false)
    stat = @ROCStaticLocalArray(Float32, (2,), false)
    w = tid ÷ 32; l = tid % 32
    m = -1f30
    c = tid
    while c < nv
        @inbounds m = max(m, WM[row + 1, c + 1, kvh + 1, b + 1])
        c += 256
    end
    for off in (1, 2, 4, 8, 16)
        m = max(m, AMDGPU.Device.shfl_xor(m, Cint(off)))
    end
    l == 0 && (@inbounds red[w + 1] = m)
    sync_workgroup()
    M = -1f30
    for i in 1:8
        @inbounds M = max(M, red[i])
    end
    sync_workgroup()
    ls = 0f0
    c = tid
    while c < nv
        @inbounds f = exp2(WM[row + 1, c + 1, kvh + 1, b + 1] - M)
        @inbounds fc[c + 1] = f
        @inbounds ls += f * WL[row + 1, c + 1, kvh + 1, b + 1]
        c += 256
    end
    for off in (1, 2, 4, 8, 16)
        ls += AMDGPU.Device.shfl_xor(ls, Cint(off))
    end
    l == 0 && (@inbounds red[w + 1] = ls)
    sync_workgroup()
    if tid == 0
        L = 0f0
        for i in 1:8
            @inbounds L += red[i]
        end
        @inbounds stat[1] = 1f0 / L
    end
    sync_workgroup()
    if tid < D
        acc = 0f0
        c = 0
        while c + 7 < nv
            cb = c
            ws = ntuple(u -> @inbounds(WO[tid + 1, row + 1, cb + u, kvh + 1, b + 1]), Val(8))
            fs = ntuple(u -> @inbounds(fc[cb + u]), Val(8))
            @unroll for u in 1:8
                acc += fs[u] * ws[u]
            end
            c += 8
        end
        while c < nv
            @inbounds acc += fc[c + 1] * WO[tid + 1, row + 1, c + 1, kvh + 1, b + 1]
            c += 1
        end
        @inbounds out[tid + 1, h + 1, s + t0, b + 1] = acc * stat[1]
    end
    return
end

# out (D, H, S, B) Float32; q (D, S, H, B); the cache pk/pv (D, T, KVH, B) Float16; the new tokens' kn/vn
# (D, S, KVH, B); posv the per-sequence cache length before the step. Returns false if the shapes/arrays
# don't qualify (the caller falls back to the one-workgroup-per-head kernel).
function decode_attention_split!(out, q, pk, pv, kn, vn, posv, scale::Float32, write_cache::Bool)
    q8 = pk isa ROCArray{Int8,4} && pv isa ROCArray{Int8,4}
    (out isa ROCArray{Float32} && (q8 || (pk isa ROCArray{Float16,4} && pv isa ROCArray{Float16,4})) && wmma_enabled() &&
     get(ENV, "LUMINAL_DECODE_SPLIT", "1") != "0") || return false
    D, S, H, B = size(q, 1), size(q, 2), size(q, 3), size(q, 4)
    KVH = size(pk, 3); T = size(pk, 2)
    (D in (128, 256) && H % KVH == 0 && size(pk, 1) == D + (q8 ? DA_Q8_PAD : 0) && cld(T, DA_CH) <= 1024) || return false
    Gphys = H ÷ KVH
    # a step whose rows (tokens x heads of a KV group) do not fit the LDS: split the group's heads over `rep` workgroups (single pass over the cache), or, failing that,
    # split the tokens over several launches (a pass over the cache each)
    S <= DA_MAX_TOKENS || return false
    divs = [r for r in 1:Gphys if Gphys % r == 0]
    ri = findfirst(r -> S * (Gphys ÷ r) <= DA_MAX_ROWS, divs)
    rep = ri === nothing ? 1 : divs[ri]
    Gq = Gphys ÷ rep                          # heads per workgroup
    tc = DA_MAX_ROWS ÷ Gq                     # tokens per launch
    tc >= 1 || return false
    QR = min(S, tc) * Gq
    QR <= DA_MAX_ROWS && (QR <= 12 || D == 256) || return false
    Ksc = Vsc = posv                          # (unused placeholders unless the cache is int8)
    if q8
        Ksc = reinterpret(Float32, pk); Vsc = reinterpret(Float32, pv)
        write_cache && @roc groupsize=32 gridsize=S * KVH * B _da_write_q8_kernel!(pk, pv, Ksc, Vsc, kn, vn, posv, S)
    elseif write_cache
        @roc groupsize=256 gridsize=cld(D * S * KVH * B, 256) _da_write_kernel!(pk, pv, kn, vn, posv, S)
    end
    nsplit = cld(T, DA_CH)
    Kv = q8 ? reinterpret(NTuple{4,UInt32}, pk) : reinterpret(WMMA_V8H, pk)
    Vp = q8 ? reinterpret(UInt16, pv) : reinterpret(NTuple{2,Float16}, pv)
    q4 = q isa ROCArray ? q : copy(q)
    o4 = Base.reshape(out, D, H, S, B)
    for t0 in 0:tc:S-1
        sc = min(tc, S - t0)                  # this launch's tokens
        QRc = sc * Gq
        WT = _da_wo_f16() ? Float16 : Float32
        WO, WM, WL = get!(DA_WS, (typeof(out), D, nsplit, KVH * rep, B, QRc, WT)) do
            (AMDGPU.zeros(WT, D, QRc, nsplit, KVH * rep, B), AMDGPU.zeros(Float32, QRc, nsplit, KVH * rep, B), AMDGPU.zeros(Float32, QRc, nsplit, KVH * rep, B))
        end
        @roc groupsize=256 gridsize=(rep * nsplit, KVH, B) _da_part_kernel!(
            WO, WM, WL, q4, Kv, Vp, Ksc, Vsc, posv, scale * 1.4426950408889634f0, S, t0, Gq, rep, Val(D), Val(QRc), Val(q8))
        @roc groupsize=256 gridsize=(H, sc * B) _da_merge_kernel!(o4, WO, WM, WL, posv, sc, t0, Gq)
    end
    return true
end
