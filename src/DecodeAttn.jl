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

using AMDGPU
using KernelAbstractions.Extras: @unroll

const DA_CH = 256                # keys per workgroup
const DA_MAX_ROWS = 24           # S * Gq limit (LDS; 4 tokens x 6 query heads per KV head)
const DA_WS = Dict{Any, Any}()   # partial-result workspaces (a captured graph bakes in the pointers)

# helpers with their own scopes (closures in the kernel body must not capture reassigned variables)
@inline _da_fma(a::NTuple{N,Float32}, p::NTuple{N,Float32}, v::Float32) where {N} = ntuple(q -> a[q] + p[q] * v, Val(N))
@inline _da_lds(scT, base, ::Val{N}) where {N} = ntuple(q -> @inbounds(scT[base + q]), Val(N))
@inline _da_vload8(Vp, dp, k0, ngrp, kvh, b) =
    ntuple(u -> @inbounds(Vp[dp + 1, k0 + (u - 1) * ngrp + 1, kvh + 1, b + 1]), Val(8))
@inline _da_kb(Kv, ks, kk, kvh, b) = _wmma_cat(@inbounds(Kv[2ks + 1, kk, kvh + 1, b + 1]), @inbounds(Kv[2ks + 2, kk, kvh + 1, b + 1]))
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

function _da_part_kernel!(WO, WM, WL, Qa, Kv, Vp, posv, scale2, S, Gq, ::Val{D}, ::Val{QR}) where {D, QR}
    tid = workitemIdx().x - 1
    l = tid % 32; r = l % 16; hh = l ÷ 16; w = tid ÷ 32
    c = workgroupIdx().x - 1; kvh = workgroupIdx().y - 1; b = workgroupIdx().z - 1
    pos = unsafe_trunc(Int, @inbounds posv[length(posv) == 1 ? 1 : b + 1])
    nmax = pos + S                           # slots 1..n are attended (the new tokens are already in the cache)
    k0 = c * DA_CH
    k0 < nmax || return
    qs = @ROCStaticLocalArray(Float16, (256, 32), false)
    scT = @ROCStaticLocalArray(Float32, (6144,), false)      # (key, row) = DA_CH * DA_MAX_ROWS
    # a key group's partial sums = D * QR floats. With many rows they reuse the scores' LDS (dead after the first sync of the
    # reduction), which lets a third workgroup fit on a CU; with few rows a separate array is faster (fewer concurrent streams).
    exch = QR > 8 ? scT : (@ROCStaticLocalArray(Float32, (4608,), false))
    mq = @ROCStaticLocalArray(Float32, (24,), false)
    lq = @ROCStaticLocalArray(Float32, (24,), false)
    # stage the queries: row = (token s, head g); rows >= QR are zero
    idx = tid
    while idx < 32 * D
        rr = idx ÷ D; d = idx % D
        @inbounds qs[d + 1, rr + 1] = rr < QR ? Float16(Qa[d + 1, rr ÷ Gq + 1, kvh * Gq + rr % Gq + 1, b + 1]) : Float16(0)
        idx += 256
    end
    sync_workgroup()
    # scores: the wave's two tiles (tt = w, w + 8) of 16 keys, K loads issued ahead of the WMMAs
    kkA = min(k0 + w * 16 + r, nmax - 1) + 1
    kkB = min(k0 + (w + 8) * 16 + r, nmax - 1) + 1
    klA = w * 16 + r; klB = (w + 8) * 16 + r
    for rt in 0:(QR - 1) ÷ 16
        scA = _wmma_zero(); scB = _wmma_zero()
        for ks in 0:2:D ÷ 16 - 1
            bA0 = _da_kb(Kv, ks, kkA, kvh, b); bB0 = _da_kb(Kv, ks, kkB, kvh, b)
            bA1 = _da_kb(Kv, ks + 1, kkA, kvh, b); bB1 = _da_kb(Kv, ks + 1, kkB, kvh, b)
            qa0 = _da_qa(qs, ks, 16rt + r + 1); qa1 = _da_qa(qs, ks + 1, 16rt + r + 1)
            scA = _wmma16(qa0, bA0, scA); scB = _wmma16(qa0, bB0, scB)
            scA = _wmma16(qa1, bA1, scA); scB = _wmma16(qa1, bB1, scB)
        end
        for i in 0:7
            row = 16rt + 2i + hh
            if row < QR
                n = pos + row ÷ Gq + 1            # this row's token attends slots 1..n
                @inbounds scT[klA * QR + row + 1] = k0 + klA < n ? scA[i + 1].value * scale2 : -1f30
                @inbounds scT[klB * QR + row + 1] = k0 + klB < n ? scB[i + 1].value * scale2 : -1f30
            end
        end
    end
    sync_workgroup()
    # softmax of each row over the chunk (a wave per row)
    row = w
    while row < QR
        m = -1f30
        i = l
        while i < DA_CH
            @inbounds m = max(m, scT[i * QR + row + 1])
            i += 32
        end
        for off in (1, 2, 4, 8, 16)
            m = max(m, AMDGPU.Device.shfl_xor(m, Cint(off)))
        end
        ssum = 0f0
        i = l
        while i < DA_CH
            @inbounds e = scT[i * QR + row + 1] <= -1f29 ? 0f0 : exp2(scT[i * QR + row + 1] - m)
            @inbounds scT[i * QR + row + 1] = e
            ssum += e
            i += 32
        end
        for off in (1, 2, 4, 8, 16)
            ssum += AMDGPU.Device.shfl_xor(ssum, Cint(off))
        end
        if l == 0
            @inbounds mq[row + 1] = m; @inbounds lq[row + 1] = ssum
        end
        row += 8
    end
    sync_workgroup()
    if QR > 12
        _da_vhalf!(WO, scT, Vp, tid, min(nmax - k0, DA_CH), k0, c, kvh, b, Val(D), Val(QR))
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
                a0 = _da_fma(a0, p, Float32(vs[u][1])); a1 = _da_fma(a1, p, Float32(vs[u][2]))
            end
            kj += 8 * ngrp
        end
        while kj < nv
            v = @inbounds Vp[dp + 1, k0 + kj + 1, kvh + 1, b + 1]
            p = _da_lds(scT, kj * QR, Val(QR))
            a0 = _da_fma(a0, p, Float32(v[1])); a1 = _da_fma(a1, p, Float32(v[2]))
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
                @inbounds WO[2dp + 1, q, c + 1, kvh + 1, b + 1] = a0[q]
                @inbounds WO[2dp + 2, q, c + 1, kvh + 1, b + 1] = a1[q]
            end
        end
    end
    if tid < QR
        @inbounds WM[tid + 1, c + 1, kvh + 1, b + 1] = mq[tid + 1]
        @inbounds WL[tid + 1, c + 1, kvh + 1, b + 1] = lq[tid + 1]
    end
    return
end
# Many rows (QR > 12, D = 256): two thread groups, each takes half of the rows and all the keys (half the accumulators a thread
# and no reduction across groups; the two groups read the same V). Thread (dp, g): head dims 2dp, 2dp + 1, rows g*QH + 1..
function _da_vhalf!(WO, scT, Vp, tid, nv, k0, c, kvh, b, ::Val{D}, ::Val{QR}) where {D, QR}
    QH = (QR + 1) ÷ 2
    dp = tid % (D ÷ 2); g = tid ÷ (D ÷ 2)
    r0 = g * QH
    a0 = ntuple(_ -> 0f0, Val(QH)); a1 = ntuple(_ -> 0f0, Val(QH))
    kj = 0
    while kj + 7 < nv
        vs = _da_vload8(Vp, dp, k0 + kj, 1, kvh, b)
        @unroll for u in 1:8
            p = _da_lds(scT, (kj + u - 1) * QR + r0, Val(QH))
            a0 = _da_fma(a0, p, Float32(vs[u][1])); a1 = _da_fma(a1, p, Float32(vs[u][2]))
        end
        kj += 8
    end
    while kj < nv
        v = @inbounds Vp[dp + 1, k0 + kj + 1, kvh + 1, b + 1]
        p = _da_lds(scT, kj * QR + r0, Val(QH))
        a0 = _da_fma(a0, p, Float32(v[1])); a1 = _da_fma(a1, p, Float32(v[2]))
        kj += 1
    end
    @unroll for q in 1:QH
        if r0 + q <= QR
            @inbounds WO[2dp + 1, r0 + q, c + 1, kvh + 1, b + 1] = a0[q]
            @inbounds WO[2dp + 2, r0 + q, c + 1, kvh + 1, b + 1] = a1[q]
        end
    end
    return
end
@inline _da_add_exch(a::NTuple{N,Float32}, exch, off, D) where {N} = ntuple(q -> a[q] + @inbounds(exch[(q - 1) * D + off]), Val(N))

# one workgroup per (head, token, sequence): out[:, h, s, b] = the chunks' outputs weighted by exp2(m_c - M) / L
function _da_merge_kernel!(out, WO, WM, WL, posv, S, Gq)
    tid = workitemIdx().x - 1
    h = workgroupIdx().x - 1; sb = workgroupIdx().y - 1
    s = sb % S + 1; b = sb ÷ S
    D = size(out, 1)
    pos = unsafe_trunc(Int, @inbounds posv[length(posv) == 1 ? 1 : b + 1])
    nv = cld(pos + s, DA_CH)
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
        @inbounds out[tid + 1, h + 1, s, b + 1] = acc * stat[1]
    end
    return
end

# out (D, H, S, B) Float32; q (D, S, H, B); the cache pk/pv (D, T, KVH, B) Float16; the new tokens' kn/vn
# (D, S, KVH, B); posv the per-sequence cache length before the step. Returns false if the shapes/arrays
# don't qualify (the caller falls back to the one-workgroup-per-head kernel).
function decode_attention_split!(out, q, pk, pv, kn, vn, posv, scale::Float32, write_cache::Bool)
    (out isa ROCArray{Float32} && pk isa ROCArray{Float16,4} && pv isa ROCArray{Float16,4} && wmma_enabled() &&
     get(ENV, "LUMINAL_DECODE_SPLIT", "1") != "0") || return false
    D, S, H, B = size(q, 1), size(q, 2), size(q, 3), size(q, 4)
    KVH = size(pk, 3); T = size(pk, 2)
    (D in (128, 256) && H % KVH == 0 && size(pk, 1) == D && cld(T, DA_CH) <= 1024) || return false
    Gq = H ÷ KVH
    QR = S * Gq
    QR <= DA_MAX_ROWS && (QR <= 12 || D == 256) || return false
    if write_cache
        @roc groupsize=256 gridsize=cld(D * S * KVH * B, 256) _da_write_kernel!(pk, pv, kn, vn, posv, S)
    end
    nsplit = cld(T, DA_CH)
    WO, WM, WL = get!(DA_WS, (typeof(out), D, nsplit, KVH, B, QR)) do
        (AMDGPU.zeros(Float32, D, QR, nsplit, KVH, B), AMDGPU.zeros(Float32, QR, nsplit, KVH, B), AMDGPU.zeros(Float32, QR, nsplit, KVH, B))
    end
    Kv = reinterpret(WMMA_V8H, pk)
    Vp = reinterpret(NTuple{2,Float16}, pv)
    q4 = q isa ROCArray ? q : copy(q)
    @roc groupsize=256 gridsize=(nsplit, KVH, B) _da_part_kernel!(
        WO, WM, WL, q4, Kv, Vp, posv, scale * 1.4426950408889634f0, S, Gq, Val(D), Val(QR))
    o4 = Base.reshape(out, D, H, S, B)
    @roc groupsize=256 gridsize=(H, S * B) _da_merge_kernel!(o4, WO, WM, WL, posv, S, Gq)
    return true
end
