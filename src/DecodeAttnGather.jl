# Gather variant of the split-K decode attention (DecodeAttn.jl), for Latent Sparse Attention: one decode token reads the keys and
# values a list of cache slots names, instead of the cache's first `pos + 1`. The cache stays where it is - no gathered copy.
#
# Same structure as `_da_part_kernel!`: the list is cut into chunks of DA_CH entries and every (chunk, KV head, sequence) is a workgroup
# that computes its chunk's scores with WMMA, softmaxes them in LDS and sums the probability-weighted V. The merge kernel is
# `_da_merge_kernel!` unchanged, fed `count - 1` as the position. What differs from the dense kernel:
#   - a chunk's 256 slots are read from `ids` once, into LDS, and every K and V load goes through them;
#   - a list entry is valid if its index is below the sequence's count, not below a position;
#   - one token a step (the query heads of a KV group are the rows), a Float16 cache, D = 256.
# Slots past a sequence's count are never read (the staging clamps to the last valid entry, and the scores of those entries are masked).
#
# This file is `include`d after DecodeAttn.jl and uses its helpers.

# 8 keys' V loads in flight together, each through the chunk's staged slots (`sl`, 1-based slots) - kj is the list offset in the chunk
@inline _dg_vload8(Vp, sl, dp, kj, ngrp, kvh, b) =
    ntuple(u -> @inbounds(Vp[dp + 1, sl[kj + (u - 1) * ngrp + 1], kvh + 1, b + 1]), Val(8))

function _dg_part_kernel!(WO, WM, WL, Qa, Kv, Vp, ids, lastpos, scale2, Gq, ::Val{D}, ::Val{QR}) where {D, QR}
    tid = workitemIdx().x - 1
    l = tid % 32; r = l % 16; hh = l ÷ 16; w = tid ÷ 32
    c = workgroupIdx().x - 1
    kvh = workgroupIdx().y - 1; b = workgroupIdx().z - 1
    nmax = unsafe_trunc(Int, @inbounds lastpos[length(lastpos) == 1 ? 1 : b + 1]) + 1     # entries of the list that are valid
    k0 = c * DA_CH
    k0 < nmax || return
    qs = @ROCStaticLocalArray(Float16, (256, 32), false)
    scT = @ROCStaticLocalArray(Float32, (6144,), false)      # (key, row) = DA_CH * DA_MAX_ROWS
    exch = QR > 8 ? scT : (@ROCStaticLocalArray(Float32, (4608,), false))
    mq = @ROCStaticLocalArray(Float32, (24,), false)
    lq = @ROCStaticLocalArray(Float32, (24,), false)
    red = @ROCStaticLocalArray(Float32, (256,), false)
    sl = @ROCStaticLocalArray(Int32, (256,), false)           # the chunk's slots, 1-based
    @inbounds sl[tid + 1] = ids[min(k0 + tid, nmax - 1) + 1, b + 1] + Int32(1)
    # stage the queries: row = the query head of the group; rows >= QR are zero
    idx = tid
    while idx < 32 * D
        rr = idx ÷ D; d = idx % D
        @inbounds qs[d + 1, rr + 1] = rr < QR ? Float16(Qa[d + 1, 1, kvh * Gq + rr + 1, b + 1]) : Float16(0)
        idx += 256
    end
    sync_workgroup()
    # scores: the wave's two tiles (tt = w, w + 8) of 16 keys, K loads issued ahead of the WMMAs
    klA = w * 16 + r; klB = (w + 8) * 16 + r
    kkA = @inbounds Int(sl[klA + 1]); kkB = @inbounds Int(sl[klB + 1])
    for rt in 0:(QR - 1) ÷ 16
        scA = _wmma_zero(); scB = _wmma_zero()
        for ks in 0:2:D ÷ 16 - 1
            bA0 = _da_kb(Val(false), Kv, ks, kkA, kvh, b); bB0 = _da_kb(Val(false), Kv, ks, kkB, kvh, b)
            bA1 = _da_kb(Val(false), Kv, ks + 1, kkA, kvh, b); bB1 = _da_kb(Val(false), Kv, ks + 1, kkB, kvh, b)
            qa0 = _da_qa(qs, ks, 16rt + r + 1); qa1 = _da_qa(qs, ks + 1, 16rt + r + 1)
            scA = _wmma16(qa0, bA0, scA); scB = _wmma16(qa0, bB0, scB)
            scA = _wmma16(qa1, bA1, scA); scB = _wmma16(qa1, bB1, scB)
        end
        for i in 0:7
            row = 16rt + 2i + hh
            if row < QR
                @inbounds scT[klA * QR + row + 1] = k0 + klA < nmax ? scA[i + 1].value * scale2 : -1f30
                @inbounds scT[klB * QR + row + 1] = k0 + klB < nmax ? scB[i + 1].value * scale2 : -1f30
            end
        end
    end
    sync_workgroup()
    # softmax of each row over the chunk, as in the dense kernel
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
            @inbounds scT[kk * QR + srow + 1] = e
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
    # out[row, d] = sum_key p[row, key] V[d, slot(key)]: thread (dp, g) takes the key subset g, d pair dp
    ndp = D ÷ 2
    ngrp = 256 ÷ ndp
    dp = tid % ndp; g = tid ÷ ndp
    nv = min(nmax - k0, DA_CH)
    a0 = ntuple(_ -> 0f0, Val(QR)); a1 = ntuple(_ -> 0f0, Val(QR))
    kj = g
    while kj + 7 * ngrp < nv               # 8 keys at a time: their V loads are in flight together
        vs = _dg_vload8(Vp, sl, dp, kj, ngrp, kvh, b)
        @unroll for u in 1:8
            p = _da_lds(scT, (kj + (u - 1) * ngrp) * QR, Val(QR))
            a0 = _da_fma(a0, p, _da_v1(vs[u])); a1 = _da_fma(a1, p, _da_v2(vs[u]))
        end
        kj += 8 * ngrp
    end
    while kj < nv
        v = @inbounds Vp[dp + 1, sl[kj + 1], kvh + 1, b + 1]
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
            @inbounds WO[2dp + 1, q, c + 1, kvh + 1, b + 1] = a0[q]
            @inbounds WO[2dp + 2, q, c + 1, kvh + 1, b + 1] = a1[q]
        end
    end
    if tid < QR
        @inbounds WM[tid + 1, c + 1, kvh + 1, b + 1] = mq[tid + 1]
        @inbounds WL[tid + 1, c + 1, kvh + 1, b + 1] = lq[tid + 1]
    end
    return
end

"""
    decode_attention_gather!(out, q, pk, pv, ids, lastpos, scale) -> Bool

One decode token's attention over the cache slots a list names. `out` (D, H, 1, B) Float32; `q` (D, 1, H, B) Float32; the whole cache
`pk`, `pv` (D, T, KVH, B) Float16, read in place; `ids` (W, B) Int32, 0-based slots (entries past a sequence's count are never read);
`lastpos` (B,) Float32, one less than the number of valid entries in each sequence's list (the decode kernel's `posv` convention).
Needs D = 256, a single token and at most 12 query heads a KV group. Returns false if the arrays or shapes don't qualify.
"""
function decode_attention_gather!(out, q, pk, pv, ids, lastpos, scale::Float32)
    (out isa ROCArray{Float32} && pk isa ROCArray{Float16,4} && pv isa ROCArray{Float16,4} && ids isa ROCArray{Int32,2} &&
     lastpos isa ROCArray{Float32} && wmma_enabled()) || return false
    D, S, H, B = size(q, 1), size(q, 2), size(q, 3), size(q, 4)
    KVH = size(pk, 3)
    (D == 256 && S == 1 && H % KVH == 0 && size(pk, 1) == D && size(pv) == size(pk) && size(pk, 4) == B &&
     size(ids, 2) == B && length(lastpos) in (1, B)) || return false
    Gq = H ÷ KVH
    Gq <= 12 || return false
    nsplit = cld(size(ids, 1), DA_CH)
    nsplit <= 1024 || return false
    WT = _da_wo_f16() ? Float16 : Float32
    WO, WM, WL = get!(DA_WS, (:gather, typeof(out), D, nsplit, KVH, B, Gq, WT)) do
        (AMDGPU.zeros(WT, D, Gq, nsplit, KVH, B), AMDGPU.zeros(Float32, Gq, nsplit, KVH, B), AMDGPU.zeros(Float32, Gq, nsplit, KVH, B))
    end
    Kv = reinterpret(WMMA_V8H, pk)
    Vp = reinterpret(NTuple{2,Float16}, pv)
    o4 = Base.reshape(out, D, H, 1, B)
    @roc groupsize=256 gridsize=(nsplit, KVH, B) _dg_part_kernel!(
        WO, WM, WL, q, Kv, Vp, ids, lastpos, scale * 1.4426950408889634f0, Gq, Val(D), Val(Gq))
    @roc groupsize=256 gridsize=(H, B) _da_merge_kernel!(o4, WO, WM, WL, lastpos, 1, 0, Gq)
    return true
end
