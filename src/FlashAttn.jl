# Fused causal self-attention for prefill on gfx11 (RDNA3 / 3.5) matrix cores: online
# softmax over key tiles, so neither the (S, S, heads) score matrix nor the repeated GQA
# keys/values exist. Used through the `CausalAttention` op (see `SelfAttention`).
#
# q (D, S, H, B), k and v (D, S, KVH, B) -> out (D, S, H, B) = softmax(scale q'k + causal mask) v'.
# Block = 4 waves = 32 query rows: wave w handles query rows 16 * (w ÷ 2) + 0:15 and the
# half (w % 2) of the head dimension of the output (both waves of a row group compute the
# same scores, which costs ~1.5x the ideal MMAs and saves any exchange between waves).
# Operands come straight from global memory (L2-resident): Q and K as Float16 with D
# contiguous, V transposed to (keys, D) so a lane's 16 keys are contiguous; the probabilities
# go through a per-wave LDS tile to change from the accumulator to the A-operand layout.
# Key tiles of 64, up to the diagonal of the block's last row. D must be a multiple of 32.

using AMDGPU

@inline _fa_scale8(v::WMMA_V8F, a::NTuple{8,Float32}) = ntuple(i -> VecElement(v[i].value * a[i]), Val(8))
@inline function _fa_max16(x::Float32)   # max over the 16 lanes sharing a half-wave row group
    for m in (1, 2, 4, 8)
        x = max(x, AMDGPU.Device.shfl_xor(x, Cint(m)))
    end
    return x
end
@inline function _fa_sum16(x::Float32)
    for m in (1, 2, 4, 8)
        x += AMDGPU.Device.shfl_xor(x, Cint(m))
    end
    return x
end

function _flash_prefill_kernel!(O, Qv, Kv, Vv, S, H, KVH, scale2, qoff, ::Val{NT}) where {NT}
    D = 32 * NT
    tid = workitemIdx().x - 1
    l = tid % 32; r = l % 16; hh = l ÷ 16; w = tid ÷ 32; rg = w ÷ 2; dh = w % 2
    qblk = workgroupIdx().x - 1; hb = workgroupIdx().y - 1
    head = hb % H; b = hb ÷ H; kvh = head ÷ (H ÷ KVH)
    q0 = qblk * 32 + rg * 16
    qrow = min(q0 + r, S - 1)
    Ps = @ROCStaticLocalArray(Float16, (72, 64), false)
    m = ntuple(_ -> -1f30, Val(8))
    lsum = ntuple(_ -> 0f0, Val(8))
    acc = ntuple(_ -> _wmma_zero(), Val(NT))
    NK = qoff + S                      # keys: the qoff cached ones, then the S queries' own
    ktmax = (qoff + min(S, qblk * 32 + 32) - 1) ÷ 64
    for kt in 0:ktmax
        # scores of the wave's 16 rows against 64 keys: 4 tiles
        s0 = _wmma_zero(); s1 = _wmma_zero(); s2 = _wmma_zero(); s3 = _wmma_zero()
        for ks in 0:2NT-1
            qa = _wmma_cat(Qv[2ks + 1, qrow + 1, head + 1, b + 1], Qv[2ks + 2, qrow + 1, head + 1, b + 1])
            k0 = min(kt * 64 + r, NK - 1) + 1;      k1 = min(kt * 64 + 16 + r, NK - 1) + 1
            k2 = min(kt * 64 + 32 + r, NK - 1) + 1; k3 = min(kt * 64 + 48 + r, NK - 1) + 1
            s0 = _wmma16(qa, _wmma_cat(Kv[2ks + 1, k0, kvh + 1, b + 1], Kv[2ks + 2, k0, kvh + 1, b + 1]), s0)
            s1 = _wmma16(qa, _wmma_cat(Kv[2ks + 1, k1, kvh + 1, b + 1], Kv[2ks + 2, k1, kvh + 1, b + 1]), s1)
            s2 = _wmma16(qa, _wmma_cat(Kv[2ks + 1, k2, kvh + 1, b + 1], Kv[2ks + 2, k2, kvh + 1, b + 1]), s2)
            s3 = _wmma16(qa, _wmma_cat(Kv[2ks + 1, k3, kvh + 1, b + 1], Kv[2ks + 2, k3, kvh + 1, b + 1]), s3)
        end
        # online softmax; lane holds rows 2i + hh of each tile, key column r of tile t
        alpha = m
        mn = m
        for i in 0:7
            row = q0 + 2i + hh
            sv(t, x) = (kk = kt * 64 + 16t + r; (kk <= qoff + row && kk < NK) ? x * scale2 : -1f30)
            e0 = sv(0, s0[i + 1].value); e1 = sv(1, s1[i + 1].value)
            e2 = sv(2, s2[i + 1].value); e3 = sv(3, s3[i + 1].value)
            tm = _fa_max16(max(max(e0, e1), max(e2, e3)))
            mi = m[i + 1]; mnew = max(mi, tm)
            al = exp2(mi - mnew)
            p0 = e0 <= -1f29 ? 0f0 : exp2(e0 - mnew); p1 = e1 <= -1f29 ? 0f0 : exp2(e1 - mnew)
            p2 = e2 <= -1f29 ? 0f0 : exp2(e2 - mnew); p3 = e3 <= -1f29 ? 0f0 : exp2(e3 - mnew)
            prow = w * 16 + 2i + hh + 1
            Ps[r + 1, prow] = Float16(p0);      Ps[17 + r, prow] = Float16(p1)
            Ps[33 + r, prow] = Float16(p2);     Ps[49 + r, prow] = Float16(p3)
            mn = Base.setindex(mn, mnew, i + 1); alpha = Base.setindex(alpha, al, i + 1)
            lsum = Base.setindex(lsum, lsum[i + 1] * al + (p0 + p1) + (p2 + p3), i + 1)
        end
        m = mn
        acc = let a = acc, al = alpha
            ntuple(j -> _fa_scale8(a[j], al), Val(NT))
        end
        sync_workgroup()
        # acc += P V over the 4 chunks of 16 keys
        for c in 0:3
            pa = ntuple(j -> VecElement(Ps[16c + j, w * 16 + r + 1]), Val(16))
            kc = 2 * (kt * 4 + c)
            acc = let a = acc
                ntuple(Val(NT)) do j
                    d = dh * (D ÷ 2) + (j - 1) * 16 + r + 1
                    vb = _wmma_cat(Vv[kc + 1, d, kvh + 1, b + 1], Vv[kc + 2, d, kvh + 1, b + 1])
                    _wmma16(pa, vb, a[j])
                end
            end
        end
        sync_workgroup()
    end
    for i in 0:7
        row = q0 + 2i + hh
        inv = 1f0 / _fa_sum16(lsum[i + 1])
        row < S || continue
        for j in 0:NT-1
            d = dh * (D ÷ 2) + 16j + r
            O[d + 1, row + 1, head + 1, b + 1] = acc[j + 1][i + 1].value * inv
        end
    end
    return
end

# Flash prefill attention with K and V staged in LDS (design after gufo's WmmaCausalAttention, ported to Julia).
# Block = 256 threads = 8 waves, 64 rows = 2 query heads (sharing a KV head) x 32 queries, one 16-key tile per iteration:
#   S = Q K^T: wave w computes the 16x16 score tile of row block rb = w % 4 over the dim half w ÷ 4 (Q fragments stay in registers);
#   the two halves are summed by the softmax reader (LDS); softmax: 4 threads per row; O += P V: wave w owns dim tiles w and w + 8
#   for all 4 row blocks. K tile: coalesced into padded LDS; V tile: one key per lane, written TRANSPOSED to LDS (V^T[d][key]).
#   The next tile's K/V are loaded into registers right after the K staging barrier.
@inline function _lds_frag(arr, i)           # 16 contiguous Float16 starting at 1-based element i (a multiple of 8) -> WMMA fragment
    p = reinterpret(Core.LLVMPtr{WMMA_V8H,3}, pointer(arr))
    a = unsafe_load(p, (i - 1) ÷ 8 + 1, Val(16)); b = unsafe_load(p, (i - 1) ÷ 8 + 2, Val(16))
    return _wmma_cat(a, b)
end

function _flash_prefill_lds!(O, Qv, Kc, Vc, S, H, KVH, scale2, qoff, ::Val{NT}) where {NT}
    D = 32 * NT; NC = D ÷ 8                   # 16-byte chunks per key row
    KS = NT                                   # k steps (16 dims) per wave of the S phase: half of D / 16
    DT = NT ÷ 4                               # output dim tiles per wave (D / 16 / 8) = V chunks per thread
    KR = NC ÷ 16                              # K chunks per thread: 16 keys x NC chunks / 256 threads
    tid = workitemIdx().x - 1
    lane = tid % 32; sub = lane % 16; half_id = lane ÷ 16; wave = tid ÷ 32
    qblk = workgroupIdx().x - 1; hp = workgroupIdx().y - 1
    npair = H ÷ 2
    b = hp ÷ npair; pair = hp % npair
    first_head = 2 * pair; kvh = first_head ÷ (H ÷ KVH)
    query_start = qblk * 32
    NK = qoff + S
    max_visible = min(NK, qoff + query_start + 32)
    ntile = cld(max_visible, 16)
    s_tile = wave % 4; s_kh = wave ÷ 4         # this wave's score tile (row block) and dim half
    s_head = first_head + s_tile % 2; s_qoff = (s_tile ÷ 2) * 16

    Ks = @ROCStaticLocalArray(WMMA_V8H, (NC + 1, 16), false)
    Vt = @ROCStaticLocalArray(Float16, (24, D), false)
    s_lds = @ROCStaticLocalArray(Float32, (17, 16, 4, 2), false)
    p_lds = @ROCStaticLocalArray(Float16, (24, 64), false)
    row_scale = @ROCStaticLocalArray(Float32, (64,), false)
    row_sum = @ROCStaticLocalArray(Float32, (64,), false)

    @inbounds begin
    qrow = min(query_start + s_qoff + sub, S - 1)
    qf = ntuple(Val(KS)) do ks
        c = 2 * (s_kh * KS + ks - 1)
        _wmma_cat(Qv[c + 1, qrow + 1, s_head + 1, b + 1], Qv[c + 2, qrow + 1, s_head + 1, b + 1])
    end
    oacc = ntuple(_ -> _wmma_zero(), Val(4 * DT))
    running_max = -Inf32; running_sum = 0f0
    zero8 = ntuple(_ -> VecElement(Float16(0)), Val(8))

    v_key = lane % 16; v_slice = (tid ÷ 16) * DT                 # V: lane's key, and the first of its DT chunks
    # first tile's K and V
    kcur = ntuple(n -> (idx = tid + 256 * (n - 1); Kc[idx % NC + 1, min(idx ÷ NC, NK - 1) + 1, kvh + 1, b + 1]), Val(KR))
    vlive = v_key < NK
    vcur = ntuple(j -> vlive ? Vc[v_slice + j, v_key + 1, kvh + 1, b + 1] : zero8, Val(DT))

    for kt in 0:ntile - 1
        key_start = kt * 16
        sync_workgroup()
        for n in 1:KR
            idx = tid + 256 * (n - 1)
            Ks[idx % NC + 1, idx ÷ NC + 1] = kcur[n]
        end
        sync_workgroup()
        # prefetch the next tile (clamped, harmless past the end)
        nstart = key_start + 16
        kpre = ntuple(n -> (idx = tid + 256 * (n - 1); Kc[idx % NC + 1, min(nstart + idx ÷ NC, NK - 1) + 1, kvh + 1, b + 1]), Val(KR))
        nkey = nstart + v_key; nlive = nkey < NK
        vpre = ntuple(j -> nlive ? Vc[v_slice + j, nkey + 1, kvh + 1, b + 1] : zero8, Val(DT))
        # S = Q K^T over this wave's half of the dims
        s_acc = _wmma_zero()
        for ks in 0:KS - 1
            c = 2 * (s_kh * KS + ks)
            kf = _wmma_cat(Ks[c + 1, sub + 1], Ks[c + 2, sub + 1])
            s_acc = _wmma16(qf[ks + 1], kf, s_acc)
        end
        for i in 0:7
            s_lds[sub + 1, 2i + half_id + 1, s_tile + 1, s_kh + 1] = s_acc[i + 1].value
        end
        sync_workgroup()
        # online softmax: 4 threads per row, 4 keys each
        rg = tid ÷ 4; seg = tid % 4
        rb = rg ÷ 16; row = rg % 16
        lq = query_start + (rb ÷ 2) * 16 + row
        v0 = -Inf32; v1 = -Inf32; v2 = -Inf32; v3 = -Inf32
        col = seg * 4
        t4 = col ÷ 16; cc = col % 16
        base_valid = lq < S
        kpos = key_start + col
        qabs = qoff + lq
        sv(m) = (base_valid && kpos + m <= qabs) ? (s_lds[cc + m + 1, row + 1, rb + 1, 1] + s_lds[cc + m + 1, row + 1, rb + 1, 2]) * scale2 : -Inf32
        v0 = sv(0); v1 = sv(1); v2 = sv(2); v3 = sv(3)
        pmax = max(max(v0, v1), max(v2, v3))
        pmax = max(pmax, AMDGPU.Device.shfl_xor(pmax, Cint(1)))
        pmax = max(pmax, AMDGPU.Device.shfl_xor(pmax, Cint(2)))
        nmax = max(running_max, pmax)
        prior = isfinite(running_max) ? exp2(running_max - nmax) : 0f0
        w0 = isfinite(v0) ? exp2(v0 - nmax) : 0f0; w1 = isfinite(v1) ? exp2(v1 - nmax) : 0f0
        w2 = isfinite(v2) ? exp2(v2 - nmax) : 0f0; w3 = isfinite(v3) ? exp2(v3 - nmax) : 0f0
        p_lds[col + 1, rg + 1] = Float16(w0); p_lds[col + 2, rg + 1] = Float16(w1)
        p_lds[col + 3, rg + 1] = Float16(w2); p_lds[col + 4, rg + 1] = Float16(w3)
        psum = (w0 + w1) + (w2 + w3)
        psum += AMDGPU.Device.shfl_xor(psum, Cint(1)); psum += AMDGPU.Device.shfl_xor(psum, Cint(2))
        running_max = nmax
        running_sum = running_sum * prior + psum
        seg == 0 && (row_scale[rg + 1] = prior)
        sync_workgroup()
        # rescale the output accumulators
        oacc = let o = oacc
            ntuple(Val(4 * DT)) do n
                rbn = (n - 1) ÷ DT
                _fa_scale8(o[n], ntuple(i -> row_scale[rbn * 16 + 2 * (i - 1) + half_id + 1], Val(8)))
            end
        end
        # stage V transposed: V^T[d][key], this thread's key and its 16 dims
        for j in 1:DT, i in 0:7
            Vt[v_key + 1, (v_slice + j - 1) * 8 + i + 1] = vcur[j][i + 1].value
        end
        sync_workgroup()
        # O += P V: dim tiles wave + 8t for all 4 row blocks
        for t in 0:DT - 1
            dim = (wave + 8t) * 16 + sub
            vf = _lds_frag(Vt, dim * 24 + 1)
            oacc = let o = oacc
                ntuple(Val(4 * DT)) do n
                    rbn = (n - 1) ÷ DT; tn = (n - 1) % DT
                    tn == t ? _wmma16(_lds_frag(p_lds, (rbn * 16 + sub) * 24 + 1), vf, o[n]) : o[n]
                end
            end
        end
        kcur = kpre; vcur = vpre
    end
    tid % 4 == 0 && (row_sum[tid ÷ 4 + 1] = running_sum)
    sync_workgroup()
    for n in 1:4 * DT
        rbn = (n - 1) ÷ DT; tn = (n - 1) % DT
        head = first_head + rbn % 2
        for i in 0:7
            row = 2i + half_id
            lq = query_start + (rbn ÷ 2) * 16 + row
            if lq < S
                den = row_sum[rbn * 16 + row + 1]
                d = (wave + 8tn) * 16 + sub
                O[d + 1, lq + 1, head + 1, b + 1] = den > 0f0 ? oacc[n][i + 1].value / den : 0f0
            end
        end
    end
    end # inbounds
    return
end

# The LDS kernel needs D % 128 == 0 (its 8 waves split the output dims in 16-wide tiles) and an even number of query heads per KV head.
_lds_attn_ok(D, H, KVH) = D % 128 == 0 && H % KVH == 0 && (H ÷ KVH) % 2 == 0

# out (D, S, H, B) Float32 <- attention over q (D, S, H, B), k, v (D, S, KVH, B) (any real eltype,
# arrays or views); D % 32 == 0, S >= 1.
function flash_prefill!(out::ROCArray{Float32,4}, q, k, v, scale::Float32)
    D, S, H, B = size(q); KVH = size(k, 3)
    D % 32 == 0 && H % KVH == 0 && size(out) == size(q) ||
        throw(ArgumentError("flash_prefill!: need D % 32 == 0 and H % KVH == 0"))
    q16 = similar(out, Float16); q16 .= q
    k16 = similar(out, Float16, D, S, KVH, B); k16 .= k
    scale2 = scale * 1.4426950408889634f0
    if _lds_attn_ok(D, H, KVH)
        v16 = similar(out, Float16, D, S, KVH, B); v16 .= v
        @roc groupsize=256 gridsize=(cld(S, 32), (H ÷ 2) * B) _flash_prefill_lds!(
            out, reinterpret(WMMA_V8H, q16), reinterpret(WMMA_V8H, k16), reinterpret(WMMA_V8H, v16), S, H, KVH, scale2, 0, Val(D ÷ 32))
        AMDGPU.unsafe_free!(v16)
    else
        Spad = cld(S, 64) * 64
        vt = AMDGPU.zeros(Float16, Spad, D, KVH, B)
        view(vt, 1:S, :, :, :) .= PermutedDimsArray(v, (2, 1, 3, 4))
        Qv = reinterpret(WMMA_V8H, q16); Kv = reinterpret(WMMA_V8H, k16); Vv = reinterpret(WMMA_V8H, vt)
        @roc groupsize=128 gridsize=(cld(S, 32), H * B) _flash_prefill_kernel!(out, Qv, Kv, Vv, S, H, KVH, scale2, 0, Val(D ÷ 32))
        AMDGPU.unsafe_free!(vt)
    end
    AMDGPU.unsafe_free!(q16); AMDGPU.unsafe_free!(k16)
    return out
end

# Reference (CPU arrays, or any other device through the host).
function causal_attention_ref(q, k, v, scale::Float32)
    q = Array{Float32}(q); k = Array{Float32}(k); v = Array{Float32}(v)
    D, S, H, B = size(q); KVH = size(k, 3); G = H ÷ KVH
    out = similar(q)
    for b in 1:B, h in 1:H
        kv = (h - 1) ÷ G + 1
        sc = (q[:, :, h, b]' * k[:, :, kv, b]) .* scale          # (queries, keys)
        for j in 1:S, i in 1:S
            j > i && (sc[i, j] = -Inf32)
        end
        sc .-= maximum(sc; dims = 2)
        p = exp.(sc); p ./= sum(p; dims = 2)
        out[:, :, h, b] = v[:, :, kv, b] * p'
    end
    return out
end


# vt[s, d, h] = pv[d, s, h] for s <= NK: the V prefix transposed to (keys, D) for `_flash_prefill_kernel!`, through a 32x32 LDS tile
# (reads contiguous along d, writes contiguous along s). Groups of 32x8 threads; grid (key tiles, D tiles, KV heads).
function _vtranspose_kernel!(vt, pv, NK)
    D = size(pv, 1)
    tx = workitemIdx().x - 1; ty = workitemIdx().y - 1
    s0 = (workgroupIdx().x - 1) * 32; d0 = (workgroupIdx().y - 1) * 32; h = workgroupIdx().z
    tile = @ROCStaticLocalArray(Float16, (33, 32), false)
    for j in 0:8:24
        d = d0 + tx + 1; s = s0 + ty + j + 1
        if d <= D && s <= NK
            @inbounds tile[tx + 1, ty + j + 1] = pv[d, s, h, 1]
        end
    end
    sync_workgroup()
    for j in 0:8:24
        s = s0 + tx + 1; d = d0 + ty + j + 1
        if d <= D && s <= NK
            @inbounds vt[s, d, h, 1] = tile[ty + j + 1, tx + 1]
        end
    end
    return
end

# Attention of a chunk of S new tokens (a prefill chunk) over the cache: the chunk's K/V are written to
# slots pos + 1.. of the Float16 cache pk/pv (D, T, KVH, B) first, then query i attends to slots 1..pos + i.
# q (D, S, H, B) Float32; writes out (D, H, S, B). Sequence b's pos is read from posv on the host (a chunk is
# a big step, and this is not captured in a graph).
function chunk_attention!(out, q, pk, pv, kn, vn, posv, scale::Float32)
    D, S, H, B = size(q); KVH = size(pk, 3)
    q8 = pk isa ROCArray{Int8,4} && pv isa ROCArray{Int8,4}
    (D % 32 == 0 && H % KVH == 0 && B == 1 && (q8 || (pk isa ROCArray{Float16,4} && pv isa ROCArray{Float16,4}))) || return false
    ksc = vsc = nothing
    if q8
        ksc = reinterpret(Float32, pk); vsc = reinterpret(Float32, pv)
        @roc groupsize=32 gridsize=S * KVH * B _da_write_q8_kernel!(pk, pv, ksc, vsc, kn, vn, posv, S)
    else
        @roc groupsize=256 gridsize=cld(D * S * KVH * B, 256) _da_write_kernel!(pk, pv, kn, vn, posv, S)
    end
    o4 = Base.reshape(out, D, H, S, B)
    posh = Int.(round.(Array(posv)))
    for b in 1:B
        pos = posh[length(posh) == 1 ? 1 : b]
        NK = pos + S
        q16 = AMDGPU.zeros(Float16, D, S, H, 1); q16 .= q
        ob = AMDGPU.zeros(Float32, D, S, H, 1)
        scale2 = scale * 1.4426950408889634f0
        if _lds_attn_ok(D, H, KVH)
            kt, vd = pk, pv                        # the cache itself: the kernel stages K and V tiles through LDS
            if q8                                   # attend to the dequantized cache (the same values the decode kernel uses)
                kt = AMDGPU.zeros(Float16, D, NK, KVH, 1); vd = AMDGPU.zeros(Float16, D, NK, KVH, 1)
                @roc groupsize=256 gridsize=cld(D * NK * KVH, 256) _kv_dequant_kernel!(kt, vd, pk, pv, ksc, vsc, NK, Val(false))
            end
            @roc groupsize=256 gridsize=(cld(S, 32), H ÷ 2) _flash_prefill_lds!(
                ob, reinterpret(WMMA_V8H, q16), reinterpret(WMMA_V8H, kt), reinterpret(WMMA_V8H, vd), S, H, KVH, scale2, pos, Val(D ÷ 32))
            q8 && (AMDGPU.unsafe_free!(kt); AMDGPU.unsafe_free!(vd))
        else
            Spad = cld(NK, 64) * 64
            vt = AMDGPU.zeros(Float16, Spad, D, KVH, 1)
            kt = pk
            if q8
                kt = AMDGPU.zeros(Float16, D, Spad, KVH, 1)
                @roc groupsize=256 gridsize=cld(D * NK * KVH, 256) _kv_dequant_kernel!(kt, vt, pk, pv, ksc, vsc, NK, Val(true))
            else
                @roc groupsize=(32, 8) gridsize=(cld(NK, 32), cld(D, 32), KVH) _vtranspose_kernel!(vt, pv, NK)
            end
            Qv = reinterpret(WMMA_V8H, q16); Kb = reinterpret(WMMA_V8H, kt); Vv = reinterpret(WMMA_V8H, vt)
            @roc groupsize=128 gridsize=(cld(S, 32), H) _flash_prefill_kernel!(
                ob, Qv, Kb, Vv, S, H, KVH, scale2, pos, Val(D ÷ 32))
            AMDGPU.unsafe_free!(vt); q8 && AMDGPU.unsafe_free!(kt)
        end
        o4 .= PermutedDimsArray(ob, (1, 3, 2, 4))
        AMDGPU.unsafe_free!(q16); AMDGPU.unsafe_free!(ob)
    end
    return true
end
