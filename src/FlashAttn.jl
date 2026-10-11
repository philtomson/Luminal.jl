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

# out (D, S, H, B) Float32 <- attention over q (D, S, H, B), k, v (D, S, KVH, B) (any real eltype,
# arrays or views); D % 32 == 0, S >= 1.
function flash_prefill!(out::ROCArray{Float32,4}, q, k, v, scale::Float32)
    D, S, H, B = size(q); KVH = size(k, 3)
    D % 32 == 0 && H % KVH == 0 && size(out) == size(q) ||
        throw(ArgumentError("flash_prefill!: need D % 32 == 0 and H % KVH == 0"))
    q16 = similar(out, Float16); q16 .= q
    k16 = similar(out, Float16, D, S, KVH, B); k16 .= k
    Spad = cld(S, 64) * 64
    vt = AMDGPU.zeros(Float16, Spad, D, KVH, B)
    view(vt, 1:S, :, :, :) .= PermutedDimsArray(v, (2, 1, 3, 4))
    Qv = reinterpret(WMMA_V8H, q16); Kv = reinterpret(WMMA_V8H, k16); Vv = reinterpret(WMMA_V8H, vt)
    @roc groupsize=128 gridsize=(cld(S, 32), H * B) _flash_prefill_kernel!(
        out, Qv, Kv, Vv, S, H, KVH, scale * 1.4426950408889634f0, 0, Val(D ÷ 32))
    AMDGPU.unsafe_free!(q16); AMDGPU.unsafe_free!(k16); AMDGPU.unsafe_free!(vt)
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


# Attention of a chunk of S new tokens (a prefill chunk) over the cache: the chunk's K/V are written to
# slots pos + 1.. of the Float16 cache pk/pv (D, T, KVH, B) first, then query i attends to slots 1..pos + i.
# q (D, S, H, B) Float32; writes out (D, H, S, B). Sequence b's pos is read from posv on the host (a chunk is
# a big step, and this is not captured in a graph).
function chunk_attention!(out, q, pk, pv, kn, vn, posv, scale::Float32)
    D, S, H, B = size(q); KVH = size(pk, 3)
    (D % 32 == 0 && H % KVH == 0 && B == 1 && pk isa ROCArray{Float16,4} && pv isa ROCArray{Float16,4}) || return false
    @roc groupsize=256 gridsize=cld(D * S * KVH * B, 256) _da_write_kernel!(pk, pv, kn, vn, posv, S)
    o4 = Base.reshape(out, D, H, S, B)
    posh = Int.(round.(Array(posv)))
    for b in 1:B
        pos = posh[length(posh) == 1 ? 1 : b]
        NK = pos + S
        q16 = AMDGPU.zeros(Float16, D, S, H, 1); q16 .= q
        Spad = cld(NK, 64) * 64
        vt = AMDGPU.zeros(Float16, Spad, D, KVH, 1)
        view(vt, 1:NK, :, :, :) .= PermutedDimsArray(view(pv, :, 1:NK, :, :), (2, 1, 3, 4))
        Qv = reinterpret(WMMA_V8H, q16); Kb = reinterpret(WMMA_V8H, pk); Vv = reinterpret(WMMA_V8H, vt)
        ob = AMDGPU.zeros(Float32, D, S, H, 1)
        @roc groupsize=128 gridsize=(cld(S, 32), H) _flash_prefill_kernel!(
            ob, Qv, Kb, Vv, S, H, KVH, scale * 1.4426950408889634f0, pos, Val(D ÷ 32))
        o4 .= PermutedDimsArray(ob, (1, 3, 2, 4))
        AMDGPU.unsafe_free!(q16); AMDGPU.unsafe_free!(vt); AMDGPU.unsafe_free!(ob)
    end
    return true
end
