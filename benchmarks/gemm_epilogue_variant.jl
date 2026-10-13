@inline function _abl_store16!(Cv, acc, m0, n0, r, h, N)
    for j in 0:3
        col = n0 + 16j + r
        for i in 0:1
            a = acc[4i + j + 1]
            lo = h == 0
            v1, v2, v3, v4, v5, v6, v7, v8 = a[1].value, a[2].value, a[3].value, a[4].value, a[5].value, a[6].value, a[7].value, a[8].value
            q1 = AMDGPU.Device.shfl_xor(ifelse(lo, v5, v1), Cint(16)); q2 = AMDGPU.Device.shfl_xor(ifelse(lo, v6, v2), Cint(16))
            q3 = AMDGPU.Device.shfl_xor(ifelse(lo, v7, v3), Cint(16)); q4 = AMDGPU.Device.shfl_xor(ifelse(lo, v8, v4), Cint(16))
            f1 = ifelse(lo, v1, q1); f2 = ifelse(lo, q1, v5); f3 = ifelse(lo, v2, q2); f4 = ifelse(lo, q2, v6)
            f5 = ifelse(lo, v3, q3); f6 = ifelse(lo, q3, v7); f7 = ifelse(lo, v4, q4); f8 = ifelse(lo, q4, v8)
            if col < N
                Cv[(m0 + 16i + 8h) ÷ 8 + 1, col + 1] = (VecElement(Float16(f1)), VecElement(Float16(f2)), VecElement(Float16(f3)), VecElement(Float16(f4)),
                                                         VecElement(Float16(f5)), VecElement(Float16(f6)), VecElement(Float16(f7)), VecElement(Float16(f8)))
            end
        end
    end
end
function _wmma_q4_gs_kernel!(C, Pw, Sc, Mn, Bv, K, N, group, ::Val{SYM}, R, hr::Val, ::Val{GS}, ::Val{EP}, C16) where {SYM,GS,EP}
    @inbounds begin
    tid = workitemIdx().x - 1
    wave = tid ÷ 32; lane = tid % 32; sl = lane % 16; h = lane ÷ 16
    wr = wave ÷ 4; wt = wave % 4
    gx = gridGroupDim().x; gy = gridGroupDim().y
    lin = (workgroupIdx().y - 1) * gx + (workgroupIdx().x - 1)
    if GS > 0
        grp = lin ÷ (GS * gx); r0 = lin - grp * GS * gx
        gh = min(GS, gy - grp * GS)
        ty = grp * GS + r0 % gh; tx = r0 ÷ gh
    else
        ty = workgroupIdx().y - 1; tx = workgroupIdx().x - 1
    end
    bm = ty * 256; bn = tx * 256
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
    if EP == 0
        _wmma_store_big!(C, acc, bm + wr * 32, bn + wt * 64, sl, h, N, R, hr)
    elseif EP == 1       # no store: keep the accumulators live with one conditional write
        s = 0f0
        for t in 1:8, e in 1:8; s += acc[t][e].value; end
        s == 1.2345679f0 && (C[1, 1] = (VecElement(s), VecElement(s), VecElement(s), VecElement(s)))
    else                 # Float16 output: one 16-byte store of 8 rows per tile
        _abl_store16!(C16, acc, bm + wr * 32, bn + wt * 64, sl, h, N)
    end
    return
    end
end
