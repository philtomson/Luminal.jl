# fragments of one K16 step (A: two row tiles, B: four token tiles), and the 8 MMAs on them
@inline function _gp_frags(As, Bs, wr, wt, sl, ks)
    p = ((sl + 2 * ks) & 15) + 1
    a0 = _wmma_cat(As[p, 2wr + 1, ks + 1, 1], As[p, 2wr + 1, ks + 1, 2])
    a1 = _wmma_cat(As[p, 2wr + 2, ks + 1, 1], As[p, 2wr + 2, ks + 1, 2])
    b0 = _wmma_cat(Bs[p, 4wt + 1, ks + 1, 1], Bs[p, 4wt + 1, ks + 1, 2])
    b1 = _wmma_cat(Bs[p, 4wt + 2, ks + 1, 1], Bs[p, 4wt + 2, ks + 1, 2])
    b2 = _wmma_cat(Bs[p, 4wt + 3, ks + 1, 1], Bs[p, 4wt + 3, ks + 1, 2])
    b3 = _wmma_cat(Bs[p, 4wt + 4, ks + 1, 1], Bs[p, 4wt + 4, ks + 1, 2])
    return (a0, a1, b0, b1, b2, b3)
end
@inline function _gp_mma(acc::NTuple{8,WMMA_V8F}, f)
    c0, c1, c2, c3, c4, c5, c6, c7 = acc
    a0, a1, b0, b1, b2, b3 = f
    c0 = _wmma16(a0, b0, c0); c4 = _wmma16(a1, b0, c4)
    c1 = _wmma16(a0, b1, c1); c5 = _wmma16(a1, b1, c5)
    c2 = _wmma16(a0, b2, c2); c6 = _wmma16(a1, b2, c6)
    c3 = _wmma16(a0, b3, c3); c7 = _wmma16(a1, b3, c7)
    return (c0, c1, c2, c3, c4, c5, c6, c7)
end
function _wmma_q4_gp_kernel!(C, Pw, Sc, Mn, Bv, K, N, group, ::Val{SYM}, R, hr::Val, ::Val{V}) where {SYM,V}
    @inbounds begin
    tid = workitemIdx().x - 1
    wave = tid ÷ 32; lane = tid % 32; sl = lane % 16; h = lane ÷ 16
    wr = wave ÷ 4; wt = wave % 4
    bm = (workgroupIdx().y - 1) * 256; bn = (workgroupIdx().x - 1) * 256
    As = @ROCStaticLocalArray(WMMA_V8H, (16, 16, 4, 2), false)     # (row in tile, row tile, k16 step, k half): 16 lanes read 256 contiguous bytes of one half
    Bs = @ROCStaticLocalArray(WMMA_V8H, (16, 16, 4, 2), false)
    acc = ntuple(_ -> _wmma_zero(), Val(8))
    row = tid ÷ 4; kk = tid % 4                                   # this thread's weight row / token and k16 step of every K64 tile
    grow = min(bm + row, _vec_rows(C) * size(C, 1) - 1)    # (C is the vector view: 4 rows of Float32 or 8 of Float16 per element)
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
        # next tile's global loads, in flight during the MMAs. Unconditional (the last iteration re-loads the last tile): the `q + 1 < nk &&`
        # form made the compiler wrap the loads in exec-mask branches (~10 branches and ~100 scalar instructions per iteration; 3-5% slower)
        ld = _g_load(Pw, Sc, Mn, Bv, grow, tok, kk, min(q + 1, nk - 1), group, Val(SYM))
        if V == 0
            acc = _g_ks(acc, As, Bs, wr, wt, sl, 0)
            acc = _g_ks(acc, As, Bs, wr, wt, sl, 1)
            acc = _g_ks(acc, As, Bs, wr, wt, sl, 2)
            dec = _g_decode(ld)
            acc = _g_ks(acc, As, Bs, wr, wt, sl, 3)
        else    # the fragments of step ks + 1 are read from LDS before the MMAs of step ks
            f0 = _gp_frags(As, Bs, wr, wt, sl, 0)
            f1 = _gp_frags(As, Bs, wr, wt, sl, 1)
            acc = _gp_mma(acc, f0)
            f2 = _gp_frags(As, Bs, wr, wt, sl, 2)
            acc = _gp_mma(acc, f1)
            f3 = _gp_frags(As, Bs, wr, wt, sl, 3)
            acc = _gp_mma(acc, f2)
            dec = _g_decode(ld)
            acc = _gp_mma(acc, f3)
        end
        sync_workgroup()
    end
    _wmma_store_big!(C, acc, bm + wr * 32, bn + wt * 64, sl, h, N, R, hr)
    return
    end
end
