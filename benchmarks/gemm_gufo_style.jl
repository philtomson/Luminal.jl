# Experimental int4 GEMM with gufo's HalfPrefillGemmKernel structure (see TODO.md, "What gufo's prefill does"): not wired in.
# include into module Luminal:  Base.include(Luminal, "benchmarks/gemm_gufo_style.jl"); launch with groupsize=1024, gridsize=(Npad÷256, M÷256)
# (M % 256 == 0, K % 64 == 0, Npad % 256 == 0, power-of-two group): L._wmma_q4_g_kernel!(Cv, Pw, scale, mn, Bv, K, N, group, Val(sym), Rv, Val(has_residual))
# gufo-style int4 GEMM: 256 x 256 block, 32 waves (8 x 4, a 32 x 64 wave tile), K64 per iteration, one LDS stage (64 KB) filled from registers
# between two barriers; every wave stages its share of the next tile while the matrix units run the current one.
@inline function _g_load(Pw, Sc, Mn, Bv, grow, tok, kk, q, group, ::Val{SYM}) where {SYM}
    gi = (64 * q + 16 * kk) >> trailing_zeros(group) + 1
    sa = Float32(Sc[gi, grow + 1])
    ma = SYM ? -8f0 * sa : Float32(Mn[gi, grow + 1])
    return (Pw[8q + 2kk + 1, grow + 1], Pw[8q + 2kk + 2, grow + 1], sa, ma, Bv[8q + 2kk + 1, tok + 1], Bv[8q + 2kk + 2, tok + 1])
end
@inline function _g_ks(acc::NTuple{8,WMMA_V8F}, As, Bs, wr, wt, sl, ks)
    c0, c1, c2, c3, c4, c5, c6, c7 = acc      # (A tile i, B tile j) is acc[4i + j + 1]
    p = ((sl + 2 * ks) & 15) + 1
    a0 = _wmma_cat(As[p, 2wr + 1, ks + 1, 1], As[p, 2wr + 1, ks + 1, 2])
    a1 = _wmma_cat(As[p, 2wr + 2, ks + 1, 1], As[p, 2wr + 2, ks + 1, 2])
    b = _wmma_cat(Bs[p, 4wt + 1, ks + 1, 1], Bs[p, 4wt + 1, ks + 1, 2]); c0 = _wmma16(a0, b, c0); c4 = _wmma16(a1, b, c4)
    b = _wmma_cat(Bs[p, 4wt + 2, ks + 1, 1], Bs[p, 4wt + 2, ks + 1, 2]); c1 = _wmma16(a0, b, c1); c5 = _wmma16(a1, b, c5)
    b = _wmma_cat(Bs[p, 4wt + 3, ks + 1, 1], Bs[p, 4wt + 3, ks + 1, 2]); c2 = _wmma16(a0, b, c2); c6 = _wmma16(a1, b, c6)
    b = _wmma_cat(Bs[p, 4wt + 4, ks + 1, 1], Bs[p, 4wt + 4, ks + 1, 2]); c3 = _wmma16(a0, b, c3); c7 = _wmma16(a1, b, c7)
    return (c0, c1, c2, c3, c4, c5, c6, c7)
end
@inline function _g_decode(ld)
    w1, w2, sa, ma, b1, b2 = ld
    return (_wmma_dq4w(w1, sa, ma), _wmma_dq4w(w2, sa, ma), b1, b2)
end
function _wmma_q4_g_kernel!(C, Pw, Sc, Mn, Bv, K, N, group, ::Val{SYM}, R, hr::Val) where {SYM}
    @inbounds begin
    tid = workitemIdx().x - 1
    wave = tid ÷ 32; lane = tid % 32; sl = lane % 16; h = lane ÷ 16
    wr = wave ÷ 4; wt = wave % 4
    bm = (workgroupIdx().y - 1) * 256; bn = (workgroupIdx().x - 1) * 256
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
    _wmma_store_big!(C, acc, bm + wr * 32, bn + wt * 64, sl, h, N, R, hr)
    return
    end
end
