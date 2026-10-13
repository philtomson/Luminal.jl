@inline _sgb(mask, size) = ccall("llvm.amdgcn.sched.group.barrier", llvmcall, Cvoid, (Int32, Int32, Int32), Int32(mask), Int32(size), Int32(0))
@inline _setprio(p) = ccall("llvm.amdgcn.s.setprio", llvmcall, Cvoid, (Int16,), Int16(p))
# masks: 0x8 matrix (WMMA), 0x100 DS read, 0x200 DS write, 0x2 VALU, 0x20 VMEM read
@inline _sgb_pattern(::Val{0}) = nothing
@inline function _sgb_pattern(::Val{1})        # 3 LDS reads : 2 WMMA, 16 times
    for _ in 1:16; _sgb(0x100, 3); _sgb(0x8, 2); end
end
@inline function _sgb_pattern(::Val{2})        # 2 LDS reads : 1 WMMA (24 : 12) then the rest of the WMMAs
    for _ in 1:24; _sgb(0x100, 2); _sgb(0x8, 1); end
end
@inline function _sgb_pattern(::Val{3})        # 1 : 1 until the reads are exhausted
    for _ in 1:32; _sgb(0x100, 2); _sgb(0x8, 2); end
end
@inline function _sgb_pattern(::Val{4})        # WMMA-first blocks of 4 interleaved with the reads of the next step
    for _ in 1:8; _sgb(0x100, 6); _sgb(0x8, 4); end
end
function _wmma_q4_gs_kernel!(C, Pw, Sc, Mn, Bv, K, N, group, ::Val{SYM}, R, hr::Val, ::Val{V}) where {SYM,V}
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
        acc = _g_ks(acc, As, Bs, wr, wt, sl, 0)
        acc = _g_ks(acc, As, Bs, wr, wt, sl, 1)
        acc = _g_ks(acc, As, Bs, wr, wt, sl, 2)
        dec = _g_decode(ld)                                       # (VALU work on the arrived data, in the same block as the last MMAs)
        acc = _g_ks(acc, As, Bs, wr, wt, sl, 3)
        _sgb_pattern(Val(V))
        sync_workgroup()
    end
    _wmma_store_big!(C, acc, bm + wr * 32, bn + wt * 64, sl, h, N, R, hr)
    return
    end
end
