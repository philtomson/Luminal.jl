function _flash_prefill_ab!(O, Qv, Kc, Vc, S, H, KVH, scale2, qoff, ::Val{NT}, ::Val{ABL}) where {NT,ABL}
    D = 32 * NT; NC = D ÷ 8                   # NT = 8: D = 256, 32 16-byte chunks per key row
    KS = NT                                   # k steps (16 dims) per wave of the S phase: half of D / 16
    tid = workitemIdx().x - 1
    lane = tid % 32; sub = lane % 16; half_id = lane ÷ 16; wave = tid ÷ 32
    qblk = workgroupIdx().x - 1; hp = workgroupIdx().y - 1
    npair = H ÷ 2
    b = hp ÷ npair; pair = hp % npair
    first_head = 2 * pair; kvh = first_head ÷ (H ÷ KVH)
    query_start = qblk * 64
    NK = qoff + S
    max_visible = min(NK, qoff + query_start + 64)
    ntile = cld(max_visible, 16)
    s_tile = wave % 8; s_kh = wave ÷ 8
    s_head = first_head + s_tile % 2; s_qoff = (s_tile ÷ 2) * 16

    Ks = @ROCStaticLocalArray(WMMA_V8H, (NC + 1, 16), false)
    Vt = @ROCStaticLocalArray(Float16, (24, D), false)
    s_lds = @ROCStaticLocalArray(Float32, (17, 16, 8, 2), false)
    p_lds = @ROCStaticLocalArray(Float16, (24, 128), false)
    row_scale = @ROCStaticLocalArray(Float32, (128,), false)
    row_sum = @ROCStaticLocalArray(Float32, (128,), false)

    @inbounds begin
    qrow = min(query_start + s_qoff + sub, S - 1)
    qf = ntuple(Val(KS)) do ks
        c = 2 * (s_kh * KS + ks - 1)
        _wmma_cat(Qv[c + 1, qrow + 1, s_head + 1, b + 1], Qv[c + 2, qrow + 1, s_head + 1, b + 1])
    end
    oacc = ntuple(_ -> _wmma_zero(), Val(8))
    running_max = -Inf32; running_sum = 0f0
    zero8 = ntuple(_ -> VecElement(Float16(0)), Val(8))

    v_key = lane % 16; v_chunk = tid ÷ 16 + 1                    # V: lane's key, and its 16-byte chunk (8 dims) of the row
    kcur = Kc[tid % NC + 1, min(tid ÷ NC, NK - 1) + 1, kvh + 1, b + 1]
    vcur = v_key < NK ? Vc[v_chunk, v_key + 1, kvh + 1, b + 1] : zero8

    for kt in 0:ntile - 1
        key_start = kt * 16
        Ks[tid % NC + 1, tid ÷ NC + 1] = kcur
        sync_workgroup()                # (all waves are past the previous P V: Vt may be restaged)
        for i in 0:(ABL == 1 ? 0 : 7)
            Vt[v_key + 1, (v_chunk - 1) * 8 + i + 1] = vcur[i + 1].value
        end
        nstart = key_start + 16
        if ABL != 4
            kcur = Kc[tid % NC + 1, min(nstart + tid ÷ NC, NK - 1) + 1, kvh + 1, b + 1]
            nkey = nstart + v_key
            vcur = nkey < NK ? Vc[v_chunk, nkey + 1, kvh + 1, b + 1] : zero8
        end
        # S = Q K^T over this wave's half of the dims
        s_acc = _qk_dot(qf, Ks, 2 * s_kh * KS, sub)
        for i in 0:7
            s_lds[sub + 1, 2i + half_id + 1, s_tile + 1, s_kh + 1] = s_acc[i + 1].value
        end
        sync_workgroup()
        # online softmax: 4 threads per row, 4 keys each
        rg = tid ÷ 4; seg = tid % 4
        rb = rg ÷ 16; row = rg % 16
        lq = query_start + (rb ÷ 2) * 16 + row
        col = seg * 4
        cc = col % 16
        base_valid = lq < S
        kpos = key_start + col
        qabs = qoff + lq
        sv(m) = (base_valid && kpos + m <= qabs) ? (s_lds[cc + m + 1, row + 1, rb + 1, 1] + s_lds[cc + m + 1, row + 1, rb + 1, 2]) * scale2 : -Inf32
        v0 = sv(0); v1 = sv(1); v2 = sv(2); v3 = sv(3)
        pmax = max(max(v0, v1), max(v2, v3))
        if ABL != 3
            pmax = max(pmax, AMDGPU.Device.shfl_xor(pmax, Cint(1)))
            pmax = max(pmax, AMDGPU.Device.shfl_xor(pmax, Cint(2)))
        end
        nmax = max(running_max, pmax)
        prior = (ABL == 3 || isfinite(running_max)) ? exp2(running_max - nmax) : 0f0
        w0 = ABL == 3 ? v0 : (isfinite(v0) ? exp2(v0 - nmax) : 0f0); w1 = ABL == 3 ? v1 : (isfinite(v1) ? exp2(v1 - nmax) : 0f0)
        w2 = ABL == 3 ? v2 : (isfinite(v2) ? exp2(v2 - nmax) : 0f0); w3 = ABL == 3 ? v3 : (isfinite(v3) ? exp2(v3 - nmax) : 0f0)
        p_lds[col + 1, rg + 1] = Float16(w0)
        if ABL != 2
            p_lds[col + 2, rg + 1] = Float16(w1); p_lds[col + 3, rg + 1] = Float16(w2); p_lds[col + 4, rg + 1] = Float16(w3)
        end
        psum = (w0 + w1) + (w2 + w3)
        psum += AMDGPU.Device.shfl_xor(psum, Cint(1)); psum += AMDGPU.Device.shfl_xor(psum, Cint(2))
        running_max = nmax
        running_sum = running_sum * prior + psum
        seg == 0 && (row_scale[rg + 1] = prior)
        sync_workgroup()
        # rescale the output accumulators, then O += P V for this wave's dim tile and all 8 row blocks
        vf = _lds_frag(Vt, (wave * 16 + sub) * 24 + 1)
        oacc = let o = oacc
            ntuple(Val(8)) do n
                rbn = n - 1
                sc = _fa_scale8(o[n], ntuple(i -> row_scale[rbn * 16 + 2 * (i - 1) + half_id + 1], Val(8)))
                _wmma16(_lds_frag(p_lds, (rbn * 16 + sub) * 24 + 1), vf, sc)
            end
        end
    end
    tid % 4 == 0 && (row_sum[tid ÷ 4 + 1] = running_sum)
    sync_workgroup()
    let oa = oacc
        ntuple(Val(8)) do n
            rbn = n - 1
            head = first_head + rbn % 2
            for i in 0:7
                row = 2i + half_id
                lq = query_start + (rbn ÷ 2) * 16 + row
                if lq < S
                    den = row_sum[rbn * 16 + row + 1]
                    d = wave * 16 + sub
                    O[d + 1, lq + 1, head + 1, b + 1] = den > 0f0 ? oa[n][i + 1].value / den : 0f0
                end
            end
            nothing
        end
    end
    end # inbounds
    return
end
