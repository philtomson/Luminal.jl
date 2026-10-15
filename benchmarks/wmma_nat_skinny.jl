# Skinny matrix-core GEMM for the native K-quant weights (:q5, :q6, :iq4): C (M, N <= 32) = W (M, K) * B (K, N), the weights read once.
# One workgroup = NW waves over the same 16 weight rows; wave w takes every NW-th K16 step. The A fragment (16 rows x 16 k) is the row's
# own words decoded to Float16 in registers (the decoders of the 256 x 256 prefill kernel), the B fragment the activations (L2 resident).
@inline function _ns_load(::Val{:q5}, Pw, Meta, Blk, grow, t)
    g = (t >> 1) + 1
    r1, r2, r3, r4 = Blk[((g - 1) >> 3) + 1, grow + 1]
    return (Pw[2t + 1, grow + 1], Pw[2t + 2, grow + 1], Meta[g, grow + 1], r1, r2, r3, r4, 0, 0)
end
@inline function _ns_load(::Val, Pw, Meta, Blk, grow, t)
    g = (t >> 1) + 1
    return (Pw[2t + 1, grow + 1], Pw[2t + 2, grow + 1], Meta[g, grow + 1], 0, 0)
end
@inline function _ns_decode(fmt::Val, ld, t)
    a1, a2, _, _ = _gx_decode(fmt, ld, t & 1, (t >> 1) & 7)
    return _wmma_cat(a1, a2)
end
function _wmma_nat_skinny_kernel!(C, Pw, Meta, Blk, Bv, K, N, fmt::Val, ::Val{NW}, ::Val{NT}) where {NW, NT}
    @inbounds begin
    tid = workitemIdx().x - 1
    l = tid % 32; r = l % 16; h = l ÷ 16; w = tid ÷ 32
    M = size(C, 1)
    row0 = (workgroupIdx().x - 1) * 16
    grow = min(row0 + r, M - 1)
    Rs = @ROCStaticLocalArray(Float32, (8, 32, NW), false)
    nt = K ÷ 16
    acc = ntuple(_ -> _wmma_zero(), Val(NT))
    ld = _ns_load(fmt, Pw, Meta, Blk, grow, min(w, nt - 1))
    for t in w:NW:nt-1
        ldn = _ns_load(fmt, Pw, Meta, Blk, grow, min(t + NW, nt - 1))
        A = _ns_decode(fmt, ld, t)
        acc = let a = acc
            ntuple(j -> _wmma16(A, _wmma_cat(Bv[2t + 1, 16 * (j - 1) + r + 1], Bv[2t + 2, 16 * (j - 1) + r + 1]), a[j]), Val(NT))
        end
        ld = ldn
    end
    for j in 1:NT
        for i in 0:7
            Rs[i + 1, l + 1, w + 1] = acc[j][i + 1].value
        end
        sync_workgroup()
        if w == 0 && 16 * (j - 1) + r < N
            for i in 0:7
                row = row0 + 2i + h
                v = 0f0
                for k in 1:NW
                    v += Rs[i + 1, l + 1, k]
                end
                row < M && (C[row + 1, 16 * (j - 1) + r + 1] = v)
            end
        end
        sync_workgroup()
    end
    return
    end
end
function wmma_nat_skinny!(C::ROCArray{Float32,2}, W, B16::ROCArray{Float16,2}, N::Int; nw::Int = 4)
    M, K = size(W)
    NT = cld(N, 16)
    (M % 16 == 0 && K % 256 == 0 && size(B16, 1) == K && size(B16, 2) >= 16 * NT && N <= 32 && size(C) == (M, N)) || throw(ArgumentError("wmma_nat_skinny!: shape"))
    Pw = reinterpret(UInt32, _nat_v(W))
    Bv = reinterpret(WMMA_V8H, B16)
    @roc groupsize=32 * nw gridsize=cld(M, 16) _wmma_nat_skinny_kernel!(C, Pw, _nat_meta(W), _nat_blk(W), Bv, K, N, Val(_nat_fmt(W)), Val(nw), Val(NT))
    return C
end
