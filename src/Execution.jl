# Implements a simple interpreter to execute a computation graph.



using CUDA
using AMDGPU
using LinearAlgebra
using GPUArrays
using KernelAbstractions
using KernelAbstractions.Extras: @unroll


export execute_op, execute_op!, realize_view, execute, eval_dim, FusedElementwiseOp, HalfWeight, HalfWeightN, QuantWeight

# Helper for batch matrix multiplication
function batch_matmul(A, B)
    # ... (Keep existing implementation for interpreter) ...
    # A: (..., M, K), B: (..., K, N)
    
    # Handle simple 2D case
    if ndims(A) == 2 && ndims(B) == 2
        return A * B
    end
    
    # Handle 3D * 2D (Linear on batch): (B, S, D) * (D, V) -> (B, S, V)
    if ndims(A) == 3 && ndims(B) == 2
        B1, S, D = size(A)
        res = similar(A, B1, S, size(B, 2))
        for i in 1:B1
            res[i, :, :] = A[i, :, :] * B
        end
        return res
    end
    
    # Handle 3D batch case: (B, M, K) * (B, K, N) -> (B, M, N)
    if ndims(A) == 3 && ndims(B) == 3
        @assert size(A, 1) == size(B, 1) "Batch dimensions (3D) must match: $(size(A, 1)) vs $(size(B, 1))"
        batch_size = size(A, 1)
        res = similar(A, batch_size, size(A, 2), size(B, 3))
        for i in 1:batch_size
            res[i, :, :] = A[i, :, :] * B[i, :, :]
        end
        return res
    end
    
    # Handle 4D (Llama attention): (B, H, S, D) * (B, H, D, S) -> (B, H, S, S)
    if ndims(A) == 4 && ndims(B) == 4
        @assert size(A, 1) == size(B, 1) && size(A, 2) == size(B, 2) "Batch and head dimensions must match: A $(size(A)), B $(size(B))"
        B1, B2 = size(A, 1), size(A, 2)
        res = similar(A, B1, B2, size(A, 3), size(B, 4))
        for i in 1:B1, j in 1:B2
            res[i, j, :, :] = A[i, j, :, :] * B[i, j, :, :]
        end
        return res
    end
    
    error("Batch matmul not implemented for $(ndims(A))D and $(ndims(B))D (Shapes: $(size(A)) and $(size(B)))")
end

function _ensure_contiguous(x)
    if typeof(x) <: PermutedDimsArray && parent(x) isa AnyGPUArray
        return copy(x)
    end
    return x
end

# An Expand whose every consumer is elementwise is not materialized: its result is
# the input with a size-1 dim inserted, which broadcasting expands on the fly.
# Wrapped so realize_view can tell it apart from a genuinely size-1 tensor.
struct BroadcastView{A}
    a::A
end
Base.size(b::BroadcastView) = size(b.a)
Base.length(b::BroadcastView) = length(b.a)

# --- Rotary embedding ---------------------------------------------------------
# x, out: (D, S, H, B). Tables c, s: (half, S) shared by the batch, or
# (half, S, B) with one set of positions per sequence (batched decode).
@kernel function _rotary_kernel!(out, @Const(x), @Const(c), @Const(s), half)
    I = @index(Global, Cartesian)
    i = I[1]; t = I[2]
    b = size(c, 3) == 1 ? 1 : I[4]
    @inbounds if i <= half
        out[I] = x[I] * c[i, t, b] - x[i + half, t, I[3], I[4]] * s[i, t, b]
    else
        j = i - half
        out[I] = x[I] * c[j, t, b] + x[j, t, I[3], I[4]] * s[j, t, b]
    end
end

function execute_op!(out, op::RotaryEmbed, x, c, s)
    x4 = ndims(x) == 4 ? x : Base.reshape(x, size(x)..., ntuple(_ -> 1, 4 - ndims(x))...)
    o4 = ndims(out) == 4 ? out : Base.reshape(out, size(x4))
    S = size(x4, 2)
    c3 = Base.reshape(c, size(c, 1), S, :)
    s3 = Base.reshape(s, size(s, 1), S, :)
    _rotary_kernel!(KernelAbstractions.get_backend(o4), 256)(o4, x4, c3, s3, size(x4, 1) ÷ 2; ndrange = size(o4))
    return out
end

# --- RMS normalization --------------------------------------------------------
# One workgroup per column (all dims after the first): sum of squares over dim 1
# with a tree reduction, then the normalized, weighted write.
@kernel function _rmsnorm_kernel!(out, @Const(x), @Const(w), eps)
    col = @index(Group, Linear); lid = @index(Local, Linear)
    T = @uniform @groupsize()[1]
    H = @uniform size(x, 1)
    red = @localmem Float32 (256,)
    acc = 0f0
    i = lid
    while i <= H
        @inbounds v = x[i, col]
        acc += v * v
        i += T
    end
    @inbounds red[lid] = acc
    @synchronize
    stride = T ÷ 2
    while stride > 0
        lid <= stride && (@inbounds red[lid] += red[lid + stride])
        @synchronize
        stride ÷= 2
    end
    @inbounds r = 1f0 / sqrt(red[1] / H + eps)
    i = lid
    while i <= H
        @inbounds out[i, col] = x[i, col] * r * w[i]
        i += T
    end
end

function execute_op!(out, op::RMSNormOp, x, w)
    H = size(x, 1)
    xc = x isa DenseArray ? x : copy(x)
    x2 = Base.reshape(xc, H, :)
    o2 = Base.reshape(out, H, :)
    wv = vec(w)
    if !(out isa AnyGPUArray)          # CPU: plain loops (the kernel's reduction is GPU-shaped)
        for c in axes(x2, 2)
            r = 1f0 / sqrt(sum(abs2, view(x2, :, c)) / H + op.epsilon)
            o2[:, c] .= view(x2, :, c) .* r .* wv
        end
        return out
    end
    _rmsnorm_kernel!(KernelAbstractions.get_backend(o2), 256)(o2, x2, wv, op.epsilon; ndrange = size(x2, 2) * 256)
    return out
end

# --- Decode attention ---------------------------------------------------------
# One workgroup per (query head, batch). Scores for the n = pos + 1 positions
# (cache slots 1..pos, then the new token) live in local memory: dot products,
# then a parallel max and sum for the softmax, then the probability-weighted sum
# of V with the workgroup split into G ÷ D slices over positions.
const DECODE_ATTN_MAX_CTX = 8192
@kernel function _decode_attn_kernel!(out, @Const(q), pk, pv, @Const(kn), @Const(vn),
                                      @Const(posv), scale, Gq, write_cache)
    # Values shared across barriers must be @uniform (the CPU backend splits the
    # kernel into loops at each @synchronize); the softmax statistics go through
    # local memory (`stats`) for the same reason.
    grp = @index(Group, Linear); lid = @index(Local, Linear)
    T = @uniform @groupsize()[1]
    D = @uniform size(q, 1)
    H = @uniform size(q, 3)
    h = @uniform (@index(Group, Linear) - 1) % size(q, 3) + 1
    b = @uniform (@index(Group, Linear) - 1) ÷ size(q, 3) + 1
    kv = @uniform (h - 1) ÷ Gq + 1
    pos = @uniform unsafe_trunc(Int, @inbounds posv[length(posv) == 1 ? 1 : b])
    n = @uniform pos + 1
    # write_cache: the first query head of each KV group stores this token's K/V in
    # slot pos + 1 (no workgroup reads that slot from the cache).
    if write_cache && (h - 1) % Gq == 0 && lid <= D
        @inbounds pk[lid, pos + 1, kv, b] = kn[lid, 1, kv, b]
        @inbounds pv[lid, pos + 1, kv, b] = vn[lid, 1, kv, b]
    end
    sc = @localmem Float32 (DECODE_ATTN_MAX_CTX,)
    red = @localmem Float32 (256,)
    stats = @localmem Float32 (2,)
    # 1. scores
    j = lid
    while j <= n
        acc = 0f0
        if j <= pos
            for d in 1:D
                @inbounds acc += q[d, 1, h, b] * pk[d, j, kv, b]
            end
        else
            for d in 1:D
                @inbounds acc += q[d, 1, h, b] * kn[d, 1, kv, b]
            end
        end
        @inbounds sc[j] = acc * scale
        j += T
    end
    @synchronize
    # 2. max
    m = -Inf32
    j = lid
    while j <= n
        @inbounds m = max(m, sc[j])
        j += T
    end
    @inbounds red[lid] = m
    @synchronize
    stride = T ÷ 2
    while stride > 0
        lid <= stride && (@inbounds red[lid] = max(red[lid], red[lid + stride]))
        @synchronize
        stride ÷= 2
    end
    lid == 1 && (@inbounds stats[1] = red[1])
    @synchronize
    # 3. exp and sum
    ssum = 0f0
    j = lid
    while j <= n
        @inbounds e = exp(sc[j] - stats[1])
        @inbounds sc[j] = e
        ssum += e
        j += T
    end
    @inbounds red[lid] = ssum
    @synchronize
    stride = T ÷ 2
    while stride > 0
        lid <= stride && (@inbounds red[lid] += red[lid + stride])
        @synchronize
        stride ÷= 2
    end
    lid == 1 && (@inbounds stats[2] = 1f0 / red[1])
    @synchronize
    # 4. out[d] = sum_j p_j V[d, j]; thread (d, part) covers positions part, part + P, ...
    P = T ÷ D
    d = (lid - 1) % D + 1
    part = (lid - 1) ÷ D + 1
    acc = 0f0
    if part <= P
        j = part
        while j <= n
            if j <= pos
                @inbounds acc += sc[j] * pv[d, j, kv, b]
            else
                @inbounds acc += sc[j] * vn[d, 1, kv, b]
            end
            j += P
        end
    end
    @inbounds red[lid] = acc
    @synchronize
    if lid <= D
        tot = 0f0
        for p in 0:P-1
            @inbounds tot += red[lid + p * D]
        end
        @inbounds out[lid, h, b] = tot * stats[2]
    end
end

function execute_op!(out, op::DecodeAttention, q, pk, pv, kn, vn, posv)
    D, H, B = size(q, 1), size(q, 3), size(q, 4)
    KVH = size(pk, 3)
    if !(out isa AnyGPUArray)          # CPU: plain loops (the kernel's reductions are GPU-shaped)
        G = H ÷ KVH
        o3 = Base.reshape(out, D, H, B)
        for b in 1:B, h in 1:H
            pos = Int(posv[length(posv) == 1 ? 1 : b])
            kv = (h - 1) ÷ G + 1
            s = Float32[op.scale * sum(q[d, 1, h, b] * (j <= pos ? pk[d, j, kv, b] : kn[d, 1, kv, b]) for d in 1:D)
                        for j in 1:pos+1]
            p = exp.(s .- maximum(s)); p ./= sum(p)
            for d in 1:D
                o3[d, h, b] = sum(p[j] * (j <= pos ? pv[d, j, kv, b] : vn[d, 1, kv, b]) for j in 1:pos+1)
            end
        end
        if op.write_cache
            for b in 1:B
                pos = Int(posv[length(posv) == 1 ? 1 : b])
                pk[:, pos + 1, :, b] .= kn[:, 1, :, b]
                pv[:, pos + 1, :, b] .= vn[:, 1, :, b]
            end
        end
        return out
    end
    size(pk, 2) + 1 <= DECODE_ATTN_MAX_CTX || error("DecodeAttention: cache longer than $(DECODE_ATTN_MAX_CTX)")
    (D <= 256 && 256 % D == 0) || error("DecodeAttention: head_dim must divide 256")
    o3 = Base.reshape(out, D, H, B)
    _decode_attn_kernel!(KernelAbstractions.get_backend(o3), 256)(
        o3, q, pk, pv, kn, vn, posv, op.scale, H ÷ KVH, op.write_cache; ndrange = H * B * 256)
    return out
end

# --- Float16 matmul weights ---------------------------------------------------
# A matmul weight stored as Float16, transposed to (In, Out) so that each output
# row is contiguous, and read through 16-byte (8 x Float16) vector loads. It keeps
# the logical (Out, In) size of the Float32 weight it replaces; activations and
# accumulation stay Float32. Decode is bound by weight bandwidth, so halving the
# bytes read is roughly a 2x speedup on the large projections.
struct HalfWeight{A, V} <: AbstractMatrix{Float16}
    t::A   # (In, Out) Float16
    v::V   # (In/8, Out) NTuple{8, Float16} view of `t`
end
function HalfWeight(W::AbstractMatrix)
    t = similar(W, Float16, size(W, 2), size(W, 1))
    t .= permutedims(W, (2, 1))
    v = Base.reshape(reinterpret(NTuple{8, Float16}, vec(t)), size(t, 1) ÷ 8, size(t, 2))
    return HalfWeight(t, v)
end
Base.size(w::HalfWeight) = (size(w.t, 2), size(w.t, 1))
Base.getindex(w::HalfWeight, i::Int, j::Int) = w.t[j, i]

# A matmul weight stored as Float16 in its own (Out, In) layout, for rocBLAS's
# mixed-precision GEMM (MatMulF16(_, :gemm_ex)): untransposed, gemm_ex is ~1.7-2.3x
# faster than Float32 GEMM on TinyLlama's shapes; transposed it is not.
struct HalfWeightN{A} <: AbstractMatrix{Float16}
    w::A   # (Out, In) Float16
end
# (The inner constructor is called explicitly: this outer method has the same
# signature as the default one-argument constructor and replaces it.)
HalfWeightN(W::AbstractMatrix) = (w = similar(W, Float16, size(W)...); w .= W; HalfWeightN{typeof(w)}(w))
Base.size(w::HalfWeightN) = size(w.w)
Base.getindex(w::HalfWeightN, i::Int, j::Int) = w.w[i, j]

# A matmul weight quantized to int8 with symmetric Float32 scales, one per group of
# `group` consecutive inputs of each output row (scale = max|group| / 127; a
# group equal to In means one scale per row). Transposed to (In, Out) so each row
# is contiguous and read as 16-byte vectors of 16 int8. Weight-only: activations
# stay Float32. Half the bytes of Float16, for bandwidth-bound decode. Smaller
# groups track the weights' range more closely (a single large weight no longer
# coarsens a whole 2048- or 5632-element row) for ~4 bytes per group.
const DEFAULT_Q8_GROUP = 128     # inputs per int8 scale
# int8 GEMV threads per workgroup; 0 = choose per call (`_q8_threads`).
const DEFAULT_Q8_THREADS = 0

# Threads per workgroup for an int8 GEMV with K16 = K/16 vector loads per row and N
# columns. A thread handles K16/threads loads, so 256 threads only pay off when
# that is ~1-2: measured with weights read from DRAM (not cache), 256 threads are
# +20-55% on K = 4096 (Llama-3-8B) with 1-4 columns, but slower on K = 2048
# (TinyLlama: half the threads idle), on K = 14336, and at 8 columns.
_q8_threads(K16::Int, N::Int) =
    (N == 1 && 256 <= K16 < 512) || (N <= 4 && 256 <= K16 < 320) ? 256 : 128
struct QuantWeight{Q, V, S} <: AbstractMatrix{Float32}
    q::Q         # (In, Out) Int8
    v::V         # (In/16, Out) NTuple{16, Int8} view of `q`
    scale::S     # (In/group, Out) Float32
    group::Int
end
function QuantWeight(W::AbstractMatrix; group::Int = DEFAULT_Q8_GROUP)
    M, K = size(W)
    (group % 16 == 0 && K % group == 0) || (group = K)       # fall back to one scale per row
    K ÷ group <= 512 || (group = K)                           # the kernel stages <= 512 scales
    Wt = permutedims(W, (2, 1))                              # (In, Out)
    Wg = Base.reshape(Wt, group, K ÷ group, M)
    scale = Base.reshape(maximum(abs, Wg; dims=1), K ÷ group, M) ./ 127f0
    scale .= max.(scale, floatmin(Float32))
    q = similar(W, Int8, K, M)
    Base.reshape(q, group, K ÷ group, M) .= unsafe_trunc.(Int8, round.(Wg ./ Base.reshape(scale, 1, K ÷ group, M)))
    v = Base.reshape(reinterpret(NTuple{16, Int8}, vec(q)), K ÷ 16, M)
    return QuantWeight(q, v, scale, group)
end
Base.size(w::QuantWeight) = (size(w.q, 2), size(w.q, 1))
Base.getindex(w::QuantWeight, i::Int, j::Int) = Float32(w.q[j, i]) * w.scale[(j - 1) ÷ w.group + 1, i]

# Y[:, c] = Wq * X[:, c] with per-group scales. Each workgroup computes R output
# rows (R = 2 when the row count is even): a thread loads its 16-element slice of
# x once and reuses it for both rows. With one row per workgroup the kernel was
# limited by activation loads (32 bytes of x per 16 bytes of weights) at ~135
# GB/s; with two it reaches ~205 GB/s on a Radeon 8060S (R = 4 was slower).
# 16 int8 weights per load, Float32 accumulation, scaled once per load.
# `shift` >= 0: loads per scale group is 2^shift (index by bit shift).
# Columns are processed in chunks of up to C (C = N for N <= Q8_MAX_COLS, so
# batched decode reads each weight vector once per chunk). Accumulators are tuples, so
# they stay in registers: a thread holds R x C partial sums, reduced together once
# per chunk. Weights are converted to Float32 once per load, not per column.
@inline _dot16(w, xa, xb) =
    w[1] * xa[1] + w[2] * xa[2] + w[3] * xa[3] + w[4] * xa[4] + w[5] * xa[5] + w[6] * xa[6] +
    w[7] * xa[7] + w[8] * xa[8] + w[9] * xb[1] + w[10] * xb[2] + w[11] * xb[3] + w[12] * xb[4] +
    w[13] * xb[5] + w[14] * xb[6] + w[15] * xb[7] + w[16] * xb[8]

@kernel function _q8_matmul_kernel!(Y, @Const(Wv), @Const(scale), @Const(Xv), K16, N,
                                    loads_per_group, shift, ::Val{R}, ::Val{C}) where {R, C}
    grp = @index(Group, Linear); lid = @index(Local, Linear); G = @groupsize()[1]
    s = @localmem Float32 (256, 2 * C)
    row0 = (grp - 1) * R
    c0 = 0
    while c0 < N
        acc1 = ntuple(_ -> 0f0, Val(C))
        acc2 = ntuple(_ -> 0f0, Val(C))
        k = lid
        while k <= K16
            gi = shift >= 0 ? ((k - 1) >> shift) + 1 : (k - 1) ÷ loads_per_group + 1
            @inbounds w1 = Wv[k, row0 + 1]
            @inbounds w2 = Wv[k, row0 + R]
            f1 = ntuple(i -> Float32(w1[i]), Val(16))
            f2 = ntuple(i -> Float32(w2[i]), Val(16))
            @inbounds sc1 = scale[gi, row0 + 1]
            @inbounds sc2 = scale[gi, row0 + R]
            # Each column's 16 activations are loaded once for both rows. Columns past N
            # (a partial last chunk) repeat column c0 + 1: computed, never written.
            # (Kept inline: as an @inline helper this was ~2 ms/token slower at batch 1.)
            ks = 2k; cb = c0   # fresh bindings: closures must not capture reassigned variables
            acc1, acc2 = let a1 = acc1, a2 = acc2
                xs = ntuple(j -> cb + j <= N ? (@inbounds (Xv[ks - 1, cb + j], Xv[ks, cb + j])) :
                                              (@inbounds (Xv[ks - 1, cb + 1], Xv[ks, cb + 1])), Val(C))
                (ntuple(j -> a1[j] + _dot16(f1, xs[j][1], xs[j][2]) * sc1, Val(C)),
                 R == 2 ? ntuple(j -> a2[j] + _dot16(f2, xs[j][1], xs[j][2]) * sc2, Val(C)) : a2)
            end
            k += G
        end
        @unroll for j in 1:C
            @inbounds s[lid, (j - 1) * R + 1] = acc1[j]
            R == 2 && (@inbounds s[lid, (j - 1) * R + R] = acc2[j])
        end
        @synchronize
        stride = G ÷ 2
        while stride > 0
            if lid <= stride
                @unroll for i in 1:R*C
                    @inbounds s[lid, i] += s[lid + stride, i]
                end
            end
            @synchronize
            stride ÷= 2
        end
        if lid <= R * C
            r = (lid - 1) % R + 1; j = (lid - 1) ÷ R + 1
            c0 + j <= N && (@inbounds Y[row0 + r, c0 + j] = s[1, lid])
        end
        @synchronize
        c0 += C
    end
end

const GEMV_MAX_COLS = 8   # columns accumulated per pass in the GEMV kernels
# The int8 kernel holds 16 activations per column: at 8 columns the register
# pressure costs more than re-reading the (by then cached) weights for a second
# pass, so it takes at most 4 columns per pass (+25-40% at 8 columns on
# Llama-3-8B shapes).
const Q8_MAX_COLS = 4

function _q8_matmul!(C, W::QuantWeight, B; group::Int = DEFAULT_Q8_THREADS)
    K, M = size(W.q)
    N = length(B) ÷ K
    Bc = B isa DenseArray ? B : copy(B)
    Xv = Base.reshape(reinterpret(NTuple{8, Float32}, vec(Bc)), K ÷ 8, N)
    R = iseven(M) ? 2 : 1
    lpg = W.group ÷ 16
    group == 0 && (group = _q8_threads(K ÷ 16, N))
    group <= 256 || error("int8 GEMV: at most 256 threads per workgroup")
    _q8_matmul_kernel!(KernelAbstractions.get_backend(C), group)(
        Base.reshape(C, M, N), W.v, W.scale, Xv, K ÷ 16, N, lpg,
        ispow2(lpg) ? trailing_zeros(lpg) : -1, Val(R), Val(min(N, Q8_MAX_COLS)); ndrange = (M ÷ R) * group)
    return C
end

# Float16 copies of activations for gemm_ex, reused per (device array type, size):
# a captured graph bakes in the pointer, so they must outlive every call.
const _GEMM_EX_WORKSPACE = Dict{Any, Any}()

# C (Out, N) = W * B via rocBLAS gemm_ex: Float16 W and B (B rounded into a
# workspace), Float32 accumulation and output. Trailing dims of B/C flatten into N.
function _gemm_ex_f16!(C, W::HalfWeightN, B)
    M, K = size(W.w)
    N = length(B) ÷ K
    Bc = B isa DenseArray ? B : copy(B)
    x16 = get!(() -> similar(Bc, Float16, K, N), _GEMM_EX_WORKSPACE, (typeof(Bc), K, N))
    x16 .= Base.reshape(Bc, K, N)
    C2 = Base.reshape(C, M, N)
    RB = AMDGPU.rocBLAS
    α = Ref{Float32}(1f0); β = Ref{Float32}(0f0)
    p(A) = reinterpret(Ptr{Cvoid}, pointer(A))
    RB.rocblas_gemm_ex_64(RB.handle(), RB.rocblas_operation_none, RB.rocblas_operation_none,
        M, N, K, α, p(W.w), RB.rocblas_datatype_f16_r, M, p(x16), RB.rocblas_datatype_f16_r, K,
        β, p(C2), RB.rocblas_datatype_f32_r, M, p(C2), RB.rocblas_datatype_f32_r, M,
        RB.rocblas_datatype_f32_r, RB.rocblas_gemm_algo_standard, Int32(0), UInt32(0))
    return C
end

# Y[:, c] = W * X[:, c]. One workgroup per output row; each weight vector is
# loaded (and converted to Float32) once per chunk of up to C columns (C = N for
# N <= 8), whose partial sums stay in registers and are reduced together.
@inline _dot8(w, x) = w[1] * x[1] + w[2] * x[2] + w[3] * x[3] + w[4] * x[4] +
                      w[5] * x[5] + w[6] * x[6] + w[7] * x[7] + w[8] * x[8]

# acc[j] += w · X[k, c0 + j] for the chunk's columns. Columns past N (a partial last
# chunk) repeat column N: computed without a branch, never written.
@inline function _half_acc(acc::NTuple{C, Float32}, f, Xv, k, c0, N) where {C}
    return ntuple(j -> acc[j] + _dot8(f, @inbounds Xv[k, min(c0 + j, N)]), Val(C))
end

@kernel function _half_matmul_kernel!(Y, @Const(Wv), @Const(Xv), K8, N, ::Val{C}) where {C}
    row = @index(Group, Linear); lid = @index(Local, Linear); G = @groupsize()[1]
    s = @localmem Float32 (256, C)
    c0 = 0
    while c0 < N
        acc = ntuple(_ -> 0f0, Val(C))
        k = lid
        while k <= K8
            @inbounds w = Wv[k, row]
            f = ntuple(i -> Float32(w[i]), Val(8))
            acc = _half_acc(acc, f, Xv, k, c0, N)
            k += G
        end
        @unroll for j in 1:C
            @inbounds s[lid, j] = acc[j]
        end
        @synchronize
        stride = G ÷ 2
        while stride > 0
            if lid <= stride
                @unroll for j in 1:C
                    @inbounds s[lid, j] += s[lid + stride, j]
                end
            end
            @synchronize
            stride ÷= 2
        end
        lid <= C && c0 + lid <= N && (@inbounds Y[row, c0 + lid] = s[1, lid])
        @synchronize
        c0 += C
    end
end

# C (Out, N) = W (Out, In) * B (In, N), with any trailing dims of B/C flattened into N.
function _half_matmul!(C, W::HalfWeight, B; group::Int = DEFAULT_HALF_GROUP)
    K, M = size(W.t)
    N = length(B) ÷ K
    Bc = B isa DenseArray ? B : copy(B)
    Xv = Base.reshape(reinterpret(NTuple{8, Float32}, vec(Bc)), K ÷ 8, N)
    group <= 256 || error("Float16 GEMV: at most 256 threads per workgroup")
    _half_matmul_kernel!(KernelAbstractions.get_backend(C), group)(
        Base.reshape(C, M, N), W.v, Xv, K ÷ 8, N, Val(min(N, GEMV_MAX_COLS)); ndrange = M * group)
    return C
end

# Convert each Float32 weight at most once, so graphs sharing weights (e.g.
# prefill and decode) share one Float16 copy. Keyed by identity without holding
# the source array. An objectid can be reused once its array is collected (and
# the finalizer that drops the entry runs asynchronously), so each entry keeps a
# WeakRef to its source and only counts as a hit if it still points to `W`.
const _HALF_WEIGHTS = Dict{UInt, Tuple{WeakRef, Any}}()
const _HALF_WEIGHTS_LOCK = ReentrantLock()
function half_weight(W, ::Type{T} = HalfWeight) where {T}
    key = hash(T, objectid(W))
    lock(_HALF_WEIGHTS_LOCK) do
        entry = get(_HALF_WEIGHTS, key, nothing)
        entry !== nothing && entry[1].value === W && return entry[2]
        hw = T(W)
        _HALF_WEIGHTS[key] = (WeakRef(W), hw)
        finalizer(W) do _
            @async lock(_HALF_WEIGHTS_LOCK) do
                e = get(_HALF_WEIGHTS, key, nothing)
                # only drop the entry if it still belongs to this (now dead) array
                e !== nothing && e[1].value === nothing && delete!(_HALF_WEIGHTS, key)
            end
        end
        return hw
    end
end

# C[:, :, i] = A[:, :, i] * B[:, :, i] as one strided-batched BLAS call on GPU.
_batched_gemm!(C::ROCArray{T,3}, A::ROCArray{T,3}, B::ROCArray{T,3}) where {T<:Union{Float32,Float64}} =
    AMDGPU.rocBLAS.gemm_strided_batched!('N', 'N', one(T), A, B, zero(T), C)
_batched_gemm!(C::CuArray{T,3}, A::CuArray{T,3}, B::CuArray{T,3}) where {T<:Union{Float32,Float64}} =
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', one(T), A, B, zero(T), C)
_batched_gemm!(C, A, B) = nothing
_batched_gemm!(C::ROCArray{T,3}, A::ROCArray{T,3}, B::ROCArray{T,3}, ta::Char, tb::Char) where {T<:Union{Float32,Float64}} =
    AMDGPU.rocBLAS.gemm_strided_batched!(ta, tb, one(T), A, B, zero(T), C)
_batched_gemm!(C::CuArray{T,3}, A::CuArray{T,3}, B::CuArray{T,3}, ta::Char, tb::Char) where {T<:Union{Float32,Float64}} =
    CUDA.CUBLAS.gemm_strided_batched!(ta, tb, one(T), A, B, zero(T), C)
_batched_gemm!(C, A, B, ta, tb) = nothing

_swap12(X) = ndims(X) == 2 ? permutedims(X, (2, 1)) : permutedims(X, (2, 1, 3:ndims(X)...))

# op(A) * op(B) with transpose flags on the first two dims (see MatMulT). Dense
# operands with matching batch dims use BLAS flags directly; anything else
# materializes the transpose and takes the regular path.
function batch_matmul_t!(C, A, B, ta::Bool, tb::Bool)
    if A isa DenseArray && B isa DenseArray && C isa DenseArray && ndims(A) == ndims(B) == ndims(C)
        if ndims(A) == 2
            mul!(C, ta ? transpose(A) : A, tb ? transpose(B) : B)
            return C
        elseif size(A)[3:end] == size(B)[3:end]
            nb = prod(size(A)[3:end])
            r3(X) = Base.reshape(X, size(X, 1), size(X, 2), nb)
            _batched_gemm!(r3(C), r3(A), r3(B), ta ? 'T' : 'N', tb ? 'T' : 'N') === nothing || return C
        end
    end
    return batch_matmul!(C, ta ? _swap12(A) : A, tb ? _swap12(B) : B)
end

# (M, K) * (K, N): a single-column right-hand side (e.g. one decode token) runs as
# GEMV, which is ~1.5-2x faster than an n=1 GEMM for large weight matrices.
function _matmul2d!(C, A, B)
    if size(B, 2) == 1 && B isa DenseArray && C isa DenseArray
        mul!(vec(C), A, vec(B))
    else
        mul!(C, A, B)
    end
    return C
end

function batch_matmul!(C, A, B)
    A isa HalfWeight && return _half_matmul!(C, A, B)
    A isa QuantWeight && return _q8_matmul!(C, A, B)
    A = _ensure_contiguous(A)
    B = _ensure_contiguous(B)
    # println("DEBUG matmul: C=$(size(C)) ($(typeof(C))), A=$(size(A)) ($(typeof(A))), B=$(size(B)) ($(typeof(B)))")
    
    # 2D * 2D
    if ndims(A) == 2 && ndims(B) == 2
        return _matmul2d!(C, A, B)
    end
    
    # Linear: 2D * 3D (Out, In) * (In, S, B) -> (Out, S, B)
    if ndims(A) == 2 && ndims(B) == 3
        # Use single large gemm by flattening batch and sequence
        out_dim, in_dim = size(A)
        in_dim2, s, b = size(B)
        @assert in_dim == in_dim2 "Inner dimensions must match: $in_dim vs $in_dim2"
        
        # mul! works on reshaped views
        _matmul2d!(Base.reshape(C, out_dim, s * b), A, Base.reshape(B, in_dim, s * b))
        return C
    end

    # Linear: 3D * 2D (B, S, In) * (In, Out) -> (B, S, Out)
    if ndims(A) == 3 && ndims(B) == 2
        batch, s, in_dim = size(A)
        in_dim2, out_dim = size(B)
        @assert in_dim == in_dim2 "Inner dimensions must match: $in_dim vs $in_dim2"
        
        # mul! works on reshaped views
        mul!(Base.reshape(C, batch * s, out_dim), Base.reshape(A, batch * s, in_dim), B)
        return C
    end
    
    # Attention: 3D * 3D (B, M, K) * (B, K, N) -> (B, M, N)
    if ndims(A) == 3 && ndims(B) == 3
        B1 = size(A, 1)
        for i in 1:B1
            mul!(view(C, i, :, :), view(A, i, :, :), view(B, i, :, :))
        end
        return C
    end
    
    # Attention: 4D * 4D (H, B, S, D) * (H, B, D, S) -> (H, B, S, S)
    # Note: In Julia column-major, (S, D) should be the fastest dimensions for efficient view matmul.
    # So (S, D, H, B) or (D, S, H, B) are better.
    if ndims(A) == 4 && ndims(B) == 4
        # One strided-batched GEMM over all (head, batch) pairs when layouts allow
        if size(A)[3:4] == size(B)[3:4] && A isa DenseArray && B isa DenseArray && C isa DenseArray
            nb = size(A, 3) * size(A, 4)
            r3(X) = Base.reshape(X, size(X, 1), size(X, 2), nb)
            _batched_gemm!(r3(C), r3(A), r3(B)) === nothing || return C
        end
        # Loop over the last two dimensions
        B1, B2 = size(A, 3), size(A, 4)
        for i in 1:B1, j in 1:B2
            mul!(view(C, :, :, i, j), view(A, :, :, i, j), view(B, :, :, i, j))
        end
        return C
    end
    
    error("Batch matmul! not implemented for $(ndims(A))D and $(ndims(B))D (Shapes: $(size(A)) and $(size(B)))")
end

function realize_view(data, st::ShapeTracker)
    (data isa HalfWeight || data isa HalfWeightN || data isa QuantWeight) && return data  # consumed whole, as a matmul weight
    # If buffer already matches logical size, it's likely already realized (common in interpreter)
    r_dims = realized_dims(st)
    # Already exactly this shape (e.g. a strided slice view): pass through unchanged
    # rather than reshaping it into a wrapper GPU broadcasts may not handle.
    data isa AbstractArray && ndims(data) == length(r_dims) &&
        all(i -> r_dims[i] isa Integer && size(data, i) == r_dims[i], 1:ndims(data)) && return data
    # A broadcastable view (an elided Expand: size 1 where the graph says k) is also
    # passed through; only elementwise ops read these, and broadcasting expands it.
    data isa BroadcastView && return data.a
    if length(data) == prod(Int.(Luminal.eval_dim.(r_dims))) && length(data) > 1
        return Base.reshape(data, Int.(Luminal.eval_dim.(r_dims))...)
    end

    if length(data) == 1
        if data isa Number
            data = [data] # Wrap scalar in an array to allow reshape
        end
        return Base.reshape(data, fill(1, length(st.indexes))...)
    end

    # 1. Reshape to physical rank (excluding fake dims)
    physical_dims = Int[]
    for i in 1:length(st.dims)
        if !st.fake[i]
            push!(physical_dims, Int(Luminal.eval_dim(st.dims[i])))
        end
    end
    
    if length(data) != prod(physical_dims)
        # Fallback: check if it matches logical size or something
        if length(data) == prod([Int(Luminal.eval_dim(d)) for d in st.dims])
             arr = Base.reshape(data, [Int(Luminal.eval_dim(d)) for d in st.dims]...)
        else
             return data
        end
    else
        arr = Base.reshape(data, physical_dims...)
    end
    
    # 2. Restore full rank by adding back fake dims as 1s
    full_arr = arr
    for i in 1:length(st.dims)
        if st.fake[i]
            sz = [size(full_arr)...]
            insert!(sz, i, 1)
            full_arr = Base.reshape(full_arr, sz...)
        end
    end
    
    # 3. Apply Indexing (Permutation and Rank selection)
    if st.indexes == 1:length(st.indexes)
        res = full_arr
    else
        res = PermutedDimsArray(full_arr, Tuple(st.indexes))
    end
    
    # 4. Apply Mask (Slicing)
    r_dims = realized_dims(st)
    if [size(res)...] != r_dims
        ranges = Any[]
        for i in 1:length(st.indexes)
            idx = st.indexes[i]
            s, e = st.mask[idx]
            dim_size = size(res, i)
            start = max(1, Int(s) + 1)
            stop = min(Int(e), dim_size)
            push!(ranges, start:stop)
        end
        res = view(res, ranges...)
    end

    return res
end

function execute_slice(input, ranges)
    res = input
    for (i, (s, e)) in enumerate(ranges)
        start_idx = max(1, s + 1)
        end_idx = min(size(res, i), e)
        res = selectdim(res, i, start_idx:end_idx)
    end
    return copy(res)
end

function execute_slice!(out, input, ranges)
    # In-place slice is tricky: copyto!(out, sliced_view)
    # Construct view
    v = input
    for (i, (s, e)) in enumerate(ranges)
        start_idx = max(1, s + 1)
        end_idx = min(size(v, i), e)
        v = selectdim(v, i, start_idx:end_idx)
    end
    copyto!(out, v)
    return out
end

function execute_pad(input, padding)
    old_size = size(input)
    new_size = [old_size[i] + padding[i][1] + padding[i][2] for i in 1:length(old_size)]
    res = similar(input, new_size...)
    fill!(res, 0)
    dest_ranges = [ (padding[i][1]+1):(padding[i][1]+old_size[i]) for i in 1:length(old_size) ]
    res[dest_ranges...] = input
    return res
end

function execute_pad!(out, input, padding)
    fill!(out, 0)
    old_size = size(input)
    dest_ranges = [ (padding[i][1]+1):(padding[i][1]+old_size[i]) for i in 1:length(old_size) ]
    # Copy input to center
    # out[dest_ranges...] = input # This might allocate?
    # view(out, dest_ranges...) .= input # In-place
    view(out, dest_ranges...) .= input
    return out
end


# --- Dispatchable Execute functions (Functional) ---

function align_broadcast_ranks(a, b)
    if ndims(a) == ndims(b) return a, b end
    if ndims(a) < ndims(b)
        return Base.reshape(a, ones(Int, ndims(b) - ndims(a))..., size(a)...), b
    else
        return a, Base.reshape(b, ones(Int, ndims(a) - ndims(b))..., size(b)...)
    end
end

execute_op(op::Add, a, b) = begin (a_a, b_a) = align_broadcast_ranks(a, b); a_a .+ b_a end
execute_op(op::Mul, a, b) = begin (a_a, b_a) = align_broadcast_ranks(a, b); a_a .* b_a end
execute_op(op::Mod, a, b) = begin (a_a, b_a) = align_broadcast_ranks(a, b); a_a .% b_a end
execute_op(op::LessThan, a, b) = begin (a_a, b_a) = align_broadcast_ranks(a, b); Float32.(a_a .< b_a) end
execute_op(op::FusedMulAdd, a, b, c) = (a .* b) .+ c
execute_op(op::FusedAddReLU, a, b) = Base.max.(a .+ b, 0)
execute_op(op::Log2, a) = log2.(a)
execute_op(op::Exp2, a) = exp2.(a)
execute_op(op::Sin, a) = sin.(a)
execute_op(op::Cos, a) = cos.(a)
execute_op(op::Sqrt, a) = sqrt.(a)
execute_op(op::Recip, a) = 1.0f0 ./ a
execute_op(op::Reshape, a) = Base.reshape(a, op.shape...)
execute_op(op::Permute, a) = Base.permutedims(a, op.dims)
execute_op(op::Contiguous, a) = copy(a)
execute_op(op::MatMul, a, b) = batch_matmul(a, b)
execute_op(op::SumReduce, a) = dropdims(Base.sum(a, dims=op.dim), dims=op.dim)
execute_op(op::MaxReduce, a) = dropdims(Base.maximum(a, dims=op.dim), dims=op.dim)
execute_op(op::Slice, a) = execute_slice(a, op.ranges)
execute_op(op::Pad, a) = execute_pad(a, op.padding)
execute_op(op::Constant, device) = to_device(op.value, device)

function execute_op(op::Expand, a)
    curr_sz = [size(a)...]
    insert!(curr_sz, op.dim, 1)
    reshaped = Base.reshape(a, curr_sz...)
    repeats = ones(Int, length(curr_sz))
    repeats[op.dim] = op.size
    return repeat(reshaped, outer=repeats)
end

execute_op(op::FusedElementwiseOp, inputs...) = broadcast(op.f, inputs...)

function execute_op(op::FlashAttentionOp, q, k, v)
    out = similar(q)
    return execute_op!(out, op, q, k, v)
end

function execute_op(op::Function, inputs...)
    if op.name == "InputTensor"
        return nothing 
    elseif op.name == "ARange"
        error("ARange requires context")
    elseif op.name == "Gather"
        indices = Int.(inputs[2]) .+ 1
        return inputs[1][indices, :]
    elseif op.name == "CumSum"
        return cumsum(inputs[1], dims=ndims(inputs[1]))
    else
        error("Function op with name $(op.name) not implemented.")
    end
end

# --- Dispatchable Execute functions (In-Place) ---

function _align_broadcast(x, out)
    nd = ndims(x)
    nout = ndims(out)
    if nd > 0 && nd < nout
        return Base.reshape(x, (size(x)..., ntuple(i->1, nout - nd)...))
    end
    return x
end

function execute_op!(out, op::Add, a, b)
    a_aligned = _align_broadcast(a, out)
    b_aligned = _align_broadcast(b, out)
    broadcast!(+, out, a_aligned, b_aligned)
end

function execute_op!(out, op::Mul, a, b)
    a_aligned = _align_broadcast(a, out)
    b_aligned = _align_broadcast(b, out)
    broadcast!(*, out, a_aligned, b_aligned)
end

execute_op!(out, op::Mod, a, b) = broadcast!(%, out, _align_broadcast(a, out), _align_broadcast(b, out))
execute_op!(out, op::LessThan, a, b) = broadcast!((x,y)->Float32(x<y), out, _align_broadcast(a, out), _align_broadcast(b, out))
execute_op!(out, op::FusedMulAdd, a, b, c) = broadcast!((x,y,z)->x*y+z, out, _align_broadcast(a, out), _align_broadcast(b, out), _align_broadcast(c, out))
execute_op!(out, op::FusedAddReLU, a, b) = broadcast!((x,y)->max(x+y, 0), out, _align_broadcast(a, out), _align_broadcast(b, out))
execute_op!(out, op::Log2, a) = broadcast!(log2, out, a)
execute_op!(out, op::Exp2, a) = broadcast!(exp2, out, a)
execute_op!(out, op::Sin, a) = broadcast!(sin, out, a)
execute_op!(out, op::Cos, a) = broadcast!(cos, out, a)
execute_op!(out, op::Sqrt, a) = broadcast!(sqrt, out, a)
execute_op!(out, op::Recip, a) = broadcast!(x->1.0f0/x, out, a)
execute_op!(out, op::ReLU, a) = broadcast!(x->max(x, zero(x)), out, a)
execute_op!(out, op::Max, a, b) = broadcast!(max, out, _align_broadcast(a, out), _align_broadcast(b, out))

function execute_op!(out, op::Reshape, a)
    # copyto! allows different shapes if length matches? 
    # Usually copyto!(dest, src).
    # ensure 'a' is viewed as matching length.
    if length(out) != length(a)
        error("Length mismatch in Reshape!: $(length(out)) vs $(length(a)). size(out)=$(size(out)), size(a)=$(size(a))")
    end
    copyto!(out, a)
    return out
end

function execute_op!(out, op::Permute, a)
    permutedims!(out, a, op.dims)
    return out
end

execute_op!(out, op::Contiguous, a) = copyto!(out, a)
execute_op!(out, op::MatMul, a, b) = batch_matmul!(out, a, b)
function execute_op!(out, op::MatMulF16, a, b)   # `a` is a HalfWeight(N) once compiled
    a isa HalfWeight && return _half_matmul!(out, a, b; group=op.group)
    a isa HalfWeightN && a.w isa ROCArray && return _gemm_ex_f16!(out, a, b)
    a isa HalfWeightN && return batch_matmul!(out, Float32.(a.w), b)
    return batch_matmul!(out, a, b)
end
execute_op!(out, op::MatMulQ8, a, b) =   # `a` is a QuantWeight once compiled
    a isa QuantWeight ? _q8_matmul!(out, a, b; group=op.group) : batch_matmul!(out, a, b)
execute_op!(out, op::MatMulT, a, b) = batch_matmul_t!(out, a, b, op.ta, op.tb)

# Reduce `a` over dimension `dim` into `out` (which has that dim dropped or 1).
# GPU: one workgroup per output element, strided accumulation, tree reduction in
# local memory, written straight into `out` -- no temporaries, so it is safe to
# record into a captured HIP graph. Elements are addressed linearly as
# a[i + pre*(k-1) + pre*r*(j-1)] for output (i, j) and reduced index k.
@kernel function _reduce_dim_kernel!(out, @Const(a), pre, r, op, init)
    g = @index(Group, Linear); lid = @index(Local, Linear); G = @groupsize()[1]
    i = (g - 1) % pre + 1
    j = (g - 1) ÷ pre + 1
    base = i + pre * r * (j - 1)
    acc = init
    k = lid
    while k <= r
        @inbounds acc = op(acc, Float32(a[base + pre * (k - 1)]))
        k += G
    end
    s = @localmem Float32 (256,)
    @inbounds s[lid] = acc
    @synchronize
    stride = G ÷ 2
    while stride > 0
        lid <= stride && (@inbounds s[lid] = op(s[lid], s[lid + stride]))
        @synchronize
        stride ÷= 2
    end
    lid == 1 && (@inbounds out[g] = s[1])
end

function _reduce_dim!(out, a, dim, op, init)
    if out isa AnyGPUArray
        pre = prod(size(a)[1:dim-1]; init=1)
        r = size(a, dim)
        G = r >= 256 ? 256 : max(32, nextpow(2, r))
        _reduce_dim_kernel!(KernelAbstractions.get_backend(out), G)(
            out, a, pre, r, op, init; ndrange = length(out) * G)
    else
        rsz = ntuple(d -> d == dim ? 1 : size(a, d), ndims(a))
        op === (+) ? sum!(Base.reshape(out, rsz), a) : maximum!(Base.reshape(out, rsz), a)
    end
    return out
end

execute_op!(out, op::SumReduce, a) = _reduce_dim!(out, a, op.dim, +, 0f0)
execute_op!(out, op::MaxReduce, a) = _reduce_dim!(out, a, op.dim, max, -Inf32)

execute_op!(out, op::Slice, a) = execute_slice!(out, a, op.ranges)
execute_op!(out, op::Pad, a) = execute_pad!(out, a, op.padding)
execute_op!(out, op::Constant, device) = copyto!(out, to_device(op.value, device))

function execute_op!(out, op::Expand, a)
    # expand(a) -> repeat to fills out.
    # Just broadcast!?
    # `out .= a` should work if dimensions align or broadcast rules apply.
    # Expand changes defaults.
    # Julia broadcast automatically expands singleton dims.
    # So if `a` has size 1 where `out` has size N, `out .= a` works.
    # BUT `Expand` op might insert a dimension that wasn't there?
    # If `a` is (N,), Expand(dim=2) -> (N, 1) -> (N, M).
    # `a` needs to be reshaped to (N, 1) first if it isn't already compatible.
    
    # Check if we need to reshape `a`
    # The `compile` realize_view logic might handle inputs? 
    # But `Expand` takes the direct input from another node.
    # In `execute_op`: `curr_sz = insert!(...); reshaped = reshape(a, ...)`
    # We should do the same here.
    
    curr_sz = [size(a)...]
    insert!(curr_sz, op.dim, 1)
    reshaped_a = Base.reshape(a, curr_sz...)
    
    # helper for broadcast copy
    # copyto!(out, reshaped_a) -> this only works if sizes match.
    # broadcast!(identity, out, reshaped_a) -> this does expansion!
    broadcast!(identity, out, reshaped_a)
    return out
end

# Backend-agnostic Flash Attention Forward
@kernel function flash_attn_fwd_kernel(O, Q, K, V, B, H, N, d, scale, causal)
    # sequence index (1 to N) and batch*head index (1 to B*H)
    group = @index(Group, Cartesian)
    i = group[1]
    bh = group[2]
    
    b = (bh - 1) ÷ H + 1
    h = (bh - 1) % H + 1
    tid = @index(Local, Linear)
    
    # Shared memory for row accumulation
    # Note: Using a fixed size for d (head_dim) and acc (reduction) to ensure compatibility.
    # Typical head dims: 64, 128. acc max threads: 256.
    s_Q = @localmem Float32 (128,)
    s_O = @localmem Float32 (128,)
    s_acc = @localmem Float32 (256,)

    # Load Q row
    if tid <= d
        s_Q[tid] = Q[tid, i, h, b]
        s_O[tid] = 0.0f0
    end
    
    m_i = -1f32 / 0f32
    l_i = 0.0f0
    
    @synchronize
    
    for j in 1:N
        if causal && j > i continue end
        
        # 1. Compute S_ij = sum(Q[i, :] * K[j, :]) * scale
        val = 0.0f0
        if tid <= d
            val = s_Q[tid] * K[tid, j, h, b]
        end
        s_acc[tid] = val
        @synchronize
        
        # Parallel reduction for dot product
        s = 1
        # Use groupsize directly if available
        gs = @groupsize()[1]
        while s < gs
            s *= 2
        end
        s ÷= 2
        while s >= 1
            if tid <= s && tid + s <= gs
                s_acc[tid] += s_acc[tid + s]
            end
            @synchronize
            s ÷= 2
        end
        dot = s_acc[1] * scale
        
        # 2. Update stats (Online Softmax)
        m_curr = dot
        m_next = max(m_i, m_curr)
        p = exp(m_curr - m_next)
        scale_old = exp(m_i - m_next)
        if isnan(scale_old) scale_old = 0.0f0 end
        
        l_next = l_i * scale_old + p
        
        # 3. Update Output row (unnormalized)
        if tid <= d
            s_O[tid] = s_O[tid] * scale_old + p * V[tid, j, h, b]
        end
        
        m_i = m_next
        l_i = l_next
        @synchronize
    end
    
    # Final normalization and write back
    if tid <= d
        O[tid, i, h, b] = s_O[tid] / l_i
    end
end

# CPU Fallback for Flash Attention
# Same layout as the GPU kernel: (HeadDim, Seq, Head, Batch)
function flash_attn_cpu(q, k, v, scale, causal)
    D, N, H, B = size(q)
    out = similar(q)
    for b in 1:B, h in 1:H
        for i in 1:N
            m_i = -Inf32
            l_i = 0.0f0
            o_row = zeros(Float32, D)
            for j in 1:N
                if causal && j > i continue end
                # Dot product
                dot = sum(q[:, i, h, b] .* k[:, j, h, b]) * scale
                # Online softmax
                m_next = max(m_i, dot)
                p = exp(dot - m_next)
                scale_old = exp(m_i - m_next)
                if isnan(scale_old) scale_old = 0.0f0 end
                
                o_row = o_row .* scale_old .+ p .* v[:, j, h, b]
                l_i = l_i * scale_old + p
                m_i = m_next
            end
            out[:, i, h, b] = o_row ./ l_i
        end
    end
    return out
end

function unfold_1d_cpu!(out_2d, a_2d, O, K, S, D)
    BC = size(out_2d, 1)
    for bc in 1:BC
        for o in 1:O
            for k in 1:K
                a_idx = (o - 1) * S + (k - 1) * D + 1
                out_2d[bc, o, k] = a_2d[bc, a_idx]
            end
        end
    end
    return
end

@kernel function unfold_1d_kernel(out_2d, a_2d, O, K, S, D)
    k, o, bc = @index(Global, NTuple)
    
    if k <= K && o <= O && bc <= size(a_2d, 1)
        a_idx = (o - 1) * S + (k - 1) * D + 1
        @inbounds out_2d[bc, o, k] = a_2d[bc, a_idx]
    end
end

function execute_op!(out, op::Unfold, a)
    fill!(out, 0)
    spatial = length(op.kernel_shape)
    if spatial == 1
        K = op.kernel_shape[1]
        S = op.stride_shape[1]
        D = op.dilation_shape[1]
        
        O = size(out)[end-1]
        BC = prod(size(out)[1:end-2])
        L = size(a)[end]
        
        out_2d = Base.reshape(out, BC, O, K)
        a_2d = Base.reshape(a, BC, L)
        
        if a isa AnyGPUArray
            backend = KernelAbstractions.get_backend(a)
            kernel! = unfold_1d_kernel(backend)
            kernel!(out_2d, a_2d, O, K, S, D, ndrange=(K, O, BC))
            KernelAbstractions.synchronize(backend)
        else
            unfold_1d_cpu!(out_2d, a_2d, O, K, S, D)
        end
    else
        error("Unfold > 1D not implemented")
    end
    return out
end

function execute_op(op::Unfold, a)
    spatial = length(op.kernel_shape)
    rank = ndims(a)
    batch_len = rank - spatial - 1
    out_spatial = Int[]
    for i in 1:spatial
        s_i = size(a, batch_len + 1 + i)
        k_i = op.kernel_shape[i]
        d_i = op.dilation_shape[i]
        st_i = op.stride_shape[i]
        o_i = (s_i - d_i * (k_i - 1) - 1) ÷ st_i + 1
        push!(out_spatial, o_i)
    end
    out_shape = [size(a)[1:batch_len+1]..., out_spatial..., op.kernel_shape...]
    out = similar(a, out_shape...)
    return execute_op!(out, op, a)
end

function execute_op!(out, op::FusedElementwiseOp, inputs...)
    try
        broadcast!(op.f, out, inputs...)
    catch e
        println("FAILED TO COMPILE OR EXECUTE KERNEL FOR: ", op.name)
        println("Output Type: ", typeof(out), " Size: ", size(out))
        for (i, inp) in enumerate(inputs)
            println("Input $i Type: ", typeof(inp), " Size: ", size(inp))
        end
        rethrow(e)
    end
end

function execute_op!(out, op::FlashAttentionOp, q, k, v)
    if q isa AnyGPUArray
        d, N, H, B = size(q)
        backend = KernelAbstractions.get_backend(q)
        # Grid: (N, B*H)
        # Block: (max(d, 32),)
        threads = max(32, 1 << (31 - leading_zeros(d - 1) + 1)) # Next power of 2
        # Ensure we don't exceed max threads
        threads = min(threads, 256)
        
        kernel! = flash_attn_fwd_kernel(backend)
        kernel!(out, q, k, v, B, H, N, d, op.scale, op.causal, 
                ndrange=(N * threads, B*H), workgroupsize=(threads, 1))
        KernelAbstractions.synchronize(backend)
    else
        copyto!(out, flash_attn_cpu(q, k, v, op.scale, op.causal))
    end
    return out
end

@kernel function _gather_kernel(out, x, indices)
    i = @index(Global)
    if i <= length(out)
        S = length(indices)
        row = (i - 1) % S + 1
        col = (i - 1) ÷ S + 1
        
        idx_val = Int(indices[row]) + 1
        if idx_val >= 1 && idx_val <= size(x, 1)
            @inbounds out[i] = x[idx_val, col]
        else
            @inbounds out[i] = 0.0f0
        end
    end
end

function execute_op!(out, op::Function, inputs...)
    if op.name == "InputTensor"
        return nothing
    elseif op.name == "ARange"
        error("ARange requires context")
    elseif op.name == "Gather"
        # inputs[1][indices, :] -> out
        if typeof(out) <: AnyGPUArray
            backend = KernelAbstractions.get_backend(out)
            kernel! = _gather_kernel(backend)
            kernel!(out, inputs[1], inputs[2], ndrange=length(out))
        else
            indices = Int.(inputs[2]) .+ 1
            copyto!(out, inputs[1][indices, :])
        end
    elseif op.name == "CumSum"
        if inputs[1] isa AMDGPU.ROCArray
            AMDGPU.@allowscalar cumsum!(out, inputs[1], dims=ndims(inputs[1]))
        else
            cumsum!(out, inputs[1], dims=ndims(inputs[1]))
        end
    else
        error("Function op with name $(op.name) not implemented.")
    end
    return out
end

function execute(graph::Graph, output_ids::Vector{Int}, initial_inputs::Dict, device::AbstractDevice=get_device())
    results = to_device(initial_inputs, device)
    
    # Include pre-loaded weights from the graph
    for ((node_id, output_idx), data) in graph.tensors
        if !haskey(results, node_id) && output_idx == 1
            results[node_id] = data
        end
    end

    for (node_id, node) in enumerate(graph.nodes)
        haskey(results, node_id) && continue

        op = node.op
        
        # Realize each input according to its ShapeTracker
        input_values = []
        for (id, _, st) in node.inputs
            raw_data = results[id]
            push!(input_values, realize_view(raw_data, st))
        end
        
        node_shape = graph.shapes[node_id]

        if op isa Function && op.name == "ARange"
             n = eval_dim(realized_dims(node_shape)[1])
             current_result = to_device(Float32.(collect(0:n-1)), device)
        else
            # For Constant, we need device
            if op isa Constant
                 current_result = execute_op(op, device)
            elseif hasmethod(execute_op!, Tuple{Any, typeof(op), map(typeof, input_values)...})
                 # Same in-place kernels as the compiled path, into an output of the
                 # node's shape (so every op compile() supports runs here too)
                 dims = Int[eval_dim(d) for d in realized_dims(node_shape)]
                 current_result = zero_tensor(device, Float32, dims...)
                 execute_op!(current_result, op, input_values...)
            else
                 current_result = execute_op(op, input_values...)
            end
        end
        
        results[node_id] = current_result
    end

    # Return a dict of results
    final_results = Dict{Int, Any}()
    for id in output_ids
        final_results[id] = from_device(results[id])
    end
    return final_results
end

function execute(graph::Graph, output_id::Int, initial_inputs::Dict, device::AbstractDevice=get_device())
    res_dict = execute(graph, [output_id], initial_inputs, device)
    return res_dict[output_id]
end

function eval_dim(d, sym_vals::Dict{Symbol, Int}=Dict{Symbol, Int}())
    if d isa Int
        return d
    elseif d isa BasicSymbolic
        # Substitute symbols using the provided dictionary
        # Convert Dict{Symbol, Int} to Dict{Any, Any} for SymbolicUtils.substitute
        subs_dict = Dict{Any, Any}(Sym{Int}(k) => v for (k, v) in sym_vals)
        val = substitute(d, subs_dict)
        if val isa Int
            return val
        else
            error("Cannot evaluate symbolic dimension $d to Int. Missing context? Residual part: $val")
        end
    else
        return Int(d)
    end
end
