# Implements a simple interpreter to execute a computation graph.



using AMDGPU
using LinearAlgebra
using GPUArrays
using KernelAbstractions
using KernelAbstractions.Extras: @unroll


export execute_op, execute_op!, realize_view, execute, eval_dim, FusedElementwiseOp, HalfWeight, HalfWeightN, QuantWeight, Q4Weight, Q5Weight, IQ4Weight, Q6Weight, NativeWeight, Int4, dequantize

# Functional batched matmul, left-aligned as the MatMul op: (M, K, batch...) *
# (K, N, batch...) -> (M, N, batch...), batch dims broadcasting.
function batch_matmul(A, B)
    ndims(A) == 2 && ndims(B) == 2 && return A * B
    bA, bB = size(A)[3:end], size(B)[3:end]
    n = max(length(bA), length(bB))
    get1(t, k) = k <= length(t) ? t[k] : 1
    batch = ntuple(k -> max(get1(bA, k), get1(bB, k)), n)
    C = similar(B, promote_type(eltype(A), eltype(B)), size(A, 1), size(B, 2), batch...)
    return batch_matmul!(C, A, B)
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
@kernel function _rmsnorm_kernel!(out, @Const(x), @Const(w), eps, ::Val{T}) where {T}
    col = @index(Group, Linear); lid = @index(Local, Linear)
    H = @uniform size(x, 1)
    red = @localmem Float32 (T,)
    part = @localmem Float32 (32,)
    acc = 0f0
    i = lid
    while i <= H
        @inbounds v = x[i, col]
        acc += v * v
        i += T
    end
    @inbounds red[lid] = acc
    @synchronize
    # two-level sum (32 partial sums, then one pass over them): 3 barriers instead of log2(T)
    if lid <= 32
        a = 0f0
        j = lid
        while j <= T
            @inbounds a += red[j]
            j += 32
        end
        @inbounds part[lid] = a
    end
    @synchronize
    s = 0f0
    for j in 1:32
        @inbounds s += part[j]
    end
    r = 1f0 / sqrt(s / H + eps)
    i = lid
    while i <= H
        @inbounds out[i, col] = eltype(out)(x[i, col] * r * w[i])
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
    T = H >= 2048 ? 512 : 256        # one workgroup per column; wide hidden sizes want more threads
    _rmsnorm_kernel!(KernelAbstractions.get_backend(o2), T)(o2, x2, wv, op.epsilon, Val(T); ndrange = size(x2, 2) * T)
    return out
end

# --- L2 normalization over dim 1 ---------------------------------------------------
# One wave (32 lanes) per column: each lane sums the squares of every 32nd element, a shuffle tree adds the
# lanes, and the normalized values are written in the output's element type.
function _l2norm_kernel!(out, x, D, ncols, eps)
    tid = workitemIdx().x - 1
    lane = tid % 32
    col = (workgroupIdx().x - 1) * (workgroupDim().x ÷ 32) + tid ÷ 32
    col < ncols || return
    base = col * D
    acc = 0f0
    i = lane
    while i < D
        v = Float32(@inbounds x[base + i + 1])
        acc += v * v
        i += 32
    end
    acc += AMDGPU.Device.shfl_xor(acc, Cint(16))
    acc += AMDGPU.Device.shfl_xor(acc, Cint(8))
    acc += AMDGPU.Device.shfl_xor(acc, Cint(4))
    acc += AMDGPU.Device.shfl_xor(acc, Cint(2))
    acc += AMDGPU.Device.shfl_xor(acc, Cint(1))
    r = 1f0 / sqrt(acc + eps)
    T = eltype(out)
    i = lane
    while i < D
        @inbounds out[base + i + 1] = T(Float32(x[base + i + 1]) * r)
        i += 32
    end
    return
end

function execute_op!(out, op::L2NormOp, x)
    D = size(x, 1)
    xc = x isa DenseArray ? x : copy(x)
    if !(out isa AMDGPU.ROCArray)
        x2 = Base.reshape(xc, D, :); o2 = Base.reshape(out, D, :)
        for c in axes(x2, 2)
            v = Float32.(view(x2, :, c))
            o2[:, c] .= v .* (1f0 / sqrt(sum(abs2, v) + op.epsilon))
        end
        return out
    end
    ncols = length(xc) ÷ D
    @roc groupsize=256 gridsize=cld(ncols, 8) _l2norm_kernel!(vec(out), vec(xc), D, ncols, op.epsilon)
    return out
end

# --- SwiGLU: silu(gate) * up -----------------------------------------------------------
# (a scalar function: `-g` on an array would allocate a temporary, which a captured HIP graph cannot do)
@inline _swiglu(g, u) = (g / (1f0 + exp(-g))) * u
@inline _swiglu(g::Float16, u::Float16) = _swiglu(Float32(g), Float32(u))
# Float16 in and out (the gate / up GEMM results and the input of the down GEMM): eight elements per thread with 16-byte
# loads and stores, so it runs at the memory bandwidth (a broadcast over Float16 arrays loads two bytes at a time).
function _swiglu_f16_kernel!(out, g, u, n8)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    i <= n8 || return
    @inbounds begin
        gv = g[i]; uv = u[i]
        out[i] = ntuple(k -> VecElement(Float16(_swiglu(Float32(gv[k].value), Float32(uv[k].value)))), Val(8))
    end
    return
end
const _V8F16 = NTuple{8,VecElement{Float16}}
function execute_op!(out, op::SwiGLUOp, g, u)
    if out isa AMDGPU.ROCArray{Float16} && g isa AMDGPU.ROCArray{Float16} && u isa AMDGPU.ROCArray{Float16} &&
       length(out) == length(g) == length(u) && length(out) % 8 == 0
        n8 = length(out) ÷ 8
        @roc groupsize=256 gridsize=cld(n8, 256) _swiglu_f16_kernel!(reinterpret(_V8F16, vec(out)), reinterpret(_V8F16, vec(g)), reinterpret(_V8F16, vec(u)), n8)
        return out
    end
    out .= _swiglu.(g, u)
    return out
end

# --- RMSNorm times silu of a gate -----------------------------------------------------
# One wave (32 lanes) per column of D elements: sum of squares by a shuffle tree, then out = x * r * w * (z / (1 + exp(-z))),
# written in the output's element type (Float16 when it feeds a matrix-core GEMM).
function _gated_rmsnorm_kernel!(out, x, w, z, D, ncols, eps, nv, tnk)
    tid = workitemIdx().x - 1
    lane = tid % 32
    col = (workgroupIdx().x - 1) * (workgroupDim().x ÷ 32) + tid ÷ 32
    col < ncols || return
    base = col * D
    obase = base
    if tnk > 0                                   # tiled output slots: head h = k G + g goes to g nk + k
        h = col % nv; G = nv ÷ tnk
        obase = (col - h + (h % G) * tnk + h ÷ G) * D
    end
    acc = 0f0
    i = lane
    while i < D
        v = Float32(@inbounds x[base + i + 1])
        acc += v * v
        i += 32
    end
    acc += AMDGPU.Device.shfl_xor(acc, Cint(16))
    acc += AMDGPU.Device.shfl_xor(acc, Cint(8))
    acc += AMDGPU.Device.shfl_xor(acc, Cint(4))
    acc += AMDGPU.Device.shfl_xor(acc, Cint(2))
    acc += AMDGPU.Device.shfl_xor(acc, Cint(1))
    r = 1f0 / sqrt(acc / D + eps)
    T = eltype(out)
    i = lane
    while i < D
        zi = Float32(@inbounds z[base + i + 1])
        @inbounds out[obase + i + 1] = T(Float32(x[base + i + 1]) * r * Float32(w[i + 1]) * (zi / (1f0 + exp(-zi))))
        i += 32
    end
    return
end

function execute_op!(out, op::GatedRMSNormOp, x, w, z)
    D = size(x, 1)
    xc = x isa DenseArray ? x : copy(x)
    zc = z isa DenseArray ? z : copy(z)
    if !(out isa AMDGPU.ROCArray)
        x2 = Base.reshape(xc, D, :); z2 = Base.reshape(zc, D, :); o2 = Base.reshape(out, D, :)
        nvh = op.nv
        for c in axes(x2, 2)
            v = Float32.(view(x2, :, c)); zz = Float32.(view(z2, :, c))
            oc = c
            if op.tiled_nk > 0
                h = (c - 1) % nvh; G = nvh ÷ op.tiled_nk
                oc = c - h + (h % G) * op.tiled_nk + h ÷ G
            end
            o2[:, oc] .= v .* (1f0 / sqrt(sum(abs2, v) / D + op.epsilon)) .* Float32.(vec(w)) .* (zz ./ (1f0 .+ exp.(-zz)))
        end
        return out
    end
    ncols = length(xc) ÷ D
    @roc groupsize=256 gridsize=cld(ncols, 8) _gated_rmsnorm_kernel!(vec(out), vec(xc), vec(w), vec(zc), D, ncols, op.epsilon, op.nv, op.tiled_nk)
    return out
end

# --- Softmax over dim 1 --------------------------------------------------------
# One workgroup per column (all dims after the first): the column max and the
# sum of exponentials with tree reductions, then the normalized write.
@kernel function _softmax_kernel!(out, @Const(x))
    col = @index(Group, Linear); lid = @index(Local, Linear)
    T = @uniform @groupsize()[1]
    L = @uniform size(x, 1)
    red = @localmem Float32 (256,)
    stats = @localmem Float32 (2,)
    m = -Inf32
    i = lid
    while i <= L
        @inbounds m = max(m, x[i, col])
        i += T
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
    acc = 0f0
    i = lid
    while i <= L
        @inbounds acc += exp(x[i, col] - stats[1])
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
    lid == 1 && (@inbounds stats[2] = 1f0 / red[1])
    @synchronize
    i = lid
    while i <= L
        @inbounds out[i, col] = exp(x[i, col] - stats[1]) * stats[2]
        i += T
    end
end

function execute_op!(out, op::SoftmaxOp, x)
    L = size(x, 1)
    xc = x isa DenseArray ? x : copy(x)
    x2 = Base.reshape(xc, L, :)
    o2 = Base.reshape(out, L, :)
    if !(out isa AnyGPUArray)          # CPU: plain loops (the kernel's reductions are GPU-shaped)
        for c in axes(x2, 2)
            col = view(x2, :, c)
            e = exp.(col .- maximum(col))
            o2[:, c] .= e ./ sum(e)
        end
        return out
    end
    _softmax_kernel!(KernelAbstractions.get_backend(o2), 256)(o2, x2; ndrange = size(x2, 2) * 256)
    return out
end

# --- Decode attention ---------------------------------------------------------
# One workgroup per (query head, batch). Scores for the n = pos + 1 positions
# (cache slots 1..pos, then the new token) live in local memory: dot products,
# then a parallel max and sum for the softmax, then the probability-weighted sum
# of V with the workgroup split into G ÷ D slices over positions.
const DECODE_ATTN_MAX_CTX = 8192
@kernel function _decode_attn_kernel!(out, @Const(q), pk, pv, @Const(kn), @Const(vn),
                                      @Const(posv), scale, Gq, write_cache, window)
    # Values shared across barriers must be @uniform (the CPU backend splits the
    # kernel into loops at each @synchronize); the softmax statistics go through
    # local memory (`stats`) for the same reason.
    grp = @index(Group, Linear); lid = @index(Local, Linear)
    T = @uniform @groupsize()[1]
    D = @uniform size(q, 1)
    H = @uniform size(q, 3)
    S = @uniform size(q, 2)                          # new tokens per sequence (1, or a speculative step's 2+)
    h = @uniform (@index(Group, Linear) - 1) % size(q, 3) + 1
    s = @uniform ((@index(Group, Linear) - 1) ÷ size(q, 3)) % size(q, 2) + 1
    b = @uniform (@index(Group, Linear) - 1) ÷ (size(q, 3) * size(q, 2)) + 1
    kv = @uniform (h - 1) ÷ Gq + 1
    pos = @uniform unsafe_trunc(Int, @inbounds posv[length(posv) == 1 ? 1 : b])
    n = @uniform pos + s                             # attended positions: the cache's pos slots, new tokens 1..s
    lo = @uniform window > 0 ? max(1, n - window + 1) : 1   # first position attended to
    # write_cache: the first query head of each KV group stores this token's K/V in
    # slot pos + 1 (no workgroup reads that slot from the cache).
    if write_cache && (h - 1) % Gq == 0 && lid <= D
        @inbounds pk[lid, pos + s, kv, b] = kn[lid, s, kv, b]
        @inbounds pv[lid, pos + s, kv, b] = vn[lid, s, kv, b]
    end
    sc = @localmem Float32 (DECODE_ATTN_MAX_CTX,)
    red = @localmem Float32 (256,)
    stats = @localmem Float32 (2,)
    # 1. scores
    j = lid
    while j <= n
        acc = 0f0
        if j < lo
            acc = -Inf32                                   # outside the window
        elseif j <= pos
            for d in 1:D
                @inbounds acc += q[d, s, h, b] * pk[d, j, kv, b]
            end
        else
            for d in 1:D
                @inbounds acc += q[d, s, h, b] * kn[d, j - pos, kv, b]
            end
        end
        @inbounds sc[j] = j < lo ? -Inf32 : acc * scale
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
                @inbounds acc += sc[j] * vn[d, j - pos, kv, b]
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
        @inbounds out[lid, h, s, b] = tot * stats[2]
    end
end

function execute_op!(out, op::DecodeAttention, q, pk, pv, kn, vn, posv)
    D, S, H, B = size(q, 1), size(q, 2), size(q, 3), size(q, 4)
    KVH = size(pk, 3)
    if !(out isa AnyGPUArray)          # CPU: plain loops (the kernel's reductions are GPU-shaped)
        G = H ÷ KVH
        o4 = Base.reshape(out, D, H, S, B)
        for b in 1:B, h in 1:H, t in 1:S
            pos = Int(posv[length(posv) == 1 ? 1 : b])
            kv = (h - 1) ÷ G + 1
            n = pos + t
            lo = op.window > 0 ? max(1, n - op.window + 1) : 1
            kat(j, d) = j <= pos ? pk[d, j, kv, b] : kn[d, j - pos, kv, b]
            vat(j, d) = j <= pos ? pv[d, j, kv, b] : vn[d, j - pos, kv, b]
            s = Float32[j < lo ? -Inf32 : op.scale * sum(q[d, t, h, b] * kat(j, d) for d in 1:D) for j in 1:n]
            p = exp.(s .- maximum(s)); p ./= sum(p)
            for d in 1:D
                o4[d, h, t, b] = sum(p[j] * vat(j, d) for j in 1:n)
            end
        end
        if op.write_cache
            for b in 1:B
                pos = Int(posv[length(posv) == 1 ? 1 : b])
                pk[:, pos + 1:pos + S, :, b] .= kn[:, 1:S, :, b]
                pv[:, pos + 1:pos + S, :, b] .= vn[:, 1:S, :, b]
            end
        end
        return out
    end
    op.window == 0 && decode_attention_split!(out, q, pk, pv, kn, vn, posv, op.scale, op.write_cache) && return out
    op.window == 0 && op.write_cache && out isa ROCArray && S * (H ÷ KVH) > DA_MAX_ROWS &&
        chunk_attention!(out, q, pk, pv, kn, vn, posv, op.scale) && return out
    eltype(pk) === Int8 && error("DecodeAttention: an int8 KV cache needs the split-K kernel (AMD GPU, no window, <= 24 query rows per KV head, or a write_cache chunk step)")
    size(pk, 2) <= DECODE_ATTN_MAX_CTX || error("DecodeAttention: cache longer than $(DECODE_ATTN_MAX_CTX)")
    (D <= 256 && 256 % D == 0) || error("DecodeAttention: head_dim must divide 256")
    o4 = Base.reshape(out, D, H, S, B)
    _decode_attn_kernel!(KernelAbstractions.get_backend(o4), 256)(
        o4, q, pk, pv, kn, vn, posv, op.scale, H ÷ KVH, op.write_cache, op.window; ndrange = H * S * B * 256)
    return out
end

# --- Linear attention: causal convolution and the gated delta rule -------------
# (see CausalConv and DeltaRule in Ops.jl)

_silu(x) = x / (1 + exp(-x))

@kernel function _causal_conv_kernel!(out, @Const(x), state, @Const(w), @Const(lens), snap, snap_t, ::Val{K}, fresh) where K
    I = @index(Global, Linear)
    C = size(x, 1); S = size(x, 2)
    c = (I - 1) % C + 1; b = (I - 1) ÷ C + 1
    L = min(S, unsafe_trunc(Int, @inbounds lens[b]))
    # the last K inputs, oldest first
    win = ntuple(j -> fresh ? 0f0 : @inbounds(Float32(state[c, j, b])), Val(K))
    for t in 1:L
        @inbounds xt = Float32(x[c, t, b])
        win = (Base.tail(win)..., xt)
        acc = 0f0
        for j in 1:K
            @inbounds acc += Float32(w[c, 1, j]) * win[j]
        end
        @inbounds out[c, t, b] = _silu(acc)
        if t <= snap_t          # snapshot slot t (slot t of the batch dim) = the state after t tokens
            for j in 1:K
                @inbounds snap[c, j, b + (t - 1) * size(x, 3)] = win[j]
            end
        end
    end
    for tt in L+1:snap_t        # fewer tokens consumed than snapshots: those slots hold the final state
        for j in 1:K
            @inbounds snap[c, j, b + (tt - 1) * size(x, 3)] = win[j]
        end
    end
    for t in L+1:S
        @inbounds out[c, t, b] = 0f0
    end
    for j in 1:K
        @inbounds state[c, j, b] = win[j]
    end
end

@kernel function _last_token_kernel!(out, @Const(x), @Const(lens))
    I = @index(Global, Cartesian)
    h = I[1]; b = I[3]
    t = clamp(Int(lens[b]), 1, size(x, 2))
    @inbounds out[h, 1, b] = x[h, t, b]
end

function execute_op!(out, op::LastToken, x, lens)
    length(out) == 0 && return out
    _last_token_kernel!(KernelAbstractions.get_backend(out), 256)(out, _dense(x), _dense(lens); ndrange=size(out))
    return out
end

# One workgroup per (position, batch) column: each thread scans a strided slice for its best (value, index),
# then a tree reduction keeps the larger value, and the smaller index on ties (the first maximum).
@kernel function _argmax_kernel!(out, @Const(x))
    grp = @index(Group, Linear); lid = @index(Local, Linear)
    T = @uniform @groupsize()[1]
    V = @uniform size(x, 1)
    col = @uniform (@index(Group, Linear) - 1)
    s = @uniform col % size(x, 2) + 1
    b = @uniform col ÷ size(x, 2) + 1
    rv = @localmem Float32 (256,)
    ri = @localmem Int32 (256,)
    bv = -Inf32; bi = Int32(typemax(Int32))
    i = lid
    while i <= V
        @inbounds v = x[i, s, b]
        if v > bv || (v == bv && Int32(i) < bi)
            bv = v; bi = Int32(i)
        end
        i += T
    end
    @inbounds rv[lid] = bv; @inbounds ri[lid] = bi
    @synchronize
    stride = T ÷ 2
    while stride > 0
        if lid <= stride
            @inbounds ov = rv[lid + stride]; @inbounds oi = ri[lid + stride]
            @inbounds cv = rv[lid]; @inbounds ci = ri[lid]
            if ov > cv || (ov == cv && oi < ci)
                @inbounds rv[lid] = ov; @inbounds ri[lid] = oi
            end
        end
        @synchronize
        stride ÷= 2
    end
    lid == 1 && (@inbounds out[s, b] = Float32(ri[1] - 1))
end

# Wide vocabularies: P workgroups per column each scan a strided slice (one workgroup per column leaves
# the GPU idle: 0.34 ms for 2 columns of 248k), then a second kernel merges the P partial results.
const ARGMAX_PARTS = 64
const _ARGMAX_WORKSPACE = Dict{Any, Any}()
@kernel function _argmax_part_kernel!(pv, pidx, @Const(x), ::Val{P}) where {P}
    grp = @index(Group, Linear); lid = @index(Local, Linear)
    T = @uniform @groupsize()[1]
    V = @uniform size(x, 1)
    g = grp - 1
    col = g ÷ P
    part = g % P
    s = col % size(x, 2) + 1
    b = col ÷ size(x, 2) + 1
    rv = @localmem Float32 (256,)
    ri = @localmem Int32 (256,)
    bv = -Inf32; bi = Int32(typemax(Int32))
    i = part * T + lid
    while i <= V
        @inbounds v = x[i, s, b]
        if v > bv || (v == bv && Int32(i) < bi)
            bv = v; bi = Int32(i)
        end
        i += P * T
    end
    @inbounds rv[lid] = bv; @inbounds ri[lid] = bi
    @synchronize
    stride = T ÷ 2
    while stride > 0
        if lid <= stride
            @inbounds ov = rv[lid + stride]; @inbounds oi = ri[lid + stride]
            @inbounds cv = rv[lid]; @inbounds ci = ri[lid]
            if ov > cv || (ov == cv && oi < ci)
                @inbounds rv[lid] = ov; @inbounds ri[lid] = oi
            end
        end
        @synchronize
        stride ÷= 2
    end
    if lid == 1
        @inbounds pv[part + 1, col + 1] = rv[1]; @inbounds pidx[part + 1, col + 1] = ri[1]
    end
end

@kernel function _argmax_final_kernel!(out, @Const(pv), @Const(pidx), ::Val{P}) where {P}
    col = @index(Group, Linear); lid = @index(Local, Linear)
    bv = -Inf32; bi = Int32(typemax(Int32))
    if lid <= P
        @inbounds bv = pv[lid, col]; @inbounds bi = pidx[lid, col]
    end
    rv = @localmem Float32 (P,)
    ri = @localmem Int32 (P,)
    lid <= P && (@inbounds rv[lid] = bv; @inbounds ri[lid] = bi)
    @synchronize
    stride = P ÷ 2
    while stride > 0
        if lid <= stride
            @inbounds ov = rv[lid + stride]; @inbounds oi = ri[lid + stride]
            @inbounds cv = rv[lid]; @inbounds ci = ri[lid]
            if ov > cv || (ov == cv && oi < ci)
                @inbounds rv[lid] = ov; @inbounds ri[lid] = oi
            end
        end
        @synchronize
        stride ÷= 2
    end
    lid == 1 && (@inbounds out[col] = Float32(ri[1] - 1))
end

function execute_op!(out, op::ArgMax, x)
    V, S, B = size(x, 1), size(x, 2), size(x, 3)
    if !(out isa AnyGPUArray)
        for b in 1:B, s in 1:S
            out[s, b] = Float32(argmax(view(x, :, s, b)) - 1)
        end
        return out
    end
    backend = KernelAbstractions.get_backend(out)
    if V >= 32 * 256 * ARGMAX_PARTS ÷ 8 && ispow2(ARGMAX_PARTS) && get(ENV, "LUMINAL_ARGMAX_PARTS", "1") != "0"
        P = ARGMAX_PARTS; ncol = S * B
        # (reused across calls: a captured graph bakes in the pointers)
        pv, pidx = get!(_ARGMAX_WORKSPACE, (typeof(out), ncol)) do
            (similar(out, Float32, P, ncol), similar(out, Int32, P, ncol))
        end
        _argmax_part_kernel!(backend, 256)(pv, pidx, _dense(x), Val(P); ndrange = ncol * P * 256)
        _argmax_final_kernel!(backend, P)(Base.reshape(out, :), pv, pidx, Val(P); ndrange = ncol * P)
        return out
    end
    _argmax_kernel!(backend, 256)(out, _dense(x); ndrange = S * B * 256)
    return out
end

function execute_op!(out, op::CausalAttention, q, k, v)
    D = size(q, 1)
    if out isa AMDGPU.ROCArray && wmma_supported() && D % 32 == 0
        flash_prefill!(out, q, k, v, op.scale)
    else
        copyto!(out, causal_attention_ref(q, k, v, op.scale))
    end
    return out
end

function execute_op!(out, op::CausalConv, x, state, w, lens, snap = state)
    C, S, B = size(x, 1), size(x, 2), size(x, 3)
    K = size(w, 3)
    if !(out isa AnyGPUArray)
        fill!(out, 0)
        for b in 1:B, c in 1:C
            win = op.fresh ? zeros(Float32, K) : Float32[state[c, j, b] for j in 1:K]
            L = min(S, Int(lens[b]))
            for t in 1:L
                win = vcat(win[2:end], Float32(x[c, t, b]))
                out[c, t, b] = _silu(sum(Float32(w[c, 1, j]) * win[j] for j in 1:K))
                t <= op.snap_t && (snap[c, :, b + (t - 1) * B] .= win)
            end
            for tt in L+1:op.snap_t
                snap[c, :, b + (tt - 1) * B] .= win
            end
            state[c, :, b] .= win
        end
        return out
    end
    _causal_conv_kernel!(KernelAbstractions.get_backend(out), 256)(out, _dense(x), state, _dense(w), _dense(lens),
                                                                    snap, op.snap_t, Val(K), op.fresh; ndrange = C * B)
    return out
end

# One workgroup per (value head, batch), one thread per value dim j, owning
# column j of S (row j of the stored S^T, so a step's accesses are coalesced).
const DELTA_MAX_DK = 256
@kernel function _delta_rule_kernel!(out, @Const(q), @Const(k), @Const(v), @Const(g), @Const(beta), st,
                                     @Const(lens), snap, snap_t, scale, G, fresh)
    j = @index(Local, Linear)
    T = @uniform @groupsize()[1]
    dk = @uniform size(q, 1); dv = @uniform size(v, 1)
    nv = @uniform size(v, 2); S = @uniform size(v, 3)
    # (@uniform values are computed from @index directly: they're hoisted)
    h = @uniform (@index(Group, Linear) - 1) % size(v, 2) + 1
    b = @uniform (@index(Group, Linear) - 1) ÷ size(v, 2) + 1
    hk = @uniform (h - 1) ÷ G + 1
    L = @uniform min(size(v, 3), unsafe_trunc(Int, @inbounds lens[(@index(Group, Linear) - 1) ÷ size(v, 2) + 1]))
    qs = @localmem Float32 (DELTA_MAX_DK,)
    ks = @localmem Float32 (DELTA_MAX_DK,)
    if fresh && j <= dv
        for i in 1:dk
            @inbounds st[j, i, h, b] = 0f0
        end
    end
    for t in 1:L
        i = j
        while i <= dk
            @inbounds qs[i] = Float32(q[i, hk, t, b]) * scale
            @inbounds ks[i] = Float32(k[i, hk, t, b])
            i += T
        end
        @synchronize
        if j <= dv
            @inbounds dec = exp(Float32(g[h, t, b]))
            kv = 0f0
            for i in 1:dk                                   # decay, and S^T k
                @inbounds s_ = st[j, i, h, b] * dec
                @inbounds st[j, i, h, b] = s_
                @inbounds kv += s_ * ks[i]
            end
            @inbounds delta = (Float32(v[j, h, t, b]) - kv) * Float32(beta[h, t, b])
            o = 0f0
            for i in 1:dk                                   # rank-1 update, and S^T q
                @inbounds s_ = st[j, i, h, b] + ks[i] * delta
                @inbounds st[j, i, h, b] = s_
                @inbounds o += s_ * qs[i]
            end
            @inbounds out[j, h, t, b] = o
            if t <= snap_t
                for i in 1:dk
                    @inbounds snap[j, i, h, b + (t - 1) * size(v, 4)] = st[j, i, h, b]
                end
            end
        end
        @synchronize
    end
    if j <= dv
        for tt in L+1:snap_t
            for i in 1:dk
                @inbounds snap[j, i, h, b + (tt - 1) * size(v, 4)] = st[j, i, h, b]
            end
        end
        for t in L+1:S
            @inbounds out[j, h, t, b] = 0f0
        end
    end
end

# The same recurrence with the state held in registers for the whole sequence: the state (dv x dk
# Float32, 64 KB per head at 128 x 128) is read once and written once, instead of read and written
# in global memory at every token (~45 us per token step). Column j of S^T (one value dim) is shared
# by a quad of 4 adjacent lanes, each holding every 4th of the dk key rows (32 registers at dk = 128;
# with 2 lanes per column the 64-register state plus temporaries spilled to scratch). Keys are
# interleaved across the quad (lane l holds keys l, l+4, ...), so the quad reads 4 consecutive words
# of the staged k / q and the reads are free of LDS bank conflicts (with a contiguous split every lane
# of the quad hit the same bank: 2.4x slower). The three per-step sums needed from the state
# (S^T k, S^T q, and k.q, which does not depend on the state) are all taken on the state *before*
# the step's decay and update, so a step is
#     kv = dec * S^T k,   delta = (v - kv) * beta,   o = dec * S^T q + delta * (k.q),   S = dec * S + k delta^T
# with one quad reduction (two DPP lane swaps, no LDS round trip) per sum. q and k (and v, decay,
# beta, k.q) are staged in local memory DELTA_REG_CHUNK steps at a time: a few barriers per chunk.
# The sums associate differently from `_delta_rule_kernel!`, so results agree to rounding, not bitwise.
const DELTA_REG_MAX_DK = 128
const DELTA_REG_CHUNK = 16
const DELTA_REG_LANES = 4
# Dot product of a register tuple with x[off], x[off + ST], ... as straight-line code, with 4 independent accumulators
@generated function _dot_st(a::NTuple{N,Float32}, x, off, ::Val{ST}) where {N,ST}
    accs = Any[:(0f0) for _ in 1:4]
    for i in 1:N
        r = (i - 1) % 4 + 1
        accs[r] = :(muladd(a[$i], Base.@inbounds(x[off + $((i - 1) * ST)]), $(accs[r])))
    end
    return :(($(accs[1]) + $(accs[2])) + ($(accs[3]) + $(accs[4])))
end
# The same for two arrays in memory, x[off+1:off+N] . y[off+1:off+N]
@generated function _dot_mem(x, y, off, ::Val{N}) where {N}
    accs = Any[:(0f0) for _ in 1:4]
    for i in 1:N
        r = (i - 1) % 4 + 1
        accs[r] = :(muladd(Base.@inbounds(x[off + $i]), Base.@inbounds(y[off + $i]), $(accs[r])))
    end
    return :(($(accs[1]) + $(accs[2])) + ($(accs[3]) + $(accs[4])))
end
# Swap with the neighbouring lane(s) inside a quad of 4 lanes (DPP quad_perm; control = perm[0] | perm[1] << 2 | ...)
@inline function _dpp_quad(x::Float32, ::Val{CTRL}) where {CTRL}
    xi = reinterpret(Int32, x)
    r = ccall("llvm.amdgcn.update.dpp.i32", llvmcall, Int32, (Int32, Int32, Int32, Int32, Int32, Bool),
              xi, xi, Int32(CTRL), Int32(0xF), Int32(0xF), false)
    return reinterpret(Float32, r)
end
@inline function _quad_sum(x::Float32)
    x += _dpp_quad(x, Val(0xB1))            # lanes (1, 0, 3, 2)
    x += _dpp_quad(x, Val(0x4E))            # lanes (2, 3, 0, 1)
    return x
end
# (A plain AMDGPU kernel, not a KernelAbstractions one: KA's lowering loses the types of the Val
# parameter here. Grid: one workgroup per (value head, batch), 4 * dv threads. SNAP: whether any
# per-step state snapshots are wanted (snap_t > 0); compiled out otherwise, which is faster.)
function _delta_rule_reg_kernel!(out, q, k, v, g, beta, st, lens, snap, snap_t, scale, G, fresh, ::Val{HD}, ::Val{SNAP}, ::Val{NUQ}, ::Val{NUV}) where {HD,SNAP,NUQ,NUV}
    NL = DELTA_REG_LANES
    tid = workitemIdx().x
    T = workgroupDim().x
    j = (tid - 1) ÷ NL + 1                 # value dim of this quad
    hf = (tid - 1) % NL                    # this lane holds key rows hf + 1, hf + 1 + NL, ...
    DK = NL * HD
    grp = workgroupIdx().x - 1
    dv = size(v, 1); nv = size(v, 2); S = size(v, 3)
    h = grp % nv + 1
    b = grp ÷ nv + 1
    hk = (h - 1) ÷ G + 1
    L = min(S, unsafe_trunc(Int, @inbounds lens[b]))
    qs = @ROCStaticLocalArray(Float32, (DELTA_REG_MAX_DK * DELTA_REG_CHUNK,), false)
    ks = @ROCStaticLocalArray(Float32, (DELTA_REG_MAX_DK * DELTA_REG_CHUNK,), false)
    vsm = @ROCStaticLocalArray(Float32, (128 * DELTA_REG_CHUNK,), false)
    ds = @ROCStaticLocalArray(Float32, (DELTA_REG_CHUNK,), false)
    bs = @ROCStaticLocalArray(Float32, (DELTA_REG_CHUNK,), false)
    kqs = @ROCStaticLocalArray(Float32, (DELTA_REG_CHUNK,), false)
    active = j <= dv
    s = ntuple(i -> (active && !fresh) ? (@inbounds st[j, (i - 1) * NL + hf + 1, h, b]) : 0f0, Val(HD))::NTuple{HD,Float32}
    t0 = 1
    while t0 <= L
        n = min(DELTA_REG_CHUNK, L - t0 + 1)
        # Stage this chunk's q (scaled), k and v: all of a thread's loads are issued together and only then stored (a loop that
        # loaded and stored one element at a time paid a memory latency per iteration: ~8 per chunk, a third of the kernel's time).
        qk, vv = let n = n, t0 = t0       # (rebound: closures over variables reassigned in this loop would box them)
            qk = ntuple(Val(NUQ)) do u
                lidx = tid + T * (u - 1)
                lok = lidx <= n * DK
                ltt = lok ? (lidx - 1) ÷ DK + 1 : 1
                li = lok ? lidx - (ltt - 1) * DK : 1
                (lok, @inbounds(Float32(q[li, hk, t0 + ltt - 1, b])) * scale, @inbounds(Float32(k[li, hk, t0 + ltt - 1, b])))
            end
            vv = ntuple(Val(NUV)) do u
                lidx = tid + T * (u - 1)
                lok = lidx <= n * dv
                ltt = lok ? (lidx - 1) ÷ dv + 1 : 1
                ljj = lok ? lidx - (ltt - 1) * dv : 1
                (lok, @inbounds(Float32(v[ljj, h, t0 + ltt - 1, b])))
            end
            (qk, vv)
        end
        dd = tid <= n ? exp(@inbounds(Float32(g[h, t0 + tid - 1, b]))) : 0f0
        bb = tid <= n ? @inbounds(Float32(beta[h, t0 + tid - 1, b])) : 0f0
        let qk = qk, vv = vv
            ntuple(Val(NUQ)) do u
                sok, sqv, skv = qk[u]
                if sok
                    sidx = tid + T * (u - 1)
                    @inbounds qs[sidx] = sqv
                    @inbounds ks[sidx] = skv
                end
                nothing
            end
            ntuple(Val(NUV)) do u
                vok, vx = vv[u]
                vok && (@inbounds vsm[tid + T * (u - 1)] = vx)
                nothing
            end
        end
        if tid <= n
            @inbounds ds[tid] = dd
            @inbounds bs[tid] = bb
        end
        sync_workgroup()
        wv = (tid - 1) ÷ 32 + 1; ln = (tid - 1) % 32      # k . q of each step (independent of the state): one wave per step
        while wv <= n
            part = 0f0
            for m in 0:cld(DK, 32)-1
                if ln + 32m < DK
                    @inbounds part = muladd(ks[(wv - 1) * DK + ln + 32m + 1], qs[(wv - 1) * DK + ln + 32m + 1], part)
                end
            end
            part += AMDGPU.Device.shfl_xor(part, Cint(16)); part += AMDGPU.Device.shfl_xor(part, Cint(8))
            part += AMDGPU.Device.shfl_xor(part, Cint(4)); part += AMDGPU.Device.shfl_xor(part, Cint(2))
            part += AMDGPU.Device.shfl_xor(part, Cint(1))
            ln == 0 && (@inbounds kqs[wv] = part)
            wv += T ÷ 32
        end
        sync_workgroup()
        if active
            for tt in 1:n
                t = t0 + tt - 1
                off = (tt - 1) * DK + hf + 1
                @inbounds dec = ds[tt]
                @inbounds bet = bs[tt]
                @inbounds vj = vsm[(tt - 1) * dv + j]
                @inbounds kq = kqs[tt]
                kv = dec * _quad_sum(_dot_st(s, ks, off, Val(NL))::Float32)                  # S^T k
                sq = dec * _quad_sum(_dot_st(s, qs, off, Val(NL))::Float32)                  # S^T q
                delta = (vj - kv) * bet
                o = sq + delta * kq
                s = let s0 = s                                                               # decay and rank-1 update
                    ntuple(i -> muladd(@inbounds(ks[off + (i - 1) * NL]), delta, @inbounds(s0[i]) * dec), Val(HD))::NTuple{HD,Float32}
                end
                hf == 0 && (@inbounds out[j, h, t, b] = o)
                if SNAP && t <= snap_t
                    for i in 1:HD
                        @inbounds snap[j, (i - 1) * NL + hf + 1, h, b + (t - 1) * size(v, 4)] = s[i]
                    end
                end
            end
        end
        sync_workgroup()
        t0 += DELTA_REG_CHUNK
    end
    if active
        for i in 1:HD
            @inbounds st[j, (i - 1) * NL + hf + 1, h, b] = s[i]
        end
        for tt in L+1:(SNAP ? snap_t : 0)
            for i in 1:HD
                @inbounds snap[j, (i - 1) * NL + hf + 1, h, b + (tt - 1) * size(v, 4)] = s[i]
            end
        end
        if hf == 0
            for t in L+1:S
                @inbounds out[j, h, t, b] = 0f0
            end
        end
    end
    return
end

# --- Delta rule, row-split: one wave per 8 state rows ----------------------------------
# gufo's structure for dk = dv = 128 (Float32 q / k / v, no snapshots): a wave owns 8 rows of the state matrix (4 lanes per
# row with 32 keys each in registers), k and q come from memory (L1) at every step through 16-byte loads, and there is no
# LDS and no barrier: the waves of a head are independent. Written with raw pointers and constant offsets: the same kernel
# with 4-D array indexing generated ~2000 address instructions per step instead of ~200 and ran 3x slower than this one.
# Grid (dv / 64, nv, B), 256 threads. 1.43 -> 1.1 us per token over the register kernel above.
@generated function _dot_tt32(a::NTuple{N,Float32}, b::NTuple{N,Float32}) where {N}
    accs = Any[:(0f0) for _ in 1:4]
    for i in 1:N
        r = (i - 1) % 4 + 1
        accs[r] = :(muladd(a[$i], b[$i], $(accs[r])))
    end
    return :(($(accs[1]) + $(accs[2])) + ($(accs[3]) + $(accs[4])))
end
const _DR_P4 = Core.LLVMPtr{NTuple{4,VecElement{Float32}},1}
# 32 consecutive Float32 from `p` as a tuple, by eight 16-byte loads
@inline function _dr_load32(p::Core.LLVMPtr{Float32,1})
    pv = reinterpret(_DR_P4, p)
    return ntuple(Val(32)) do i
        v = unsafe_load(pv, (i - 1) ÷ 4 + 1, Val(16))
        @inbounds v[(i - 1) % 4 + 1].value
    end
end
function _delta_rule_rows_kernel!(out, q, k, v, g, beta, st, lens, scale, G, fresh, voff, vts, vbs)
    tid = workitemIdx().x - 1
    wave = tid >> 5; lane = tid & 31; seg = lane & 3; grp = lane >> 2
    j = (workgroupIdx().x - 1) * 64 + wave * 8 + grp + 1
    h = workgroupIdx().y; b = workgroupIdx().z
    dk = size(q, 1); nk = size(q, 2); dv = size(out, 1); nv = size(out, 2); S = size(out, 3)
    hk = (h - 1) ÷ G + 1
    L = min(S, unsafe_trunc(Int, @inbounds lens[b]))
    kq_stride = nk * dk
    s = let fresh = fresh, st = st, j = j, seg = seg, h = h, b = b
        ntuple(i -> fresh ? 0f0 : (@inbounds st[j, seg * 32 + i, h, b]), Val(32))::NTuple{32,Float32}
    end
    kp = pointer(k) + 4 * (((b - 1) * S) * kq_stride + (hk - 1) * dk + seg * 32)
    qp = pointer(q) + 4 * (((b - 1) * S) * kq_stride + (hk - 1) * dk + seg * 32)
    vp = pointer(v) + 4 * (voff + (j - 1) + dv * (h - 1) + vbs * (b - 1))        # v: element (j, h, t, b) at voff + j - 1 + dv (h - 1) + vts t + vbs (b - 1)
    gp = pointer(g) + 4 * ((h - 1) + nv * S * (b - 1))
    bp = pointer(beta) + 4 * ((h - 1) + nv * S * (b - 1))
    op = pointer(out) + 4 * ((j - 1) + dv * ((h - 1) + nv * S * (b - 1)))
    for t in 0:L-1
        kv = _dr_load32(kp + 4 * kq_stride * t); qv = _dr_load32(qp + 4 * kq_stride * t)
        dec = exp(unsafe_load(gp, t * nv + 1)); bet = unsafe_load(bp, t * nv + 1); vj = unsafe_load(vp, t * vts + 1)
        kq = _quad_sum(_dot_tt32(kv, qv))
        kvs = dec * _quad_sum(_dot_tt32(s, kv))
        sq = dec * _quad_sum(_dot_tt32(s, qv))
        delta = (vj - kvs) * bet
        seg == 0 && unsafe_store!(op, (sq + delta * kq) * scale, t * nv * dv + 1)
        s = let s0 = s, kv = kv, delta = delta, dec = dec
            ntuple(i -> muladd(@inbounds(kv[i]), delta, @inbounds(s0[i]) * dec), Val(32))::NTuple{32,Float32}
        end
    end
    for i in 1:32
        @inbounds st[j, seg * 32 + i, h, b] = s[i]
    end
    if seg == 0
        for t in L+1:S
            @inbounds out[j, h, t, b] = 0f0
        end
    end
    return
end

function _launch_delta_rows!(out, q, k, v, g, beta, st, lens, scale, G, fresh, nv, B, dv; voff = 0, vts = nv * dv, vbs = nv * dv * size(out, 3))
    @roc groupsize=256 gridsize=(dv ÷ 64, nv, B) _delta_rule_rows_kernel!(out, q, k, v, g, beta, st, lens, scale, G, fresh, voff, vts, vbs)
    return nothing
end
function _delta_rows_ok(out, q, k, v, g, beta, st, snap_t)
    return out isa AMDGPU.ROCArray{Float32} && q isa AMDGPU.ROCArray{Float32} && k isa AMDGPU.ROCArray{Float32} && v isa AMDGPU.ROCArray{Float32} &&
           g isa AMDGPU.ROCArray{Float32} && beta isa AMDGPU.ROCArray{Float32} && st isa AMDGPU.ROCArray{Float32} && snap_t == 0 &&
           size(q, 1) == 128 && size(v, 1) == 128 && size(k) == size(q) && ndims(q) == 4 && size(q, 3) == size(v, 3) &&
           UInt(pointer(q)) % 16 == 0 && UInt(pointer(k)) % 16 == 0 && get(ENV, "LUMINAL_DELTA_RS", "1") != "0"
end

# Launch of the register kernel. On an AMDGPU.jl that has the `flat_work_group_size` / `waves_per_eu` compiler
# options (the luminal-launch-bounds branch, see TODO.md) the kernel is compiled for exactly this workgroup size
# and (at dk = 128) a 12-waves-per-SIMD register budget (~120 VGPRs, so that all 48 heads of the 27B model are
# resident at once; ~1.4x faster than without); stock AMDGPU.jl launches it without either.
function _launch_delta_reg!(out, q, k, v, g, beta, st, lens, snap, snap_t, scale, G, fresh, nv, B, dk, dv)
    nt = cld(DELTA_REG_LANES * dv, 32) * 32
    args = (out, q, k, v, g, beta, st, lens, snap, snap_t, scale, G, fresh, Val(dk ÷ DELTA_REG_LANES), Val(snap_t > 0),
            Val(cld(DELTA_REG_CHUNK * dk, nt)), Val(cld(DELTA_REG_CHUNK * dv, nt)))   # loads per thread of a chunk's q / k and v
    wpe = dk == 128 ? 12 : 1
    @roc_bounded nt wpe groupsize=nt gridsize=nv * B _delta_rule_reg_kernel!(args...)
    return nothing
end

function execute_op!(out, op::DeltaRule, q, k, v, g, beta, st, lens, snap = st)
    dk, nk = size(q, 1), size(q, 2)
    dv, nv, S, B = size(v, 1), size(v, 2), size(v, 3), size(v, 4)
    G = nv ÷ nk
    if !(out isa AnyGPUArray)
        fill!(out, 0)
        for b in 1:B, h in 1:nv
            hk = (h - 1) ÷ G + 1
            M = op.fresh ? zeros(Float32, dv, dk) : Float32.(st[:, :, h, b])     # S^T
            L = min(S, Int(lens[b]))
            for t in 1:L
                kt = Float32.(k[:, hk, t, b]); qt = Float32.(q[:, hk, t, b]) .* op.scale
                M .*= exp(Float32(g[h, t, b]))
                delta = (Float32.(v[:, h, t, b]) .- M * kt) .* Float32(beta[h, t, b])
                M .+= delta * kt'
                out[:, h, t, b] .= M * qt
                t <= op.snap_t && (snap[:, :, h, b + (t - 1) * B] .= M)
            end
            for tt in L+1:op.snap_t
                snap[:, :, h, b + (tt - 1) * B] .= M
            end
            st[:, :, h, b] .= M
        end
        return out
    end
    (dk <= DELTA_MAX_DK && dv <= 256) || error("DeltaRule: head dims must be <= 256 (key $dk, value $dv)")
    gs = min(256, max(32, nextpow(2, max(dv, dk))))
    if out isa AMDGPU.ROCArray && dk <= DELTA_REG_MAX_DK && dk % DELTA_REG_LANES == 0 && dv <= 128 && get(ENV, "LUMINAL_DELTA_REG", "1") != "0"
        qd, kd, vd, gd, bd, ld = _dense(q), _dense(k), _dense(v), _dense(g), _dense(beta), _dense(lens)
        if _delta_rows_ok(out, qd, kd, vd, gd, bd, st, op.snap_t)
            _launch_delta_rows!(out, qd, kd, vd, gd, bd, st, ld, op.scale, G, op.fresh, nv, B, dv)
        else
            _launch_delta_reg!(out, qd, kd, vd, gd, bd, st, ld, snap, op.snap_t, op.scale, G, op.fresh, nv, B, dk, dv)
        end
        return out
    end
    _delta_rule_kernel!(KernelAbstractions.get_backend(out), gs)(
        out, _dense(q), _dense(k), _dense(v), _dense(g), _dense(beta), st, _dense(lens), snap, op.snap_t, op.scale, G, op.fresh;
        ndrange = nv * B * gs)
    return out
end

# --- DeltaRuleQKV: the recurrence straight from the convolution's output ---------------------------
# q and k (rows 0:nk*dk and nk*dk:2*nk*dk of every token's column) are L2-normalized into dense scratch arrays by one kernel (one
# wave per head and token, the same summation order as `_l2norm_kernel!`); v is read in place by the row-split kernel (token stride C).
function _l2norm_qk_kernel!(qn, kn, qkv, C, nk, dk, S, B, eps)
    tid = workitemIdx().x - 1
    lane = tid % 32
    w = (workgroupIdx().x - 1) * (workgroupDim().x ÷ 32) + tid ÷ 32       # (which, head, token, batch), which fastest
    w < 2 * nk * S * B || return
    which = w % 2; r = w ÷ 2
    hd = r % nk; r = r ÷ nk
    t = r % S; b = r ÷ S
    src = (b * S + t) * C + which * nk * dk + hd * dk                      # 0-based element offset of the head's first key
    acc = 0f0
    i = lane
    while i < dk
        v = @inbounds qkv[src + i + 1]
        acc += v * v
        i += 32
    end
    acc += AMDGPU.Device.shfl_xor(acc, Cint(16))
    acc += AMDGPU.Device.shfl_xor(acc, Cint(8))
    acc += AMDGPU.Device.shfl_xor(acc, Cint(4))
    acc += AMDGPU.Device.shfl_xor(acc, Cint(2))
    acc += AMDGPU.Device.shfl_xor(acc, Cint(1))
    r = 1f0 / sqrt(acc + eps)
    dst = ((b * S + t) * nk + hd) * dk
    dstarr = which == 0 ? qn : kn
    i = lane
    while i < dk
        @inbounds dstarr[dst + i + 1] = qkv[src + i + 1] * r
        i += 32
    end
    return
end

function execute_op!(out, op::DeltaRuleQKV, qkv, g, beta, st, lens, snap = st)
    C, S, B = size(qkv, 1), size(qkv, 2), size(qkv, 3)
    nk, dk = op.nk, op.dk
    dv, nv = size(out, 1), size(out, 2)
    G = nv ÷ nk
    if out isa AMDGPU.ROCArray{Float32} && qkv isa AMDGPU.ROCArray{Float32} && qkv isa DenseArray && dk == 128 && dv == 128 &&
       g isa AMDGPU.ROCArray{Float32} && beta isa AMDGPU.ROCArray{Float32} && st isa AMDGPU.ROCArray{Float32} && op.snap_t == 0 &&
       get(ENV, "LUMINAL_DELTA_RS", "1") != "0" && UInt(pointer(qkv)) % 16 == 0
        qn = similar(qkv, Float32, dk, nk, S, B); kn = similar(qkv, Float32, dk, nk, S, B)
        @roc groupsize=256 gridsize=cld(2 * nk * S * B, 8) _l2norm_qk_kernel!(vec(qn), vec(kn), vec(qkv), C, nk, dk, S, B, op.epsilon)
        gd, bd, ld = _dense(g), _dense(beta), _dense(lens)
        _launch_delta_rows!(out, qn, kn, qkv, gd, bd, st, ld, op.scale, G, op.fresh, nv, B, dv; voff = 2 * nk * dk, vts = C, vbs = C * S)
        _free_now!(qn); _free_now!(kn)
        return out
    end
    # anything else: slice, normalize and run the separate ops
    qkvd = qkv isa DenseArray ? qkv : copy(qkv)
    cut(lo, hi, d1, d2) = copy(Base.reshape(view(qkvd, lo+1:hi, :, :), d1, d2, S, B))
    q = cut(0, nk * dk, dk, nk); k = cut(nk * dk, 2nk * dk, dk, nk); v = cut(2nk * dk, C, dv, nv)
    qn = similar(q, Float32); kn = similar(k, Float32)
    execute_op!(qn, L2NormOp(op.epsilon), q); execute_op!(kn, L2NormOp(op.epsilon), k)
    execute_op!(out, DeltaRule(op.scale, op.fresh, op.snap_t), qn, kn, v, g, beta, st, lens, snap)
    foreach(_free_now!, (q, k, v, qn, kn))
    return out
end

# --- ConvDeltaRule: convolution + SiLU + L2 norm of q and k in one kernel ---------------------------
# One wave per (head of the stacked q / k / v rows, chunk of TC tokens, batch); a lane owns 4 channels of the head's 128 (coalesced).
# Every output depends only on the previous K - 1 inputs, so the tokens of a chunk are independent of the other chunks (the LDS-free
# sliding window starts from x, or from the conv state for the first K - 1 tokens). q / k heads are normalized by their norm over the
# 128 channels (the same summation order as `_l2norm_kernel!`) and written dense; v heads are written as they are.
const CONV_TC = 16
# One token of the sliding window for the four channels of a lane: the SiLU of the dot product with the weights, and the shifted window.
@inline function _conv_token(win, wk, xp, t, C, ::Val{K}) where {K}
    xs = ntuple(i -> unsafe_load(xp, (t - 1) * C + 32 * (i - 1) + 1), Val(4))
    nw = ntuple(i -> (Base.tail(win[i])..., xs[i]), Val(4))
    ys = ntuple(Val(4)) do i
        acc = 0f0
        for j in 1:K
            acc += wk[i][j] * nw[i][j]
        end
        _silu(acc)
    end
    return ys, nw
end
function _conv_norm_kernel!(qn, kn, vo, x, state, w, lens, nk, nv, S, C, eps, fresh, ::Val{K}, ::Val{TC}) where {K,TC}
    lane = workitemIdx().x - 1
    hidx = workgroupIdx().x - 1; tc = workgroupIdx().y - 1; b = workgroupIdx().z - 1
    L = min(S, unsafe_trunc(Int, @inbounds lens[b + 1]))
    t0 = tc * TC + 1
    t0 <= L || return
    c0 = hidx * 128 + lane                                       # 0-based channel of lane slot i: c0 + 32 i
    xp = pointer(x) + 4 * (b * S * C + c0)                       # x[c, t, b] at (t - 1) * C + c
    sp = pointer(state) + 4 * (b * C * K + c0)                   # state[c, j, b] at (j - 1) * C + c
    wp = pointer(w) + 4 * c0                                     # w[c, 1, j] at (j - 1) * C + c
    wk = ntuple(Val(4)) do i
        ntuple(j -> unsafe_load(wp, (j - 1) * C + 32 * (i - 1) + 1), Val(K))
    end
    # window of the K inputs before token t0 (oldest first, K - 1 of them are used: the last slot is shifted in)
    win = ntuple(Val(4)) do i
        ntuple(Val(K)) do j
            pos = t0 - K + j - 1                                 # input position of slot j before shifting: t0 - K + j - 1 (<= t0 - 1 for j <= K)
            if pos >= 1
                unsafe_load(xp, (pos - 1) * C + 32 * (i - 1) + 1)
            elseif fresh
                0f0
            else
                unsafe_load(sp, (K + pos - 1) * C + 32 * (i - 1) + 1)   # state slot K + pos (pos <= 0)
            end
        end
    end
    isq = hidx < 2 * nk
    nkd = nk * 128; nvd = nv * 128
    tend = min(t0 + TC - 1, L)
    for t in t0:tend
        ys, win = _conv_token(win, wk, xp, t, C, Val(K))
        if isq
            ss = 0f0
            for i in 1:4
                ss += ys[i] * ys[i]
            end
            ss += AMDGPU.Device.shfl_xor(ss, Cint(16)); ss += AMDGPU.Device.shfl_xor(ss, Cint(8)); ss += AMDGPU.Device.shfl_xor(ss, Cint(4))
            ss += AMDGPU.Device.shfl_xor(ss, Cint(2)); ss += AMDGPU.Device.shfl_xor(ss, Cint(1))
            r = 1f0 / sqrt(ss + eps)
            dst = hidx < nk ? pointer(qn) : pointer(kn)
            hk = hidx < nk ? hidx : hidx - nk
            base = ((b * S + t - 1) * nk + hk) * 128 + lane
            for i in 1:4
                unsafe_store!(dst, ys[i] * r, base + 32 * (i - 1) + 1)
            end
        else
            vh = hidx - 2 * nk
            base = ((b * S + t - 1) * nv + vh) * 128 + lane
            vp = pointer(vo)
            for i in 1:4
                unsafe_store!(vp, ys[i], base + 32 * (i - 1) + 1)
            end
        end
    end
    return
end

# The conv state after the tokens consumed: the last K inputs (those before the first token come from the old state, or zeros when fresh).
function _conv_state_kernel!(state, x, lens, S, C, fresh, ::Val{K}) where {K}
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    nth = size(state, 1) * size(state, 3)
    i <= nth || return
    c = (i - 1) % C + 1; b = (i - 1) ÷ C + 1
    L = min(S, unsafe_trunc(Int, @inbounds lens[b]))
    vals = ntuple(Val(K)) do j
        p = L - K + j
        p >= 1 ? @inbounds(x[c, p, b]) : (fresh ? 0f0 : @inbounds(state[c, K + p, b]))
    end
    for j in 1:K
        @inbounds state[c, j, b] = vals[j]
    end
    return
end

function execute_op!(out, op::ConvDeltaRule, x, cstate, cw, g, beta, st, lens, snaps...)
    C, S, B = size(x, 1), size(x, 2), size(x, 3)
    nk, dk = op.nk, op.dk
    dv, nv = size(out, 1), size(out, 2)
    K = size(cw, 3)
    if out isa AMDGPU.ROCArray{Float32} && x isa AMDGPU.ROCArray{Float32} && x isa DenseArray && cw isa AMDGPU.ROCArray{Float32} && cw isa DenseArray &&
       cstate isa AMDGPU.ROCArray{Float32} && g isa AMDGPU.ROCArray{Float32} && beta isa AMDGPU.ROCArray{Float32} && st isa AMDGPU.ROCArray{Float32} &&
       op.snap_t == 0 && dk == 128 && dv == 128 && C == (2nk + nv) * 128 && S >= 16 && K <= 8 && get(ENV, "LUMINAL_CONV_FUSED", "1") != "0" &&
       get(ENV, "LUMINAL_DELTA_RS", "1") != "0"
        qn = similar(x, Float32, dk, nk, S, B); kn = similar(x, Float32, dk, nk, S, B); vo = similar(x, Float32, dv, nv, S, B)
        ld = _dense(lens)
        @roc groupsize=32 gridsize=(2nk + nv, cld(S, CONV_TC), B) _conv_norm_kernel!(qn, kn, vo, x, cstate, cw, ld, nk, nv, S, C, op.epsilon, op.fresh, Val(K), Val(CONV_TC))
        @roc groupsize=256 gridsize=cld(C * B, 256) _conv_state_kernel!(cstate, x, ld, S, C, op.fresh, Val(K))
        _launch_delta_rows!(out, qn, kn, vo, _dense(g), _dense(beta), st, ld, op.scale, nv ÷ nk, op.fresh, nv, B, dv)
        foreach(_free_now!, (qn, kn, vo))
        return out
    end
    # anything else: the two separate ops
    cs = isempty(snaps) ? cstate : snaps[1]
    rs = isempty(snaps) ? st : snaps[2]
    conv = similar(x, Float32, C, S, B)
    execute_op!(conv, CausalConv(op.fresh, op.snap_t), x, cstate, cw, lens, cs)
    execute_op!(out, DeltaRuleQKV(op.scale, op.fresh, op.snap_t, nk, dk, op.epsilon), conv, g, beta, st, lens, rs)
    _free_now!(conv)
    return out
end

# --- AttnPrep / GatedAttnOut: the gated attention layer's query / key preparation and output gate ---------------
# One wave per (head, token, batch); a lane owns head_dim / 32 elements of the head (e = lane + 32 i), so the two elements a RoPE rotation
# pairs (e and e + rotary_dim / 2) sit in the same lane when rotary_dim % 64 == 0.
function _attn_prep_kernel!(out, x, w, cs, sn, D, H, S, rd, rowstride, ncs3, eps, ::Val{NS}) where {NS}
    tid = workitemIdx().x - 1
    lane = tid % 32
    wv = (workgroupIdx().x - 1) * (workgroupDim().x ÷ 32) + tid ÷ 32
    wv < H * S * (length(out) ÷ (D * H * S)) || return
    h = wv % H; s = (wv ÷ H) % S; b = wv ÷ (H * S)
    src = ((b * S + s) * H + h) * rowstride
    xp = pointer(x) + 4 * src; wp = pointer(w)
    vals = ntuple(i -> unsafe_load(xp, lane + 32 * (i - 1) + 1), Val(NS))
    ss = 0f0
    for i in 1:NS
        ss += vals[i] * vals[i]
    end
    ss += AMDGPU.Device.shfl_xor(ss, Cint(16)); ss += AMDGPU.Device.shfl_xor(ss, Cint(8)); ss += AMDGPU.Device.shfl_xor(ss, Cint(4))
    ss += AMDGPU.Device.shfl_xor(ss, Cint(2)); ss += AMDGPU.Device.shfl_xor(ss, Cint(1))
    r = 1f0 / sqrt(ss / D + eps)
    ys = ntuple(i -> vals[i] * r * unsafe_load(wp, lane + 32 * (i - 1) + 1), Val(NS))
    half = rd ÷ 2; hs = half ÷ 32                              # the partner of slot i (< hs) is slot i + hs
    cb = ncs3 == 1 ? 0 : b
    cp = pointer(cs) + 4 * (half * (s + S * cb)); sp = pointer(sn) + 4 * (half * (s + S * cb))
    op = pointer(out) + 4 * (((b * H + h) * S + s) * D)
    T = eltype(out)
    for i in 1:NS
        e = lane + 32 * (i - 1)
        y = ys[i]
        if i <= hs                                             # first half of the rotated dims
            c = unsafe_load(cp, e + 1); sv = unsafe_load(sp, e + 1)
            y = ys[i] * c - ys[i + hs] * sv
        elseif i <= 2hs                                        # second half
            e0 = e - half
            c = unsafe_load(cp, e0 + 1); sv = unsafe_load(sp, e0 + 1)
            y = ys[i] * c + ys[i - hs] * sv
        end
        unsafe_store!(op, T(y), e + 1)
    end
    return
end

function execute_op!(out, op::AttnPrep, x, w, cs, sn)
    D, H, rd = op.head_dim, op.heads, op.rotary_dim
    S = size(out, 2); B = size(out, 4)
    if out isa AMDGPU.ROCArray && x isa AMDGPU.ROCArray{Float32} && w isa AMDGPU.ROCArray{Float32} && cs isa AMDGPU.ROCArray{Float32} &&
       sn isa AMDGPU.ROCArray{Float32} && D % 32 == 0 && rd % 64 == 0 && 0 < rd <= D && D <= 256
        xd, wd, cd, sd = _dense(x), _dense(w), _dense(cs), _dense(sn)
        ncs3 = length(cd) ÷ ((rd ÷ 2) * S)
        @roc groupsize=256 gridsize=cld(H * S * B, 8) _attn_prep_kernel!(out, xd, wd, cd, sd, D, H, S, rd, op.rowstride, ncs3, op.epsilon, Val(D ÷ 32))
        return out
    end
    # reference path (CPU, other shapes): per head, RMSNorm then rotate-half on the first rd dims
    xa = Array(x); wa = vec(Array(w)); ca = Base.reshape(Array(cs), rd ÷ 2, S, :); sa = Base.reshape(Array(sn), rd ÷ 2, S, :)
    xr = Base.reshape(xa, op.rowstride * H, S, B)
    res = zeros(Float32, D, S, H, B)
    for b in 1:B, h in 1:H, s in 1:S
        v = Float32.(xr[(h-1)*op.rowstride+1:(h-1)*op.rowstride+D, s, b])
        y = v .* (1f0 / sqrt(sum(abs2, v) / D + op.epsilon)) .* wa
        cb = size(ca, 3) == 1 ? 1 : b
        half = rd ÷ 2
        z = copy(y)
        for i in 1:half
            z[i] = y[i] * ca[i, s, cb] - y[i+half] * sa[i, s, cb]
            z[i+half] = y[i+half] * ca[i, s, cb] + y[i] * sa[i, s, cb]
        end
        res[:, s, h, b] = z
    end
    copyto!(out, res)
    return out
end

function _gated_attn_out_kernel!(out, a, qp, D, H, n)
    i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x - 1
    i < n || return
    r = i % (D * H); col = i ÷ (D * H)
    h = r ÷ D; d = r % D
    g = @inbounds qp[col * (2 * D * H) + h * 2 * D + D + d + 1]
    @inbounds out[i + 1] = eltype(out)(a[i + 1] * (1f0 / (1f0 + exp(-g))))
    return
end

function execute_op!(out, op::GatedAttnOut, a, qp)
    D = op.head_dim
    H = size(a, 1) ÷ D
    if out isa AMDGPU.ROCArray && a isa AMDGPU.ROCArray{Float32} && qp isa AMDGPU.ROCArray{Float32}
        ad, qd = _dense(a), _dense(qp)
        n = length(out)
        @roc groupsize=256 gridsize=cld(n, 256) _gated_attn_out_kernel!(vec(out), vec(ad), vec(qd), D, H, n)
        return out
    end
    aa = Array(a); qa = Base.reshape(Array(qp), 2D * H, :)
    ar = Base.reshape(aa, D * H, :)
    res = similar(ar, Float32)
    for col in axes(ar, 2), h in 0:H-1, d in 1:D
        g = qa[h * 2D + D + d, col]
        res[h * D + d, col] = ar[h * D + d, col] / (1f0 + exp(-g))
    end
    copyto!(out, Base.reshape(res, size(out)))
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
    permutedims!(t, W, (2, 1))                     # converts in place, no Float32 temporary
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

_free_now!(x) = (x isa AnyGPUArray && GPUArrays.unsafe_free!(x); nothing)

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
    Wt = similar(W, K, M)
    permutedims!(Wt, W, (2, 1))                              # (In, Out)
    Wg = Base.reshape(Wt, group, K ÷ group, M)
    mx = maximum(abs, Wg; dims=1)
    mxr = Base.reshape(mx, K ÷ group, M)
    scale = mxr ./ 127f0
    scale .= max.(scale, floatmin(Float32))
    q = similar(W, Int8, K, M)
    qg = Base.reshape(q, group, K ÷ group, M)
    sg = Base.reshape(scale, 1, K ÷ group, M)
    qg .= unsafe_trunc.(Int8, round.(Wg ./ sg))
    v = Base.reshape(reinterpret(NTuple{16, Int8}, vec(q)), K ÷ 16, M)
    # Free the Float32 temporaries now. Left to the GC, GPU buffers are released
    # only when it gets to them, and converting a whole model piles them up (~4x
    # the int8 size held on TinyLlama; on an APU that is system RAM). A buffer is
    # shared by its reshapes, so every one of them has to be freed.
    foreach(_free_now!, (Wg, Wt, mxr, mx))
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

# The (residual, Val(has_residual)) kernel arguments; without a residual the
# output array stands in (never read).
# Many columns (prefill) on a GEMV-stored weight: the GEMV kernels read the
# weight once per few columns, so instead dequantize it to a temporary Float32
# matrix, run one GEMM (rocBLAS on GPU) and free the temporary at once (the peak
# is one weight's Float32 size, not the model's). CPU arrays always go this way
# (the GEMV kernels are GPU kernels).
const QUANT_GEMM_COLS = 16
function _dequant_gemm!(C, W, B, residual)
    M, K = size(W)
    N = length(B) ÷ K
    W32 = dequantize(W)
    Bc = B isa DenseArray ? B : copy(B)
    C2 = Base.reshape(C, M, N)
    mul!(C2, W32, Base.reshape(Bc, K, N))
    residual === nothing || (C2 .+= Base.reshape(residual isa DenseArray ? residual : copy(residual), M, N))
    _free_now!(W32)
    Bc === B || _free_now!(Bc)
    return C
end

function _residual_arg(residual, C, M, N)
    residual === nothing && return (Base.reshape(C, M, N), Val(false))
    length(residual) == M * N || error("residual has $(length(residual)) elements, expected $(M * N)")
    r = residual isa DenseArray ? residual : copy(residual)
    return (Base.reshape(r, M, N), Val(true))
end

@kernel function _q8_matmul_kernel!(Y, @Const(Wv), @Const(scale), @Const(Xv), K16, N,
                                    loads_per_group, shift, ::Val{R}, ::Val{C},
                                    @Const(Res), ::Val{RES}) where {R, C, RES}
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
            if c0 + j <= N
                @inbounds Y[row0 + r, c0 + j] = RES ? s[1, lid] + Res[row0 + r, c0 + j] : s[1, lid]
            end
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

# `residual` (same size as C, optional): C = W * B + residual, added in the kernel's
# final write (a fused residual connection).
function _q8_matmul!(C, W::QuantWeight, B; group::Int = DEFAULT_Q8_THREADS, cols::Int = 0,
                     residual = nothing)
    K, M = size(W.q)
    N = length(B) ÷ K
    if eltype(B) === Float16          # a Float16 activation (see `dty` in compile)
        N > QUANT_GEMM_COLS && _wmma_q8_gemm!(C, W, B, residual) && return C
        B = Float32.(B)
    end
    N > QUANT_GEMM_COLS && _wmma_q8_gemm!(C, W, B, residual) && return C
    (N > QUANT_GEMM_COLS || !(C isa AnyGPUArray)) && return _dequant_gemm!(C, W, B, residual)
    Bc = B isa DenseArray ? B : copy(B)
    Xv = Base.reshape(reinterpret(NTuple{8, Float32}, vec(Bc)), K ÷ 8, N)
    R = iseven(M) ? 2 : 1
    lpg = W.group ÷ 16
    group == 0 && (group = _q8_threads(K ÷ 16, N))
    group <= 256 || error("int8 GEMV: at most 256 threads per workgroup")
    _q8_matmul_kernel!(KernelAbstractions.get_backend(C), group)(
        Base.reshape(C, M, N), W.v, W.scale, Xv, K ÷ 16, N, lpg,
        ispow2(lpg) ? trailing_zeros(lpg) : -1, Val(R), Val(min(N, cols == 0 ? Q8_MAX_COLS : cols)),
        _residual_arg(residual, C, M, N)...; ndrange = (M ÷ R) * group)
    return C
end

# --- 4-bit matmul weights ---------------------------------------------------------
# Weight-only 4-bit quantization per group of `group` consecutive inputs of each
# output row, w = q * scale + min with q in 0..15: symmetric (min = -8 * scale,
# scale = max|w| / 7, only the scale stored) or asymmetric (the group's
# [min, max], both stored), as Float16. Transposed to (In, Out) and packed two per
# byte (input 2i-1 in the low nibble, 2i in the high one), so a 16-byte load
# brings 32 weights. Symmetric groups of 32 (the default) are 0.5625 bytes per
# weight (int8: ~1.03, Float16: 2).
"""
    Int4

Marker for 4-bit weight storage: `compile(...; weight_dtype=Luminal.Int4)`,
`LlamaSession(...; decode_weights=Luminal.Int4)`. See `Q4Weight`.
"""
abstract type Int4 end

# Default: symmetric, groups of 32 -- the most accurate at 0.5625 bytes per weight
# on WikiText-2 perplexity for both TinyLlama and Llama-3-8B (examples/quant_formats.jl:
# +5.2% / +5.3% vs Float32, against +7.6% / +9.0% for asymmetric groups of 64 at the
# same size; asymmetric groups of 32 is slightly better on 8B but 11% larger).
const DEFAULT_Q4_GROUP = 32
const DEFAULT_Q4_SYMMETRIC = true
struct Q4Weight{P, V, S, M} <: AbstractMatrix{Float32}
    p::P          # (In/2, Out) UInt8, two 4-bit values per byte
    v::V          # (In/32, Out) NTuple{16, UInt8} view of `p`
    scale::S      # (In/group, Out) Float16
    mn::M         # (In/group, Out) Float16 group minimum, or `nothing` when symmetric (-8 * scale)
    group::Int
end
# `symmetric`: codes q - 8 in -8..7 around zero, scale = max|w| / 7, stored as
# min = -8 * scale, so the same kernel runs both. Asymmetric fits each group's
# [min, max] instead.
function Q4Weight(W::AbstractMatrix; group::Int = DEFAULT_Q4_GROUP, symmetric::Bool = DEFAULT_Q4_SYMMETRIC)
    M, K = size(W)
    K % 32 == 0 || error("Q4Weight: input dimension $K is not a multiple of 32")
    (group % 32 == 0 && K % group == 0) || (group = K)       # one group per row
    G = K ÷ group
    Wt = similar(W, K, M)
    permutedims!(Wt, W, (2, 1))                              # (In, Out)
    Wg = Base.reshape(Wt, group, G, M)
    lo = minimum(Wg; dims=1); hi = maximum(Wg; dims=1)
    # Float16 scale and minimum, and the codes computed from those rounded values,
    # so dequantization reproduces exactly what was fitted
    if symmetric
        sc16 = Float16.(max.(max.(abs.(lo), abs.(hi)) ./ 7f0, Float32(floatmin(Float16))))
        mn16 = Float16.(-8f0 .* Float32.(sc16))                # exact: a power-of-two multiple
    else
        mn16 = Float16.(lo)
        sc16 = Float16.(max.((hi .- lo) ./ 15f0, Float32(floatmin(Float16))))
    end
    q = similar(W, UInt8, K, M)
    qg = Base.reshape(q, group, G, M)
    qg .= unsafe_trunc.(UInt8, clamp.(round.((Wg .- Float32.(mn16)) ./ Float32.(sc16)), 0f0, 15f0))
    p = similar(W, UInt8, K ÷ 2, M)
    q2 = Base.reshape(q, 2, K ÷ 2, M)
    lo_nib = view(q2, 1, :, :); hi_nib = view(q2, 2, :, :)
    p .= lo_nib .| (hi_nib .<< 0x04)
    v = Base.reshape(reinterpret(NTuple{16, UInt8}, vec(p)), K ÷ 32, M)
    scale = Base.reshape(sc16, G, M)
    mn = symmetric ? nothing : Base.reshape(mn16, G, M)
    foreach(_free_now!, (Wg, Wt, lo, hi, qg, q2, q))            # temporaries and their reshapes
    symmetric && _free_now!(mn16)
    return Q4Weight(p, v, scale, mn, group)
end
Base.size(w::Q4Weight) = (size(w.p, 2), 2 * size(w.p, 1))
function Base.getindex(w::Q4Weight, i::Int, j::Int)
    b = w.p[(j + 1) ÷ 2, i]
    q = isodd(j) ? b & 0x0f : b >> 0x04
    gi = (j - 1) ÷ w.group + 1
    s = Float32(w.scale[gi, i])
    return Float32(q) * s + (w.mn === nothing ? -8f0 * s : Float32(w.mn[gi, i]))
end

"""
    dequantize(w) -> Float32 matrix (Out, In)

The Float32 values a reduced-precision matmul weight (`HalfWeight`, `QuantWeight`,
`Q4Weight`) stands for, on the same device: exactly what the kernels compute with.
Lets a Float32 model be evaluated with any mix of quantized tensors.
"""
dequantize(w::HalfWeight) = permutedims(Float32.(w.t), (2, 1))
# (Temporaries are freed at once: dequantizing a whole model otherwise piles up
# tens of GB of garbage before the GC runs.)
function dequantize(w::QuantWeight)
    K, M = size(w.q); G = K ÷ w.group
    x = Float32.(Base.reshape(w.q, w.group, G, M)) .* Base.reshape(w.scale, 1, G, M)
    xr = Base.reshape(x, K, M)
    out = permutedims(xr, (2, 1))
    foreach(_free_now!, (xr, x))
    return out
end
function dequantize(w::Q4Weight)
    M, K = size(w); G = K ÷ w.group
    codes = similar(w.p, UInt8, 2, K ÷ 2, M)
    view(codes, 1, :, :) .= w.p .& 0x0f
    view(codes, 2, :, :) .= w.p .>> 0x04
    cg = Base.reshape(codes, w.group, G, M)
    s = Base.reshape(w.scale, 1, G, M)
    x = w.mn === nothing ? (Float32.(cg) .- 8f0) .* Float32.(s) :
        Float32.(cg) .* Float32.(s) .+ Float32.(Base.reshape(w.mn, 1, G, M))
    xr = Base.reshape(x, K, M)
    out = permutedims(xr, (2, 1))
    foreach(_free_now!, (xr, x, cg, codes))
    return out
end

"""
    Q5Weight

GGUF's Q5_K in the file's own values and size: `p` / `v` are the low four bits of the codes in `Q4Weight`'s layout, `hi` (In/32, Out) holds one
`UInt32` of fifth bits per group of 32 inputs (bit `4 + b` is the fifth bit of the low-nibble code of byte `b` of the group's 16 code bytes and bit
`(20 + b) % 32` that of the high-nibble code: a kernel masks `rotr(word, b) & 0x00100010` and ORs it into the code's Float16 bit pattern), and
`blk` (In/256, Out) the file's own 16-byte block header: d and dmin as Float16 (word 1) and the twelve bytes of 6-bit scales and minimums
(words 2-4), see `_kq_scale_min`.
"""
struct Q5Weight{P, V, H, B} <: AbstractMatrix{Float32}
    p::P
    v::V
    hi::H
    blk::B
end
Base.size(w::Q5Weight) = (size(w.p, 2), 2 * size(w.p, 1))
@inline _q5_bitpos(l) = (b = l >> 1; iseven(l) ? 4 + b : (20 + b) & 31)    # l: 0-based code index in its group of 32
@inline _f16lo(u::UInt32) = Float32(reinterpret(Float16, UInt16(u & 0xffff)))
@inline _f16hi(u::UInt32) = Float32(reinterpret(Float16, UInt16(u >> 16)))
# the step (d * sc) and minimum (-dmin * m) of sub-block j (0..7) of a Q4_K / Q5_K block header `rec` = (d | dmin << 16, bytes 0-3, 4-7, 8-11 of the scales)
@inline function _kq_scale_min(rec::NTuple{4, UInt32}, j)
    ju = UInt32(j)
    w0, w1, w2 = rec[2], rec[3], rec[4]
    sh = (ju & 0x03) << 0x03
    a = (w0 >> sh) & 0xff; b = (w1 >> sh) & 0xff; c = (w2 >> sh) & 0xff
    hiq = (ju & 0x04) != 0x00                                   # sub-blocks 4..7 take their low bits from w2 and their top two from w0 / w1
    s = ifelse(hiq, (c & 0x0f) | ((a >> 0x06) << 0x04), a & 0x3f)
    m = ifelse(hiq, (c >> 0x04) | ((b >> 0x06) << 0x04), b & 0x3f)
    return _f16lo(rec[1]) * Float32(s), -_f16hi(rec[1]) * Float32(m)
end
function Base.getindex(w::Q5Weight, i::Int, j::Int)
    b = w.p[(j + 1) ÷ 2, i]
    c = isodd(j) ? b & 0x0f : b >> 0x04
    g = (j - 1) ÷ 32
    s, m = _kq_scale_min(w.blk[g ÷ 8 + 1, i], g % 8)
    bit = (w.hi[g + 1, i] >> _q5_bitpos((j - 1) % 32)) & 0x01
    return Float32(c + 0x10 * bit) * s + m
end
function dequantize(w::Q5Weight)
    M, K = size(w); G = K ÷ 32
    codes = similar(w.p, UInt8, 2, K ÷ 2, M)
    view(codes, 1, :, :) .= w.p .& 0x0f
    view(codes, 2, :, :) .= w.p .>> 0x04
    u = Base.reshape(reinterpret(UInt32, vec(w.blk)), 4, K ÷ 256, M)
    jv = similar(w.p, Int, 8); copyto!(jv, collect(0:7))
    u1, u2, u3, u4 = (Base.reshape(u[r, :, :], 1, K ÷ 256, M) for r in 1:4)
    sm = _kq_sm_arr.(u1, u2, u3, u4, Base.reshape(jv, 8, 1, 1))          # (8, K/256, M) of (step, min)
    sc = Base.reshape(first.(sm), 1, G, M); mn = Base.reshape(last.(sm), 1, G, M)
    pos = similar(w.p, UInt32, 32); copyto!(pos, UInt32[_q5_bitpos(l) for l in 0:31])
    hi = Base.reshape(w.hi, 1, G, M)
    cg = Base.reshape(codes, 32, G, M)
    x = (Float32.(cg) .+ 16f0 .* Float32.((hi .>> Base.reshape(pos, 32, 1, 1)) .& 0x01)) .* sc .+ mn
    xr = Base.reshape(x, K, M)
    out = permutedims(xr, (2, 1))
    foreach(_free_now!, (xr, x, cg, codes, pos, sm, hi, sc, mn, jv))
    return out
end
@inline _kq_sm_arr(u1::UInt32, u2::UInt32, u3::UInt32, u4::UInt32, j) = _kq_scale_min((u1, u2, u3, u4), j)
_free_now!(w::QuantWeight) = (_free_now!(w.q); _free_now!(w.scale); nothing)
_free_now!(w::Q4Weight) = (foreach(_free_now!, (w.p, w.scale)); w.mn === nothing || _free_now!(w.mn); nothing)
_free_now!(w::Q5Weight) = (_free_now!(w.p); _free_now!(w.hi); _free_now!(w.blk); nothing)

"""
    IQ4Weight

A 4-bit weight with a value table (GGUF's IQ4_XS / IQ4_NL): the layout of a symmetric-less `Q4Weight` (`q`, whose `mn` is `nothing` and whose
Float16 `scale` is the group's step) whose codes index `IQ4NL_VALUES`: value = scale * IQ4NL_VALUES[code + 1].
"""
struct IQ4Weight{Q <: Q4Weight} <: AbstractMatrix{Float32}
    q::Q
end
const IQ4NL_VALUES = Float32[-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
Base.size(w::IQ4Weight) = size(w.q)
function Base.getindex(w::IQ4Weight, i::Int, j::Int)
    b = w.q.p[(j + 1) ÷ 2, i]
    c = isodd(j) ? b & 0x0f : b >> 0x04
    return IQ4NL_VALUES[Int(c) + 1] * Float32(w.q.scale[(j - 1) ÷ 32 + 1, i])
end
const IQ4NL_TUPLE = Tuple(IQ4NL_VALUES)
@inline _iq4_val(c::UInt8) = @inbounds IQ4NL_TUPLE[Int(c) + 1]
function dequantize(w::IQ4Weight)                              # kernels only (no host reads: it runs inside graph capture)
    M, K = size(w); G = K ÷ 32
    lo = Base.reshape(_iq4_val.(w.q.p .& 0x0f), 1, K ÷ 2, M)
    hi = Base.reshape(_iq4_val.(w.q.p .>> 0x04), 1, K ÷ 2, M)
    v = cat(lo, hi; dims = 1)
    x = Base.reshape(v, 32, G, M) .* Base.reshape(Float32.(w.q.scale), 1, G, M)
    xr = Base.reshape(x, K, M)
    out = permutedims(xr, (2, 1))
    foreach(_free_now!, (xr, x, v, lo, hi))
    return out
end
_free_now!(w::IQ4Weight) = (_free_now!(w.q); nothing)
"""
    Q6Weight

GGUF's Q6_K: `p` / `v` the low four bits of the codes as in `Q4Weight`, and `meta` (In/32, Out) of 12 bytes per group of 32 inputs: a word with the two
Float16 steps (d * sc of its two K16 halves, low half first) and one word per K16 half with the codes' two high bits (the fields of byte b's
low-nibble code at bits 4 + 2b, 5 + 2b, those of its high-nibble code at (20 + 2b) mod 32 and the next; see `_wmma_dq6w`).
Value = (code - 32) * step.
"""
struct Q6Weight{P, V, T} <: AbstractMatrix{Float32}
    p::P
    v::V
    meta::T
end
Base.size(w::Q6Weight) = (size(w.p, 2), 2 * size(w.p, 1))
@inline _q6_bitpos(l) = (b = l >> 1; iseven(l) ? 4 + 2b : (20 + 2b) & 31)   # l: 0-based code index in its K16 step
function Base.getindex(w::Q6Weight, i::Int, j::Int)
    b = w.p[(j + 1) ÷ 2, i]
    c = isodd(j) ? b & 0x0f : b >> 0x04
    m = w.meta[(j - 1) ÷ 32 + 1, i]
    half = ((j - 1) % 32) ÷ 16
    s = half == 0 ? _f16lo(m[1]) : _f16hi(m[1])
    h2 = (m[2 + half] >> _q6_bitpos((j - 1) % 16)) & 0x03
    return Float32(Int(c) + 16 * Int(h2) - 32) * s
end
@inline function _q6_val(c::UInt8, l::UInt32, w1::UInt32, w2::UInt32, w3::UInt32)
    half = l >> 4
    s = half == 0 ? _f16lo(w1) : _f16hi(w1)
    pos = (l & 1) == 0 ? 4 + 2 * ((l & 15) >> 1) : (20 + 2 * ((l & 15) >> 1)) & 31
    h2 = ((half == 0 ? w2 : w3) >> pos) & 0x03
    return Float32(Int(c) + 16 * Int(h2) - 32) * s
end
function dequantize(w::Q6Weight)
    M, K = size(w); G = K ÷ 32
    codes = similar(w.p, UInt8, 2, K ÷ 2, M)
    view(codes, 1, :, :) .= w.p .& 0x0f
    view(codes, 2, :, :) .= w.p .>> 0x04
    u = Base.reshape(reinterpret(UInt32, vec(w.meta)), 3, G, M)
    w1 = Base.reshape(u[1, :, :], 1, G, M); w2 = Base.reshape(u[2, :, :], 1, G, M); w3 = Base.reshape(u[3, :, :], 1, G, M)
    lv = similar(w.p, UInt32, 32); copyto!(lv, UInt32.(0:31))
    cg = Base.reshape(codes, 32, G, M)
    x = _q6_val.(cg, Base.reshape(lv, 32, 1, 1), w1, w2, w3)
    xr = Base.reshape(x, K, M)
    out = permutedims(xr, (2, 1))
    foreach(_free_now!, (xr, x, cg, codes, lv, w1, w2, w3))
    return out
end
_free_now!(w::Q6Weight) = (_free_now!(w.p); _free_now!(w.meta); nothing)
"""A weight in a GGUF file's own values, repacked on the GPU (see src/NativeWeights.jl)."""
const NativeWeight = Union{Q5Weight, IQ4Weight, Q6Weight}

# GEMV for 5-bit weights: one row per workgroup, a 16-byte code chunk, its fifth-bit word and its scale / minimum per thread and
# K tile, each column in turn (the row's weights stay in cache across columns).
@inline _wb(w::NTuple{16, UInt8}, b::Int) = @inbounds w[b]
@inline _wb(w::NTuple{4, UInt32}, b::Int) = UInt8((w[(b + 3) >> 2] >> UInt32(8 * ((b - 1) & 3))) & 0xff)
@inline function _dot_q5(w::Union{NTuple{16, UInt8}, NTuple{4, UInt32}}, hi::UInt32, xa, xb, xc, xd)
    x = (xa..., xb..., xc..., xd...)
    return sum(ntuple(b -> Float32((_wb(w, b) & 0x0f) + 0x10 * UInt8((hi >> UInt32(4 + b - 1)) & 0x01)) * x[2b - 1] +
                           Float32((_wb(w, b) >> 0x04) + 0x10 * UInt8((hi >> UInt32((20 + b - 1) & 31)) & 0x01)) * x[2b], Val(16)))
end
# Q6_K: bytes 1..8 are the first K16 step (high bits `h`, step `s1`), bytes 9..16 the second
@inline function _dot_q6(w::Union{NTuple{16, UInt8}, NTuple{4, UInt32}}, h::UInt32, h2::UInt32, s1::Float32, s2::Float32, xa, xb, xc, xd)
    x = (xa..., xb..., xc..., xd...)
    t(b, hw, r) = (Float32(Int(_wb(w, b) & 0x0f) + 16 * Int(bitrotate(hw, -(4 + 2r)) & 0x03) - 32) * x[2b - 1] +
                   Float32(Int(_wb(w, b) >> 0x04) + 16 * Int(bitrotate(hw, -(20 + 2r)) & 0x03) - 32) * x[2b])
    a1 = sum(ntuple(b -> t(b, h, b - 1), Val(8)))
    a2 = sum(ntuple(b -> t(b + 8, h2, b - 1), Val(8)))
    return s1 * a1 + s2 * a2
end
@inline function _dot_iq4(w::Union{NTuple{16, UInt8}, NTuple{4, UInt32}}, xa, xb, xc, xd)
    x = (xa..., xb..., xc..., xd...)
    return sum(ntuple(b -> IQ4NL_TUPLE[Int(_wb(w, b) & 0x0f) + 1] * x[2b - 1] + IQ4NL_TUPLE[Int(_wb(w, b) >> 0x04) + 1] * x[2b], Val(16)))
end
const IQ4NL_TUPLE = (-127f0, -104f0, -83f0, -65f0, -49f0, -35f0, -22f0, -10f0, 1f0, 13f0, 25f0, 38f0, 53f0, 69f0, 89f0, 113f0)
# GEMV for the native formats: one wave computes 4 rows (the activation loads of a K32 group are shared by the 4 rows), its 32 lanes stride over
# the row's K32 groups, and a shuffle reduction replaces the workgroup tree (no LDS, no barriers). 8 waves = 32 rows per workgroup of 256.
@inline function _gv_load(::Val, Wv, Meta, Blk, g, row)
    @inbounds begin
        return (Wv[g, row], Meta[g, row])
    end
end
@inline function _gv_load(::Val{:q5}, Wv, Meta, Blk, g, row)
    @inbounds begin
        return (Wv[g, row], Meta[g, row], Blk[((g - 1) >> 3) + 1, row])
    end
end
@inline function _gv_use(::Val{:q5}, ld, g, xa, xb, xc, xd, xs)
    w, hi, rec = ld
    s, m = _kq_scale_min(rec, (g - 1) & 7)
    return s * _dot_q5(w, hi, xa, xb, xc, xd) + m * xs
end
@inline function _gv_use(::Val{:iq4}, ld, g, xa, xb, xc, xd, xs)
    w, m = ld
    return Float32(m) * _dot_iq4(w, xa, xb, xc, xd)
end
@inline function _gv_use(::Val{:q6}, ld, g, xa, xb, xc, xd, xs)
    w, m = ld
    return _dot_q6(w, m[2], m[3], _f16lo(m[1]), _f16hi(m[1]), xa, xb, xc, xd)
end
# One wave computes 4 rows. The next K32 group's weights are loaded (unconditionally: the last group is re-read) before the current one is
# used, so a group's loads are in flight during the previous group's decode and dot products.
function _nat_gemv2_kernel!(Y, Wv, Meta, Blk, Xv, K32, M, N, Res, ::Val{RES}, ::Val{F}) where {RES, F}
    t = workitemIdx().x - 1; lane = t & 31; wave = t >> 5
    row0 = ((workgroupIdx().x - 1) * 8 + wave) * 4 + 1          # first of this wave's 4 rows (1-based)
    if row0 <= M
        for c in 1:N
            a1 = 0f0; a2 = 0f0; a3 = 0f0; a4 = 0f0
            g = lane + 1
            l1 = _gv_load(Val(F), Wv, Meta, Blk, g, row0); l2 = _gv_load(Val(F), Wv, Meta, Blk, g, row0 + 1)
            l3 = _gv_load(Val(F), Wv, Meta, Blk, g, row0 + 2); l4 = _gv_load(Val(F), Wv, Meta, Blk, g, row0 + 3)
            while g <= K32
                gn = min(g + 32, K32)
                n1 = _gv_load(Val(F), Wv, Meta, Blk, gn, row0); n2 = _gv_load(Val(F), Wv, Meta, Blk, gn, row0 + 1)
                n3 = _gv_load(Val(F), Wv, Meta, Blk, gn, row0 + 2); n4 = _gv_load(Val(F), Wv, Meta, Blk, gn, row0 + 3)
                kk = 4g
                @inbounds xa = (Xv[kk - 3, c], Xv[kk - 2, c], Xv[kk - 1, c], Xv[kk, c])
                xs = F === :q5 ? _sum32(xa...) : 0f0
                xa1, xb1, xc1, xd1 = xa
                a1 += _gv_use(Val(F), l1, g, xa1, xb1, xc1, xd1, xs)
                a2 += _gv_use(Val(F), l2, g, xa1, xb1, xc1, xd1, xs)
                a3 += _gv_use(Val(F), l3, g, xa1, xb1, xc1, xd1, xs)
                a4 += _gv_use(Val(F), l4, g, xa1, xb1, xc1, xd1, xs)
                l1 = n1; l2 = n2; l3 = n3; l4 = n4
                g += 32
            end
            for off in (16, 8, 4, 2, 1)
                a1 += AMDGPU.Device.shfl_xor(a1, Cint(off)); a2 += AMDGPU.Device.shfl_xor(a2, Cint(off))
                a3 += AMDGPU.Device.shfl_xor(a3, Cint(off)); a4 += AMDGPU.Device.shfl_xor(a4, Cint(off))
            end
            if lane < 4
                v = lane == 0 ? a1 : lane == 1 ? a2 : lane == 2 ? a3 : a4
                @inbounds Y[row0 + lane, c] = RES ? v + Res[row0 + lane, c] : v
            end
        end
    end
    return
end

# The same for C (2..4) columns c0 + 1 .. c0 + C at once: each K32 group's weights are loaded once and used for every column (the speculative
# verify steps run 2..4 columns; the per-column loop above re-reads and re-decodes the weights for each). The decode is streamed in four
# chunks of 8 inputs (one 32-bit word of codes each), every chunk applied to all columns before the next: a whole group decoded at once
# stays live across the columns and spills.
@inline _gv_pre(::Val{:iq4}, ld, g) = (ld[1], Float32(ld[2]))
@inline function _gv_pre(::Val{:q5}, ld, g)
    s, m = _kq_scale_min(ld[3], (g - 1) & 7)
    return (ld[1], ld[2], s, m)
end
@inline _gv_pre(::Val{:q6}, ld, g) = (ld[1], ld[2][2], ld[2][3], _f16lo(ld[2][1]), _f16hi(ld[2][1]))
@inline _byte(w::NTuple{4, UInt32}, ::Val{S}, bi::Int) where {S} = UInt8((w[S] >> UInt32(8 * (bi - 1))) & 0xff)
@inline function _dotc(::Val{:iq4}, pre, ::Val{S}, x) where {S}
    w, m = pre
    return m * sum(ntuple(bi -> (b = _byte(w, Val(S), bi); IQ4NL_TUPLE[Int(b & 0x0f) + 1] * x[2bi - 1] + IQ4NL_TUPLE[Int(b >> 0x04) + 1] * x[2bi]), Val(4)))
end
@inline function _dotc(::Val{:q5}, pre, ::Val{S}, x) where {S}
    w, hi, s, m = pre
    q = sum(ntuple(Val(4)) do bi
        b = 4 * (S - 1) + bi
        by = _byte(w, Val(S), bi)
        Float32((by & 0x0f) + 0x10 * UInt8((hi >> UInt32(4 + b - 1)) & 0x01)) * x[2bi - 1] +
            Float32((by >> 0x04) + 0x10 * UInt8((hi >> UInt32((20 + b - 1) & 31)) & 0x01)) * x[2bi]
    end)
    return s * q + m * sum(x)
end
@inline function _dotc(::Val{:q6}, pre, ::Val{S}, x) where {S}
    w, h, h2, s1, s2 = pre
    hw = S <= 2 ? h : h2; st = S <= 2 ? s1 : s2
    q = sum(ntuple(Val(4)) do bi
        r = 4 * ((S - 1) & 1) + bi - 1
        by = _byte(w, Val(S), bi)
        Float32(Int(by & 0x0f) + 16 * Int(bitrotate(hw, -(4 + 2r)) & 0x03) - 32) * x[2bi - 1] +
            Float32(Int(by >> 0x04) + 16 * Int(bitrotate(hw, -(20 + 2r)) & 0x03) - 32) * x[2bi]
    end)
    return st * q
end
@inline function _gv_chunk(::Val{F}, ::Val{S}, acc::NTuple{C, NTuple{4, Float32}}, p1, p2, p3, p4, Xv, g, c0) where {F, S, C}
    return ntuple(Val(C)) do j
        x = @inbounds Xv[4g - 4 + S, c0 + j]
        a1, a2, a3, a4 = acc[j]
        (a1 + _dotc(Val(F), p1, Val(S), x), a2 + _dotc(Val(F), p2, Val(S), x), a3 + _dotc(Val(F), p3, Val(S), x), a4 + _dotc(Val(F), p4, Val(S), x))
    end
end
@inline function _gv_group(::Val{F}, acc, l1, l2, l3, l4, Xv, g, c0) where {F}
    p1 = _gv_pre(Val(F), l1, g); p2 = _gv_pre(Val(F), l2, g); p3 = _gv_pre(Val(F), l3, g); p4 = _gv_pre(Val(F), l4, g)
    acc = _gv_chunk(Val(F), Val(1), acc, p1, p2, p3, p4, Xv, g, c0)
    acc = _gv_chunk(Val(F), Val(2), acc, p1, p2, p3, p4, Xv, g, c0)
    acc = _gv_chunk(Val(F), Val(3), acc, p1, p2, p3, p4, Xv, g, c0)
    return _gv_chunk(Val(F), Val(4), acc, p1, p2, p3, p4, Xv, g, c0)
end
function _nat_gemvN_kernel!(Y, Wv, Meta, Blk, Xv, K32, M, c0, Res, ::Val{RES}, ::Val{F}, ::Val{C}) where {RES, F, C}
    t = workitemIdx().x - 1; lane = t & 31; wave = t >> 5
    row0 = ((workgroupIdx().x - 1) * 8 + wave) * 4 + 1
    if row0 <= M
        acc = ntuple(_ -> (0f0, 0f0, 0f0, 0f0), Val(C))
        g = lane + 1
        l1 = _gv_load(Val(F), Wv, Meta, Blk, g, row0); l2 = _gv_load(Val(F), Wv, Meta, Blk, g, row0 + 1)
        l3 = _gv_load(Val(F), Wv, Meta, Blk, g, row0 + 2); l4 = _gv_load(Val(F), Wv, Meta, Blk, g, row0 + 3)
        while g <= K32
            gn = min(g + 32, K32)
            n1 = _gv_load(Val(F), Wv, Meta, Blk, gn, row0); n2 = _gv_load(Val(F), Wv, Meta, Blk, gn, row0 + 1)
            n3 = _gv_load(Val(F), Wv, Meta, Blk, gn, row0 + 2); n4 = _gv_load(Val(F), Wv, Meta, Blk, gn, row0 + 3)
            acc = _gv_group(Val(F), acc, l1, l2, l3, l4, Xv, g, c0)
            l1 = n1; l2 = n2; l3 = n3; l4 = n4
            g += 32
        end
        for j in 1:C
            a1, a2, a3, a4 = acc[j]
            for off in (16, 8, 4, 2, 1)
                a1 += AMDGPU.Device.shfl_xor(a1, Cint(off)); a2 += AMDGPU.Device.shfl_xor(a2, Cint(off))
                a3 += AMDGPU.Device.shfl_xor(a3, Cint(off)); a4 += AMDGPU.Device.shfl_xor(a4, Cint(off))
            end
            if lane < 4
                v = lane == 0 ? a1 : lane == 1 ? a2 : lane == 2 ? a3 : a4
                @inbounds Y[row0 + lane, c0 + j] = RES ? v + Res[row0 + lane, c0 + j] : v
            end
        end
    end
    return
end

# The same with Float16 activations and packed dot products (the Q5_K / Q6_K counterpart of `_q4d_kernel!` below): a weight is decoded once into
# Float16 pairs (input 2j, input 2j+1) of a byte, as 1024 + q (the 0x6400 mantissa trick; the fifth / fifth-and-sixth bits are or-ed in at bits 4-5
# and 20-21), and each column then costs one v_dot2_f32_f16 per 2 weights. The 1024 offset (and Q6's -32) is folded into the group's minimum term,
# which needs the activations' sum per group (a dot2 against ones, shared by the 4 rows): s*sum((1024+q) x) + (m - 1024 s) sum(x).
@inline _d2_ones(x::NTuple{4, UInt32}, a::Float32) = _fdot2(0x3c003c00, x[4], _fdot2(0x3c003c00, x[3], _fdot2(0x3c003c00, x[2], _fdot2(0x3c003c00, x[1], a))))
@inline _d2_dot(p::NTuple{4, UInt32}, x::NTuple{4, UInt32}, a::Float32) = _fdot2(p[4], x[4], _fdot2(p[3], x[3], _fdot2(p[2], x[2], _fdot2(p[1], x[1], a))))
@inline _d2_nib(w::UInt32, ::Val{J}) where {J} = (u = w >> UInt32(8J); ((u & 0x0000000f) | ((u << 12) & 0x000f0000)) | 0x64006400)
@inline function _d2_pairs(::Val{:q5}, pre, ::Val{S}) where {S}
    w = pre[1][S]; hw = pre[2]
    return ntuple(jj -> _d2_nib(w, Val(jj - 1)) | (bitrotate(hw, -(4S + jj - 5)) & 0x00100010), Val(4))
end
@inline function _d2_pairs(::Val{:q6}, pre, ::Val{S}) where {S}
    w = pre[1][S]; hw = S <= 2 ? pre[2] : pre[3]
    return ntuple(jj -> _d2_nib(w, Val(jj - 1)) | (bitrotate(hw, -2 * (4 * ((S - 1) & 1) + jj - 1)) & 0x00300030), Val(4))
end
# IQ4_XS / IQ4_NL: the four codes of each byte pair through the table lookup of the matrix-core path (`_lut_iq4`: KV + 128 as bytes), put in the
# mantissa of 1024.0 as above; the group's 1024 + 128 offset (1152) leaves with its activation sum: s * (sum((1024 + u) x) - 1152 sum(x)).
@inline function _d2_pairs(::Val{:iq4}, pre, ::Val{S}) where {S}
    w = pre[1][S]
    ul = _lut_iq4(w & 0x0f0f0f0f); uh = _lut_iq4((w >> 0x04) & 0x0f0f0f0f)
    return ntuple(jj -> _perm(uh, ul, UInt32(jj - 1) | 0x00000c00 | (UInt32(jj + 3) << 16) | 0x0c000000) | 0x64006400, Val(4))
end
@inline function _d2_chunk(::Val{F}, ::Val{S}, d0, xs0, pres, Xq, g, c0, ::Val{C}) where {F, S, C}
    x = ntuple(j -> (@inbounds Xq[4g - 4 + S, c0 + j]), Val(C))
    ps = map(p -> _d2_pairs(Val(F), p, Val(S)), pres)
    xs1 = ntuple(j -> _d2_ones(x[j], xs0[j]), Val(C))
    d1 = ntuple(r -> ntuple(j -> _d2_dot(ps[r], x[j], d0[r][j]), Val(C)), Val(4))
    return d1, xs1
end
@inline _d2_zero(::Val{C}) where {C} = (ntuple(_ -> ntuple(_ -> 0f0, Val(C)), Val(4)), ntuple(_ -> 0f0, Val(C)))
@inline function _d2_group(::Val{:q5}, acc, pres, Xq, g, c0, vc::Val{C}) where {C}
    d0, xs0 = _d2_zero(vc)
    d1, xs1 = _d2_chunk(Val(:q5), Val(1), d0, xs0, pres, Xq, g, c0, vc)
    d2, xs2 = _d2_chunk(Val(:q5), Val(2), d1, xs1, pres, Xq, g, c0, vc)
    d3, xs3 = _d2_chunk(Val(:q5), Val(3), d2, xs2, pres, Xq, g, c0, vc)
    d4, xs4 = _d2_chunk(Val(:q5), Val(4), d3, xs3, pres, Xq, g, c0, vc)
    return ntuple(j -> ntuple(r -> acc[j][r] + pres[r][3] * d4[r][j] + (pres[r][4] - 1024f0 * pres[r][3]) * xs4[j], Val(4)), Val(C))
end
@inline function _d2_group(::Val{:iq4}, acc, pres, Xq, g, c0, vc::Val{C}) where {C}
    d0, xs0 = _d2_zero(vc)
    d1, xs1 = _d2_chunk(Val(:iq4), Val(1), d0, xs0, pres, Xq, g, c0, vc)
    d2, xs2 = _d2_chunk(Val(:iq4), Val(2), d1, xs1, pres, Xq, g, c0, vc)
    d3, xs3 = _d2_chunk(Val(:iq4), Val(3), d2, xs2, pres, Xq, g, c0, vc)
    d4, xs4 = _d2_chunk(Val(:iq4), Val(4), d3, xs3, pres, Xq, g, c0, vc)
    return ntuple(j -> ntuple(r -> acc[j][r] + pres[r][2] * (d4[r][j] - 1152f0 * xs4[j]), Val(4)), Val(C))
end
@inline function _d2_group(::Val{:q6}, acc, pres, Xq, g, c0, vc::Val{C}) where {C}
    d0, xs0 = _d2_zero(vc)
    d1, xs1 = _d2_chunk(Val(:q6), Val(1), d0, xs0, pres, Xq, g, c0, vc)
    d2, xs2 = _d2_chunk(Val(:q6), Val(2), d1, xs1, pres, Xq, g, c0, vc)
    acc1 = ntuple(j -> ntuple(r -> acc[j][r] + pres[r][4] * (d2[r][j] - 1056f0 * xs2[j]), Val(4)), Val(C))
    e0, ys0 = _d2_zero(vc)
    e1, ys1 = _d2_chunk(Val(:q6), Val(3), e0, ys0, pres, Xq, g, c0, vc)
    e2, ys2 = _d2_chunk(Val(:q6), Val(4), e1, ys1, pres, Xq, g, c0, vc)
    return ntuple(j -> ntuple(r -> acc1[j][r] + pres[r][5] * (e2[r][j] - 1056f0 * ys2[j]), Val(4)), Val(C))
end
function _nat_gemvD_kernel!(Y, Wv, Meta, Blk, Xq, K32, M, c0, Res, ::Val{RES}, ::Val{F}, ::Val{C}) where {RES, F, C}
    t = workitemIdx().x - 1; lane = t & 31; wave = t >> 5
    row0 = ((workgroupIdx().x - 1) * 8 + wave) * 4 + 1
    if row0 <= M
        acc = ntuple(_ -> (0f0, 0f0, 0f0, 0f0), Val(C))
        g = lane + 1
        l1 = _gv_load(Val(F), Wv, Meta, Blk, g, row0); l2 = _gv_load(Val(F), Wv, Meta, Blk, g, row0 + 1)
        l3 = _gv_load(Val(F), Wv, Meta, Blk, g, row0 + 2); l4 = _gv_load(Val(F), Wv, Meta, Blk, g, row0 + 3)
        while g <= K32
            gn = min(g + 32, K32)
            n1 = _gv_load(Val(F), Wv, Meta, Blk, gn, row0); n2 = _gv_load(Val(F), Wv, Meta, Blk, gn, row0 + 1)
            n3 = _gv_load(Val(F), Wv, Meta, Blk, gn, row0 + 2); n4 = _gv_load(Val(F), Wv, Meta, Blk, gn, row0 + 3)
            pres = (_gv_pre(Val(F), l1, g), _gv_pre(Val(F), l2, g), _gv_pre(Val(F), l3, g), _gv_pre(Val(F), l4, g))
            acc = _d2_group(Val(F), acc, pres, Xq, g, c0, Val(C))
            l1 = n1; l2 = n2; l3 = n3; l4 = n4
            g += 32
        end
        for j in 1:C
            a1, a2, a3, a4 = acc[j]
            for off in (16, 8, 4, 2, 1)
                a1 += AMDGPU.Device.shfl_xor(a1, Cint(off)); a2 += AMDGPU.Device.shfl_xor(a2, Cint(off))
                a3 += AMDGPU.Device.shfl_xor(a3, Cint(off)); a4 += AMDGPU.Device.shfl_xor(a4, Cint(off))
            end
            if lane < 4
                v = lane == 0 ? a1 : lane == 1 ? a2 : lane == 2 ? a3 : a4
                @inbounds Y[row0 + lane, c0 + j] = RES ? v + Res[row0 + lane, c0 + j] : v
            end
        end
    end
    return
end

# sum_i q_i x_i over one 32-weight load: byte b holds inputs 2b-1 (low nibble) and
# 2b (high nibble); x is the load's 32 activations as four 8-tuples. (Unpacking the
# load once to 32 Float32s for reuse across columns spills registers: ~2x slower.)
@inline function _dot_q4(w::NTuple{16, UInt8}, xa, xb, xc, xd)
    x = (xa..., xb..., xc..., xd...)
    return sum(ntuple(b -> Float32(w[b] & 0x0f) * x[2b - 1] + Float32(w[b] >> 0x04) * x[2b], Val(16)))
end
@inline _sum32(xa, xb, xc, xd) = sum(xa) + sum(xb) + sum(xc) + sum(xd)

# Y[:, c] = Wq * X[:, c], like the int8 kernel: R rows per workgroup share each
# column's activations, C columns per pass. Per 32-weight load a row adds
# scale * sum(q x) + min * sum(x), and sum(x) is shared by the rows.
# SYM: symmetric weights, whose minimum is -8 * scale (`mn` is not read).
@kernel function _q4_matmul_kernel!(Y, @Const(Wv), @Const(sc), @Const(mn), @Const(Xv), K32, N,
                                    shift, ::Val{R}, ::Val{C}, @Const(Res), ::Val{RES},
                                    ::Val{SYM}) where {R, C, RES, SYM}
    grp = @index(Group, Linear); lid = @index(Local, Linear); G = @groupsize()[1]
    s = @localmem Float32 (256, 2 * C)
    row0 = (grp - 1) * R
    c0 = 0
    while c0 < N
        acc1 = ntuple(_ -> 0f0, Val(C))
        acc2 = ntuple(_ -> 0f0, Val(C))
        k = lid
        while k <= K32
            gi = ((k - 1) >> shift) + 1
            @inbounds w1 = Wv[k, row0 + 1]
            @inbounds w2 = Wv[k, row0 + R]
            @inbounds s1 = Float32(sc[gi, row0 + 1])
            @inbounds s2 = Float32(sc[gi, row0 + R])
            m1 = SYM ? -8f0 * s1 : (@inbounds Float32(mn[gi, row0 + 1]))
            m2 = SYM ? -8f0 * s2 : (@inbounds Float32(mn[gi, row0 + R]))
            kk = 4k; cb = c0   # fresh bindings: closures must not capture reassigned variables
            acc1, acc2 = let a1 = acc1, a2 = acc2
                xs = ntuple(j -> (c = min(cb + j, N);
                                  @inbounds (Xv[kk - 3, c], Xv[kk - 2, c], Xv[kk - 1, c], Xv[kk, c])), Val(C))
                sums = ntuple(j -> _sum32(xs[j]...), Val(C))
                (ntuple(j -> a1[j] + s1 * _dot_q4(w1, xs[j]...) + m1 * sums[j], Val(C)),
                 R == 2 ? ntuple(j -> a2[j] + s2 * _dot_q4(w2, xs[j]...) + m2 * sums[j], Val(C)) : a2)
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
            if c0 + j <= N
                @inbounds Y[row0 + r, c0 + j] = RES ? s[1, lid] + Res[row0 + r, c0 + j] : s[1, lid]
            end
        end
        @synchronize
        c0 += C
    end
end

# The same GEMV with 4 rows per workgroup (each activation load is shared by 4 rows instead of 2): the
# activations are read as 128 bytes per column per 16-byte weight chunk, and that L1 traffic is what limits
# the multi-column kernel. Written out per row (a1..a4) like the 2-row kernel: nested closures over a
# tuple of row accumulators did not compile for the GPU.
@kernel function _q4_matmul_kernel_r4!(Y, @Const(Wv), @Const(sc), @Const(mn), @Const(Xv), K32, N,
                                       shift, ::Val{C}, @Const(Res), ::Val{RES}, ::Val{SYM}) where {C, RES, SYM}
    grp = @index(Group, Linear); lid = @index(Local, Linear); G = @groupsize()[1]
    s = @localmem Float32 (256, 4 * C)
    row0 = (grp - 1) * 4
    c0 = 0
    while c0 < N
        acc1 = ntuple(_ -> 0f0, Val(C)); acc2 = ntuple(_ -> 0f0, Val(C))
        acc3 = ntuple(_ -> 0f0, Val(C)); acc4 = ntuple(_ -> 0f0, Val(C))
        k = lid
        while k <= K32
            gi = ((k - 1) >> shift) + 1
            @inbounds w1 = Wv[k, row0 + 1]; @inbounds w2 = Wv[k, row0 + 2]
            @inbounds w3 = Wv[k, row0 + 3]; @inbounds w4 = Wv[k, row0 + 4]
            @inbounds s1 = Float32(sc[gi, row0 + 1]); @inbounds s2 = Float32(sc[gi, row0 + 2])
            @inbounds s3 = Float32(sc[gi, row0 + 3]); @inbounds s4 = Float32(sc[gi, row0 + 4])
            m1 = SYM ? -8f0 * s1 : (@inbounds Float32(mn[gi, row0 + 1]))
            m2 = SYM ? -8f0 * s2 : (@inbounds Float32(mn[gi, row0 + 2]))
            m3 = SYM ? -8f0 * s3 : (@inbounds Float32(mn[gi, row0 + 3]))
            m4 = SYM ? -8f0 * s4 : (@inbounds Float32(mn[gi, row0 + 4]))
            kk = 4k; cb = c0   # fresh bindings: closures must not capture reassigned variables
            acc1, acc2, acc3, acc4 = let a1 = acc1, a2 = acc2, a3 = acc3, a4 = acc4
                xs = ntuple(j -> (c = min(cb + j, N);
                                  @inbounds (Xv[kk - 3, c], Xv[kk - 2, c], Xv[kk - 1, c], Xv[kk, c])), Val(C))
                sums = ntuple(j -> _sum32(xs[j]...), Val(C))
                (ntuple(j -> a1[j] + s1 * _dot_q4(w1, xs[j]...) + m1 * sums[j], Val(C)),
                 ntuple(j -> a2[j] + s2 * _dot_q4(w2, xs[j]...) + m2 * sums[j], Val(C)),
                 ntuple(j -> a3[j] + s3 * _dot_q4(w3, xs[j]...) + m3 * sums[j], Val(C)),
                 ntuple(j -> a4[j] + s4 * _dot_q4(w4, xs[j]...) + m4 * sums[j], Val(C)))
            end
            k += G
        end
        @unroll for j in 1:C
            @inbounds s[lid, (j - 1) * 4 + 1] = acc1[j]; @inbounds s[lid, (j - 1) * 4 + 2] = acc2[j]
            @inbounds s[lid, (j - 1) * 4 + 3] = acc3[j]; @inbounds s[lid, (j - 1) * 4 + 4] = acc4[j]
        end
        @synchronize
        stride = G ÷ 2
        while stride > 0
            if lid <= stride
                @unroll for i in 1:4*C
                    @inbounds s[lid, i] += s[lid + stride, i]
                end
            end
            @synchronize
            stride ÷= 2
        end
        if lid <= 4 * C
            r = (lid - 1) % 4 + 1; j = (lid - 1) ÷ 4 + 1
            if c0 + j <= N
                @inbounds Y[row0 + r, c0 + j] = RES ? s[1, lid] + Res[row0 + r, c0 + j] : s[1, lid]
            end
        end
        @synchronize
        c0 += C
    end
end

# Columns per pass and threads, measured on Llama-3-8B shapes with weights read
# from DRAM: batch 1 wants 256 threads only where K/32 >= 256 (every thread busy);
# with several columns, 2 per pass and 128 threads are best (~2x 4 per pass at 4
# columns). Batched int4 is still bound by activation reads (128 bytes per column
# per 16-byte load, twice int8's ratio); unpacking the codes once per load instead
# of per column did not help, and neither did more rows per workgroup (4 or 8 rows
# sharing each activation load were slower). The kernel is instruction-bound with
# several columns (nibble extract, convert and multiply-add per weight and column,
# ~60% of the GPU's scalar issue rate at 8 columns): the next step is packed Float16
# math (two weights per instruction) with Float16 activations.
const Q4_MAX_COLS = 2
const Q4_ROWS_MIN_N = Ref(2)       # the 4-row kernel is used from this many columns
const Q4_ROWS = Ref(4)           # rows per workgroup of the multi-column int4 GEMV (see `_q4_matmul_kernel_r!`)
const DEFAULT_Q4_THREADS = 0     # 0: chosen per call (`_q4_threads`)
_q4_threads(K32::Int, N::Int) = (N == 1 && K32 >= 256) ? 256 : 128
# threads of the 4-row kernel: 64 for tall matrices with a short K (many workgroups), else 128 (measured at the 27B's shapes)
_q4_threads_r4(M::Int, K32::Int) = (K32 <= 256 && M >= 8192) ? 64 : 128

# --- int4 GEMV with Float16 activations and v_dot2_f32_f16 ------------------------------------------------
# The Float32 kernels above spend ~3 instructions per weight and column (nibble extract, convert, FMA), so
# they are instruction-bound from 2 columns on. Here the activations are Float16 in a pair order matching
# the weights' nibbles, a weight word is dequantized once per 8 weights into packed Float16 pairs ((1024 + q)
# from the 0x6400 mantissa trick, minus 1032 -> q - 8, exact), and every column then costs one
# v_dot2_f32_f16 per 2 weights: 2, 3 columns run at the 1-column speed (bandwidth). Symmetric weights only.
# (Float16 activations: relative error ~2e-4 of the result, like the WMMA kernels.)
@inline function _fdot2(a::UInt32, b::UInt32, c::Float32)
    Base.llvmcall(("declare float @llvm.amdgcn.fdot2(<2 x half>, <2 x half>, float, i1)
define float @entry(i32 %a, i32 %b, float %c) #0 {
  %x = bitcast i32 %a to <2 x half>
  %y = bitcast i32 %b to <2 x half>
  %r = call float @llvm.amdgcn.fdot2(<2 x half> %x, <2 x half> %y, float %c, i1 false)
  ret float %r
}
attributes #0 = { alwaysinline }", "entry"), Float32, Tuple{UInt32,UInt32,Float32}, a, b, c)
end
@inline function _pkadd(a::UInt32, b::UInt32)
    Base.llvmcall(("define i32 @entry(i32 %a, i32 %b) #0 {
  %x = bitcast i32 %a to <2 x half>
  %y = bitcast i32 %b to <2 x half>
  %r = fadd <2 x half> %x, %y
  %o = bitcast <2 x half> %r to i32
  ret i32 %o
}
attributes #0 = { alwaysinline }", "entry"), UInt32, Tuple{UInt32,UInt32}, a, b)
end
const _Q4D_NEG1032 = 0xe408e408     # -1032 in f16 pairs: (1024 + q) - 1032 = q - 8
@inline _q4d_dq(w::UInt32, sh) = _pkadd(((w >> sh) & 0x000f000f) | 0x64006400, _Q4D_NEG1032)

@inline function _q4d_wdot(wi::UInt32, x1::UInt32, x2::UInt32, x3::UInt32, x4::UInt32, acc::Float32)
    q0 = _q4d_dq(wi, 0x00000000); q1 = _q4d_dq(wi, 0x00000004); q2 = _q4d_dq(wi, 0x00000008); q3 = _q4d_dq(wi, 0x0000000c)
    return _fdot2(q3, x4, _fdot2(q2, x3, _fdot2(q1, x2, _fdot2(q0, x1, acc))))
end
# one row's chunk dot against one column's 16 activation pairs
@inline function _q4d_coldot(w::NTuple{4,UInt32}, x::NTuple{16,UInt32})
    a = _q4d_wdot(w[1], x[1], x[2], x[3], x[4], 0f0)
    a = _q4d_wdot(w[2], x[5], x[6], x[7], x[8], a)
    a = _q4d_wdot(w[3], x[9], x[10], x[11], x[12], a)
    return _q4d_wdot(w[4], x[13], x[14], x[15], x[16], a)
end
@inline _q4d_rowdots(w::NTuple{4,UInt32}, xs::NTuple{C,NTuple{16,UInt32}}) where {C} = ntuple(c -> _q4d_coldot(w, xs[c]), Val(C))

@inline _q4d_cat4(a::NTuple{4,UInt32}, b::NTuple{4,UInt32}, c::NTuple{4,UInt32}, d::NTuple{4,UInt32}) =
    (a[1], a[2], a[3], a[4], b[1], b[2], b[3], b[4], c[1], c[2], c[3], c[4], d[1], d[2], d[3], d[4])
@inline _q4d_xcol(Xv, kk, c) = @inbounds _q4d_cat4(Xv[4kk - 3, c], Xv[4kk - 2, c], Xv[4kk - 1, c], Xv[4kk, c])
@inline _q4d_xcols(Xv, kk, cb, N, ::Val{C}) where {C} = ntuple(j -> _q4d_xcol(Xv, kk, min(cb + j, N)), Val(C))
@kernel function _q4d_kernel!(Y, @Const(Wv), @Const(sc), @Const(Xv), K32, N, shift, ::Val{C}, @Const(Res), ::Val{RES}) where {C, RES}
    grp = @index(Group, Linear); lid = @index(Local, Linear); G = @groupsize()[1]
    s = @localmem Float32 (256, 4 * C)
    row0 = (grp - 1) * 4
    c0 = 0
    while c0 < N
        acc1 = ntuple(_ -> 0f0, Val(C)); acc2 = ntuple(_ -> 0f0, Val(C))
        acc3 = ntuple(_ -> 0f0, Val(C)); acc4 = ntuple(_ -> 0f0, Val(C))
        k = lid
        while k <= K32
            gi = ((k - 1) >> shift) + 1
            @inbounds w1 = Wv[k, row0 + 1]; @inbounds w2 = Wv[k, row0 + 2]
            @inbounds w3 = Wv[k, row0 + 3]; @inbounds w4 = Wv[k, row0 + 4]
            @inbounds s1 = Float32(sc[gi, row0 + 1]); @inbounds s2 = Float32(sc[gi, row0 + 2])
            @inbounds s3 = Float32(sc[gi, row0 + 3]); @inbounds s4 = Float32(sc[gi, row0 + 4])
            kk = k; cb = c0
            acc1, acc2, acc3, acc4 = let a1 = acc1, a2 = acc2, a3 = acc3, a4 = acc4
                xs = _q4d_xcols(Xv, kk, cb, N, Val(C))
                d1 = _q4d_rowdots(w1, xs); d2 = _q4d_rowdots(w2, xs); d3 = _q4d_rowdots(w3, xs); d4 = _q4d_rowdots(w4, xs)
                (ntuple(j -> a1[j] + s1 * d1[j], Val(C)), ntuple(j -> a2[j] + s2 * d2[j], Val(C)),
                 ntuple(j -> a3[j] + s3 * d3[j], Val(C)), ntuple(j -> a4[j] + s4 * d4[j], Val(C)))
            end
            k += G
        end
        @unroll for j in 1:C
            @inbounds s[lid, (j - 1) * 4 + 1] = acc1[j]; @inbounds s[lid, (j - 1) * 4 + 2] = acc2[j]
            @inbounds s[lid, (j - 1) * 4 + 3] = acc3[j]; @inbounds s[lid, (j - 1) * 4 + 4] = acc4[j]
        end
        @synchronize
        stride = G ÷ 2
        while stride > 0
            if lid <= stride
                @unroll for i in 1:4*C
                    @inbounds s[lid, i] += s[lid + stride, i]
                end
            end
            @synchronize
            stride ÷= 2
        end
        if lid <= 4 * C
            r = (lid - 1) % 4 + 1; j = (lid - 1) ÷ 4 + 1
            if c0 + j <= N
                @inbounds Y[row0 + r, c0 + j] = RES ? s[1, lid] + Res[row0 + r, c0 + j] : s[1, lid]
            end
        end
        @synchronize
        c0 += C
    end
end

# f16 activations in the kernel's pair order: within each 8, [0,4,1,5,2,6,3,7]
@kernel function _q4d_xprep_kernel!(X16, @Const(X))
    i = @index(Global, Linear)
    K = size(X, 1)
    n = (i - 1) ÷ K; k = (i - 1) % K          # 0-based
    g = k ÷ 8; r = k % 8                       # position within the 8-group in the new order
    src = 8g + (r ÷ 2) + 4 * (r % 2)
    @inbounds X16[k + 1, n + 1] = Float16(X[src + 1, n + 1])
end


const Q4_DOT2_MIN_N = Ref(2)
const Q4_DOT2_MAX_N = Ref(4)
const _Q4D_WORKSPACE = Dict{Any, Any}()
# Y (M, N) = Wq * B (+ residual) with the kernel above; false if the weight or arrays don't qualify.
function _q4d_matmul!(C, W::Q4Weight, B, residual; threads::Int = 128)
    (C isa ROCArray{Float32} && W.p isa ROCArray{UInt8} && W.mn === nothing && W.scale isa ROCArray{Float16} &&
     get(ENV, "LUMINAL_Q4_DOT2", "1") != "0") || return false
    M, K = size(W)
    (M % 4 == 0 && K % 32 == 0) || return false
    N = length(B) ÷ K
    Bc = B isa DenseArray ? B : copy(B)
    X16 = get!(() -> similar(Bc, Float16, K, N), _Q4D_WORKSPACE, (typeof(Bc), K, N))
    backend = KernelAbstractions.get_backend(C)
    _q4d_xprep_kernel!(backend, 256)(X16, Base.reshape(Bc, K, N); ndrange = K * N)
    Wq = Base.reshape(reinterpret(NTuple{4, UInt32}, vec(W.p)), K ÷ 32, M)
    Xq = Base.reshape(reinterpret(NTuple{4, UInt32}, vec(X16)), K ÷ 8, N)
    lpg = W.group ÷ 32
    ispow2(lpg) || return false
    _q4d_kernel!(backend, threads)(Base.reshape(C, M, N), Wq, W.scale, Xq, K ÷ 32, N, trailing_zeros(lpg), Val(min(N, 4)),
                                   _residual_arg(residual, C, M, N)...; ndrange = (M ÷ 4) * threads)
    Bc === B || _free_now!(Bc)
    return true
end

function _q4_matmul!(C, W::Q4Weight, B; group::Int = DEFAULT_Q4_THREADS, cols::Int = 0,
                     residual = nothing)
    M, K = size(W)
    N = length(B) ÷ K
    if eltype(C) === Float16          # a Float16 result: the 256 x 256 matrix-core GEMM stores it directly, anything else goes through Float32
        N > QUANT_GEMM_COLS && residual === nothing && _wmma_q4_gemm!(C, W, B, nothing) && return C
        C32 = similar(C, Float32)
        _q4_matmul!(C32, W, B; group, cols, residual)
        copyto!(C, C32); _free_now!(C32)
        return C
    end
    if eltype(B) === Float16          # a Float16 activation (see `dty` in compile): only the matrix-core GEMM takes it as it is
        N > QUANT_GEMM_COLS && _wmma_q4_gemm!(C, W, B, residual) && return C
        B = Float32.(B)
    end
    SKINNY_MIN_COLS <= N <= SKINNY_MAX_COLS && _wmma_q4_skinny_gemm!(C, W, B, residual) && return C
    Q4_DOT2_MIN_N[] <= N <= Q4_DOT2_MAX_N[] && _q4d_matmul!(C, W, B, residual) && return C
    N > QUANT_GEMM_COLS && _wmma_q4_gemm!(C, W, B, residual) && return C
    (N > QUANT_GEMM_COLS || !(C isa AnyGPUArray)) && return _dequant_gemm!(C, W, B, residual)
    Bc = B isa DenseArray ? B : copy(B)
    Xv = Base.reshape(reinterpret(NTuple{8, Float32}, vec(Bc)), K ÷ 8, N)
    R = iseven(M) ? 2 : 1
    lpg = W.group ÷ 32
    ispow2(lpg) || error("Q4Weight: group $(W.group) must be 32 times a power of two")
    group == 0 && (group = _q4_threads(K ÷ 32, N))
    group <= 256 || error("int4 GEMV: at most 256 threads per workgroup")
    if Q4_ROWS[] == 4 && N >= Q4_ROWS_MIN_N[] && M % 4 == 0
        group == _q4_threads(K ÷ 32, N) && (group = _q4_threads_r4(M, K ÷ 32))
        _q4_matmul_kernel_r4!(KernelAbstractions.get_backend(C), group)(
            Base.reshape(C, M, N), W.v, W.scale, W.mn === nothing ? W.scale : W.mn, Xv, K ÷ 32, N,
            trailing_zeros(lpg), Val(min(N, cols == 0 ? Q4_MAX_COLS : cols)),
            _residual_arg(residual, C, M, N)..., Val(W.mn === nothing); ndrange = (M ÷ 4) * group)
        return C
    end
    _q4_matmul_kernel!(KernelAbstractions.get_backend(C), group)(
        Base.reshape(C, M, N), W.v, W.scale, W.mn === nothing ? W.scale : W.mn, Xv, K ÷ 32, N,
        trailing_zeros(lpg), Val(R), Val(min(N, cols == 0 ? Q4_MAX_COLS : cols)),
        _residual_arg(residual, C, M, N)..., Val(W.mn === nothing); ndrange = (M ÷ R) * group)
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

@kernel function _half_matmul_kernel!(Y, @Const(Wv), @Const(Xv), K8, N, ::Val{C},
                                      @Const(Res), ::Val{RES}) where {C, RES}
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
        if lid <= C && c0 + lid <= N
            @inbounds Y[row, c0 + lid] = RES ? s[1, lid] + Res[row, c0 + lid] : s[1, lid]
        end
        @synchronize
        c0 += C
    end
end

# Activations rounded to Float16 and zero-padded to a multiple of 64 (128 from 192 columns, 256 from 384) columns, for the
# WMMA GEMM; reused per (array type, K, padded N) like _GEMM_EX_WORKSPACE. The padding
# columns are never written, so they stay zero.
const _WMMA_WORKSPACE = Dict{Any, Any}()

# B (K, N) rounded to Float16 in the zero-padded workspace; returns (x16, Bc), Bc being
# the dense copy to free afterwards when B had to be copied.
function _wmma_activations(B, K, N; wide::Bool=false)
    blk = wide || N >= 384 ? 256 : N >= 192 ? 128 : 64   # the int4 kernels' block widths: 128x256, 128x128, 64x64 (wide: the 256 x 256 format kernel only)
    Npad = cld(N, blk) * blk
    # already Float16 and a whole number of blocks wide (the compiler computed it that way, see `dty` in compile): no copy
    eltype(B) === Float16 && B isa DenseArray && Npad == N && return Base.reshape(B, K, N), B
    Bc = B isa DenseArray ? B : copy(B)
    x16 = get!(() -> (x = similar(Bc, Float16, K, Npad); fill!(x, Float16(0)); x),
               _WMMA_WORKSPACE, (typeof(Bc), K, Npad))
    copyto!(view(x16, :, 1:N), Base.reshape(Bc, K, N))
    return x16, Bc
end

_wmma_add_residual!(C2, residual, M, N) =
    residual === nothing || (C2 .+= Base.reshape(residual isa DenseArray ? residual : copy(residual), M, N))

# C (Out, N) = W * B on the matrix cores (src/WMMA.jl), with the optional residual
# added afterwards. Returns false when the shapes or the device do not fit, and the
# caller falls back to the dequantize-and-rocBLAS path.
function _wmma_half_gemm!(C, W::HalfWeight, B, residual)
    (C isa ROCArray{Float32} && W.t isa ROCArray{Float16} && wmma_enabled()) || return false
    K, M = size(W.t)
    (M % 64 == 0 && K % 32 == 0) || return false
    N = length(B) ÷ K
    x16, Bc = _wmma_activations(B, K, N)
    C2 = Base.reshape(C, M, N)
    wmma_gemm!(C2, W.t, x16, N)
    _wmma_add_residual!(C2, residual, M, N)
    Bc === B || _free_now!(Bc)
    return true
end

# The same for 4-bit weights.
function _wmma_q4_gemm!(C, W::Q4Weight, B, residual)
    (C isa ROCArray{<:Union{Float32,Float16}} && W.p isa ROCArray{UInt8} && W.scale isa ROCArray{Float16} && wmma_enabled()) || return false
    M, K = size(W)
    (M % 16 == 0 && K % 32 == 0 && W.group % 32 == 0) || return false
    N = length(B) ÷ K
    if C isa ROCArray{Float16}      # a Float16 result (see `dty` in compile): only the 256 x 256 kernel stores it
        (residual === nothing && M % 256 == 0 && K % 64 == 0 && ispow2(W.group) && N % 256 == 0 && get(ENV, "LUMINAL_WMMA_G", "1") != "0") || return false
    end
    x16, Bc = _wmma_activations(B, K, N)
    C2 = Base.reshape(C, M, N)
    # a dense Float32 residual is added in the kernel's store (no separate pass over C)
    R = residual isa ROCArray{Float32} && length(residual) == M * N ? Base.reshape(residual, M, N) : nothing
    wmma_q4_gemm!(C2, W.v, W.scale, W.mn, W.group, x16, N; residual = R)
    R === nothing && _wmma_add_residual!(C2, residual, M, N)
    Bc === B || _free_now!(Bc)
    return true
end

# The same for the native formats (the 256 x 256 kernel only; other shapes dequantize).
_nat_fmt(::Q5Weight) = :q5
_nat_fmt(::IQ4Weight) = :iq4
_nat_fmt(::Q6Weight) = :q6
_nat_p(w::Union{Q5Weight, Q6Weight}) = w.p
_nat_p(w::IQ4Weight) = w.q.p
_nat_v(w::Union{Q5Weight, Q6Weight}) = w.v
_nat_v(w::IQ4Weight) = w.q.v
_nat_meta(w::Q6Weight) = w.meta
_nat_meta(w::Q5Weight) = w.hi                              # (the fifth bits)
_nat_blk(w::Q5Weight) = w.blk
_nat_blk(w) = _nat_meta(w)                                    # (unused by the other formats)
_nat_meta(w::IQ4Weight) = w.q.scale                       # (the step per group: all it has)
function _wmma_nat_gemm!(C, W::NativeWeight, B, residual)
    (C isa ROCArray{<:Union{Float32,Float16}} && _nat_p(W) isa ROCArray{UInt8} && wmma_enabled() && get(ENV, "LUMINAL_WMMA_G", "1") != "0") || return false
    M, K = size(W)
    N = length(B) ÷ K
    (M % 256 == 0 && K % 64 == 0 && (residual === nothing || C isa ROCArray{Float32})) || return false   # (any N: the activations are padded to 256 columns)
    C isa ROCArray{Float16} && residual !== nothing && return false
    x16, Bc = _wmma_activations(B, K, N; wide = true)
    C2 = Base.reshape(C, M, N)
    R = residual isa ROCArray{Float32} && length(residual) == M * N ? Base.reshape(residual, M, N) : nothing
    wmma_nat_gemm!(C2, _nat_v(W), _nat_meta(W), _nat_blk(W), x16, N; fmt = _nat_fmt(W), residual = R)
    R === nothing && _wmma_add_residual!(C2, residual, M, N)
    Bc === B || _free_now!(Bc)
    return true
end

const NAT_D2_COLS = Ref(parse(Int, get(ENV, "LUMINAL_NAT_D2_COLS", "8")))      # columns per pass of the Float16 dot2 GEMV (1..8)
function _nat_matmul!(C, W::NativeWeight, B; residual = nothing)
    M, K = size(W)
    N = length(B) ÷ K
    if eltype(C) === Float16
        N > QUANT_GEMM_COLS && residual === nothing && _wmma_nat_gemm!(C, W, B, nothing) && return C
        C32 = similar(C, Float32)
        _nat_matmul!(C32, W, B; residual)
        copyto!(C, C32); _free_now!(C32)
        return C
    end
    N > QUANT_GEMM_COLS && _wmma_nat_gemm!(C, W, B, residual) && return C
    eltype(B) === Float16 && (B = Float32.(B))
    (N <= QUANT_GEMM_COLS && C isa ROCArray{Float32} && _nat_p(W) isa ROCArray{UInt8} && K % 256 == 0 && M % 4 == 0) ||
        return _dequant_gemm!(C, W, B, residual)
    Bc = B isa DenseArray ? B : copy(B)
    Xv = Base.reshape(reinterpret(NTuple{8, Float32}, vec(Bc)), K ÷ 8, N)
    R, RES = _residual_arg(residual, C, M, N)
    Wq = Base.reshape(reinterpret(NTuple{4, UInt32}, vec(_nat_p(W))), K ÷ 32, M)
    Y = Base.reshape(C, M, N)
    if N == 1 || get(ENV, "LUMINAL_NAT_GEMV_COLS", "1") == "0"
        @roc groupsize=256 gridsize=cld(M, 32) _nat_gemv2_kernel!(Y, Wq, _nat_meta(W), _nat_blk(W), Xv, K ÷ 32, M, N, R, RES, Val(_nat_fmt(W)))
        return C
    end
    c0 = 0
    if (_nat_fmt(W) !== :iq4 || get(ENV, "LUMINAL_NAT_DOT2_IQ4", "1") != "0") && get(ENV, "LUMINAL_NAT_DOT2", "1") != "0"     # Float16 activations + packed dot products
        X16 = get!(() -> similar(Bc, Float16, K, N), _Q4D_WORKSPACE, (:d2, typeof(Bc), K, N))
        X16 .= Base.reshape(Bc, K, N)
        Xq = Base.reshape(reinterpret(NTuple{4, UInt32}, vec(X16)), K ÷ 8, N)
        while c0 < N
            nc = min(NAT_D2_COLS[], N - c0)
            vc = nc >= 8 ? Val(8) : nc == 7 ? Val(7) : nc == 6 ? Val(6) : nc == 5 ? Val(5) : nc == 4 ? Val(4) : nc == 3 ? Val(3) : nc == 2 ? Val(2) : Val(1)
            @roc groupsize=256 gridsize=cld(M, 32) _nat_gemvD_kernel!(Y, Wq, _nat_meta(W), _nat_blk(W), Xq, K ÷ 32, M, c0, R, RES, Val(_nat_fmt(W)), vc)
            c0 += nc
        end
        Bc === B || _free_now!(Bc)
        return C
    end
    while c0 < N                                              # up to 4 columns per pass over the weights
        nc = min(4, N - c0)
        vc = nc == 4 ? Val(4) : nc == 3 ? Val(3) : nc == 2 ? Val(2) : Val(1)
        @roc groupsize=256 gridsize=cld(M, 32) _nat_gemvN_kernel!(Y, Wq, _nat_meta(W), _nat_blk(W), Xv, K ÷ 32, M, c0, R, RES, Val(_nat_fmt(W)), vc)
        c0 += nc
    end
    return C
end

# H (M, N...) Float16 = silu(Wg B) * (Wu B) for two 4-bit weights, in one kernel (the compiler folds `SwiGLUOp` of two matmuls into this).
function swiglu_gemm!(H, Wg::Q4Weight, Wu::Q4Weight, B)
    M, K = size(Wg)
    N = length(B) ÷ K
    x16, Bc = _wmma_activations(B, K, N)
    wmma_q4_swiglu!(Base.reshape(H, M, N), Wg.v, Wg.scale, Wg.mn, Wu.v, Wu.scale, Wu.mn, Wg.group, x16, N)
    Bc === B || _free_now!(Bc)
    return H
end

# 7..32 columns: the weights are streamed once by the skinny matrix-core kernel (the GEMV would
# read them N/2 times; measured crossover at the 27B's shapes is ~7 columns).
const SKINNY_MIN_COLS = 7
const SKINNY_MAX_COLS = 32
function _wmma_q4_skinny_gemm!(C, W::Q4Weight, B, residual)
    (C isa ROCArray{Float32} && W.p isa ROCArray{UInt8} && W.scale isa ROCArray{Float16} && wmma_enabled() &&
     get(ENV, "LUMINAL_WMMA_SKINNY", "1") != "0") || return false
    M, K = size(W)
    (M % 16 == 0 && K % 256 == 0 && W.group == 32) || return false
    N = length(B) ÷ K
    x16, Bc = _wmma_activations(B, K, N)
    C2 = Base.reshape(C, M, N)
    wmma_q4_skinny!(C2, W.v, W.scale, W.mn, W.group, x16, N)
    _wmma_add_residual!(C2, residual, M, N)
    Bc === B || _free_now!(Bc)
    return true
end

# The same for int8 weights: dequantized to Float16 as the tiles are staged, so no
# Float32 copy of the weight is made.
function _wmma_q8_gemm!(C, W::QuantWeight, B, residual)
    (C isa ROCArray{Float32} && W.q isa ROCArray{Int8} && W.scale isa ROCArray{Float32} && wmma_enabled()) || return false
    K, M = size(W.q)
    (M % 16 == 0 && K % 32 == 0 && W.group % 16 == 0) || return false
    N = length(B) ÷ K
    x16, Bc = _wmma_activations(B, K, N)
    C2 = Base.reshape(C, M, N)
    wmma_q8_gemm!(C2, W.q, W.scale, W.group, x16, N)
    _wmma_add_residual!(C2, residual, M, N)
    Bc === B || _free_now!(Bc)
    return true
end

# C (Out, N) = W (Out, In) * B (In, N), with any trailing dims of B/C flattened into N.
function _half_matmul!(C, W::HalfWeight, B; group::Int = DEFAULT_HALF_GROUP, residual = nothing)
    eltype(B) === Float16 && (B = Float32.(B))
    K, M = size(W.t)
    N = length(B) ÷ K
    N > QUANT_GEMM_COLS && _wmma_half_gemm!(C, W, B, residual) && return C
    (N > QUANT_GEMM_COLS || !(C isa AnyGPUArray)) && return _dequant_gemm!(C, W, B, residual)
    Bc = B isa DenseArray ? B : copy(B)
    Xv = Base.reshape(reinterpret(NTuple{8, Float32}, vec(Bc)), K ÷ 8, N)
    group <= 256 || error("Float16 GEMV: at most 256 threads per workgroup")
    _half_matmul_kernel!(KernelAbstractions.get_backend(C), group)(
        Base.reshape(C, M, N), W.v, Xv, K ÷ 8, N, Val(min(N, GEMV_MAX_COLS)),
        _residual_arg(residual, C, M, N)...; ndrange = M * group)
    return C
end

# out = W * x + r for a GEMV-stored weight (see the fused residual step in compile).
function matmul_residual!(out, op, W, x, r)
    W isa QuantWeight && return _q8_matmul!(out, W, x; residual = r,
                                             group = op isa MatMulQ8 ? op.group : DEFAULT_Q8_THREADS,
                                             cols = op isa MatMulQ8 ? op.cols : 0)
    W isa Q4Weight && return _q4_matmul!(out, W, x; residual = r,
                                          group = op isa MatMulQ4 ? op.group : DEFAULT_Q4_THREADS,
                                          cols = op isa MatMulQ4 ? op.cols : 0)
    W isa NativeWeight && return _nat_matmul!(out, W, x; residual = r)
    W isa HalfWeight && return _half_matmul!(out, W, x; residual = r,
                                              group = op isa MatMulF16 ? op.group : DEFAULT_HALF_GROUP)
    error("matmul_residual!: unsupported weight storage $(typeof(W))")
end

# Convert each Float32 weight at most once, so graphs sharing weights (e.g.
# prefill and decode) share one Float16 copy. Keyed by identity without holding
# the source array. An objectid can be reused once its array is collected (and
# the finalizer that drops the entry runs asynchronously), so each entry keeps a
# WeakRef to its source and only counts as a hit if it still points to `W`.
const _HALF_WEIGHTS = Dict{UInt, Tuple{WeakRef, Any}}()
const _HALF_WEIGHTS_LOCK = ReentrantLock()
# Reduced-precision weight formats are built on their device and moved as they
# are: the generic AbstractArray path would expand them to dense Float32.
for D in (CPUDevice, AMDDevice, CUDADevice)
    @eval Device.to_device(w::Union{HalfWeight, HalfWeightN, QuantWeight, Q4Weight, Q5Weight, IQ4Weight, Q6Weight}, ::$D) = w
end

# The storage format for a weight dtype (see compile's weight_dtype), or nothing
# for Float32; and whether a weight with input dimension K fits it
storage_type(T) = T === Int4 ? Q4Weight : T === Int8 ? QuantWeight : T === Float16 ? HalfWeight : nothing
storage_fits(S, K) = K % (S === Q4Weight ? 32 : S === QuantWeight ? 16 : 8) == 0

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
_batched_gemm!(C, A, B) = nothing
_batched_gemm!(C::ROCArray{T,3}, A::ROCArray{T,3}, B::ROCArray{T,3}, ta::Char, tb::Char) where {T<:Union{Float32,Float64}} =
    AMDGPU.rocBLAS.gemm_strided_batched!(ta, tb, one(T), A, B, zero(T), C)
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
    eltype(B) === Float16 && !(A isa Q4Weight || A isa NativeWeight) && (B = Float32.(B))   # (Float16 activations: see `dty` in compile)
    A isa HalfWeight && return _half_matmul!(C, A, B)
    A isa QuantWeight && return _q8_matmul!(C, A, B)
    A isa Q4Weight && return _q4_matmul!(C, A, B)
    A isa NativeWeight && return _nat_matmul!(C, A, B)
    A = _ensure_contiguous(A)
    B = _ensure_contiguous(B)
    # println("DEBUG matmul: C=$(size(C)) ($(typeof(C))), A=$(size(A)) ($(typeof(A))), B=$(size(B)) ($(typeof(B)))")
    
    # Left-aligned: (M, K, batch...) * (K, N, batch...) -> (M, N, batch...), the
    # batch dims broadcasting (size 1 against k, or missing on one side).
    ndims(A) == 2 && ndims(B) == 2 && return _matmul2d!(C, A, B)

    # Unbatched left operand (a weight): B's batch folds into its columns, one GEMM
    if ndims(A) == 2 && B isa DenseArray && C isa DenseArray
        M, K = size(A)
        _matmul2d!(Base.reshape(C, M, :), A, Base.reshape(B, K, :))
        return C
    end

    bC = size(C)[3:end]
    n = length(bC)
    pad(X) = Base.reshape(X, size(X, 1), size(X, 2), ntuple(k -> k + 2 <= ndims(X) ? size(X, k + 2) : 1, n)...)
    A, B = pad(A), pad(B)
    # Equal batch dims, dense: one strided-batched GEMM on GPU
    if size(A)[3:end] == size(B)[3:end] == bC && A isa DenseArray && B isa DenseArray && C isa DenseArray
        r3(X) = Base.reshape(X, size(X, 1), size(X, 2), :)
        _batched_gemm!(r3(C), r3(A), r3(B)) === nothing || return C
    end
    # General: one GEMM per batch element, size-1 batch dims broadcast
    for I in CartesianIndices(bC)
        ia = ntuple(k -> size(A, k + 2) == 1 ? 1 : I[k], n)
        ib = ntuple(k -> size(B, k + 2) == 1 ? 1 : I[k], n)
        mul!(view(C, :, :, Tuple(I)...), view(A, :, :, ia...), view(B, :, :, ib...))
    end
    return C
end

function realize_view(data, st::ShapeTracker)
    (data isa HalfWeight || data isa HalfWeightN || data isa QuantWeight || data isa Q4Weight || data isa NativeWeight) && return data  # consumed whole, as a matmul weight
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
execute_op(op::LessThan, a, b) = begin (a_a, b_a) = align_broadcast_ranks(a, b); a_a .< b_a end
execute_op(op::FusedMulAdd, a, b, c) = (a .* b) .+ c
execute_op(op::FusedAddReLU, a, b) = Base.max.(a .+ b, 0)
execute_op(op::Log2, a) = log2.(a)
execute_op(op::Exp2, a) = exp2.(a)
execute_op(op::Sin, a) = sin.(a)
execute_op(op::Cos, a) = cos.(a)
execute_op(op::Sqrt, a) = sqrt.(a)
execute_op(op::Recip, a) = 1.0f0 ./ a
execute_op(op::Div, a, b) = begin (a_a, b_a) = align_broadcast_ranks(a, b); a_a ./ b_a end
execute_op(op::Exp, a) = exp.(a)
execute_op(op::Floor, a) = floor.(a)
execute_op(op::Cast, a) = CastTo(op.dtype).(a)
execute_op(op::TruncCast, a) = (_check_trunc(op.dtype, a); _UTrunc{op.dtype}().(a))
function execute_op(op::Union{TruncDiv, TruncRem}, a, b)
    _anyf(iszero, b) && throw(DivideError())
    (a_a, b_a) = align_broadcast_ranks(a, b)
    return (op isa TruncDiv ? _trunc_div : _trunc_rem).(a_a, b_a)
end
execute_op(op::Ceil, a) = ceil.(a)
execute_op(op::Round, a) = round.(a)
execute_op(op::Trunc, a) = trunc.(a)
function execute_op(op::Select, c, a, b)
    n = max(ndims(c), ndims(a), ndims(b))
    pad(x) = Base.reshape(x, size(x)..., ntuple(_ -> 1, n - ndims(x))...)   # left-aligned
    return ifelse.(pad(c) .!= 0, pad(a), pad(b))
end
execute_op(op::Reshape, a) = Base.reshape(a, op.shape...)
execute_op(op::Permute, a) = Base.permutedims(a, op.dims)
execute_op(op::Contiguous, a) = copy(a)
execute_op(op::MatMul, a, b) = batch_matmul(a, b)
execute_op(op::SumReduce, a) = dropdims(Base.sum(a, dims=op.dim), dims=op.dim)
execute_op(op::MaxReduce, a) = dropdims(Base.maximum(a, dims=op.dim), dims=op.dim)
execute_op(op::Scan, a) = execute_op!(similar(a), op, a)
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
execute_op!(out, op::LessThan, a, b) = broadcast!(<, out, _align_broadcast(a, out), _align_broadcast(b, out))
execute_op!(out, op::FusedMulAdd, a, b, c) = broadcast!((x,y,z)->x*y+z, out, _align_broadcast(a, out), _align_broadcast(b, out), _align_broadcast(c, out))
execute_op!(out, op::FusedAddReLU, a, b) = broadcast!((x,y)->max(x+y, 0), out, _align_broadcast(a, out), _align_broadcast(b, out))
execute_op!(out, op::Log2, a) = broadcast!(log2, out, a)
execute_op!(out, op::Exp2, a) = broadcast!(exp2, out, a)
execute_op!(out, op::Sin, a) = broadcast!(sin, out, a)
execute_op!(out, op::Cos, a) = broadcast!(cos, out, a)
execute_op!(out, op::Sqrt, a) = broadcast!(sqrt, out, a)
execute_op!(out, op::Div, a, b) = broadcast!(/, out, _align_broadcast(a, out), _align_broadcast(b, out))
execute_op!(out, op::Exp, a) = broadcast!(exp, out, a)
execute_op!(out, op::Floor, a) = broadcast!(floor, out, a)
execute_op!(out, op::Cast, a) = broadcast!(CastTo(op.dtype), out, a)
function execute_op!(out, op::TruncCast, a)
    _check_trunc(op.dtype, a)
    return broadcast!(_UTrunc{op.dtype}(), out, a)
end
function execute_op!(out, op::Union{TruncDiv, TruncRem}, a, b)
    _anyf(iszero, b) && throw(DivideError())
    f = op isa TruncDiv ? _trunc_div : _trunc_rem
    return broadcast!(f, out, _align_broadcast(a, out), _align_broadcast(b, out))
end
execute_op!(out, op::Ceil, a) = broadcast!(ceil, out, a)
execute_op!(out, op::Round, a) = broadcast!(round, out, a)
execute_op!(out, op::Trunc, a) = broadcast!(trunc, out, a)
execute_op!(out, op::Select, c, a, b) =
    broadcast!((x, y, z) -> ifelse(x != 0, y, z), out, _align_broadcast(c, out), _align_broadcast(a, out), _align_broadcast(b, out))
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
execute_op!(out, op::MatMulQ4, a, b) =   # `a` is a Q4Weight once compiled
    a isa Q4Weight ? _q4_matmul!(out, a, b; group=op.group, cols=op.cols) : batch_matmul!(out, a, b)
execute_op!(out, op::MatMulQ8, a, b) =   # `a` is a QuantWeight once compiled
    a isa QuantWeight ? _q8_matmul!(out, a, b; group=op.group, cols=op.cols) : batch_matmul!(out, a, b)
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
        @inbounds acc = op(acc, convert(typeof(init), a[base + pre * (k - 1)]))
        k += G
    end
    s = @localmem typeof(init) (256,)
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
        # accumulated in out's type, 16-bit floats in Float32
        acc_init = eltype(out) <: Union{Float16, Core.BFloat16} ? Float32(init) : init
        _reduce_dim_kernel!(KernelAbstractions.get_backend(out), G)(
            out, a, pre, r, op, acc_init; ndrange = length(out) * G)
    else
        rsz = ntuple(d -> d == dim ? 1 : size(a, d), ndims(a))
        # in out's type throughout (sum! widens small integers, then fails to
        # narrow an overflowed Int8 sum instead of wrapping)
        Base.mapreducedim!(identity, op, fill!(Base.reshape(out, rsz), init), a)
    end
    return out
end

execute_op!(out, op::SumReduce, a) = _reduce_dim!(out, a, op.dim, +, zero(eltype(out)))
execute_op!(out, op::MaxReduce, a) = _reduce_dim!(out, a, op.dim, max, _max_neutral(eltype(out)))
_max_neutral(T) = T <: AbstractFloat ? T(-Inf) : typemin(T)

# Scans: inclusive left fold along `dim`, one thread per (before, after) coordinate pair on the GPU (the axis is read
# in order, so the result is the sequential fold's: bit-identical to the CPU path). 16-bit floats accumulate in Float32.
@kernel function _scan_kernel!(out, @Const(a), r, pre, op, init)
    g = @index(Global, Linear)
    i = (g - 1) % pre + 1
    j = (g - 1) ÷ pre + 1
    base = i + pre * r * (j - 1)
    acc = init
    for k in 1:r
        @inbounds acc = op(acc, convert(typeof(init), a[base + pre * (k - 1)]))
        @inbounds out[base + pre * (k - 1)] = convert(eltype(out), acc)
    end
end

_scan_fn(kind::Symbol) = kind === :sum ? (+) : kind === :prod ? (*) : kind === :max ? max :
    error("unknown scan kind $kind")
_scan_init(kind::Symbol, T) = kind === :sum ? zero(T) : kind === :prod ? one(T) : _max_neutral(T)

function execute_op!(out, op::Scan, a)
    f = _scan_fn(op.kind)
    T = eltype(out)
    acc_T = T <: Union{Float16, Core.BFloat16} ? Float32 : T
    init = convert(acc_T, _scan_init(op.kind, T))
    pre = prod(size(a)[1:op.dim - 1]; init=1)
    r = size(a, op.dim)
    post = prod(size(a)[op.dim + 1:end]; init=1)
    if out isa AnyGPUArray
        _scan_kernel!(KernelAbstractions.get_backend(out), 64)(out, a, r, pre, f, init; ndrange = pre * post)
    else
        A = Base.reshape(a, pre, r, post); O = Base.reshape(out, pre, r, post)
        @inbounds for j in 1:post, i in 1:pre
            acc = init
            for k in 1:r
                acc = f(acc, convert(acc_T, A[i, k, j]))
                O[i, k, j] = convert(T, acc)
            end
        end
    end
    return out
end

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

# --- Casts --------------------------------------------------------------------
# The scalar conversion Cast applies: integers wrap to a narrower integer, every
# other conversion is T(x) (float rounding, int / Bool -> float).
struct CastTo{T} end
CastTo(T::Type) = CastTo{T}()
(::CastTo{T})(x) where T = T(x)
(::CastTo{T})(x::Integer) where T <: Integer = x % T
(::CastTo{T})(x::Bool) where T <: Integer = T(x)

# x truncates into T's range [typemin, -typemin) (bounds exact in Float64)
_trunc_in_range(::Type{T}, x) where T = (y = trunc(Float64(x)); Float64(typemin(T)) <= y < -Float64(typemin(T)))
# refused loudly, as upstream: NaN, ±Inf and values truncating outside T
# (functors typed by T, not closures over it: a Type field can't reach a GPU kernel)
struct _TruncBad{T} end
(::_TruncBad{T})(x) where T = !(isfinite(x) && _trunc_in_range(T, x))
_anyf(f, a) = mapreduce(f, |, a; init=false)     # (any(f, ::ROCArray) iterates on the host)
_check_trunc(T, a) = _anyf(_TruncBad{T}(), a) && throw(DomainError("trunc_cast to $T: NaN, Inf or a value out of range"))
struct _UTrunc{T} end
(::_UTrunc{T})(x) where T = unsafe_trunc(T, x)
(::_UTrunc{T})(x::Union{Float16, Core.BFloat16}) where T = unsafe_trunc(T, Float32(x))
_trunc_div(a, b) = b == -one(b) ? -a : div(a, b)     # typemin ÷ -1 wraps instead of trapping
_trunc_rem(a, b) = b == -one(b) ? zero(a) : rem(a, b)

# --- Coordinate gather / scatter and iota -------------------------------------
# One coordinate array per data axis, each with the output's (gather) or src's
# (scatter) shape, holding 0-based indices as Float32 values.

# Linear index of element i's coordinates into an array of size `dims`, and
# whether they are in range
_coord(c::Integer) = Int(c)
_coord(c) = unsafe_trunc(Int, c)

@inline function _coord_index(coords, i, dims)
    lin = 0; stride = 1; ok = true
    for k in 1:length(coords)
        c = _coord(@inbounds coords[k][i])
        ok &= (0 <= c) & (c < dims[k])
        lin += c * stride
        stride *= dims[k]
    end
    return ok, lin + 1
end

@kernel function _gather_nd_kernel!(out, @Const(data), coords, dims)
    i = @index(Global, Linear)
    ok, j = _coord_index(coords, i, dims)
    @inbounds out[i] = ok ? data[j] : zero(eltype(out))
end

@kernel function _mapping_gather_kernel!(out, @Const(x), @Const(indices))
    b, c = @index(Global, NTuple)
    @inbounds out[b, c] = x[b, indices[c]]
end

@kernel function _scatter_nd_kernel!(out, @Const(src), coords, dims, ::Val{ADD}) where ADD
    i = @index(Global, Linear)
    ok, j = _coord_index(coords, i, dims)
    if ok
        if ADD
            KernelAbstractions.@atomic out[j] += src[i]
        else
            @inbounds out[j] = src[i]
        end
    end
end

_dense(x) = x isa DenseArray ? x : copy(x)
# `x` as a dense array of `like`'s size and device (coordinates may arrive as
# views, or with size-1 dims left by an expand)
function _dense_like(x, like)
    x isa DenseArray && size(x) == size(like) && return x
    y = similar(like)
    broadcast!(identity, y, _align_broadcast(x, y))
    return y
end

function execute_op!(out, op::GatherND, data, coords...)
    d = _dense(data)
    cs = map(c -> _dense_like(c, out), coords)
    length(out) == 0 && return out
    _gather_nd_kernel!(KernelAbstractions.get_backend(out), 256)(out, d, cs, size(d); ndrange=length(out))
    return out
end

function execute_op!(out, op::MappingGather, x, scores, indices)
    d = _dense(x)
    i = _dense(indices)
    length(out) == 0 && return out
    _mapping_gather_kernel!(KernelAbstractions.get_backend(out), 256)(out, d, i;
                                                                      ndrange=size(out))
    return out
end

# --- LutKOp: K-input LUT layer, tree unrolled in registers -----------------
#
# The truth table for one neuron is a tuple of 2^K entries; one level of the
# interpolation tree halves it: c2[j] = c[j]*(1-v) + c[j + m]*v. Everything is
# Val-unrolled (K <= 6 at the call sites), so the recursion is fully inlined
# and the tree lives in registers.

# Generated straight-line unrolling (no runtime type construction — the GPU
# cannot compile Val/apply_type calls inside kernels). Level k output entries
# are named c{k}_{j}; the backward walks the levels in reverse.
# Both are emitted at load time as concrete, fully unrolled methods for
# K = 1..6 (plain functions — @generated bodies run into Julia 1.12's
# generated-body purity check for this pattern).
function _lut_unroll_body(E::Int, K::Int; backward::Bool)
    stmts = Any[]
    cur = Any[:(w[$j]) for j in 1:E]
    width = E
    for k in 1:K
        m = width ÷ 2
        prev = cur
        cur = Any[Symbol(:c, k, :_, j) for j in 1:m]
        for j in 1:m
            push!(stmts, :($(cur[j]) = $(prev[j]) * (1 - vs[$k]) + $(prev[j + m]) * vs[$k]))
        end
        width = m
    end
    push!(stmts, :(y = $(cur[1])))
    backward || return quote $(stmts...); y end
    gcur = Any[:gz]
    for k in K:-1:1
        m = 2^(K - k)
        prev = k == 1 ? Any[:(w[$j]) for j in 1:(2m)] :
               Any[Symbol(:c, k - 1, :_, j) for j in 1:(2m)]
        dv = :(0.0f0)
        for j in 1:m
            dv = :($dv + $(gcur[j]) * ($(prev[j + m]) - $(prev[j])))
        end
        push!(stmts, :($(Symbol(:dv, k)) = $dv))
        newg = Any[Symbol(:g, k, :_, j) for j in 1:(2m)]
        for j in 1:(2m)
            src = j <= m ? gcur[j] : gcur[j - m]
            f = j <= m ? :(1 - vs[$k]) : :(vs[$k])
            push!(stmts, :($(newg[j]) = $src * $f))
        end
        gcur = newg
    end
    for j in 1:E
        push!(stmts, :($(Symbol(:dw_, j)) = $(gcur[j])))
    end
    dvt = Expr(:tuple, Any[Symbol(:dv, k) for k in 1:K]...)
    dwt = Expr(:tuple, Any[Symbol(:dw_, j) for j in 1:E]...)
    return quote $(stmts...); y, $dvt, $dwt end
end

for _K in 1:6
    _E = 2^_K
    @eval @inline function _lut_fwd_gen(w::NTuple{$_E,T}, vs::NTuple{$_K,T}) where {T}
        $(_lut_unroll_body(_E, _K; backward=false))
    end
    @eval @inline function _lut_fwbw_gen(w::NTuple{$_E,T}, vs::NTuple{$_K,T}, gz::T) where {T}
        $(_lut_unroll_body(_E, _K; backward=true))
    end
end
@kernel function _lutk_fwd_kernel!(out, @Const(pins), @Const(W), ::Val{K}, ::Val{E}) where {K, E}
    b, n = @index(Global, NTuple)
    if b <= size(pins, 1) && n <= size(W, 1)
        out_dim = size(W, 1)
        w = ntuple(e -> @inbounds(W[n, e]), Val(E))
        vs = ntuple(k -> @inbounds(pins[b, (k - 1) * out_dim + n]), Val(K))
        @inbounds out[b, n] = _lut_fwd_gen(w, vs)
    end
end

@kernel function _lutk_fwd_idx_kernel!(out, @Const(x), @Const(W), @Const(idx), ::Val{K}, ::Val{E}) where {K, E}
    b, n = @index(Global, NTuple)
    if b <= size(x, 1) && n <= size(W, 1)
        out_dim = size(W, 1)
        w = ntuple(e -> @inbounds(W[n, e]), Val(E))
        vs = ntuple(k -> @inbounds(x[b, idx[(k - 1) * out_dim + n]]), Val(K))
        @inbounds out[b, n] = _lut_fwd_gen(w, vs)
    end
end

@kernel function _lutk_bwd_pins_kernel!(dpins, @Const(pins), @Const(W), @Const(dy), ::Val{K}, ::Val{E}) where {K, E}
    b, n = @index(Global, NTuple)
    if b <= size(pins, 1) && n <= size(W, 1)
        out_dim = size(W, 1)
        w = ntuple(e -> @inbounds(W[n, e]), Val(E))
        vs = ntuple(k -> @inbounds(pins[b, (k - 1) * out_dim + n]), Val(K))
        _, dvs, _ = _lut_fwbw_gen(w, vs, dy[b, n])
        ntuple(k -> @inbounds(dpins[b, (k - 1) * out_dim + n] = dvs[k]), Val(K))
    end
end

@kernel function _lutk_bwd_pins_idx_kernel!(dpins, @Const(x), @Const(W), @Const(idx), @Const(dy), ::Val{K}, ::Val{E}) where {K, E}
    b, n = @index(Global, NTuple)
    if b <= size(x, 1) && n <= size(W, 1)
        out_dim = size(W, 1)
        w = ntuple(e -> @inbounds(W[n, e]), Val(E))
        vs = ntuple(k -> @inbounds(x[b, idx[(k - 1) * out_dim + n]]), Val(K))
        _, dvs, _ = _lut_fwbw_gen(w, vs, dy[b, n])
        ntuple(k -> @inbounds(dpins[b, (k - 1) * out_dim + n] = dvs[k]), Val(K))
    end
end

@kernel function _lutk_dw_kernel!(dW, @Const(pins), @Const(W), @Const(dy), ::Val{K}, ::Val{E}) where {K, E}
    b, n = @index(Global, NTuple)
    if b <= size(pins, 1) && n <= size(W, 1)
        out_dim = size(W, 1)
        w = ntuple(e -> @inbounds(W[n, e]), Val(E))
        vs = ntuple(k -> @inbounds(pins[b, (k - 1) * out_dim + n]), Val(K))
        _, _, dwc = _lut_fwbw_gen(w, vs, dy[b, n])
        ntuple(e -> KernelAbstractions.@atomic(dW[n, e] += dwc[e]), Val(E))
    end
end

@kernel function _lutk_dw_idx_kernel!(dW, @Const(x), @Const(W), @Const(idx), @Const(dy), ::Val{K}, ::Val{E}) where {K, E}
    b, n = @index(Global, NTuple)
    if b <= size(x, 1) && n <= size(W, 1)
        out_dim = size(W, 1)
        w = ntuple(e -> @inbounds(W[n, e]), Val(E))
        vs = ntuple(k -> @inbounds(x[b, idx[(k - 1) * out_dim + n]]), Val(K))
        _, _, dwc = _lut_fwbw_gen(w, vs, dy[b, n])
        ntuple(e -> KernelAbstractions.@atomic(dW[n, e] += dwc[e]), Val(E))
    end
end

function execute_op!(out, op::LutKOp, pins, W)
    length(out) == 0 && return out
    _lutk_fwd_kernel!(KernelAbstractions.get_backend(out), 256)(out, _dense(pins),
        _dense(W), Val(op.K), Val(2^op.K); ndrange=size(out))
    return out
end

function execute_op!(out, op::LutKOp, x, W, idx)
    length(out) == 0 && return out
    _lutk_fwd_idx_kernel!(KernelAbstractions.get_backend(out), 256)(out, _dense(x),
        _dense(W), _dense(idx), Val(op.K), Val(2^op.K); ndrange=size(out))
    return out
end

function execute_op!(out, op::LutKBwdPinsOp, pins, W, dy)
    length(out) == 0 && return out
    _lutk_bwd_pins_kernel!(KernelAbstractions.get_backend(out), 256)(out, _dense(pins),
        _dense(W), _dense(dy), Val(op.K), Val(2^op.K); ndrange=size(out))
    return out
end

function execute_op!(out, op::LutKBwdPinsOp, x, W, idx, dy)
    length(out) == 0 && return out
    _lutk_bwd_pins_idx_kernel!(KernelAbstractions.get_backend(out), 256)(out, _dense(x),
        _dense(W), _dense(idx), _dense(dy), Val(op.K), Val(2^op.K); ndrange=size(out))
    return out
end

function execute_op!(out, op::LutKDWOp, pins, W, dy)
    fill!(out, zero(eltype(out)))
    size(out) == (0,) || size(out, 2) == 0 && return out
    _lutk_dw_kernel!(KernelAbstractions.get_backend(out), 256)(out, _dense(pins),
        _dense(W), _dense(dy), Val(op.K), Val(2^op.K); ndrange=(size(pins, 1), size(W, 1)))
    return out
end

function execute_op!(out, op::LutKDWOp, x, W, idx, dy)
    fill!(out, zero(eltype(out)))
    size(out) == (0,) || size(out, 2) == 0 && return out
    _lutk_dw_idx_kernel!(KernelAbstractions.get_backend(out), 256)(out, _dense(x),
        _dense(W), _dense(idx), _dense(dy), Val(op.K), Val(2^op.K); ndrange=(size(x, 1), size(W, 1)))
    return out
end

# --- SparseMapGather: compact top-K router ---------------------------------

@kernel function _spmap_fwd_kernel!(out, @Const(x), @Const(S), @Const(cand), ::Val{K}) where {K}
    b, c = @index(Global, NTuple)
    if b <= size(x, 1) && c <= size(S, 1)
        bestv = S[c, 1]
        bestk = 1
        for kk in 2:K
            v = S[c, kk]
            if v > bestv
                bestv = v
                bestk = kk
            end
        end
        @inbounds out[b, c] = x[b, cand[c, bestk]]
    end
end

@kernel function _spmap_bwd_s_kernel!(dS, @Const(x), @Const(cand), @Const(dy), ::Val{K}) where {K}
    c, k = @index(Global, NTuple)
    if c <= size(cand, 1)
        ci = cand[c, k]
        acc = 0.0f0
        for b in 1:size(dy, 1)
            @inbounds acc += (2 * x[b, ci] - 1) * dy[b, c]
        end
        @inbounds dS[k, c] = acc
    end
end

@kernel function _spmap_bwd_x_kernel!(dx, @Const(P), @Const(cand), @Const(dy))
    b, c = @index(Global, NTuple)
    if b <= size(dy, 1) && c <= size(cand, 1)
        g = dy[b, c]
        for kk in 1:size(cand, 2)
            @inbounds KernelAbstractions.@atomic dx[b, cand[c, kk]] += g * P[c, kk]
        end
    end
end

function execute_op!(out, op::SparseMapGather, x, S, cand)
    length(out) == 0 && return out
    _spmap_fwd_kernel!(KernelAbstractions.get_backend(out), 256)(out, _dense(x),
        _dense(S), _dense(cand), Val(op.K); ndrange=size(out))
    return out
end

function execute_op!(out, op::SparseMapBwdS, x, cand, dy)
    length(out) == 0 && return out
    _spmap_bwd_s_kernel!(KernelAbstractions.get_backend(out), 256)(out, _dense(x),
        _dense(cand), _dense(dy), Val(op.K); ndrange=size(out))
    return out
end

function execute_op!(out, op::SparseMapBwdX, P, cand, dy)
    fill!(out, zero(eltype(out)))
    _spmap_bwd_x_kernel!(KernelAbstractions.get_backend(out), 256)(out,
        _dense(P), _dense(cand), _dense(dy); ndrange=(size(dy, 1), size(cand, 1)))
    return out
end

# Copy init's contents into the output buffer with a kernel launch, not
# `copyto!`: device-to-device memcpys run on the null stream and abort HIP
# graph capture (hipErrorStreamCaptureUnsupported), while KA kernels capture.
@kernel function _copy_kernel!(dst, @Const(src))
    i = @index(Global, Linear)
    @inbounds dst[i] = src[i]
end

function execute_op!(out, op::ScatterND, init, src, coords...)
    out === init ||
        _copy_kernel!(KernelAbstractions.get_backend(out), 256)(out,
                                                                _dense_like(init, out);
                                                                ndrange=length(out))
    s = _dense(src)
    cs = map(c -> _dense_like(c, s), coords)
    length(s) == 0 && return out
    _scatter_nd_kernel!(KernelAbstractions.get_backend(out), 256)(out, s, cs, size(out), Val(op.mode === :add);
                                                                 ndrange=length(s))
    return out
end

_iota_values(op::Iota, dims) = op.dtype[op.f((Tuple(I) .- 1)...) for I in CartesianIndices(Tuple(dims))]
execute_op!(out, op::Iota) = copyto!(out, _iota_values(op, size(out)))

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
    else
        error("Function op with name $(op.name) not implemented.")
    end
    return out
end

# An input value as its node's dtype (a converting copy only when it differs)
_as_dtype(x::AbstractArray, T) = eltype(x) === T ? x : T.(x)
_as_dtype(x::Number, T) = convert(T, x)
_as_dtype(x, T) = x                                   # weight formats (HalfWeight, ...)

function execute(graph::Graph, output_ids::Vector{Int}, initial_inputs::Dict, device::AbstractDevice=get_device())
    results = to_device(initial_inputs, device)
    for (id, v) in results
        results[id] = _as_dtype(v, graph.dtypes[id])
    end
    
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
                 current_result = zero_tensor(device, graph.dtypes[node_id], dims...)
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
