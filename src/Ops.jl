# Defines the core operations in the computation graph.

# Abstract parent type for all operations
abstract type Op end

# Unary Ops (A -> A)
struct Log2 <: Op end
struct Exp2 <: Op end
struct Sin <: Op end
struct Sqrt <: Op end
struct Cos <: Op end
struct Recip <: Op end
struct Contiguous <: Op end
struct ReLU <: Op end

# Binary Ops (A x A -> A)
struct Add <: Op end
struct Mul <: Op end
struct Mod <: Op end
struct Max <: Op end
struct FusedMulAdd <: Op end
struct FusedAddReLU <: Op end
struct LessThan <: Op end

# Rounding (A -> A); values stay Float32. Round is half to even.
struct Floor <: Op end
struct Ceil <: Op end
struct Round <: Op end
struct Trunc <: Op end

# Exact division (A x A -> A) and the natural exponential (A -> A)
struct Div <: Op end
struct Exp <: Op end

# Coordinate-form gather and scatter (upstream's logical gather / scatter): one
# coordinate tensor per axis of the data, all of one shape, holding 0-based
# indices as Float32 values (exact up to 2^24).
#   GatherND:  out[c] = data[coord_1[c], .., coord_r[c]], 0 where out of range
#   ScatterND: out = init, then out[coord_1[c], .., coord_r[c]] = src[c] (:replace)
#              or += src[c] (:add, atomic); out-of-range writes are dropped.
#              With :replace, repeated coordinates leave one of the writes
#              (which one is unspecified on the GPU).
struct GatherND <: Op end
struct ScatterND <: Op
    mode::Symbol
end

# Column gather with a straight-through surrogate gradient, for models whose
# input connections are selected discretely (typically argmax of a score
# matrix) while the scores themselves must stay trainable. Inputs:
#   x (B, In), scores W (In, C), indices (C,) holding 1-based Int32/Int64
#   column numbers of x.
#   forward:  out[b, c] = x[b, indices[c]]
#   backward: dx = dy @ softmax(W ./ tau, dim 1)',   dW = (2x - 1)' dy
# i.e. the forward is a discrete gather while the gradient routes through a
# softmax "router" over W at temperature `tau`, as if the output had been
# x times that router (the straight-through rule of DiffLUT-Net's
# `_LearnableMappingFunction`). The scores input exists only for the backward
# pass; indices get no gradient.
struct MappingGather <: Op
    tau::Float32
end

# Fused K-input LUT layer (DiffLUT-Net): the multilinear extension of a
# per-neuron truth table over K binary-ish inputs, in one kernel. Two input
# forms:
#   gathered  — inputs (pins (B, K*out), W (out, 2^K)); pins[b, (k-1)*out + n]
#               is input k of neuron n (as produced by MappingGather).
#   indexed   — inputs (x (B, In), W (out, 2^K), indices (K*out,) 1-based
#               Int32/Int64, pin-major); the kernel reads x directly.
# Forward: out[b, n] = multilinear extension of W[n, :] at the K pins, i.e.
# Σ_e W[n, e] * Π_k (pins or 1-pins by bit k of e-1, v1 the MSB) — the same
# entry order as the Verilog export of DiffLUT-Net.
# Backward ops (created by vjp_rules, no rules of their own):
#   LutKBwdPinsOp: pin gradients (B, K*out) via the interpolation tree in
#     reverse, dv_k = g * (c_hi - c_lo) summed per level.
#   LutKDWOp: truth-table gradient (out, 2^K) = Σ_b dy .* path products,
#     accumulated with atomics (the output buffer is zeroed each run).
struct LutKOp <: Op
    K::Int
end
struct LutKBwdPinsOp <: Op
    K::Int
end
struct LutKDWOp <: Op
    K::Int
end

# Compact top-K sparse router for DiffLUT-Net's learnable mapping. After a
# dense warmup + prune, each pin keeps only its top-K' candidate inputs; the
# score matrix shrinks from (in_dim × pins) to (K' × pins).
#   SparseMapGather:  out[b, c] = x[b, cand[c, argmax_k S[c, k]]]
#     (S (C=pins, K'), cand (C, K') 1-based Int64; forward argmax over the
#      pin's own candidate scores)
#   Backward ops (created by vjp_rules, no rules of their own):
#     SparseMapBwdSOp: dS[c, k] = Σ_b (2x[b,cand[c,k]]−1)·dy[b,c]
#       (straight-through to the kept scores, no softmax weight — same as the
#        dense dW = (2x−1)'·dy restricted to kept candidates)
#     SparseMapBwdXOp: dx[b, i] += Σ_(k: cand[c,k]=i) dy[b,c]·p[c,k]
#       with p[c,k] = softmax_k(S[c,k]/tau) — the softmax-weighted scatter
#       that dense dx = dy·p' computes over all candidates.
struct SparseMapGather <: Op
    K::Int
    tau::Float32
end
struct SparseMapBwdS <: Op
    K::Int
end
struct SparseMapBwdX <: Op
    K::Int
    tau::Float32
end

# A source: out[c_1, .., c_k] = f(c_1, .., c_k) over the 0-based output
# coordinates (upstream's iota, with any Julia function as the expression).
# (Untyped field: one op type for every function, as the e-graph bridge needs;
# evaluated on the host, so nothing is lost.)
struct Iota <: Op
    f::Any
    dtype::DataType
end
Iota(f) = Iota(f, Float32)

# Dtype conversions (upstream's logical cast / trunc_cast). Cast is lossless by
# policy except for width: float -> float rounds, int -> int wraps, int / Bool ->
# float converts; float -> int is refused (use TruncCast, which truncates toward
# zero and refuses NaN, Inf and out-of-range values at run time).
struct Cast <: Op
    dtype::DataType
end
struct TruncCast <: Op
    dtype::DataType
end

# Integer division truncated toward zero and its remainder (sign of the
# dividend); a zero divisor is refused at run time.
struct TruncDiv <: Op end
struct TruncRem <: Op end

# Ternary (C x A x B -> A): select(c, a, b) = c != 0 ? a : b, broadcasting
struct Select <: Op end

# Loop Ops
struct LoopIn <: Op
    name::String
    range::DimType
    stride::DimType
end

struct LoopOut <: Op
    name::String
    range::DimType
    stride::DimType
end

# TensorCore MatMul
struct TCMatmul <: Op
    a_k_stride::DimType
    b_k_stride::DimType
    a_row_size::DimType
    b_row_size::DimType
    c_row_size::DimType
    k_loops::DimType
end

# Reduce Ops (A -> B)
struct SumReduce <: Op
    dim::Int
end

struct MaxReduce <: Op
    dim::Int
end

# Movement Ops
struct Permute <: Op
    dims::Vector{Int}
end

struct Expand <: Op
    dim::Int
    size::DimType # Corresponds to Expression in Rust
end

struct Reshape <: Op
    shape::Vector{DimType}
end

struct Slice <: Op
    ranges::Vector{Tuple{DimType, DimType}}
end

struct Pad <: Op
    padding::Vector{Tuple{DimType, DimType}}
end

struct Unfold <: Op
    kernel_shape::Vector{DimType}
    stride_shape::Vector{DimType}
    dilation_shape::Vector{DimType}
end

# Special Ops
struct MatMul <: Op end
# Same product as MatMul, with the (weight) left operand stored as Float16 on GPU.
# A precision choice the rewrite layer can select per matmul; compute stays Float32.
# `impl` picks the kernel:
#   :gemv    -- Float16 weights, Float32 activations and accumulation; `group` is
#               the threads per output row (64, 128, 256; 256 is the default: the
#               measured search picked it for every TinyLlama projection on a
#               Radeon 8060S). Best for one or a few columns (decode).
#   :gemm_ex -- rocBLAS mixed-precision GEMM: activations are also rounded to
#               Float16 (Float32 accumulation and output). ~2x Float32 GEMM for
#               many columns (prefill), with ~2e-4 relative error per matmul.
struct MatMulF16 <: Op
    group::Int
    impl::Symbol
end
const DEFAULT_HALF_GROUP = 256
MatMulF16() = MatMulF16(DEFAULT_HALF_GROUP, :gemv)
MatMulF16(group::Int) = MatMulF16(group, :gemv)

# Same product as MatMul with the (weight) left operand stored as int8 with one
# Float32 scale per output row (weight-only quantization; activations and
# accumulation stay Float32). `group` is the GEMV's threads per workgroup
# (0: chosen per call from the shape, see `_q8_threads`).
# `cols`: columns (batched-decode sequences) accumulated per pass (0: Q8_MAX_COLS).
# Both are kernel choices with no effect on the result beyond rounding; the
# e-graph search offers several (INT8_VARIANTS) and a measured search times them.
struct MatMulQ8 <: Op
    group::Int
    cols::Int
end
MatMulQ8(group::Integer) = MatMulQ8(group, 0)
MatMulQ8() = MatMulQ8(DEFAULT_Q8_THREADS, 0)

# MatMul with a 4-bit weight (Q4Weight: group-wise, weight-only). `group`: threads
# per workgroup, `cols`: columns per pass (0: the kernel's per-shape defaults,
# `_q4_threads` / `Q4_MAX_COLS`). Offered by the e-graph search with precision=:int4.
struct MatMulQ4 <: Op
    group::Int
    cols::Int
end
MatMulQ4() = MatMulQ4(0, 0)

# Rotary position embedding ("rotate half") in one kernel. Inputs: x (D, S, H, B),
# cos and sin tables (D/2, S). For i <= D/2:
#   out[i] = x[i] * cos[i] - x[i + D/2] * sin[i],  out[i + D/2] = x[i + D/2] * cos[i] + x[i] * sin[i]
# Equivalent to slicing the halves, four multiplies, and a concat (4 kernels + 2
# copies on the decode path) -- one launch instead.
struct RotaryEmbed <: Op end

# RMS normalization over dim 1 with a weight, in one kernel:
#   out = x / sqrt(mean(x .^ 2, dims=1) + epsilon) .* w,   x (H, ...), w (H,)
# (the composite form is a reduction plus ~3 elementwise kernels).
# Softmax over dim 1, in one kernel (max, sum of exponentials and the normalized
# write, one workgroup per column). Attention with the scores laid out as
# (keys, queries, ...) normalizes along this contiguous dimension.
struct SoftmaxOp <: Op end

struct RMSNormOp <: Op
    epsilon::Float32
end

# x * rsqrt(sum(x^2) + epsilon) over dim 1 (L2 normalization as FLA's l2norm): one kernel instead of
# a square, a reduction, a reciprocal, an expand and a product (Gated DeltaNet's q and k).
struct L2NormOp <: Op
    epsilon::Float32
end

# RMSNorm over dim 1 with a weight, times silu of a gate: rmsnorm(x, w) * (z * sigmoid(z)) (Gated DeltaNet's output norm).
# Inputs x, w, z (z the shape of x).
struct GatedRMSNormOp <: Op
    epsilon::Float32
end

# silu(gate) * up = gate * sigmoid(gate) * up (the SwiGLU of the MLP), inputs (gate, up) of one shape. A node of its own, so that the
# compiler can fold it into the two GEMMs that produce its inputs.
struct SwiGLUOp <: Op end

# Linear-attention building blocks (Gated DeltaNet: Qwen3.5 / Qwen3.6). Both
# carry state across calls in a buffer they update in place, like
# DecodeAttention's cache write; `fresh` starts from zeros (prefill) instead of
# the buffer's contents (decode). A last input `lens` (B,) says how many of each
# sequence's S tokens to consume: the state ends after token lens[b] (padding
# after a shorter prompt, or a batched-decode sequence that holds its position
# with lens 0, leaves it alone); outputs past lens[b] are zero.
#
# CausalConv: depthwise causal convolution then SiLU. Inputs x (C, S, B),
#   state (C, K, B), w (C, 1, K), lens; out[c, t] = silu(sum_j w[c, j] * u[c, t - K + j])
#   where u is the state's inputs followed by x. The state becomes the last K
#   inputs.
# DeltaRule: the gated delta rule. Inputs q, k (dk, nk, S, B) (L2-normalized),
#   v (dv, nv, S, B), g, beta (nv, S, B), state (dv, dk, nv, B) holding S^T per
#   head, lens; value head h reads key head (h - 1) ÷ (nv ÷ nk) + 1. Per token:
#   S <- S exp(g); S <- S + k ((v - S^T k) beta)^T; out = S^T (q * scale).
#   Output (dv, nv, S, B); the state is left at the final S^T.
# With `snap_t` = n > 0 the op takes one more input, a buffer shaped like the state but with n times the batch
# dim, and writes into batch slot (t - 1) * B + b the state after the first t tokens, t = 1..n (the final state
# for those t the sequence is too short for): a speculative decoding step that feeds several tokens keeps the
# states a rejected draft must roll back to.
struct CausalConv <: Op
    fresh::Bool
    snap_t::Int
end
CausalConv(fresh::Bool) = CausalConv(fresh, 0)
struct DeltaRule <: Op
    scale::Float32
    fresh::Bool
    snap_t::Int
end
DeltaRule(scale::Float32, fresh::Bool) = DeltaRule(scale, fresh, 0)

# DeltaRuleQKV: the Gated DeltaNet recurrence straight from the convolution's output. Inputs qkv (C, S, B) with C = 2 nk dk + nv dv
#   (q, k and v stacked along dim 1), g, beta (nv, S, B), state, lens [, snapshots]. Equal to slicing q, k and v out of qkv, L2-normalizing
#   q and k (`epsilon`) and running `DeltaRule(scale, fresh, snap_t)`, without the three slice copies and the two separate norm kernels.
struct DeltaRuleQKV <: Op
    scale::Float32
    fresh::Bool
    snap_t::Int
    nk::Int
    dk::Int
    epsilon::Float32
end

# ConvDeltaRule: CausalConv followed by DeltaRuleQKV as one op, so that the prefill can fuse the convolution, its SiLU and the L2 norm of q and
#   k into one kernel that runs in parallel over tokens. Inputs x (C, S, B) (the in-projection's output), conv state (C, K, B), conv weights
#   (C, 1, K), g, beta, recurrent state, lens [, conv snapshots, recurrent snapshots]; the output is DeltaRuleQKV's.
struct ConvDeltaRule <: Op
    scale::Float32
    fresh::Bool
    snap_t::Int
    nk::Int
    dk::Int
    epsilon::Float32
end

# ArgMax: out (S, B) = the 0-based index of the first maximum along dim 1 of x (V, S, B), as Float32.
#   Greedy decoding picks tokens on the device instead of reading the logits back.
struct ArgMax <: Op end

# LastToken: out (H, 1, B) holds each sequence's hidden state at its last real position,
#   out[:, 1, b] = x[:, lens[b], b] (clamped to 1:S), for x (H, S, B) and lens (B,).
#   A prefill needs the logits of those positions only, so the final norm and the
#   language-model head run on B columns instead of S * B.
struct LastToken <: Op end

# CausalAttention: prefill self-attention, fused. Inputs q (D, S, H, B), k and v (D, S, KVH, B)
#   (RoPE and norms applied; query head h uses KV head (h - 1) ÷ (H ÷ KVH) + 1); output
#   (D, S, H, B) = softmax(scale * q'k + causal mask) v'. Padding positions past a prompt's end
#   only affect themselves (causal), as in the composite form.
struct CausalAttention <: Op
    scale::Float32
end

# Single-token (decode) attention over a KV cache, in one kernel. Inputs:
#   q (D, 1, H, B), past_k and past_v (D, max_seq, KVH, B), k_new and v_new
#   (D, 1, KVH, B), pos (1,) -- the number of valid cache slots (slots >= pos are
#   ignored). Query head h uses KV head (h-1) ÷ (H/KVH) + 1 (grouped-query).
# Output (D, H, B): softmax(scale * q.[K_past[:, 1:pos], k_new]) * [V_past; v_new].
# With `write_cache`, the op also stores k_new/v_new into past_k/past_v at slot
# pos + 1, in place: the decode step then updates its KV cache without separate
# launches. (Other slots are untouched, and attention reads only slots <= pos.)
# With `window` > 0 (sliding-window attention, Gemma3's local layers) only the
# last `window` positions, the new token included, are attended to.
struct DecodeAttention <: Op
    scale::Float32
    write_cache::Bool
    window::Int
end
DecodeAttention(scale) = DecodeAttention(scale, false, 0)
DecodeAttention(scale, write_cache) = DecodeAttention(scale, write_cache, 0)

# op(A) * op(B), where op transposes the first two dims when its flag is set:
# a matmul that reads a permuted operand through BLAS transpose flags instead of
# materializing the Permute.
struct MatMulT <: Op
    ta::Bool
    tb::Bool
end
struct Constant <: Op
    value::Any # Corresponds to ConstantValue in Rust
end

# A function defined by the user, equivalent to Rust's Function op
struct Function <: Op
    name::String
    # The actual function will be handled later
end

# Represents a fused element-wise operation
struct FusedElementwiseOp <: Op
    name::String
    f::Base.Function # A Julia function that takes scalar inputs and returns a scalar
end

# Fused Flash Attention Op
struct FlashAttentionOp <: Op
    scale::Float32
    causal::Bool
end

# Represents a break in the graph for compilation
struct GraphBreak <: Op end

