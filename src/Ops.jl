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

# Single-token (decode) attention over a KV cache, in one kernel. Inputs:
#   q (D, 1, H, B), past_k and past_v (D, max_seq, KVH, B), k_new and v_new
#   (D, 1, KVH, B), pos (1,) -- the number of valid cache slots (slots >= pos are
#   ignored). Query head h uses KV head (h-1) ÷ (H/KVH) + 1 (grouped-query).
# Output (D, H, B): softmax(scale * q.[K_past[:, 1:pos], k_new]) * [V_past; v_new].
# With `write_cache`, the op also stores k_new/v_new into past_k/past_v at slot
# pos + 1, in place: the decode step then updates its KV cache without separate
# launches. (Other slots are untouched, and attention reads only slots <= pos.)
struct DecodeAttention <: Op
    scale::Float32
    write_cache::Bool
end
DecodeAttention(scale) = DecodeAttention(scale, false)

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

