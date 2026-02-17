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
    size::Int # Using Int for size, corresponds to Expression in Rust
end

struct Reshape <: Op
    shape::Vector{Int} # Using Int for shape dimensions
end

struct Slice <: Op
    ranges::Vector{Tuple{Int, Int}}
end

struct Pad <: Op
    padding::Vector{Tuple{Int, Int}}
end

# Special Ops
struct MatMul <: Op end
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

