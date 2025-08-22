# Defines the core operations in the computation graph.

# Abstract parent type for all operations
abstract type Op end

# Unary Ops (A -> A)
struct Log2 <: Op end
struct Exp2 <: Op end
struct Sin <: Op end
struct Sqrt <: Op end
struct Recip <: Op end
struct Contiguous <: Op end
struct ReLU <: Op end

# Binary Ops (A x A -> A)
struct Add <: Op end
struct Mul <: Op end
struct Mod <: Op end
struct LessThan <: Op end
struct Max <: Op end

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

# Represents a break in the graph for compilation
struct GraphBreak <: Op end

# Placeholder functions for Metatheory.jl pattern matching
Luminal_Log2(x) = Expr(:call, :Luminal_Log2, x)
Luminal_Exp2(x) = Expr(:call, :Luminal_Exp2, x)
Luminal_Sin(x) = Expr(:call, :Luminal_Sin, x)
Luminal_Sqrt(x) = Expr(:call, :Luminal_Sqrt, x)
Luminal_Recip(x) = Expr(:call, :Luminal_Recip, x)
Luminal_Contiguous(x) = Expr(:call, :Luminal_Contiguous, x)
Luminal_ReLU(x) = Expr(:call, :Luminal_ReLU, x)

Luminal_Add(x, y) = Expr(:call, :Luminal_Add, x, y)
Luminal_Mul(x, y) = Expr(:call, :Luminal_Mul, x, y)
Luminal_Mod(x, y) = Expr(:call, :Luminal_Mod, x, y)
Luminal_LessThan(x, y) = Expr(:call, :Luminal_LessThan, x, y)
Luminal_Max(x, y) = Expr(:call, :Luminal_Max, x, y)

Luminal_SumReduce(x, dim) = Expr(:call, :Luminal_SumReduce, x, dim)
Luminal_MaxReduce(x, dim) = Expr(:call, :Luminal_MaxReduce, x, dim)

Luminal_Permute(x, dims) = Expr(:call, :Luminal_Permute, x, dims)
Luminal_Expand(x, dim, size) = Expr(:call, :Luminal_Expand, x, dim, size)
Luminal_Reshape(x, shape) = Expr(:call, :Luminal_Reshape, x, shape)

Luminal_MatMul(x, y) = Expr(:call, :Luminal_MatMul, x, y)
Luminal_Constant(value) = Expr(:call, :Luminal_Constant, value)
Luminal_Function(name) = Expr(:call, :Luminal_Function, name)
Luminal_GraphBreak() = Expr(:call, :Luminal_GraphBreak)
