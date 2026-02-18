module Luminal

using SymbolicUtils
using SymbolicUtils: Sym, BasicSymbolic

# Type alias for dimension values: either a concrete Int or a symbolic expression
const DimType = Union{Int, BasicSymbolic{Int}}

# Shape Tracking
include("ShapeTracker.jl")
export ShapeTracker

# Core Data Structures
include("Ops.jl")
export Op, Add, Mul, LessThan, SumReduce, MaxReduce, Constant, Reshape, Permute, Expand, MatMul, Function, Slice, Pad, FlashAttentionOp

include("Graph.jl")
export Graph, GraphTensor, add_op!, tensor, constant

# Graph Construction API
include("HighLevelOps.jl")
export matmul, relu, sigmoid, swish, silu, gelu, softmax, layer_norm, mean_norm, std_norm, arange, gather, max_reduce, flash_attention, triu

# Hardware Abstraction
include("Device.jl")
using .Device
export get_device, to_device, from_device, AbstractDevice, CPUDevice, CUDADevice, AMDDevice, VulkanDevice

# Execution Engine
include("Execution.jl")
export execute

include("NN.jl")
export NN

# SymbolicUtils.jl Integration (graph <-> symbolic expression conversion)
include("SymbolicIntegration.jl")


# Compiler (optimization rules using SymbolicUtils.jl)
include("Compiler.jl")
export compile, optimize_symbolic

# Metatheory Integration
include("MetatheoryOps.jl")
include("MetatheoryBridge.jl")
include("MetatheoryCost.jl")
include("MetatheoryRules.jl")
include("MetatheoryOptimizer.jl")

end # module Luminal
