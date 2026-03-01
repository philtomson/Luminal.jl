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
export Op, Add, Mul, LessThan, SumReduce, MaxReduce, Constant, Reshape, Permute, Expand, MatMul, Function, Slice, Pad, FlashAttentionOp, Unfold


include("Graph.jl")
export Graph, GraphTensor, add_op!, tensor, constant

# Autograd
include("Autograd.jl")
export gradients, backward, mark_trainable!

# Graph Construction API
include("HighLevelOps.jl")
export matmul, relu, sigmoid, swish, silu, gelu, softmax, layer_norm, mean_norm, std_norm, arange, gather, max_reduce, flash_attention, triu, unfold, log2, exp2, sin, cos, sqrt, abs

# Hardware Abstraction
include("Device.jl")
using .Device
export get_device, to_device, from_device, AbstractDevice, CPUDevice, CUDADevice, AMDDevice, VulkanDevice

# Execution Engine
include("Execution.jl")
export execute

# Weight Loading
include("Weights.jl")
export WeightRegistry, register_weight!, load_weights!, load_weights_hf!

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

# Tokenizers
include("LlamaTokenizer.jl")
using .LlamaTokenization
export LlamaTokenizer, encode, decode

# Decoding & Inference
include("Decoding.jl")
using .Decoding
export greedy_decode

# Training & Optimizers
include("Optimizer.jl")
using .Optimizer
export SGD, Adam, AbstractOptimizer, step!

end # module Luminal
