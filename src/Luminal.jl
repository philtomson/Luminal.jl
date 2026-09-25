module Luminal

using SymbolicUtils
using SymbolicUtils: Sym, BasicSymbolic

# Type alias for dimension values: either a concrete Int or a symbolic expression
const DimType = Union{Int, BasicSymbolic{Int}}

# Tokenizer interface: every tokenizer (Llama, Whisper) adds methods to these, so
# `encode`/`decode` dispatch on the tokenizer type instead of being two
# different, conflicting exported functions.
function encode end
function decode end

# Shape Tracking
include("ShapeTracker.jl")
export ShapeTracker

# Core Data Structures
include("Ops.jl")
export Op, Add, Mul, LessThan, SumReduce, MaxReduce, Constant, Reshape, Permute, Expand, MatMul, MatMulF16, MatMulQ8, MatMulT, RotaryEmbed, DecodeAttention, RMSNormOp, Function, Slice, Pad, FlashAttentionOp, Unfold


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

# Compiler: turns a Graph into an executable plan (fusion, buffer reuse,
# constant folding, HIP graph capture)
include("Compiler.jl")
export compile

# Graph-level rewrite layer on Metatheory e-graphs; used by compile(...; search=...)
include("EGraphRewrite.jl")

# Tokenizers
include("LlamaTokenizer.jl")
using .LlamaTokenization
export LlamaTokenizer, encode, decode

# Decoding & Inference
include("Decoding.jl")
using .Decoding
export greedy_decode, llama_generate

# Training & Optimizers
include("Optimizer.jl")
using .Optimizer
export SGD, Adam, AbstractOptimizer, step!

end # module Luminal
