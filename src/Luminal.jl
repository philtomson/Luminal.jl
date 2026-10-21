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
export Op, Add, Mul, LessThan, Floor, Ceil, Round, Trunc, Select, Div, Exp, GatherND, ScatterND, MappingGather, LutKOp, LutKBwdPinsOp, LutKDWOp, SparseMapGather, SparseMapBwdS, SparseMapBwdX, Iota, Cast, TruncCast, TruncDiv, TruncRem, SumReduce, MaxReduce, Scan, Constant, Reshape, Permute, Expand, MatMul, MatMulF16, MatMulQ8, MatMulQ4, MatMulT, RotaryEmbed, DecodeAttention, RMSNormOp, L2NormOp, GatedRMSNormOp, SwiGLUOp, LayerNormOp, GeluOp, SoftmaxOp, CausalConv, DeltaRule, DeltaRuleQKV, ConvDeltaRule, AttnPrep, GatedAttnOut, LastToken, CausalAttention, ArgMax, Function, Slice, Pad, FlashAttentionOp, Unfold


include("DTypes.jl")
export dtype

include("Graph.jl")
export Graph, GraphTensor, add_op!, tensor, constant

# Autograd
include("Autograd.jl")
export gradients, backward, mark_trainable!

# Graph Construction API
include("HighLevelOps.jl")
export matmul, relu, sigmoid, softplus, swish, silu, gelu, softmax, layer_norm, mean_norm, std_norm, arange, gather, mapping_gather, lut_layer, sparse_map_gather, max_reduce, cummax, flash_attention, triu, unfold, log2, exp2, sin, cos, sqrt, abs, select, scatter, iota, cast, trunc_cast, trunc_div, trunc_rem

# Hardware Abstraction
include("Device.jl")
using .Device
export get_device, to_device, from_device, AbstractDevice, CPUDevice, CUDADevice, AMDDevice, VulkanDevice

# Execution Engine
include("LaunchBounds.jl")
include("WMMA.jl")
include("FlashAttn.jl")
include("DecodeAttn.jl")
include("Execution.jl")
export execute

# XLA via Reactant.jl (methods in ext/LuminalReactantExt.jl, loaded with Reactant)
include("ReactantStubs.jl")
export reactant_function, reactant_compile, to_stablehlo

# Weight Loading
include("Weights.jl")
include("GGUF.jl")
include("NativeWeights.jl")
export WeightRegistry, register_weight!, tie_weight!, load_weights!, load_weights_hf!

include("NN.jl")
export NN

# Compiler: turns a Graph into an executable plan (fusion, buffer reuse,
# constant folding, HIP graph capture)
include("Compiler.jl")
export compile

# Graph-level rewrite layer on Metatheory e-graphs; used by compile(...; search=...)
include("EGraphRewrite.jl")

# The e-graph's rule matchers (the fusion patterns are large) are compiled by Julia the first time `saturate_graph!` runs: about 80 s of a
# session's first `compile(...; search=...)` (measured 220 s with the attention patterns of C1). Saturating a tiny graph while the package
# is precompiled stores that work in the package image (the graph is built with `tensor`, no device is touched).
if ccall(:jl_generating_output, Cint, ()) == 1
    let g = Graph()
        a = tensor(g, [4, 4]); b = tensor(g, [4, 4])
        out = softmax(matmul(a, b) + a, 1)
        rw = EGraphRewrite.to_egraph(g, [out.id])
        EGraphRewrite.saturate_graph!(rw)
        EGraphRewrite.extract_graph(rw)
    end
end

# Tokenizers
include("LlamaTokenizer.jl")
using .LlamaTokenization
export LlamaTokenizer, encode, decode, chat_prompt

# Decoding & Inference
include("Decoding.jl")
using .Decoding
export greedy_decode, transcribe, WhisperSession, llama_generate, LlamaSession, generate, generate_ids, PrefixCache, release_stale!, prefix_summary, prefix_stats, Sampler, sample_token, MTPDrafter, DFlash2, dflash_append!, dflash_draft!, dflash_append_rows!, generate_ids_dflash, generate_ids_speculative, generate_speculative, weight_preset, awq_scales, awq_apply!

# Next-Concept Modeling generator
include("PagedAttention.jl")
include("PagedLlama.jl")
include("Pooling.jl")
include("MoE.jl")
include("NCM.jl")
export NCM

# Training & Optimizers
include("Optimizer.jl")
using .Optimizer
export SGD, Adam, AbstractOptimizer, step!

end # module Luminal
