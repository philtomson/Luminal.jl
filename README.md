# Luminal Julia Port

[![Julia](https://img.shields.io/badge/Julia-1.12.5-9558B2?style=for-the-badge&logo=julia&logoColor=white)](https://julialang.org/)
[![Status](https://img.shields.io/badge/Status-Active%20Development-blue?style=for-the-badge)](https://github.com/jafioti/luminal)

A Julia port of [Luminal](https://github.com/jafioti/luminal), a deep learning library using **ahead-of-time compilation** for high performance.

> [!NOTE]
> This is a port of the Luminal Rust library to Julia. Some features from the original Rust version are not yet implemented. See [Missing Features](#missing-features) for details.

## Quick Start

```julia
using Luminal

# Setup graph and tensors
g = Graph()
a = tensor(g, (3, 1))
b = tensor(g, (1, 4))

# Do math...
c = matmul(a, b)

# Prepare inputs
inputs = Dict(
    a.id => Float32[1.0; 2.0; 3.0;;],
    b.id => Float32[1.0 2.0 3.0 4.0]
)

# Execute
device = get_device()
result = execute(g, c.id, inputs, device)

println("Result: ", result)
```

## Installation

```bash
cd Julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Requirements

- **Julia 1.12.5+**
- **CUDA.jl** (for NVIDIA GPUs)
- **AMDGPU.jl** (for AMD GPUs)  
- **SymbolicUtils.jl** v3.31.0
- **Metatheory.jl** v3.0 (from the `ale/3.0` branch of `JuliaSymbolics/Metatheory.jl`)

## Examples

### 🦙 Llama Inference
```bash
julia --project=. examples/llama.jl
```
Simulates the generation loop of a small 4-layer Llama model using compiled graphs. 

### 🤫 Whisper Inference
```bash
julia --project=. examples/whisper.jl
```
Runs the full Whisper speech-to-text pipeline (Audio Encoder + Text Decoder with KV Cache).

### 📈 Linear Regression (Training)
```bash
julia --project=. examples/linear_regression.jl
```
Demonstrates the **Training API**: Forward pass, Loss computation, Autograd (`backward`), and Optimizer (`Adam`) updates.

## Features

### ✅ Implemented

#### Core Architecture
- **RISC-style Ops**: 12 primitive operations:
  - Unary: `Log2, Exp2, Sin, Sqrt, Recip`
  - Binary: `Add, Mul, Mod, LessThan`
  - Other: `SumReduce, MaxReduce, Contiguous`
- **Graph-based Execution**: All operations build a static computation graph
- **Shape Tracking**: Symbolic dimension tracking with broadcasting support

#### Compilation & Optimization
- **Ahead-of-Time Compilation**: `compile(graph)` creates optimized execution plans
- **Operator Fusion**: Automatic fusion of element-wise operation chains
  - `FusedMulAdd`: `a * b + c` → single kernel
  - `FusedAddReLU`: `relu(a + b)` → single kernel
- **Static Memory Allocation**: All buffers pre-allocated at compile time
- **CUDA Graph Capture**: Supported, but currently disabled by default on NVIDIA GPUs to avoid memory pool conflicts during generation loops.
- **Search-Based Compilation**: E-Graph based optimization via **Metatheory.jl**
  - Pattern matching with custom unary and binary operators
  - Structural rewrite rules with canonicalization
  - Robust e-class equivalence verification

#### Hardware Support
- **CPU**: Fully supported via Julia's native array operations
- **NVIDIA CUDA**: Full support via CUDA.jl
- **AMD ROCm**: Partial support via AMDGPU.jl
- **Automatic Device Detection**: `get_device()` selects best available hardware

#### Neural Network Layers
High-level API in `NN.jl`:
- ✅ `Linear` - Fully connected layers (Verified)
- ✅ `Embedding` - Token embeddings (Verified)
- ✅ `LayerNorm` - Layer normalization (Verified)
- ✅ `RMSNorm` - Root mean square normalization
- ✅ `Attention` - Multi-head attention with KV cache
- ✅ `RoPE` - Rotary positional embeddings

#### Transformer Support
- **Llama Architecture**: Fully implemented
  - MLP with SiLU/Swish activation
  - Multi-query attention
  - RoPE embeddings
  - RMSNorm
- **Flash Attention**: Custom CUDA kernel for efficient attention
  - Causal and non-causal variants
  - Online softmax algorithm
  - CPU fallback

#### Weight Loading & Data
- **Safetensors Support**: Load weights directly from `.safetensors` files
- **HuggingFace Integration**: `load_weights_hf!` for automatic model downloads
- **Tokenizer**: Native BPE tokenizers for Llama and Whisper (pure Julia)

#### High-Level Operations
Comprehensive operator library in `HighLevelOps.jl`:
- Math: `+, -, *, /, ^, sqrt, exp, log, sin, cos`
- Comparison: `<, >, <=, >=, ==, !=`
- Tensor ops: `matmul, permute, reshape, expand, slice, pad`
- Activations: `relu, sigmoid, swish, gelu, softmax`
- Reductions: `sum, max, mean`

#### Training & Autograd
- **Reverse-Mode AD**: Full implementation of automatic differentiation
- **Operator Gradients**: VJP rules for all 12 primitives and broadcasting
- **Optimizers**: `SGD` and `Adam` implementation for model training
- **Integrations**: `backward(loss)` and `step!(opt, loss)` for training loops

### ⚠️ Missing Features

The following features from the Rust version are **not yet implemented**:

#### Distributed Computing
- ❌ Data parallelism
- ❌ Pipeline parallelism  
- ❌ Tensor parallelism
- ❌ Multi-GPU support

Single-device execution only.

#### Additional Models
- ✅ Whisper (speech recognition) - Full inference with KV cache
- ❌ Yolo v8 (object detection)
- ❌ Phi 3

Only Llama architecture is currently implemented.

#### Advanced Optimizations
- ❌ Tensor Core utilization on NVIDIA
- ❌ Blackwell intrinsics (TMEM, TMA)
- ❌ Quantization (INT8, FP16 support exists but not quantized inference)

#### Tooling
- ❌ Benchmarking suite
- ❌ PyTorch validation tests
- ❌ Model export/import

## Architecture

### Why Julia?

The Julia port leverages Julia's strengths:
- **Multiple Dispatch**: Natural fit for operator overloading and device-specific kernels
- **Type System**: Strong typing helps catch errors at compile time
- **CUDA/GPU Support**: First-class GPU support via CUDA.jl and AMDGPU.jl
- **Scientific Computing**: Rich ecosystem for numerical computing

### Compilation Strategy

Unlike the Rust version's search-based approach, Julia uses **SymbolicUtils.jl** for graph optimization:

1. **Graph Construction**: Operations build a `Graph` of `Node` objects
2. **E-Graph Integration**: Graph → **Metatheory.jl** E-Graph
3. **Rewrite Rules**: High-performance pattern matching and structural rewrites
4. **Compilation**: `compile()` generates fused execution plan
5. **Execution**: Device-specific kernels execute via multiple dispatch

### Directory Structure

```
Julia/
├── src/
│   ├── Luminal.jl              # Main module
│   ├── Ops.jl                  # Primitive operations
│   ├── Graph.jl                # Graph data structures
│   ├── ShapeTracker.jl         # Dimension tracking
│   ├── HighLevelOps.jl         # High-level operator library
│   ├── SymbolicIntegration.jl  # SymbolicUtils integration
│   ├── Compiler.jl             # Graph compilation & fusion
│   ├── Execution.jl            # Interpreter & kernels
│   ├── Device.jl               # Hardware abstraction
│   ├── NN.jl                   # Neural network layers
│   ├── Autograd.jl             # Reverse-mode AD
│   ├── Optimizer.jl            # SGD & Adam optimizers
│   ├── Decoding.jl             # Greedy decode logic
│   ├── Weights.jl              # Safetensors/HF weight loading
│   ├── Whisper.jl              # Whisper architecture
│   └── WhisperTokenizer.jl      # Whisper BPE tokenizer
├── tests/                      # Comprehensive test suite
└── docs/
    └── porting_plan.md         # Detailed porting status
```

## Testing

Run the full test suite:

```bash
cd Julia
for f in tests/test_*.jl; do
    echo "=== $f ==="
    julia --project=. "$f"
done
```

Individual tests:
```bash
julia --project=. tests/test_compilation.jl    # Graph compilation
julia --project=. tests/test_fusion.jl          # Operator fusion
julia --project=. tests/test_attention.jl       # Flash attention
julia --project=. tests/test_llama.jl           # Llama model
julia --project=. tests/test_autograd.jl        # Autograd verification
julia --project=. tests/test_optimizer.jl       # Optimizer verification
julia --project=. tests/test_greedy_decode.jl   # Whisper end-to-end
```

See [`tests/README.md`](tests/README.md) for detailed test documentation.

## Performance

Preliminary benchmarks on NVIDIA GTX 1070:

| Model | Device | Throughput |
|-------|--------|------------|
| TinyLlama 2L/1024H | CUDA | 131ms per forward pass |
| TinyLlama 4L/512H (Generation)| CUDA | ~47 tok/s (21ms/token) |
| Llama Attention (compiled) | CUDA | ~10x faster than interpreter |
| Whisper Decoding | CPU | ~12 steps/s |

> [!NOTE]
> Performance is still being optimized. The Rust version achieves 15-25 tok/s on Llama 3 8B (M-series Macs).

## Comparison to Rust Version

| Feature | Rust Luminal | Julia Port | Notes |
|---------|--------------|------------|-------|
| **Core Ops** | ✅ 12 primitives | ✅ 12 primitives | Identical |
| **Graph Execution** | ✅ Static graphs | ✅ Static graphs | Same approach |
| **Compilation** | ✅ Search-based | ✅ Search-based | Both use E-Graphs |
| **Operator Fusion** | ✅ Automatic | ✅ Automatic | Similar results |
| **CUDA Support** | ✅ Native | ✅ Via CUDA.jl | Slightly slower |
| **Metal Support** | ✅ Native | ❌ Not supported | Julia limitation |
| **Flash Attention** | ✅ Auto-derived | ✅ Hand-written | Both optimized |
| **Training** | ✅ Full support | ✅ SGD & Adam | Supported |
| **Llama** | ✅ 3/3.1/3.2 | ✅ Architecture only | Working |
| **Other Models** | ✅ Whisper, Yolo | ✅ Whisper only | Ported |
| **Distributed** | ✅ Planned | ❌ Not planned | Long-term |

## Roadmap

### Short-term (Q1 2026)
- ✅ Flash Attention
- ✅ Graph compilation with fusion
- ✅ CUDA graph capture
- ✅ Search-based compilation (Metatheory.jl)
- ⏳ Full Llama 3 8B inference
- ⏳ PyTorch numerical validation

### Medium-term (Q2 2026)
- ✅ Training support (autograd & optimizers)
- ✅ Whisper implementation
- ⏳ Gradient checkpointing
- ⏳ Mixed precision (FP16/BF16)

### Long-term
- ⏳ Multi-GPU support
- ⏳ Model quantization (INT8, INT4)
- ⏳ Advanced kernel auto-generation
- ⏳ Distributed training

## Documentation

- [Porting Plan](docs/porting_plan.md) - Detailed implementation status and roadmap
- [Test Suite](tests/README.md) - Test documentation and coverage
- [Rust Luminal Docs](https://docs.luminalai.com) - Original library documentation

## Contributing

This is an active port of the Rust Luminal library. Contributions welcome!

**Priority areas**:
- Training/autograd implementation
- PyTorch validation tests
- Performance benchmarking
- Additional model implementations

## License

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option.
