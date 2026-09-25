# Julia Test Suite

## Running Tests

Run all tests from the project root (each test must run in a separate Julia process due to Julia JIT compilation and graph cleanup requirements):

```bash
for f in tests/test_*.jl tests/test.jl; do
    echo "=== $f ==="
    julia --project=. "$f"
    echo
done
```

Or run individual tests:

```bash
julia --project=. tests/test_egraph_rewrite.jl
```

## Test Inventory

### Core Architecture
| Test | Description |
|------|-------------|
| `test_symbolic.jl` | Symbolic expression construction and evaluation |
| `test_shape_tracker.jl` | ShapeTracker dimension operations |
| `test.jl` | End-to-end graph build + execution (matmul) |
| `test_lazy.jl` | Building a graph computes nothing until it is compiled and run |
| `test_symbolic_slice.jl` | Slices with symbolic bounds (KV-cache update) on CPU and GPU |

### Compilation & Fusion
| Test | Description |
|------|-------------|
| `test_compilation.jl` | End-to-end graph compilation to optimized execution plan |
| `test_fusion.jl` | Fusion of element-wise operators into single kernels |
| `test_half_weights.jl` | Float16 matmul weights match Float32 |

### E-Graph Rewrite Layer (Search-Based)
| Test | Description |
|------|-------------|
| `test_egraph_rewrite.jl` | Graph ⇄ e-graph round trip, rewrite rules, merged projections with constant folding, and `compile(...; search=...)` |

### Hardware & Devices
| Test | Description |
|------|-------------|
| `test_gpu_detection.jl` | Hardware discovery (CUDA/ROCm/AMDGPU) |
| `test_gpu_execution.jl` | End-to-end kernel execution on NVIDIA GPUs |

### Neural Networks & Models
| Test | Description |
|------|-------------|
| `test_nn_layers.jl` | Verification of Linear, LayerNorm, and RMSNorm layers |
| `test_attention.jl` | Multi-head attention mechanism verification |
| `test_flash_attention_verification.jl` | Numerical validation of Flash Attention vs standard attention |
| `test_llama.jl` | 2-layer Llama model inference (interpreted) |
| `test_llama_compiled.jl` | Full Llama model inference with compilation and fusion |

## Requirements

- **Julia 1.12.5+**
- **Metatheory.jl** v3.0+
- **SymbolicUtils.jl** v3.31.0
- **CUDA.jl** (for GPU tests)
