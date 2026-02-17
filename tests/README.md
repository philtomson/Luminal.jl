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
julia --project=. tests/test_metatheory_v3.jl
```

## Test Inventory

### Core Architecture
| Test | Description |
|------|-------------|
| `test_symbolic.jl` | Symbolic expression construction and evaluation |
| `test_shape_tracker.jl` | ShapeTracker dimension operations |
| `test.jl` | End-to-end graph build + execution (matmul) |
| `test_lazy.jl` | Lazy graph evaluation — operations stay symbolic |

### Algebraic Optimization (Rule-Based)
| Test | Description |
|------|-------------|
| `test_algebraic_simplification.jl` | `(a * 1) + 0` simplifies to `a` |
| `test_associativity.jl` | `(a + 2) + 3` simplifies to `a + 5` |
| `test_commutativity.jl` | `(2 * a) * 3` simplifies to `6a` |
| `test_symbolics_integration.jl` | Graph → SymbolicUtils conversion and simplification |

### Compilation & Fusion
| Test | Description |
|------|-------------|
| `test_compilation.jl` | End-to-end graph compilation to optimized execution plan |
| `test_fusion.jl` | Fusion of element-wise operators into single kernels |
| `test_optimizer.jl` | Graph-level canonicalization and simplification rules |
| `test_new_rules.jl` | Verification of recently added optimization patterns |

### Metatheory & E-Graphs (Search-Based)
| Test | Description |
|------|-------------|
| `test_metatheory_v3.jl` | Custom operator matching and rewriting via E-Graphs |
| `test_metatheory_bridge.jl` | Integration layer between Luminal and Metatheory.jl |
| `test_metatheory_optimizer.jl` | Automated search for optimal graph structures |
| `test_metatheory_cost.jl` | Cost functions and extraction strategies for E-Graphs |
| `test_metatheory_runtime.jl` | Integration tests for E-Graph optimized execution |

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
