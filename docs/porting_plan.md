# Luminal to Julia Porting Plan

This document outlines the strategy and progress for porting the Luminal deep learning framework from Rust to Julia.

> [!NOTE]
> **Last updated: October 18, 2026** — the gap list reflects a comparison with upstream `77d3060f`; phases 1-8 are as of February 2026.

---

## Phase 1: Core Scaffolding ✅ Complete

All items verified by tests (60 tests across 9 files — see `Julia/tests/README.md`).

- **[x] Core Data Structures** — `Op`, `Node`, `Graph`, `GraphTensor` in `Julia/src/`
- **[x] Graph Construction API** — `HighLevelOps.jl` overloads `+`, `*`, `log2`, etc.
- **[x] Compiler Setup** — SymbolicUtils.jl-based rewrite rules in `Compiler.jl`
- **[x] Initial Optimization Rules**:
  - ReLU → `max(x, 0)` canonicalization
  - Constant folding (e.g., `2 + 3` → `5`)
  - Algebraic simplification (e.g., `a * 1` → `a`, `a + 0` → `a`)
  - Associativity (e.g., `(a + 2) + 3` → `a + 5`)
  - Commutativity (e.g., `(2 * a) * 3` → `6a`)
- **[x] Simple Execution Engine** — CPU interpreter in `Execution.jl`
- **[x] Testing Framework** — `Julia/tests/` with 9 test files

---

## Phase 2: Advanced Rules ✅ Complete

Verified by 81 passing tests.

### 1. ShapeTracker ✅ Complete
- `SymbolicUtils.jl` handles all symbolic dimension expressions.
- `ShapeTracker.jl` integrated into `GraphTensor` and high-level ops.

### 2. Expand Compiler Rules ✅ Complete
- Infrastructure: Robust split-pass rewriter (`Prewalk` + `Postwalk`).
- Ported Rules: Inverses, Identities, Operator Fusion, Loop Fusion, and Hardware Patterns (`TCMatmul`).

---

## Phase 3: GPU Backends ✅ Complete

- **Goal**: High-performance GPU execution (NVIDIA, AMD).
- **Status**: Automatic hardware detection and device-aware execution implemented.
- **Approach**: Uses `CUDA.jl` (NVIDIA) and `AMDGPU.jl` (AMD) with seamless memory transfer (`to_device`/`from_device`).
- **Execution**: Updated the interpreter in `Execution.jl` to use device-specialized arrays (e.g., `CuArray`), leveraging Julia's multiple dispatch for automatic kernel launching.
- **Verification**: Verified on local NVIDIA GTX 1070 hardware with `Julia/tests/test_gpu_execution.jl`.

---

## Phase 4: Neural Network Layers (`NN.jl`) ✅ Complete

- **Goal**: High-level API for building models (similar to `torch.nn` or `Flux.jl`).
- **Status**: Core layers (`Linear`, `Embedding`, `LayerNorm`) implemented and verified on GPU.
- **Approach**: Built a comprehensive high-level op library (`HighLevelOps.jl`) and implemented layer structs with callable functor support in `NN.jl`.
- **Verification**: Verified numerical bit-accuracy against manual calculations on NVIDIA hardware with `Julia/tests/test_nn_layers.jl`.

---

## Phase 5: Example & Verification (Llama-Julia) ✅ Complete

- **[x] Llama Architecture**: Fully implemented the LLaMA model in `NN.jl`, including `RMSNorm`, `Mlp` (Swish/SiLU), `RoPE` (Rotary Positional Embeddings), and `Attention` (with KV Cache).
- **[x] High-Level Ops**: Implemented necessary ops: `Slice`, `Pad`, `Permute`, `Expand`, `Broadcasting`.
- **[x] Verification**: Verified with comprehensive unit and end-to-end tests in `Julia/tests/test_llama.jl`.
- **[x] Benchmarking**: Preliminary result: 131ms for TinyLlama 2L/1024H on GPU.
- **[x] Upstream Sync**: Analyzed upstream Rust repo. Confirmed core logic is stable; recent changes are primarily custom CUDA kernels (optimizations) and backend-specific fixes.
- **[x] Standalone Example**: Created `examples/llama.jl` to demonstrate the end-to-end compilation, hardware-acceleration, and simulated generation loop of the Llama model architecture.
  - _Bug Fix Discovery_: During the example's compilation phase, it was discovered that `SymbolicUtils.jl` eagerly simplifies `x * x` to dynamic exponential calls like `Base.:^(x, 2)`, which triggered an `InvalidIRError` during `CUDA.jl` execution. A custom AST transformation was injected into `Compiler.jl` to securely map these back to element-wise multiplication before constructing GPU execution kernels.
  - _CUDA JIT & World Age Fixes_: Solved Julia 1.12 `InvalidIRError` by generating global named CUDA kernels natively (`Core.eval`) rather than `RuntimeGeneratedFunction` closures to circumvent JIT semantic errors during dynamic graph execution.
  - _CUDA Graph Capture Issues_: Disabled `CUDA.capture()` default graph instantiations during generation loops on NVIDIA due to dynamic workspace allocations inside `array view`-enabled CUBLAS routes (`batch_matmul!`) triggering graph topology invalidation on replay. Simulation now completes dynamically at ~47 tok/sec.

---

## Phase 6: Optimization & Compilation ✅ Complete

All optimization features have been implemented and verified through comprehensive testing.

### Flash Attention ✅ Complete

- **[x] Implementation**: Added `FlashAttentionOp` with CUDA kernel (online softmax algorithm) and CPU fallback.
- **[x] API**: High-level `flash_attention(q, k, v; causal=false)` in `HighLevelOps.jl`.
- **[x] Verification**: Comprehensive testing showing:
  - Non-causal attention: Bit-exact match with manual baseline
  - Causal attention: Passes with expected numerical tolerance (<15% relative error) due to algorithmic differences (early termination vs. masked softmax)
- **[x] Bug Fixes**: Fixed critical interpreter bugs:
  - Broadcasting semantics (`align_broadcast_ranks` for Julia's broadcasting rules)
  - Float literal syntax errors in tests
  - Test causal mask logic (corrected index orientation)

### Operator Fusion ✅ Complete

- **[x] Implementation**: Element-wise operations are automatically fused during graph compilation
- **[x] Fusion Rules**: `FusedMulAdd` (`a * b + c`), `FusedAddReLU` (`relu(a + b)`)
- **[x] Compiler Integration**: Fusion logic integrated into `compile()` function
- **[x] Verification**: Tested in `tests/test_fusion.jl` - successfully fuses chains of element-wise ops into single kernels

### Graph Compilation ✅ Complete

- **[x] Implementation**: `compile(graph)` function creates static execution plans
- **[x] Pre-allocation**: All buffers allocated at compile time based on `get_device()`
- **[x] Execution**: Returns callable `CompiledGraph` that bypasses interpreter overhead
- **[x] Verification**: Tested in `tests/test_compilation.jl` - significant speedup over interpreter

### CUDA Graphs ✅ Complete

- **[x] Implementation**: `execute_with_capture()` in `Device.jl` for CUDA graph capture
- **[x] Caching**: Captured graphs stored in cache for replay
- **[x] Warmup**: JIT compilation warmup before capture to avoid invalidation
- **[x] Verification**: CUDA graph capture confirmed in tests ("CUDA Graph captured successfully")

### Memory Management ✅ Complete

- **[x] Static Allocation**: All node buffers pre-allocated during `compile()`
- **[x] Device-Aware**: Allocates `CuArray` for CUDA, regular arrays for CPU
- **[x] Reuse**: Results stored in pre-allocated buffers, minimizing allocations during execution

---

## Phase 8: Search-Based Compilation ✅ Complete

- **[x] Metatheory.jl Integration**: Fixed upstream bugs in Metatheory.jl v3.0 to support custom operators (PR submitted).
- **[x] Bridge**: Implemented bi-directional conversion between `GraphTensor` and Metatheory expressions (`MetatheoryBridge.jl`).
- **[x] Optimizer**: Created `compile_with_search` pipeline using e-graph saturation and extraction (`MetatheoryOptimizer.jl`).
- **[x] Rules**: Implemented algebraic simplification, operator fusion (`ADD+RELU`, `MUL+ADD`), and constant folding.
- **[x] Performance**: Achieved **~5.5x faster** compilation time (0.17ms) vs manual baseline (0.94ms) for small fusion kernels.

---

## Gaps against Rust Luminal (checked 2026-10-18 against upstream `77d3060f`)

Most of what this section once listed has since been built: autograd and Adam (`Autograd.jl`, `tests/test_autograd.jl`),
int8 / Q4 / Q8 / AWQ weights and GGUF / safetensors import, matrix-core (WMMA) kernels, Whisper, Gemma3, Qwen3.5/3.6/3.8,
Phi-3 weight mapping, speculative decoding (MTP, DFlash2), the OpenAI-compatible Qwen server, NCM (`src/NCM.jl`),
exact scans (`Scan`: cumsum / cumprod / cummax / prod) and a ROCm backend. What upstream has and we do not, by priority:

| Status | Feature | Upstream | Notes for the port |
|--------|---------|----------|--------------------|
| next | Paged attention / shared KV pool | `luminal_nn::{cache,attention}`, `model_zoo::paged_llama3` | Functional graph primitives: a flat slot cache per layer, `scatter_rows` / `gather_rows`, GQA over the gathered context, additive biases built from positions and sequence ids. Allocation and page tables stay with the caller. |
| open | Mixture-of-experts layer | `luminal_nn::moe`, `qwen3_moe`, `gemma4_moe` | `Qwen35.jl` mentions MoE; check how much of a generic routed-expert layer exists before porting. |
| open | Pooling layers | `luminal_nn::pooling` | No pooling layer in `NN.jl` (NCM has host-side area pooling only). Needed for YOLO. |
| open | FP8 Llama | `llama3_1_fp8` | FP8 storage and dequantizing matmul; gfx1151 has no FP8 matrix cores, so a dequantize-then-GEMM path. |
| open | Gemma 4 MoE, Qwen3 MoE, Flux2, YOLO v11 | `model_zoo` | Models; each needs the layers above first. |
| open | PyTorch frontend | `crates/pytorch` (PT2 translator, SPMD, ShapeVar buckets) | Not planned: Reactant (StableHLO) is the interop route here. |
| open | Metal backend | `luminal_metal` | Not planned (no Julia Metal story we use). |
| next | Compiler and search depth | egglog rulesets, logical / layout IR, genome search, bufferization and arenas, bounds and buckets | The largest gap and the core of upstream's design; see [Compiler and search](#compiler-and-search-the-largest-gap) below. |
| open | Symbolic `split_dims` / `merge_dims` | simplifier rules `(x*y)/y -> x` (#566) | Dimensions here support `+` and `*` only (no symbolic `div`), so the failure those rules fix cannot occur yet; revisit if symbolic splits are added. |
| open | Checked integer overflow in scans | `LogicalScan*` reference semantics | Our `cumprod` on integers wraps. |
| open | Tracing / profiling export | `luminal_tracing` (Perfetto) | `LAUNCH_PROFILE`-style timers exist; no trace export. |
| not planned | Multi-GPU / distributed | | Single device. |

Done since the first version of this list: training (autograd, optimizers), quantization (int8, Q4, Q8, AWQ, GGUF native weights),
model import (GGUF, safetensors), Phi-3 and Whisper, matrix-core kernels (`WMMA.jl`), a benchmarking suite (`benchmarks/`),
exact scans (cumsum / cumprod / cummax / prod).

See [`performance_vs_luminal.md`](performance_vs_luminal.md) for measurements against upstream.

---

## Compiler and search: the largest gap

Written 2026-10-18 from upstream's `spec.md`, `docs/design/*`, `docs/architecture.md` and the module headers of
`src/{extraction,search_support,bucketing}.rs`, `src/buffer/*`, `src/layout/*`, `src/logical/*` (about 28k lines of Rust in
`src/`; I have not read every rule). Our side is `src/EGraphRewrite.jl`, `src/Compiler.jl` and
[`egraph_rewrite_layer.md`](egraph_rewrite_layer.md).

### What upstream does

```
frontend recorder -> logical SSA (primitive ops, symbolic dims with bounds)
  -> egglog saturation (named rulesets; backends add rewrites HLIR -> backend ops)   the only place rewriting happens
  -> extraction: a genome picks one producer per e-class        } search = genetic algorithm over genomes,
  -> backend profiles each candidate (fitness = measured time)  } every candidate is valid by construction
  -> layout IR (LayoutTensor: logical shape + physical layout as separate, searchable facts)
  -> destination-passing rewrite -> bufferization (alias / in-place conflict analysis, MLIR-style)
  -> arena (one storage plan; boundary buffers owned by the application) -> runtime
```

Design rules that shape it: HLIR is a closed set of about 20 primitive ops, and everything else (fused kernels, matmul
libraries, attention) is introduced by a backend's egglog rules and chosen by extraction; no post-extraction passes may
pattern-match or fuse; every extractable graph must be semantically equivalent to the logical program for all valid
dtypes, shapes, strides and dynamic dims; a CPU `ReferenceRuntime` implements every HLIR op as the semantic oracle; every rule
lives in a named ruleset and has a unique name, so run reports are keyed by rule; dynamic dimensions carry interval bounds,
and an application may compile one program per bucket and dispatch on the runtime values of the dynamic variables, with
an optional arena shared by the buckets.

### What we have

A Metatheory.jl e-graph over the whole `Graph` (not a primitive subset: the nodes are our fused ops), a rule set of about
ten theories (expand elimination, scale motion, merged projections, transposed matmul, precision / quantization per
matmul, kernel-variant alternatives), greedy DAG extraction with a refinement pass, and a **measured search** that does
coordinate descent over about 13 choice groups, timing verified candidates against the incumbent in interleaved rounds
with a cache. `compile()` is the executor: elementwise fusion, buffer reuse and aliasing, constant folding, HIP graph capture.
Search for a TinyLlama decode step took 14.54 ms per token against 14.9 for the unsearched Float16 path, and the measured
search found prefill mixes no fixed choice gives (up to 19% at 16 tokens), so the machinery pays off where it applies.

### The gaps, with what each would buy here

| # | Gap | Upstream | Ours | Why it matters for this project |
|---|-----|----------|------|----------------------------------|
| C1 | Rewriting from primitives | Fusion and lowering are rules over ~20 primitive ops | Fused ops (`SoftmaxOp`, `RMSNormOp`, `DecodeAttention`, ...) are produced by the front end or by `compile()` passes, and the rules choose between alternatives we encoded | New fusions mean new hand-written ops and `compile()` code. Recognising softmax / RMSNorm / SwiGLU from primitives would make model code backend-neutral. |
| C2 | Search algorithm | Genetic search over complete genomes (joint choices, crossover) | Coordinate descent over groups, plus a few joint decisions | Coordinate descent cannot find choices that only pay together (it needed hand-made joint decisions for merged projections). Cheap to try on the existing groups. |
| C3 | Rule hygiene and reporting | Named rulesets, unique rule names, per-rule run reports | Anonymous `@theory` blocks | We cannot say which rule made a graph larger or slower; debugging the Metatheory issues in `egraph_rewrite_layer.md` was done by hand. |
| C4 | Layout IR | Logical shape and physical layout are separate; layouts are e-class facts the search can choose | `ShapeTracker` views plus explicit `Contiguous`; weight layouts chosen by hand (`HalfWeight`, `MatMulT`, `Q4Weight`, ...) | Needed before kernels can be generated rather than hand-written, and for searching transposes and tilings. Large; defer until generated kernels are on the table. |
| C5 | Bufferization and arenas | One-shot bufferization with alias analysis, destination-passing, an arena that packs device tensors and scratch, boundary buffers owned by the caller | Reuse and aliasing inside `compile()`; buffers freed by the garbage collector (see the 27B memory notes) | Peak memory is our real limit on gfx1151 (27B, 262k context). A liveness-based plan with caller-owned boundary buffers would replace the chunked-prefill and GC workarounds. |
| C6 | Bounded dynamic dims and buckets | `SymbolBounds` intervals with interval analysis; one compiled program per bucket; dispatch on `dyn_map`; shared arena | Symbolic dims re-plan buffers when a symbol changes; prefill is padded to power-of-two buckets by hand in `llama_generate` | Replaces the ad hoc padding and the per-position reallocation; with C5, buckets can share one arena. |
| C7 | Semantic oracle and equivalence tests | `ReferenceRuntime` plus fixtures for every rule over shapes, dtypes and strides | Search verifies each candidate's outputs against the original graph on one input; the CPU interpreter exists but there are no per-rule property tests | A wrong rule is caught at search time only when it fires on that input. Per-rule tests would catch dtype and stride cases first. |
| C8 | Rewriting engine | egglog (named rulesets, schedules, multi-pattern rules, primitives) | Metatheory.jl 3.0 on a fork branch; single-rooted patterns (merged projections needed a separate pass); defined-function head names silently never match | Our rule language is the limit on C1. egglog has no Julia binding; options are an FFI to the Rust crate or staying on Metatheory with the workarounds. |
| C9 | Backend profiling interface | `profile(llir)` is part of the Runtime trait | `measured_search(rw, make_runner)` is generic, the device code is not factored as a runtime | Fine for one backend; only matters if we add another. |

Where we are ahead or equal: measured search with interleaved rounds, noise-aware acceptance (more than 1% and at least 4 of 5
rounds) and an on-disk winner cache; kernel parameters (threads, columns per pass) as search variants; quantization as a rewrite
choice. Upstream searches layouts and fusions we cannot express; we search precision and kernel shape per matmul, which is
what dominated decode here.

### Proposed order (each step ends with a go / no-go)

1. **C3, rule names and per-rule reports** in `EGraphRewrite.jl` (small). Go when a search report lists which rules created
   the chosen graph's nodes.
2. **C2, genome search** over the existing choice groups: sample joint assignments, keep a population, mutate and cross. Go
   if it beats coordinate descent by 2% or more on TinyLlama decode or prefill at equal search time; otherwise keep
   coordinate descent and record the result here.
3. **C7, rule equivalence tests**: for each rule, random shapes and dtypes, rewritten graph versus the CPU interpreter.
   Prerequisite for adding rules safely (step 4).
4. **C1, recognise fused ops from primitives** for softmax, RMSNorm and SwiGLU, which `NCM.jl` and `Gemma3.jl` currently spell
   out by hand. Go if the recognised graph runs as fast as the hand-fused one.
5. **C6 + C5, bounds, buckets and a bufferization plan**, together: bounds and buckets decide which programs exist, the plan
   decides their memory. Go if peak memory of a 32k-token 27B prefill drops without a throughput loss.
6. **C4, layout IR**, only if we commit to generated kernels.
7. **C8**: revisit the rewriting engine after step 4 shows whether Metatheory's rule language holds.

---

## Compiler Architecture

The Julia port now supports two complementary compilation strategies:
1. **Rule-Based (Default)**: Uses `SymbolicUtils.jl` for fast, deterministic rewrites.
2. **Search-Based (Advanced)**: Uses `Metatheory.jl` for e-graph saturation and global optimization.

### 1. Rule-Based Pipeline (SymbolicUtils.jl)

The compiler converts Luminal's computation graph into SymbolicUtils.jl symbolic expressions, applies deterministic rewrite rules, and returns the optimized symbolic expression.

```
Graph → luminal_to_symbolic() → SymbolicUtils Term → @rule rewrites → Optimized Term
```

**Key files:**
- `SymbolicIntegration.jl` — Converts `Graph`/`GraphTensor` to SymbolicUtils `BasicSymbolic` expressions
- `Compiler.jl` — Defines rewrite rules and applies them via split-pass rewriter (`Prewalk` + `Postwalk`)

### Comparison of Approaches

| Approach | Strength | Status |
|----------|----------|--------|
| **Metatheory.jl** (e-graphs) | Global optimization, cycle detection, automated fusion search | **SOLVED** ✅ - Integrated in Phase 8. |
| **Symbolics.jl** (full CAS) | Full computer algebra system | Not suitable (eager evaluation quirks). |
| **SymbolicUtils.jl** | Fast, deterministic, simple rule definitions | **Default** ✅ - Used for standard compilation. |

SymbolicUtils.jl also provides free algebraic normalization: `x + 0 → x`, `x * 1 → x`, constant folding, and associative/commutative flattening all happen at term construction time.

### Adding New Rules

To add a new optimization rule, add a `@rule` to `Compiler.jl`:

```julia
# Example: fuse Add followed by ReLU into a FusedAddReLU
const FUSED_ADD_RELU = @rule max(~a + ~b, 0) => fused_add_relu(~a, ~b)
```

Rules are prioritized by pass type:
1. **Prewalk pass**: Structural rules (e.g., `TCMatmul`) to catch patterns before partial fusion.
2. **Postwalk pass**: Local simplifications and fusions (e.g., `MulAdd`).

### Environment

- **Julia**: 1.12.5
- **SymbolicUtils.jl**: v3.31.0 (dev-installed from `/devel/phil/SymbolicUtils.jl`)
- **TermInterface.jl**: v2.0.0
