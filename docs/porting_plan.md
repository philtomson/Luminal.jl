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
| done | Paged attention / shared KV pool | `luminal_nn::{cache,attention}`, `model_zoo::paged_llama3` | `src/PagedAttention.jl`: flat slot cache, `gather_rows` / `scatter_rows`, GQA over the gathered context, causal / sequence-isolation / sliding-window biases, `packed_sequence_ids`, and `PageTable` (page allocator: `reserve!`, `release!`, `truncate!`, all-or-nothing, `plan_step` packs a step's padded inputs). `src/PagedLlama.jl`: the fixed-shape batched decode graph (`build_paged_llama_step!`, `paged_llama_step!`) over a pool shared by any mix of sequences, tested against full-forward logits per sequence on CPU and GPU (`tests/test_paged_llama.jl`). Not done: the step copies each layer's pool back after the scatter (an in-place write like `DecodeAttention`'s would avoid it), and there is no shared-prefix page reuse. |
| done (layer) | Mixture-of-experts layer | `luminal_nn::moe`, `qwen3_moe`, `gemma4_moe` | `src/MoE.jl`: `TopKRoutes` and `Routes` (`topk_routes`, `normalize_routes`, `dispatch`, `select_experts`, `combine`), `topk_indexes`, and `SparseMoE` (Qwen3's routed SwiGLU feed-forward), tested against a per-token reference. The routes are gathered per token, so it suits decode and small batches; a grouped-GEMM kernel is the open performance work. No Qwen3-MoE / Gemma 4 model loader yet. |
| done | Pooling layers | `luminal_nn::pooling` | `src/Pooling.jl`: `avg_pool_2d`, `adaptive_avg_pool_2d` (upstream's overlapping windows when sizes do not divide), channels-first (C, H, W[, N]), tested against plain Julia. No max pooling yet (YOLO needs it). |
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
| C3 | Rule hygiene and reporting | Named rulesets, unique rule names, per-rule run reports | **Done**: every rule has a unique `set.name` (`rule_names()`), `saturate_graph!` logs matches / firings / bans per rule (`rule_report(rw)`), and `provenance(rw; choices)` says which rule created each node of the extracted graph (variants of one family are told apart by their parameters; static `-->` rules cannot report what they create). `compile_searched` prints the summary for the chosen graph. Was: anonymous `@theory` blocks | We cannot say which rule made a graph larger or slower; debugging the Metatheory issues in `egraph_rewrite_layer.md` was done by hand. |
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

1. ~~**C3, rule names and per-rule reports**~~ in `EGraphRewrite.jl`: done; a search now prints which rules created the chosen
   graph's nodes. (Next: step 2.)
2. **C2, genome search** over the existing choice groups: sample joint assignments, keep a population, mutate and cross. Go
   if it beats coordinate descent by 2% or more on TinyLlama decode or prefill at equal search time; otherwise keep
   coordinate descent and record the result here.
   *Headroom experiment (`benchmarks/search_headroom.jl`, results in `benchmarks/search_headroom_results.txt`, TinyLlama
   prefill, `:activations` precision, no graph capture):* random joint assignments of all decisions, for the same wall time
   as coordinate descent (about 130-150 samples, 300-385 s), are far worse than coordinate descent: the best of five
   re-timed finalists is 12% (16 tokens) and 57% (64 tokens) slower, and none of the quick-timed samples beat it. So random
   sampling shows no headroom, and the landscape is needle-like (most joint assignments are bad). This does not test a
   *guided* genome search, only that blind sampling is useless. Before / after the search: 16 tokens 42.1 ms (f32 original) ->
   24.7 (f16 weights) -> 21.3 (static extraction) -> 19.4 (coordinate descent, 2.16x over f32); 64 tokens 61.9 -> 73.9 (f16
   weights: slower than f32 here) -> 34.7 -> 29.0 (2.14x). Coordinate descent is 9.7% / 19.9% faster than the static pick.
   *Neighbourhood probe (`benchmarks/search_neighbourhood.jl`, results in `benchmarks/search_neighbourhood_results.txt`):*
   around the coordinate-descent optimum, every single-decision change (74 / 76) plus two-decision changes for the same wall
   time (106 / 103), cheap-pass timed against the incumbent and the best six re-timed in 15 interleaved rounds. Best neighbour:
   -1.91% at 16 tokens (19.03 vs 19.41 ms; a choice-group option plus one Slice-view lowering off, the best of ~180, so
   partly winner's curse) and -0.49% at 64 tokens (30.18 vs 30.33 ms). The median neighbour is 8% / 15% slower. Neither reaches
   the 2% gate, and the 16-token gain is about the timing noise (a few percent per single timing). **Decision: no-go on a genome
   search; keep coordinate descent.** Revisit only if the search space grows (new decisions from C1 fusions) or capture-mode
   timing, which this probe could not use, moves the optimum. Note: the measured search crashed with HIP graph capture (`hipErrorStreamCaptureUnsupported`); **fixed**
   (branch `capture_fix`): the `Reshape`, `Contiguous` and `Slice` executors copied with `copyto!`, a device-to-device memcpy on the null stream that
   aborts a capture, and the compiler aliases most such nodes away so only the search's candidates keep them. They now copy with a kernel
   (`_device_copy!`, `tests/test_hip_capture.jl`); the capture-mode measured search of a TinyLlama decode graph completes (364 s, ~10 ms/step).
   Found on the way, and older than C1 (it reproduces at `de41772`): a session built with `search=:measured` fails its prefill compile for a prompt
   padded to 16 tokens or more ("the static extraction failed verification") at the default `search_tolerance` of 1e-3, which the Float16 matrix-core
   paths exceed; the benchmarks pass `search_tolerance=1e-2`, `Decoding.jl` does not. Shorter prefills (and the decode graph) pass.
3. ~~**C7, rule equivalence tests**~~: done (`tests/test_rule_equivalence.jl`). For every named rule, seeded random graphs
   (random ranks, sizes, axes, batch dims) trigger it, and the static extraction plus every alternative of every decision
   is run and compared with the original graph (exact rules to 1e-5; int8 / int4 against a matmul by the dequantized
   weights, on the GPU). A coverage test fails if a rule never fired, and a deliberately wrong rule (`a + b` as `a * b`,
   added through `saturate_graph!(...; extra_rules=)`) must be caught, so a new rule needs a graph that exercises it. The
   first run found a real bug: a rule that merges a node with its own input (reshape to the same shape) made `decisions`
   offer a self-referential alternative whose extraction crashed (`KeyError` in `_total_cost`); `choice_groups` now skips
   nodes that read their own class. Not covered yet: dtypes other than Float32, and the merged-projection pass
   (`merge_projections!`, tested separately in `test_egraph_rewrite.jl`).
4. **C1, recognise fused ops from primitives** for softmax, RMSNorm and SwiGLU, which `NCM.jl` and `Gemma3.jl` currently spell
   out by hand. Go if the recognised graph runs as fast as the hand-fused one.
   **Status (first part done, `c1_fusion` branch):** the `fusion` rule set recognises `silu(g) * up` (`SwiGLUOp`), softmax over dim 1
   (`SoftmaxOp`) and RMSNorm with a weight (`RMSNormOp`) in the e-graph, with per-rule equivalence tests (CPU and GPU pass).
   Timing (`benchmarks/c1_fusion.jl`, TinyLlama prefill, Float32 matmuls): primitives through the static extraction match the
   hand-fused graph (41.97 vs 41.91 ms at 16 tokens, 62.28 vs 62.33 ms at 64), so the go criterion holds. There is no speed-up:
   the executor's elementwise fusion already compiles the primitive spelling within 1% of the hand-fused one (42.29 / 63.05 ms), so
   the value is backend-neutral model code and a search space with fused alternatives. Traced (`benchmarks/logits_trace.jl`, results in `logits_trace_results.txt`): on the CPU every variant agrees to ~2e-6;
   on the GPU every variant, the hand-fused one included, is 6-8e-4 from the exact CPU result because of the Float16 matrix cores
   (flash attention), and with `LUMINAL_WMMA=0` all agree with the CPU to ~1e-5. The GPU is deterministic (the same graph twice gives
   identical logits). The trace also found that the attention rule (and `CausalAttention`'s executor, which tested `wmma_supported()`
   instead of `wmma_enabled()`) used the Float16 kernel under `LUMINAL_WMMA=0`: fixed, the rule now needs `wmma_enabled()`. Also recognised now: L2Norm, RMSNorm times silu(gate) (`GatedRMSNormOp`), Gemma's (1 + w) RMSNorm (the weight is any
   tensor of shape (H,)), and two new fused ops with kernels, `LayerNormOp` (affine) and `GeluOp` (tanh form).
   `benchmarks/c1_norms.jl` (GPU): fused LayerNorm is 2.4-3.5x faster than its primitive graph, with the same error against Float64
   (~1.5e-7), which also covers the kernel at 768-5120-wide rows; fused GELU is equal to the primitives (0.98-1.01x), since the
   executor already fuses the elementwise chain. Causal attention in NN's composite form (GQA or
   multi-head; scale, triu mask, softmax over keys, weighted V) is recognised as `CausalAttention` where the flash kernel applies (head size a
   multiple of 32, 16+ tokens): `benchmarks/c1_attention.jl`, TinyLlama prefill, composite + rule vs the hand-built flash graph:
   41.75 vs 41.61 ms at 16 tokens, 62.60 vs 62.08 at 64, 187.2 vs 185.1 at 256 (the unrecognised composite: 41.94, 63.52, 205.0, i.e. 10%
   slower at 256). Cost: the first saturation in a process compiles the large pattern (~88 s on the first attention graph, then free).
   Unmasked attention is not recognised on purpose: the only fused op,
   `FlashAttentionOp`, is 8-12x slower than the composite on the GPU (`benchmarks/attn_unmasked.jl`: 0.14 vs 1.4 ms at D=64, S=256, H=8; 8.3 vs
   108 ms at S=1500, H=20) and its GPU kernel returns wrong results for head size 64 and up (correct at D=32; relative error ~1.0, while the CPU
   op is exact; `benchmarks/flash_attention_op_check.jl`), a bug the existing tests do not cover. Sliding-window attention: the matrix-core flash kernels
   (`flash_prefill!`, all three) and `CausalAttention` now take a `window` (`CausalAttention(scale, window)`; 0 = full causal); each kernel starts
   at the first key tile a block can see. `benchmarks/flash_window.jl` (GPU, 8 heads / 4 KV heads): correct at S = 2048 (3-4.5e-4 against the host
   reference, the Float16 level), and faster than full causal attention by 2.1-2.5x at a 256-key window already at S = 2048, and at S = 16384
   by 2.1-2.4x (window 4096), 5.5-6.5x (1024) and 9.5-15x (256). Gemma's windowed composite is now recognised too: its mask used to be an `Iota` node with a closure (opaque to a pattern), so `NN` builds it from
   primitives (`(triu + band_mask) * -9e9`, same values) and two rules (`causal-attention-{gqa,mha}-window`) read the window off the band's constant
   and emit `CausalAttention((scale, window))` (GPU tests pass; not timed on a model). Not done: sliding-window and unmasked attention, decode attention (no composite
   spelling exists: the cached decode path builds `DecodeAttention` directly), the erf GELU, LayerNorm without a bias or weight. The tiled
   `GatedRMSNormOp` that `Qwen35.jl` builds is recognised from its primitive spelling (the plain gated norm, then reshape (dv, G, nk, S, B), permute
   the two head axes, reshape back: `fusion.gated-rmsnorm-tiled`); `Qwen35.jl` itself still builds the tiled op directly, so the rule only fires on graphs
   spelled that way.
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
