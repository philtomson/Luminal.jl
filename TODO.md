# TODO

## Kernel fusion for DiffLUT-Net training: crossing fusion barriers

Status: **profiled 2026-10-02 — closed for the LutKOp design** (see the
measured checkbox below: host 2%, graph 74%, mapping-Adam 21%; the tree is
already fused and the remaining cost is the learnable-mapping complex, which
barrier-crossing fusion cannot touch). Kept as background for the general
compiler mechanisms. Scope: this is a **DiffLUT-Net training** item —
LLM decode is already covered by hand ops (plus the RMSNorm fusion item in
"AMD GPU performance" below), and prefill is GEMM-bound, so a general fuser is
not expected to move LLM workloads. Motivated by the DiffLUT-Net step
(~/devel/AI/DiffLUT-Network/julia): GPU utilization < 50%, the graph runs
~32-53 ms of a ~96 ms step, and the LUT tree crosses a kernel boundary at
every non-elementwise op.

What the compiler does today: `compile(fuse=true)` merges chains of elementwise
ops into generated `FusedElementwiseOp` kernels (whitelist-based; anything else
errors out of the group). Every other op is a fusion barrier — `SumReduce` /
`MaxReduce`, `GatherND`, `MatMul`, `SoftmaxOp`, custom ops — and each barrier
materializes its output to memory: a full write of the intermediate by the
producer kernel and a full read by the consumer. Movement ops (`Slice`,
`Permute`, `Expand`) are ShapeTracker views in the naive interpreter, but in the
compiled path they still bound fusion groups, and `reshape` inserts
`Contiguous()` copies when a view cannot be represented. The e-graph layer
(`EGraphRewrite.jl`, `search=:static` / `:measured`) can already rewrite and
time candidate graphs, and several hot composites are hand-fused as single ops
(`SoftmaxOp`, `RMSNormOp`, `RotaryEmbed`, `DecodeAttention`, `FlashAttentionOp`,
`CausalConv`, `DeltaRule`) — proof that crossing these barriers pays. The
strongest data point is `LutKOp` below: a hand-written composite took the
DiffLUT step from 90 s to 33 s. That argues for hand ops where a pattern is
hot; the open question is whether the *general* mechanisms below earn their
cost.

- [x] **Profile first — including the host side. MEASURED 2026-10-02, and the
  result closes this item for the LutKOp design.** Phase-timed the LutKOp
  training step on the 8060S (`bench_step.jl` in the DiffLUT-Network julia
  tree, 50 steps, HIP graph capture on). Before/after the two quick wins:
  - graph replay: 38.2 → **26.6 ms** — the mapping-backward softmax over the
    6272×8000 score matrix now uses the fused `SoftmaxOp` (`softmax1`) instead
    of the ~6-pass composite (safe: the node is created inside a vjp pass, so
    no gradient is taken through it).
  - mapping Adam: 11.1 → 11.3 ms — the fused single-kernel version is
    bandwidth-bound (1.2 GB of state traffic through shared DDR5); kept for
    launch hygiene and because it stays Float32 end-to-end (the broadcast
    version promoted to Float64 via the Float64 eps/bias constants).
  - host side: ~1.2 ms/step (2.3%) — capture already removed launch cost.
  The remaining cost is the learnable-mapping complex: the two large GEMMs
  and the softmax over the 6272×8000 score matrix. bf16-casting the score in
  the graph is a wash (the cast pass cancels the halved GEMM read; a real win
  would need a bf16-output Softmax kernel in Luminal — possible future op).
  Further DiffLUT speed would come from mapping-side changes: top-K sparse
  mapping (shrinks the 50 M-entry score matrix), bf16 mapping state, or fewer
  candidate inputs per pin.
- [ ] **Step 1 (cheap, do first): let elementwise fusion absorb movement ops
  as views.** When the fusion walker hits a `Slice`/`Permute`/`Expand`,
  propagate the output coordinates through the view into the producer's access
  expressions instead of stopping (ShapeTracker's `indexes` machinery already
  encodes this mapping; `reshape`'s inserted `Contiguous()` copies are the
  case to handle). No in-kernel reductions — this alone removes most barriers
  in the DiffLUT LUT tree, where every level is slice+mul+add. Much cheaper
  than the full mechanism below because the kernel generator keeps emitting
  straight-line elementwise code; only its read expressions gain view
  indexing. Estimate: days to a couple of weeks, dominated by correctness
  testing of the view-index propagation (strided views, views over views,
  `Contiguous()` interactions).
- [ ] **Step 2 (Mechanism B): composite-pattern rewrites / FusedReduceOp.**
  Add e-graph rewrite rules that map recurring multi-op composites onto fused
  op nodes: elementwise-chain → `SumReduce` pairs onto a generated
  `FusedReduceOp` (workgroup per output, reduced axis evaluated in registers,
  no materialized intermediate); known composites onto the existing hand ops
  where they are not already constructed directly. The e-graph machinery
  (rules, static cost model, `:measured` timing, numeric candidate
  verification within tolerance) already exists — new rules are additive.
  - [ ] Generic generated fused-reduction kernel: correct across ranks, dtypes,
    non-contiguous views, size-1 dims and symbolic dims; workgroup/tile policy
    per reduction shape (can start from `_reduce_dim_kernel!`'s shape logic).
  - [ ] Cost-model entries for the new forms so `:static` search can choose
    them; `:measured` verifies the win on-device.
  - Estimate: realistically 2-4+ weeks — the kernel generator is the smaller
    half; verification across ranks/dtypes/views/symbolic dims is most of the
    cost (see Verification below — it applies to B too, not just to the
    big mechanism).
- [ ] **Step 3 (stretch, conditional — not a planned deliverable): full
  index-transparent barriers with in-kernel reductions.** Treat reductions as
  coordinate algebra too: workgroup-per-output loop kernels that evaluate the
  whole producer chain inline; runtime-coordinate gathers stay barriers,
  constant/affine-coordinate gathers become address arithmetic.
  **Condition:** only if the profile shows the barrier pattern still dominates
  after Steps 1-2 AND another workload exhibits the same pattern — no such
  workload is identified yet (LLM decode is covered by hand ops; prefill is
  GEMM-bound). The `LutKOp` result argues for hand ops where a pattern is
  hot, which Step 1 + this section's hand ops already deliver.
  - [ ] Coordinate-propagation through views in the kernel generator
    (reuse/adapt ShapeTracker rather than inventing a second algebra).
  - [ ] In-kernel loop nest emission for reductions, with boundary/tile
    handling; workgroup sizing policy.
  - [ ] Interaction audit: CSE keys for fused nodes, `fold`, HIP graph capture
    (captured replays bake shapes/addresses — fused kernels must be
    capture-safe), `free_intermediates` buffer sizing (fewer intermediates).
  - Estimate: 1-2 months if pursued. A mini tensor-compiler kernel-authoring
    layer; the naive interpreter must keep producing identical results.
- [ ] **Verification (all steps — this is most of the cost of each).**
  - Always-on numeric verification of fused vs unfused subgraphs in the test
    suite (reuse the `search_tolerance` checking machinery), CPU + AMD GPU,
    Float32/Float16/BFloat16, edge shapes (rank-1, size-1 dims, views over
    views).
  - Golden-output regression for the model zoo tests (Qwen35/TinyLlama/
    Whisper components) with `fuse=true` vs `fuse=false` compiled runs.
  - `free_intermediates=true` vs `false` equivalence, and HIP graph capture
    replay equivalence.
  - Perf gates: `rocprofv3` launch counts + step times on the profiled
    workloads; no regression outside noise on paths that don't match the new
    patterns (the hand ops must keep winning wherever they exist — new rules
    should not fire on subgraphs already covered by them).
- [ ] **Risks.**
  - Miscompilation of generated kernels (view-index propagation errors, tile
    boundaries, symbolic dims) — mitigated by the verification harness plus
    randomized shape/property tests.
  - Performance regressions where a generic fused kernel replaces a tuned hand
    op — cost-model-gate or suppress rules where hand ops exist.
  - Compile-time blowup from e-graph saturation on llama-scale graphs —
    iteration caps; search stays on-demand only.
  - Two sources of truth for fusion (hand ops vs discovered fusions) — write
    down the policy: hand ops stay preferred; rewrites target composites with
    no hand op.
  - Payoff is bounded by the host/graph split: fusion only shrinks the graph
    portion of the step, and payoff is concentrated in the DiffLUT training
    workload; GEMM-bound prefill gains little. Measure per workload before
    and after (see "Measurement practice" below).

## DiffLUT LutKOp: fused K-input LUT layer kernel (hand-written composite)

Status: DONE (implemented, tested, upstreamed here). `LutKOp` /
`LutKBwdPinsOp` / `LutKDWOp` in `src/Ops.jl` with Val-unrolled register
kernels in `src/Execution.jl`, vjp rules in `src/Autograd.jl`, and the
`lut_layer` constructor in `src/HighLevelOps.jl`; tests in
`tests/test_lutk.jl` (forward vs brute-force basis, gradients vs independent
basis-derivative references, both gathered and indexed input forms, CPU +
GPU + compiled). Measured on the DiffLUT-Net MNIST step (2000/1000, K=4,
batch 100): 500 iterations + eval in 33 s (was 90 s composite, 58 s through
Zygote), verified bit-exact against PyTorch. GroupSum remains separate
(Reshape + SumReduce).

- [x] **`LutKOp` forward** (`Ops.jl` + `execute_op!`): one work-item per
  (batch, neuron); the neuron's `2^K` truth-table entries in registers
  (16 for K=4, 64 for K=6); input pins read directly via the per-neuron index
  constants (fixed connections: the gather disappears into the kernel; learnable
  mapping stays a separate op feeding it). Interpolation tree in registers —
  the DiffLUT-Network julia port is the reference implementation.
- [x] **`LutKOp` backward**: `dW` kernel (per (batch, neuron), all `2^K` entry
  contributions — the tree path products, same cost shape as forward) and
  pin-gradient kernel with atomic scatter (mirror difflogic_kernel.cu's
  CSR `backward_x`/`backward_w` scheme). `vjp_rules` entry; the mapping
  surrogate gradient (softmax + two rocBLAS GEMMs) stays as it is.
- [x] **Verify** against the DiffLUT-Network julia port (it has a PyTorch
  bit-exact parity harness to reuse) on CPU + AMD GPU; then upstream like
  `MappingGather`.
- Expected: kernel count per training step from hundreds to ~a dozen; per-layer
  traffic from "every tree level round-trips a (B, out, m) intermediate" to
  "read x once, write out once"; 2-4x on the ~32-53 ms graph execution.
  Estimate: 300-400 lines including tests.

## DiffLUT compact top-K mapping router (SparseMapGather)

Status: DONE (implemented, tested, measured 2026-10-05). `SparseMapGather` /
`SparseMapBwdS` / `SparseMapBwdX` in `src/Ops.jl` with kernels in
`src/Execution.jl`, vjp rules in `src/Autograd.jl`, and the
`sparse_map_gather` constructor in `src/HighLevelOps.jl`; tests in
`tests/test_sparsemap.jl` (forward vs host gather, gradients vs host
references, interpreter + compiled CPU/GPU + HIP-graph capture). This is the
"top-K sparse mapping" the profile above pointed at: after a dense warmup the
(DiffLUT-side) switch keeps the top-k candidate inputs per pin as a frozen
(C, k) id table plus a (C, k) trainable score matrix, replacing the
(in, C) ≈ 50 M-entry score matrix — forward argmax is unchanged by
construction (the kept set contains it), backward trains the kept scores
straight-through and scatters softmax-weighted gradients to x.

- [x] Measured on the DiffLUT-Net MNIST step (6272→2000→1000, K=4, batch 100,
  k=16, HIP capture on): **35.5 → 10.2 ms/step (3.5×)** — graph replay
  26.1 → 9.3 ms, on-device mapping Adam 8.1 → 0.15 ms (state 55 M → 192 K
  entries), host prep 0.63 → 0.22 ms (per-step `column_argmax` gone).
- [x] End-to-end accuracy (same seed/tau as the dense+prune baseline):
  20000-step MNIST run, dense warmup 3000 steps then compact k=16 —
  **97.26 % best test-eval vs 95.85 %** for the dense baseline; the compact
  run also converges faster (96.7 % by step 5000).
- [x] Capture-safety: `ScatterND`'s init copy became a KA kernel
  (`_copy_kernel!`) — `copyto!`'s null-stream D2D memcpy aborts HIP graph
  capture with `hipErrorStreamCaptureUnsupported`.
- Note: the dense→compact switch lives on the DiffLUT side
  (`luminal_compactify!` in the julia tree); the accuracy run vs the
  masked-dense baseline is recorded there.

## Qwen3.6-27B

Status: runs end to end from `Qwen3.6-27B-Q8_0.gguf` (int8 load in ~140 s, coherent
output; `examples/qwen35_gguf.jl`). The README still labels it "in progress".

- [ ] **Validate against a reference.** Only a tiny random model is checked against
  transformers (`tests/test_qwen35_model.jl`). Compare 27B logits / perplexity with
  transformers or llama.cpp to catch small errors in Gated DeltaNet and partial RoPE.
  Do this first.
- [ ] **Decode performance.** No numbers recorded for Q8_0 tokens/s, prefill speed or
  memory. Bandwidth floor is ~125 ms/token (~27 GB int8 at ~215 GB/s); measure how
  close the hybrid DeltaNet decode path gets (`rocprofv3`, see `debug_decode.jl`).
- [ ] **Load / startup time.** Cut the 140 s load (parallelise the loader, or cache
  converted int8 weights) and the compilation time inside `generate`.
- [ ] **Int4 and long context.** Check `--int4` quality (`examples/quant_eval.jl`
  perplexity). `--max-seq` defaults to 512; measure long-context behaviour and the
  hybrid cache's memory use.
- [ ] **Tests that don't need the 27B file.** Cover `qwen35_config(::GGUFFile)`, the
  GGUF tokenizer path (`LlamaTokenizer(::GGUFFile)`) and the MTP-layer trimming
  from commit 889c2a8.
- [ ] **Chat template / thinking mode.** The example does raw completion. Check
  chat-template formatting, stop tokens and `<think>` handling.
- [ ] **Cleanup.** Update the README from "in progress", document the GGUF example,
  and commit or delete the scratch files (`test_gpu_ops.jl`, root `run*.log`).

## AMD GPU performance (ideas from ~/devel/AI/magnitude)

Magnitude targets AMD only through Vulkan (no HIP/ROCm), so these are ideas to adapt,
not code to reuse. Based on a skim of `inference-v4/.../vulkan/lib/projection/projection.glsl`
and `info/inference-tile-sizing-autotuning.md`; none of it has been tested here.

- [ ] **Matrix-core (WMMA) GEMM for prefill.** Biggest known gap: prefill GEMMs don't
  use the matrix cores. Magnitude uses 16x16x16 cooperative-matrix multiplies on
  RDNA3/4 with f16 operands, F32 accumulation, and weights decoded to f16 once per
  tile. gfx1151 has the same WMMA hardware.
  - [x] Step 1: AMDGPU.jl emits `llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16` on
    gfx1151 (wave32); a 16x16x16 product matches the CPU (err ~1e-6). Build operand
    vectors with `ntuple(f, Val(16))`; wrapping in `NTuple{16,VecElement}(...)` makes
    the kernel fail to compile (dynamic calls). Lane `l` holds row/column `l%16` of A/B
    and `D[2i + l÷16, l%16]` for i in 0..7. Note `@roc gridsize` counts workgroups.
  - [x] Step 2: tiled GEMM (f16 in, f32 out), correct (rel. err ~1e-5) at 256^3, 2048^3,
    512x4096x4096, 4096^3. Throughput: 16x16 tile per wave 1.3 TFLOPS; 2x2 tiles per
    wave 4.2; 64x64 block tile with LDS staging (128 threads, BK=32) 5.3-5.5 TFLOPS at
    2048^3 and 4096^3 (4.2 at 512x4096x4096; slower than the 2x2 kernel at 256^3).
    Baselines at 2048^3: rocBLAS Float32 `mul!` 2.85 TFLOPS, rocBLAS Float16 `mul!`
    0.47 (does not use the matrix cores). Still far below peak; likely limited by
    scalar 2-byte global loads, LDS stores and fragment reads (not profiled).
    Scripts were scratch files (not in the repo).
  - [~] Step 3 (partly done, ROCm 10.1.0): K-contiguous operands (activations [M,K] and
    weights [N,K] row-major, so each WMMA fragment is 32 contiguous bytes), 16-byte
    global loads, 16-byte LDS reads (LDS rows padded to 5 chunks = 80 B), and global
    loads of the next K tile issued into registers before the current tile's WMMAs.
    64x64 block, 4 waves of 32x32, BK=32 ("v4"): 12.4-13.4 TFLOPS at 2048^3, 11.3 at
    4096^3, 5.9 at 512x4096x4096, 6.4 at 4096x11008x4096 (2.2x the scalar-load kernel,
    ~4x Float32 rocBLAS; correct, rel. err ~1e-5). A 128x128 block with 64x64 per wave
    ("v5") was correct but slower (8.2 / 5.2 TFLOPS at 2048^3 / 4096^3), probably
    register pressure or low occupancy (not checked). The big-weight shape drops to
    ~6 TFLOPS when B (90 MB) exceeds the 32 MB on-chip cache.
    Not tried yet: block swizzle for cache locality, BK=64, split-K for small M,
    `rocprofv3` / VGPR counts to see what limits v5, 128x64 blocks, edge handling for
    M, N not a multiple of the tile and K not a multiple of 32 (currently required).
  - [x] Step 4 (Float16 weights): saved as `benchmarks/wmma_gemm.jl` (v4 kernel plus a
    comparison with rocBLAS) and wired in as `src/WMMA.jl`, called from
    `_half_matmul!` (HalfWeight, prefill N > 16) in `src/Execution.jl`. Activations are
    rounded to Float16 into a zero-padded workspace; N may be any size, M must be a
    multiple of 64 and K of 32 (otherwise it falls back); gfx11 only; set
    `LUMINAL_WMMA=0` to disable. Benchmark (ROCm 10.1.0), WMMA vs rocBLAS `gemm_ex`
    f16 vs sgemm, TFLOPS: 2048x2048x512 13.1 / 5.2 / 2.7; 5632x2048x512 12.4 / 5.8 / 3.0;
    32000x2048x512 14.6 / 6.5 / 3.2; 11008x4096x512 12.8 / 6.6 / 3.1; 4096x4096x4096
    11.1 / 6.8 / 3.2 (at 128 tokens 8.6-21 / 5.4-7.1 / 2.5-3.1). TinyLlama, 512-token
    prompt, weights stored as Float16: time to first token 298 ms, vs 1890 ms on the
    previous dequantize-then-sgemm path for stored-Float16 weights (and 505 ms for the
    default Float32-prefill session). Quality: TinyLlama Float16 perplexity 12.1226
    (Float32 12.1229), top-1 agreement 153/153, logit relerr 5.4e-4. Tests:
    `tests/test_wmma_gemm.jl`; `test_quantized_loading.jl` relaxes its Float16 GPU
    prefill tolerance to 1e-3 (activation rounding, ~2e-4); full suite 2150/2150.
  - [ ] Default sessions are not sped up yet: `LlamaSession` keeps Float32 weights for
    prefill (plain sgemm) and only converts decode weights. Options: prefill on
    HalfWeight (changes prefill numerics to f16 activations/weights, extra weight
    memory unless shared with decode) or `weight_storage=Float16`
    (`examples/chat_benchmark.jl ... stored` measures that).
  - [x] Int8 prefill (`QuantWeight`, `_q8_matmul!` with N > 16): `_wmma_q8_kernel!`
    dequantizes each 16-int8 chunk x its group scale to Float16 while staging the A tile
    (no Float32 copy of the weight; any group that is a multiple of 16). 8-17 TFLOPS vs
    0.2-1.7 for dequantize + sgemm (10-100x). TinyLlama int8 stored, time to first token:
    74-87 ms for 128 tokens (was ~1070 ms), 251 ms for 512 tokens (was 1436 ms); decode
    unchanged (TPOT noise 5.8-6.8 ms with other jobs on the machine). Perplexity 12.1552
    (was 12.1526; Float32 12.1229), top-1 agreement 149/153 as before.
    `benchmarks/wmma_gemm.jl` now calls the kernels in `src/WMMA.jl` and covers int8.
    Not yet tried on the Qwen3.6-27B Q8_0 GGUF (group 32; needs ~27 GB free, so check
    memory with the other jobs first).
  - [x] Int4 prefill (`Q4Weight`, `_q4_matmul!` with N > 16): `_wmma_q4_kernel!`; a K tile
    is one 16-byte chunk per row (two threads share it, each dequantizing 16 codes as
    q*scale+min, symmetric or not). 10-19 TFLOPS vs 0.2-1.9 for dequantize + sgemm.
    TinyLlama int4 stored, time to first token for 512 tokens: 241 ms (was 1440 ms);
    decode unchanged. Perplexity 13.5665 vs 13.5656 on the old path (Float32 12.1229),
    top-1 agreement 126/153 either way (`examples/quant_eval.jl` now has an int4 row;
    note `quant_formats.jl` fake-quantizes to Float32 and does not exercise the kernels).
    Full suite 2164/2164.
  - [ ] Try the Qwen3.6-27B Q8_0 GGUF (group 32): memory (~27 GB) with other jobs running.
  - [ ] Fused residual in the kernel epilogue (now a separate `C .+= residual`), M and K
    tails, gfx12 (different WMMA intrinsic), CUDA/other devices untouched.
- [ ] **Fuse RMSNorm into the following projection for decode.** Magnitude applies the
  norm while staging activations, so norm + projection is one launch. RMSNorm
  currently costs 3-4 kernels x 45 layers, and launch gaps (~5.5 us) are the
  remaining decode cost. Also fuse SiLU-mul and residual-add into the output stage.
- [ ] **Per-device autotuning with a cache.** Small sweep over threads per workgroup,
  columns per pass and split-K for small-N outputs; check correctness, time on the
  device, store winners per device. Would replace hand-tuning such as `_q8_threads`.
- [ ] **Cross-check the GEMV/GEMM split.** Magnitude switches kernel class at M <= 8
  (decode weight packet decoded once, applied to every activation row). Compare with
  Luminal's multi-column int8 GEMV, which is limited to 4 columns per pass; see
  whether B > 4 improves.
- [ ] **Split-K for small-N projections** (relevant to the smaller 27B projections).
- [ ] **Measurement practice.** Measure each component, then re-measure the whole
  workload; a layer-0 saving does not extrapolate (keep using `rocprofv3` and
  `debug_decode.jl`).

## Long context for Qwen3.6-27B (262,141 tokens)

Goal: load Qwen3.6-27B at its full 262,141-token context and measure prefill and decode.
Memory at full context: int8 weights ~28 GB + K/V cache of the 16 full-attention layers
(2 x 4 kv heads x 256 x 262,144 positions) = 34 GB in Float32, 17 GB in Float16.

- [x] **Float16 K/V cache** (`LlamaSession(...; kv_dtype=Float16)`,
  `LlamaKVCacheState(...; kv_dtype=)`, `build_llama_decode_step!(...; kv_dtype=)`): the
  recurrent states stay Float32; the decode kernel reads/writes it as Float32. The decode
  graph must declare the cache inputs with the cache's dtype: the executor converts any
  input to its declared dtype, so a Float16 cache under a Float32 declaration was silently
  replaced by a Float32 copy and the new K/V never reached it (logits drifted after the
  first decode step). TinyLlama: identical text, logit relerr ~1e-4; test in
  `tests/test_qwen35_model.jl`.
- [ ] **Decode attention is capped at 8,191 positions** (`DECODE_ATTN_MAX_CTX = 8192`: the
  per-head score buffer is workgroup-local memory). Longer caches raise an error, so full
  context needs a chunked (flash-decoding) kernel: per-chunk partial max/sum/output, then a
  combine pass; keep `write_cache`, `window` and batch semantics; test against the current
  kernel.
- [ ] **Long prompts in prefill**: check that Qwen3.5 prefill attention does not build
  slen x slen scores per head (FlashAttentionOp?), and the other prefill intermediates.
- [x] **Memory while loading / compiling / running the 27B** (MemAvailable, 125 GB machine,
  `examples/qwen35_gguf_bench.jl`, Q8_0 GGUF converted on load, Float16 K/V, max_seq 8191).
  Int4 weights: start 76 GB available -> after load 55 GB (the model takes ~21 GB, with a
  transient low of 42 GB during the load, ~12 GB above the steady state) -> compiling the
  prefill and decode graphs costs ~0.1 GB (nothing is allocated until a graph first runs) ->
  the FIRST 512-token prefill takes another ~15 GB (55 -> 40 GB) -> later calls add ~1.6 GB
  once, then ~0.1-0.2 GB each (38 GB, flat): no unbounded growth at int4. Everything is
  returned when the process exits. Int8 earlier: 71 GB -> 36.5 GB after load (~35 GB) -> 20 GB
  after the first prefill, then the kernel OOM killer ended the run during the repeat calls,
  although growth per call at int4 is tiny, so that kill is not explained (other jobs on the
  machine also draw on the same RAM). Use int4 and a watchdog for experiments; int8 needs
  ~15 GB more headroom. Open: why does the first prefill need ~15 GB at 512 tokens (graph
  intermediates are kept: `free_intermediates` is off), and does it scale with prompt length?
- [~] `examples/qwen35_gguf_bench.jl` (stage-by-stage memory log, in-process watchdog,
  `--stop-after`, `--reclaim`): first numbers, Qwen3.6-27B int4 (g32), Float16 K/V, 512-token
  prompt, WMMA prefill: time to first token 3.02 s = 170 tok/s prefill (~9 TFLOPS effective);
  (CORRECTED, see the decode profile below: the 165 ms/token reported here included the one-time HIP graph
  capture of the first decode call; steady-state decode is 79-84 ms/token = 11.9-12.6 tok/s.)
  Not yet: longer prompts (2k, 4k, 8k), int8, WMMA on/off comparison, output sanity check.

## Qwen3.6-27B int4: speed plan (what gufo shows)

Reference: `~/devel/AI/gufo`, a Strix Halo (gfx1151) inference engine. Same chip, same model class
(its numbers are Qwen3.8-27B, UD-Q4_K_XL = 16.35 GiB of weights; ours is Qwen3.6-27B requantized
to int4 g32, ~14 GB; not identical, but close enough to compare). Its published AR numbers
(`docs/models/qwen3.8-27b/BENCHMARKS.md`): Q4 prefill 656 tok/s, decode **12.4 tok/s**
(llama.cpp 12.1); Q8 485 / 7.2. Ours (int4, 512-token prompt, measured with a decode warm-up): prefill 170 tok/s, decode 12.0 tok/s.

- **Decode target.** 12.4 tok/s on 16.35 GiB is ~215 GB/s, the memory ceiling, so *autoregressive* 15 tok/s
  is at the wall even for our smaller weights (14 GB / 215 GB/s = 65 ms = 15 tok/s only at 100%).
  Realistic AR goal: match gufo, ~11-12 tok/s (2x today). Above that needs **speculative decoding**:
  gufo reaches 26 tok/s (ordinary text) and 63 (repetitive) with a DFlash2 draft model, and has an
  MTP path (`src/models/qwen/hip/mtp`, `dflash`). Our GGUF-MTP dir has the MTP head we currently drop.
- **Decode: already ~12 tok/s (see the profile below); beyond that: fewer bytes, fewer launches, speculative decoding.**
  gufo's rejected-experiments list (`docs/models/qwen3.8-27b/EXPERIMENTS.md`) warns that norm/residual
  fusion, graph replay and prefetch variants did not survive end to end at the wall. Cheap thing they found:
  mapped/huge-page weights +2% (copied weights 3-8% lower cold-weight throughput than mapped).
- **Prefill: 170 -> ~650 tok/s.** gufo's prefill GEMM is W8A8: activations quantized to int8 blocks (Q8_1
  style), weights expanded to int8, `wmma_i32_16x16x16_iu8`; it measures a 55 TOPS int8 WMMA ceiling and
  reaches 58% of it (~32 TOPS) with a 128x128 block tile, 8 waves (256 threads), BK=32 LDS stages, and
  fused epilogues (SwiGLU + quantize the next activation; residual + RMSNorm + quantize)
  (`src/models/qwen/hip/kernels/prefill_quant_gemm.hip`). Ours: f16 WMMA at 12-18 TFLOPS (~9 TFLOPS
  effective over the whole model). Experiments, in order: (1) measure the f16 vs int8 WMMA peaks on this
  chip with a microbenchmark (`tools/bench/gfx1151_peak.hip` is the model); (2) an int8 (iu8) variant of
  our kernel for the int8/int4 weights with quantized activations, checking quality (W8A8 changes the
  numerics: perplexity); (3) 128x128 blocks with 8 waves.
- **Prefill memory: chunked prefill into a preallocated arena (the answer to "why 15 GB").** Measured on
  the 27B: during a 512-token prefill every intermediate stays live until the call ends (live bytes rise
  by the sum of all allocations: +16 GB, linear in the prompt), `_release!` frees buffers but the
  reshapes/views made by `realize_view` share the buffer (refcounted) and keep it alive; after the call
  the HIP pool still holds ~12.5 GB reserved and `hipMemPoolTrimTo` returns only a little of it
  (reserved 34.6 GB vs live 22.1 GB). gufo allocates all model/KV/recurrent/graph/scratch before
  serving (`arena.cpp`) and prefills in fixed chunks (`prefill_chunk.cpp`), so memory is constant in
  the prompt length. For Luminal: compile the prefill graph for a fixed chunk (e.g. 512 tokens), run it
  per chunk with the K/V cache and recurrent states carried (needs attention over the cache for chunks
  after the first), and keep the scratch resident. That also unlocks long prompts (8k now, 262k later).
- **Quality of the weights.** gufo reads the native GGUF K-quants (Q4_K / Q5_K / Q6_K mixes) directly;
  we requantize Q8_0 to int4 g32 symmetric, which costs perplexity (+7% on TinyLlama). A K-quant loader
  would give better quality per byte (17.5 GB vs our ~14 GB, so decode would be ~20% slower at the wall).
- **Not worth it (per gufo's experiments):** more attention-kernel tuning for decode at short context,
  two-layer fusion of norm into projection, graph capture variants.

### Decode profile of the 27B (int4, measured 2026-10-09, `rocprofv3 --kernel-trace`, 16-token prompt, 22 decode steps)

Steady-state decode is **79-84 ms/token = 11.9-12.6 tok/s** (matches gufo's AR 12.4); the earlier 6.1 tok/s
came from timing the first decode call, which captures the HIP graph (seconds). Per step: ~1,750 kernel
launches, GPU busy 91% of the step window (~77 ms), kernel time ~70 ms:

| kernel | per step | share |
|---|---|---|
| `_q4_matmul_kernel_` (int4 GEMV, ~475 launches) | 62.0 ms | 88.7% |
| `_delta_rule_kernel_` (48 linear-attention layers) | 3.2 ms | 4.6% |
| `broadcast_kernel_cartesian` (unfused elementwise, ~550 launches) | 1.6 ms | 2.3% |
| `_rmsnorm_kernel_` (~200 launches) | 1.5 ms | 2.2% |
| decode attention, linear broadcasts, causal conv, reduce, copy, rotary | 1.6 ms | 2.3% |

The GEMV streams ~14 GB in 62 ms = ~226 GB/s, at the memory wall (gufo: ~215 GB/s). So 15 tok/s (67 ms)
AR needs the ~15 ms outside the GEMV (8 ms small kernels + ~8 ms launch gaps between ~1,750 launches)
cut to ~5 ms AND the GEMV unchanged, which is a stretch; the alternatives are fewer bytes (int4 g64:
-5.5% of the GEMV = -3.4 ms; fewer weights read per token) or speculative decoding (gufo 26-63 tok/s).
Cheap next steps: fuse the ~550 elementwise broadcasts + 200 RMSNorm launches per step (~3 ms of kernel
time plus their launch gaps); DeltaRule (3.2 ms, 48 launches/step = 66 us each) looks slow for its work.
Also: prefill of <= 16 tokens (`QUANT_GEMM_COLS = 16`) takes the multi-pass GEMV path (16-token prompt:
520 ms to the first token vs ~65 ms to read the weights once); the WMMA path is not used below 17 columns.

### Prefill profile of the 27B (int4, 512-token prompt, measured 2026-10-09, `rocprofv3 --kernel-trace`)

Kernel time per prefill call is ~2.9 s (TTFT 2.97 s):

| kernel | per prefill | share |
|---|---|---|
| `_wmma_q4_kernel_` (int4 GEMM, 401 launches; ~20 TFLOPS in the model, 40% of the 51 TFLOPS f16 peak) | 1,333 ms | 46% |
| `_delta_rule_kernel_` (48 linear-attention layers, ~23 ms each) | **1,102 ms** | **38%** |
| broadcast (unfused elementwise, 1,200 launches) | ~137 ms | 5% |
| rocBLAS f32 GEMM `Cijk_...MT32x32x8` (129 launches) | ~104 ms | 4% |
| `fill_kernel` (zero_tensor, ~2,000 launches), `linear_copy` (640), `permutedims` (190) | ~160 ms | 5% |
| reduce, rmsnorm, causal conv, rotary | ~45 ms | 2% |

WMMA peaks measured on this chip (20 CUs, register-only loops, `llvm.amdgcn.wmma.*`): **f16 51-52 TFLOPS,
int8 (iu8) 55-57 TOPS**, i.e. int8 is only ~8-10% faster than f16 (gufo measures 55 TOPS int8 too). So int8
prefill's value is lower operand traffic (half the LDS/register bytes, per-block scales applied after the
integer MMA), not a 2x MMA rate; the GEMM gap is blocking: ours reaches 25-40% of peak, gufo's W8A8 kernel
58% (128x128 block, 8 waves).

**The DeltaRule kernel is the biggest single item and is cheap to fix.** `_delta_rule_kernel!` launches one
workgroup per (value head, batch), 48 workgroups on a 20-CU GPU, and every token step reads and writes the
head's 128x128 Float32 state (64 KB) in global memory (`st[j, i, h, b]`, ~45 us per step). Thread j owns one
column of the state, so those 128 floats can stay in registers for the whole sequence (state loaded once,
written back at the end); per-step work is ~400 FMAs per thread from registers plus the shared q/k vectors.
Expected: ~23 ms -> well under 2 ms per layer, i.e. ~1 s off a 3 s prefill (and the same state-traffic
saving, ~1 ms, on each decode step). For long prompts a chunked (WY / matrix) form would parallelize over
the sequence as well; gufo's `prefill_ssm.hip` and `batched_ssm.hip` are the references.

Order of work for prefill (largest first): (1) DeltaRule register-resident state (~1.0 s); (2) GEMM blocking
toward ~58% of peak, optionally W8A8 (~0.5 s); (3) the ~0.3 s of small kernels: stop zero-filling outputs
that are fully overwritten (`fill_kernel`), fuse the broadcasts, avoid the copies/permutes.
