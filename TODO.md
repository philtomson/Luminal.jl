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

### DeltaRule prefill kernel (2026-10-09)

`_delta_rule_reg_kernel!` (AMD GPUs; `LUMINAL_DELTA_REG=0` selects the old global-memory kernel): the state
stays in registers for the whole sequence, two lanes per state column (64 key rows each, partial sums
combined with one `shfl_xor`), q/k staged in LDS 16 steps at a time. At the 27B's head sizes
(dk = dv = 128, 48 value heads, 512 tokens) one layer drops from 23 ms to 1.5-1.7 ms (12-15x); results agree
with the old kernel to 4e-7 relative on L2-normalized q/k (unnormalized random keys make the recurrence
unstable and amplify rounding, so don't test with them). Pitfalls hit on the way: a single thread holding
all 128 state values plus temporaries needed more than the 192 VGPRs a kernel gets by default (AMDGPU.jl
cannot set the flat-work-group-size attribute) and spilled ~4 KB/thread to scratch, which kept it at 1.8x;
a KernelAbstractions kernel lost the types of its Val parameters, so it is a plain `@roc` kernel; `foldl`
over a 128-tuple is not unrolled (`_seq_dot` is a generated function). Test: `tests/test_delta_rule_gpu.jl`.
Measured on the 27B (int4, 512 tokens, 2026-10-09): time to first token 2.97 s -> 2.07 s (172 -> 248 tok/s,
1.44x); decode unchanged (11.4-12.3 tok/s). What is left of the 2.07 s: the int4 WMMA GEMM ~1.3 s (64%), the small
kernels ~0.45 s, DeltaRule now ~0.1 s. Next: GEMM blocking, the small kernels, and prompts of <= 16 tokens
(507 ms to the first token for a 16-token prompt: below `QUANT_GEMM_COLS` the weights are read several times).

### GEMM blocking and small kernels (2026-10-09)

Measured on the 27B (int4, 512 tokens): time to first token 2.07 s -> **1.38 s** (248 -> 370 tok/s; gufo ~0.78 s);
16-token prompt 507 -> 462 ms; decode unchanged (11.1-12.3 tok/s, run-to-run noise from the other jobs).
Three changes:
- **int4 GEMM** (`_wmma_q4_big_kernel!`, used when M % 128 == 0 and the padded N % 128 == 0, i.e. N >= 192 gets
  padded to 128): 128x128 block, 8 waves of 32x64, so each dequantized weight tile feeds twice the columns;
  dequantization in packed Float16 (code ORed into the mantissa of 1024.0, one exact subtract, one packed fma with
  scale and min; ~7% of the kernel time left). At the 27B's shapes 19 -> 27-28 TFLOPS (1.4x; f16 peak 51).
  Tried and dropped: LDS double buffering (one barrier per tile; no gain) and BK = 64 (slower, ~10%).
  `LUMINAL_WMMA_BIG=0` selects the 64x64 kernel. Still ~55% of peak; the remaining gap is not VGPRs (168-192 is
  fine for occupancy on RDNA3.5) or the dequantization. Next ideas: gufo's W8A8 int8 kernel, a B tile staged once
  per 2 K tiles, `v_perm`-based nibble expansion, direct-to-LDS loads.
- **Logits only at the last position** (`LastToken` op, Qwen35 `last_only=true`; `NN._prefill_last_only`): prefill
  computed the 248k-vocab lm_head for all 512 positions (a 48 ms GEMM, 0.5 GB of logits copied to the host,
  another ~45 ms) and used one column. -90 ms, and 0.5 GB less scratch. Other models still return all positions.
- **No zero fill for fully-overwritten outputs** (`_writes_all`, `Luminal.uninit_tensor`): elementwise groups,
  reshape/permute/contiguous, rmsnorm, rotary, softmax and LastToken skip the `fill_kernel` launch (~2.5% of prefill).
- `--layers=N` in `qwen35_gguf_bench.jl` runs the first N layers of the real model (3 DeltaNet + 1 attention at
  N = 4): the same kernel mix at ~1/16 of the time and 2 GB, so it can be profiled with `rocprofv3` safely
  (a profiled run of the whole 27B tripped the memory watchdog at 35 GB of HIP pool).
Profile of the 4-layer slice after these (x16 for the 27B): GEMM ~70% of the kernel time, rocBLAS f32 attention GEMMs
(`MT32x32x8`, 8 launches of ~0.7 ms per full-attention layer at ~2 TFLOPS) ~6%, DeltaRule ~4%, the rest is
~40 small kernels per layer (4-D broadcasts, `linear_copy`, permutes, reduces) at 1-3% each.
Next: a flash-style prefill attention (WMMA, GQA, causal) to replace the f32 sgemm pair; find which `linear_copy`
and 4-D broadcast launches remain (45 copies per 4 layers); the 16-token path.

### Fused causal prefill attention (2026-10-10)

`CausalAttention` op + `flash_prefill!` (`src/FlashAttn.jl`; `SelfAttention` uses it for prefill (`return_kv`) when
`seq >= 16`, no sliding window, head_dim % 32 == 0 and WMMA is available; `LUMINAL_FLASH=0` selects the composite
form): online softmax over 64-key tiles on the matrix cores, GQA inside the kernel, no (S, S, heads) score matrix and
no repeated K/V. Block = 4 waves = 32 query rows; both waves of a 16-row group compute the same scores and each
accumulates half of the head dimension (no exchange between waves); Q/K/V operands are read from global memory
(Float16 copies, V transposed), P goes through a per-wave LDS tile. Matches the host reference to 4e-4 (Float16
operands); logits of the 4-layer 27B slice match the composite form to 5e-6, same greedy tokens.
Speed: 0.87 ms per layer at 512 tokens (D = 256, 24 heads; the composite form was ~8 ms), 7.6 ms at 2048 (~7 TFLOPS of
useful causal work, so there is room: K/V tiles via LDS shared by the row groups, 128-row blocks).
27B (int4, 512 tokens): time to first token 1.38 s -> **1.30 s** (370 -> 394 tok/s); 16 tokens 460 ms; decode unchanged.
Prefill memory is not reduced by this (the pool still holds ~22 GB used): the score matrices were a small part of it.
Next: the ~40 small kernels per layer (copies, 4-D broadcasts, permutes), then GEMM (~70% of what is left).

### Small kernels, round 2 (2026-10-10)

27B (int4, 512 tokens): time to first token 1.30 s -> **1.21 s** (394 -> 425 tok/s); 16 tokens 462 ms; decode unchanged.
- The 48-row DeltaNet gates (`in_proj_a/b`) took the dequantize + sgemm path because M % 64 != 0 (~2 ms per layer in
  the synchronized step trace). The int4 kernels now take any M % 16 == 0 (clamped weight-row loads, guarded stores).
  Int8 / Float16 weights still need M % 64.
- Quantized/half matmul outputs with more than `QUANT_GEMM_COLS` columns skip the zero fill (`_gemm_writes_all`).
- `LUMINAL_TRACE_TIME=1` (AMD): times every graph step with a synchronize and lists the most expensive (op, shape)
  groups. It overstates tiny ops (launch latency); use `rocprofv3` on `--layers=4` for kernel time.
- `test_quantized_loading.jl`: GPU int4 tolerances loosened to 1e-2 (nearly every tiny-model matmul now rounds its
  activations to Float16; the 4-layer 27B slice agrees with the exact path to 5.7e-5, same tokens).
Profile of the 4-layer slice now (kernel time 80 ms): int4 GEMM 56 ms (69%), DeltaRule 4.9, ~15 ms in small kernels:
SwiGLU multiply 2.0 (already at memory speed: only fusion into the GEMM removes it), 44 `linear_copy` launches 3.9
(mostly the Float32 -> Float16 activation conversion in `_wmma_activations`, repeated for every GEMM that reads the same
activations: 4 per DeltaNet layer, 2 per MLP), remaining fills ~1.7, assorted 3-D/4-D broadcasts ~3.5 (l2norm, gates),
causal conv 0.7, rmsnorm 1.1, embedding gather 1.5 (once per prefill). The GEMM is now the whole story: ~27 of 51 TFLOPS.
Ideas left, by size: GEMM efficiency (W8A8 / better tiling); one Float16 conversion shared by the GEMMs of a block
(needs a graph-level cast node); gate+up as one GEMM with a SwiGLU epilogue; fuse the DeltaNet gates.

### int4 GEMM tiling, round 2 (2026-10-10)

27B (int4, 512 tokens): 1.21 s -> **1.18 s** (433 tok/s; other jobs load the machine, run-to-run noise is ~3%); on the
4-layer slice 89-90 ms -> 80.8 ms (-10%); standalone GEMM 23-26 -> 28-29 TFLOPS (+12-20% in the same run).
- `_wmma_q4_wide_kernel!`: 128 x 256 block, 16 waves (512 threads, each wave 32 x 64), used when M % 128 == 0 and the
  padded N % 256 == 0 (the executor pads N >= 384 to 256, N >= 192 to 128). The dequantized weight tile now feeds 256
  columns and all 512 threads share the dequantization (one 32-bit word = 8 codes each); LDS is double-buffered (60 KB),
  one barrier per K tile; a K-tile register prefetch queue (`WMMA_Q4_PD`, depth 1 is best; 4+ spills).
  `LUMINAL_WMMA_WIDE=0` selects the 128x128 kernel. `@inbounds` in all the WMMA kernels (cuts exec-mask code; no speed change).
- How the bottleneck was found (probe kernels, scratchpad): removing the loop's global loads gives 26 -> 40 TFLOPS but
  that also lets LLVM hoist the (then loop-invariant) dequantization, so it is mostly the dequantization ALU work
  (~28% of the 128x128 kernel; ~5% now). LDS fragment reads, barriers and B loads cost ~5% each; contiguous (repacked)
  weight loads gave nothing. All-overheads-removed ceiling of this structure is ~43 TFLOPS (51 peak).
- Not done: **W8A8 int8**. gufo's blocked W8A8 kernel reaches 58% of the int8 peak (~32 TOPS) and ours is at 57% of the f16
  peak; the int8 WMMA is only 8-10% faster than f16 and each K32 block's int32 accumulators need scale conversion
  (cvt + 2 fma per accumulator element, ~75% as many VALU cycles as the WMMAs themselves), so it should not beat the
  f16 path, and activation quantization costs accuracy. Skipped unless a probe shows otherwise.
- Remaining headroom (29 vs ~43): LDS traffic (24 `ds_load_b128` per wave per 16 WMMAs) and the 2 waits per tile.
  Larger wave tiles (64x64) need > 192 VGPRs (AMDGPU.jl cannot raise the limit).

### Short prompts: the skinny int4 kernel (2026-10-10)

27B (int4): time to first token 16 tokens **478 -> 146 ms**, 32 tokens ~420 -> **188 ms**, 64 tokens 387 ms, 512 tokens 1.18 s
(unchanged); decode unchanged. Weights alone at memory speed are ~95 ms, so 16 tokens is within 1.5x of the floor.
- Cause: prompts of <= 16 tokens (N <= `QUANT_GEMM_COLS` columns) used the multi-pass GEMV, which re-reads the weights N/2
  times; 17-64 columns used the 64x64 WMMA kernel with its columns padded to 64 and low occupancy (~1.2 ms vs 0.3 ms ideal
  for a 17408 x 5120 weight).
- `wmma_q4_skinny!` (`src/WMMA.jl`, used by `_q4_matmul!` for 7..32 columns when M % 16 == 0, K % 256 == 0, group 32;
  `LUMINAL_WMMA_SKINNY=0` disables): one workgroup = 4 waves on the same 16 weight rows, each wave taking every 4th group of
  8 K tiles (split-K, partial sums added through LDS), dequantizing straight into the WMMA A fragments (no block-level
  LDS tile, no barriers in the loop). 1 or 2 16-column tiles.
- What made it fast (probes, scratchpad): 16-byte loads from 16 different rows 2.5 KB apart ran at ~70 GB/s because the
  128-byte lines were evicted before the next 7 tiles used them. Fix: lanes load 4 rows x 128 contiguous bytes (8 K tiles)
  per instruction, park them in a per-wave LDS tile (a wave's LDS accesses execute in order, so no barrier is needed
  between its own write and read) and read them back in the WMMA layout; scales as one 16-byte load per row. Split-K over the
  waves of a workgroup fixed low occupancy for small M (320 row groups for M = 5120). Now ~130-140 GB/s of weights.
- Pitfalls: LLVM hoisting the activation loads of all 8 unrolled K tiles spills (192 VGPRs, scratch), so the tile loop count
  is made opaque (`nu`); 64 columns (4 tiles) still spill and are no faster than the 64x64 kernel, so they stay there.
- GEMV vs skinny crossover at the 27B's shapes is ~7 columns (`SKINNY_MIN_COLS`); batch-of-N decode and speculative
  verification (N = 4-8) can use it too (the GEMV is better below that).
- Next for short prompts: 33-64 columns (two passes of the 2-tile kernel would only break even), the floor of ~95 ms for the
  weights, and the remaining non-GEMM work (DeltaRule, attention, ~40 small kernels per layer, ~10-15 ms at 16 tokens).

### Decode: MTP speculative decoding (2026-10-10)

27B int4, real prompt (79 tokens, 128 new tokens, greedy): decode **12.1 -> 16.8 tok/s (+39%)**, output identical to plain greedy
over all 128 tokens; 74% of drafts accepted, 1.74 tokens per verify step (`examples/qwen35_spec_bench.jl base.gguf mtp.gguf --int4`).
The 15 tok/s goal is met. Cycle = verify ~97 ms (plain step 83) + MTP draft ~6 ms + rollback ~0.5 ms (on rejects).
- Mechanism (`src/Speculative.jl`): the model's own MTP head (`blk.64.nextn.*`, in the `-MTP` GGUFs; ours come from
  `Qwen3.6-27B-UD-Q8_K_XL.gguf` of the MTP directory while the base weights stay the Q8_0 file) drafts one token; the base model
  verifies `[y, draft]` in ONE 2-token step (weights read once for both), so an accepted draft is a second token for ~+17% cost.
  Head: `x = eh_proj([enorm(embed(next token)); hnorm(hidden before it)])`, a full gated-attention layer + MLP with its own KV cache,
  `shared_head_norm`, the base `lm_head` (`Qwen35MTP`, `load_qwen35_mtp`; `hnorm` input = the base's final residual stream
  *before* its output norm, like gufo). Its prompt prefill is the pairs (x[i+1], H[i]).
- Pieces: `DecodeAttention` over S new tokens (one workgroup per (head, token), token s sees the cache + new tokens 1..s);
  `CausalConv` / `DeltaRule` write the recurrent state after token `snap_t` into a snapshot buffer (rollback of a rejected draft =
  copy snapshots back; the KV cache needs none: positions just don't advance); `build_llama_decode_step!(...; seq, snapshots,
  want_hidden, want_argmax)`; `ArgMax` op (greedy tokens picked on the device: no 2 MB logits readback / host argmax over 248k);
  the MTP draft step takes two (token, hidden) pairs (both real after an accept, the 2nd a dummy after a reject) so one graph
  and one weight read serve both cases. Verified: tests/test_speculative.jl (random MTP head: speculative == greedy; a forced-oracle
  draft makes every cycle accept; mixed accept/reject), tests/test_spec_ops.jl, the verify-step test in test_qwen35_model.jl.
- The 4-layer slice (`--layers=4 --oracle=P`) measures the cost of a cycle without a usable head; the numbers there are not the 27B's.
- Where the rest goes: the 2-column GEMV is ~157 GB/s against ~185 for 1 column (+14 ms per verify step); the MTP step's lm_head
  alone is ~3 ms. Ideas: a better 2-column int4 GEMV, drafting 2 tokens per cycle (3-column verify: the GEMV does 2 columns per
  pass, the skinny kernel starts to pay at ~3), sampling instead of greedy, batching the draft with the next verify.
- The GGUF reader now drops each tensor from the page cache right after reading it (`posix_fadvise(DONTNEED)`; set
  `LUMINAL_GGUF_KEEP_CACHE=1` to keep it): loading a 28 GB file was making the kernel swap other jobs' idle pages out.
  (Swap on this machine is full again, 8 of 8 GB, from the earlier full-model runs; `sudo swapoff -a && sudo swapon -a` brings it back.)

### Faster multi-column int4 GEMV (2026-10-10)

`_q4_matmul_kernel_r4!`: 4 weight rows per workgroup (the 2-row kernel's activation loads, 128 bytes per column per 16-byte
weight chunk, were the L1 bottleneck with more than one column), used from 2 columns (`Q4_ROWS`, `Q4_ROWS_MIN_N`; 1 column keeps
the 2-row kernel: 4 rows gave nothing there), 64 threads for tall matrices with K/32 <= 256 and M >= 8192, else 128
(`_q4_threads_r4`). At the 27B's shapes a 2-column GEMV went from 140-160 to 175-190 GB/s of weights, i.e. the same speed as
1 column (lm_head 3.9 -> 3.07 ms; 17408 x 5120: 0.30 -> 0.24 ms). Written out per row because nested closures over a tuple of row
accumulators did not compile for the GPU (the closure also must not capture the loop variable `k`).
Speculative decoding on the 27B (79-token prompt, 128 tokens): decode **16.8 -> 18.05 tok/s** (plain greedy 12.08), identical output,
74% accepted; verify step 97 -> 90 ms (plain step 83), MTP draft 5.4 ms, rollback ~0.4 ms per cycle.
What is left in the verify step over a plain step (~7 ms): the S = 2 attention, DeltaRule/conv writing the snapshots (~150 MB),
2-column rmsnorm / elementwise kernels, and 2 columns in the lm_head argmax.

### MTP draft step with a restricted vocabulary (done)
The draft step was 5.4 ms, 3 ms of it the 248k-row lm_head read. `MTPDrafter(s; draft_vocab=N)` (bench `--draft-vocab=N`)
scores only token ids 0..N-1 (the BPE's most frequent) with a row-sliced copy of the head ("mtp.draft_head"), 165 MB at
32768. On the 27B with N=32768: MTP steps 395 -> 178 ms over 73 cycles (5.4 -> 2.4 ms), acceptance unchanged (74%),
output identical; ~18.9 tok/s (from 18.05, derived from the phase times). Default is still the full vocabulary (0).

### Verify-step overhead, profiled (rocprofv3, full 27B, 40 steps each)
Plain step 82.5 ms wall / 76.9 GPU-busy; verify step 91.5 / 85.1. The extra ~8 ms of GPU time:
- int4 GEMV at 2 columns: +6.5 ms (75.2 vs 68.6). Standalone the 4-row kernel is only 3-6% over 1 column at the 27B's
  shapes (gate/up 236 vs 224 us), in-graph 8-9% (248 vs 230); the WMMA skinny kernel is slower at N=2 (333 us), so the
  GEMV is near its floor for 2 columns.
- DeltaRule +0.85 ms: the 150 MB snapshot write (3 MB per layer) at memory speed; avoiding it would need an inverse update.
- gpu_linear_copy / permutedims +0.4 ms: reshape(slice(qkv)) of q, k, v is a view at S=1 but a copy at S=2 (3 per layer).
- ArgMax +0.34 ms: one workgroup per column over 248k. Fixed: 64 workgroups per column plus a merge kernel, 182 -> 17 us.
The wall-minus-GPU gap (~5.6 ms) is launch gaps between ~1000 kernels, the same in both steps.

### Small-kernel fusion prototype (partial; measured gains are small)
Profile of the plain 27B step: 1839 kernels, 1278 of them small (4.5 ms of work, ~4 us each incl. launch; 5.6 ms of
wall time is gaps). Findings: (1) the Gemma-style `w + 1` of every norm weight ran as its own kernel each step (129 + q/k
norms): the compiler now folds Constants, so `w + 1` is computed once at compile (-39 steps per 8 layers; folding of
constant-derived nodes is capped at 4M elements); (2) the residual adds were already fused into the GEMV; (3) the RMSNorm
kernel used a log2(T)-barrier tree: now a 2-level sum (3 barriers), 512 threads for H >= 2048 (9.3 -> 5.4 us standalone).
Together: plain decode 11.88 -> 12.04 tok/s, verify cycle 90.3 -> 88.7 ms, speculative 18.47 -> 18.61 tok/s.
Removing a kernel is worth ~4 us, so SiLU*up into the down-proj GEMV (64 launches) is ~0.3 ms and add+RMSNorm (129) ~0.5 ms;
a general fusion pass would reach maybe 2-3 ms in total. Not done: add+norm into one two-output op, SiLU*up in the GEMV.
A decode "megakernel" is not attractive here: 91% of the step is int4 GEMV at ~217 of ~225 GB/s.

### int4 GEMV with Float16 activations and v_dot2_f32_f16 (done, 2-4 columns)
The Float32 multi-column GEMV is instruction-bound (3 instr per weight and column). `_q4d_kernel!`: activations as Float16
in the weights' nibble pair order (`_q4d_xprep_kernel!`), each weight word dequantized once into packed Float16 pairs
((1024+q) - 1032 = q - 8, exact), then one `v_dot2_f32_f16` per 2 weights per column. 17408x5120: N=2 232 us, N=3 229 us
(old: 236 / 308), N=4 292 us; same for the 27B's other shapes. Symmetric weights only; N=1 keeps the Float32 kernel.
Relative error ~2e-4 (Float16 activations). Verify step on the 27B: 2 tokens 88.3 -> 86.5 ms, 3 tokens 111.2 -> 87.1 ms,
so drafting more than one token per cycle no longer costs a 3-column penalty. Speculative output still identical.

### Two drafts per cycle (done): `generate_ids_speculative(...; ndraft=2)`, bench `--ndraft=2`
Verify [y, d1, d2] in one 3-token step (3 columns cost about 2 now); DeltaRule/CausalConv `snap_t = n` keeps n snapshots
(batch slot (t-1)*B+b holds the state after t tokens), rollback to slot 1 or 2, none after a full accept. The MTP head
drafts d1 from the last accepted pair (3-pair step, dummies padded) and d2 with a 1-pair chain step fed d1 and its own
pre-head-norm output state. 27B int4, 128 tokens: 23.6 tok/s decode vs 12.3 plain (18.8 with one draft), 2.19 tokens
per verify step, first drafts 76% accepted, second drafts 57% of those; verify 5350 ms, MTP 295 ms (58 cycles);
output identical to plain greedy. Not tried: the post-norm hidden for the chain, 3 drafts (4-column GEMV is 25% slower),
sampling instead of greedy.

### Long-context decode: split-K attention (done) -- `src/DecodeAttn.jl`, `examples/qwen35_context_bench.jl`
The decode attention kernel was one workgroup per query head scanning the cache serially (and capped at 8192 slots):
2.6 ms per layer at 8k keys, i.e. +42 ms per 27B step. `decode_attention_split!` cuts the cache into 256-key chunks, one
workgroup per (chunk, KV head, sequence) sharing each K/V read between the group's heads and the S speculative tokens (WMMA
for the scores, a coalesced scalar PV pass, an LDS softmax, then a merge kernel); Float16 cache, head dim 128/256, rows
S*Gq <= 18. Per layer at 32k keys: 0.76 ms (1 token), 1.05 ms (3 tokens), ~170 and ~125 GB/s. No cache-length cap.
27B int4 decode step by context (random K/V fill): plain 82.6 ms @1k, 83.7 @4k, 88.4 @16k, 93.6 @32k (12.1 -> 10.7 tok/s);
3-token verify step 91.8 / 93.5 / 100.0 / 107.7 ms (speculative ~24 -> ~20 tok/s at 2.2 tokens per step).

### Chunked prefill (done) -- `_prefill_chunked!`, `chunk_attention!`, `examples/qwen35_prefill_bench.jl`
A prompt longer than `LUMINAL_PREFILL_CHUNK` (default 512; 0 disables) goes through the model in chunks, each a multi-token
decode step (seq = chunk, `head=:last`, `lens` = the real tokens of the last chunk): the recurrent layers continue from their
states (zeroed at the start), the attention layers write the chunk's K/V into the Float16 cache and `chunk_attention!` (the flash
prefill kernel with a query offset, V transposed per call) attends to all of it. Memory is one chunk's intermediates
(~32 MB/token) for any prompt length; the MTP head is primed per chunk the same way. Batch 1, Float16 cache, Qwen3.5/3.6.
Needs max_seq >= prompt + chunk. 27B int4, chunk 256: 300 tok/s @512 tokens, 289 @2048, 271 @8192 (8192 tokens in 30 s);
logits match the one-shot prefill (rel err 0 on the 27B, 3e-4 on a smaller chunk); speculative decode after a 1067-token
prompt: 22.6 tok/s, output identical to plain greedy.
Gotcha found: buffers the graph does not free itself are only released by the GC (~3 GB per 1024 tokens on an 8-layer slice,
24 GB on the 27B), so a long chunked prefill ran the machine out of memory; `_chunk_reclaim!` now runs a young-generation GC + HIP
reclaim after every chunk (a full GC costs ~130 ms per chunk, LUMINAL_CHUNK_GC=full). The one-shot prefill path has the same
property but is bounded by the prompt length.
32k prompt (27B int4, chunk 256, real text = this repo's sources/notes): prefill 146 s = 224 tok/s (rate falls with the cache
length: 300 @512, 271 @8k, 224 @32k); memory flat (min available 36 GB with other jobs running, swap 0).
Decode after the 32k prompt, 192 new tokens, output identical plain vs speculative: plain 9.81 tok/s (~102 ms/token);
speculative (2 drafts) ~21.6 tok/s = 191 tokens / 73 cycles (2.62 tokens per cycle; 66/73 first drafts, 52/66 second drafts
accepted) at verify 108 ms + MTP 12 ms per cycle (from the loop's phase timers). The wall-clock difference of a 1-token and a
192-token speculative run is NOT usable (6.9 tok/s): the extra ~19 s is most likely the MTP priming over the 32k prompt, which
the 1-token run seems to skip (unverified). At short context: ~12 plain / 23.6 speculative, so 32k costs ~18% / ~8%.
Next: bigger chunks need more headroom than this shared machine has (512 -> ~16 GB); int8 KV cache and sparse attention are
the lossy options for the decode-side read traffic; split-K kernel is at 125-170 of ~215 GB/s.

### Split-K attention bandwidth tuning (small win) -- `src/DecodeAttn.jl`
Microbenchmark at 32k keys (D=256, 4 KV heads, 24 heads; copy ceiling ~212 GB/s): S=1 (plain step) 728 us = 180 GB/s;
S=3 (verify step) 1043 us = 126 GB/s, i.e. the 3-token case is where the headroom is. Part kernel = 96% of the time (merge 38 us,
cache write 4 us); 192 VGPRs, 53.8 KB LDS per workgroup. Ablations at S=1: without the V pass -200 us, without the K/score phase
-436 us; the phases run one after another inside a workgroup, so the memory pipe idles between them.
Tried: (1) `exch` aliased onto the scores' LDS (frees 18 KB, a third workgroup fits): S=3 1043 -> 946 us but S=1 728 -> 800 us, so
it is applied only when QR = S*Gq > 8 (kept: S=3 954 us, S=1 732 us). (2) Score row stride padded to a multiple of 4 (vector LDS
reads): no change. (3) Software-pipelined V loads (next batch issued before the current is consumed, first batch before the
softmax): much slower (S=3 1431 us; register pressure). Both (2) and (3) reverted.
End to end on the 27B the verify step at 32k is 107.5 ms (was 107.7): the ~1.4 ms predicted is inside the run-to-run noise of the
shared machine (plain step at 1k read 111-114 ms in these runs against 82.6 before, so another job was using the GPU).
Untried: DA_CH=128 (twice the workgroups, half the LDS; needs the tile loops generalised), staging K through LDS to overlap the
phases, int8 KV (halves the traffic).
The MTP-priming gap from the 32k speculative run is confirmed in the code: `generate_ids_speculative` returns before `_mtp_prime!`
when max_new_tokens <= 1, so the ~19 s priming cost (32k tokens) is a one-time cost of any longer generation.

### Three drafts per cycle (done, option `ndraft=3`) -- `_spec_loopN!`
`generate_ids_speculative(...; ndraft=1|2|3)`: the loop for 2 and 3 drafts is one generic function (`_spec_loopN!`, replaces
`_spec_loop2!`): verify [y, d1..dND] in one (ND+1)-token step, emit the leading matches plus the model's own next token, roll the
recurrent states back to the snapshot after the last emitted token; the MTP head drafts d1 from the last accepted token and the
rest as a chain of one-pair steps (d.pos is advanced during the chain and restored). Stats: :accepted, :accepted2, :accepted3.
What it took for the 4-token verify step to be cheap: (1) decode attention allowed 24 rows (DA_MAX_ROWS; 4 tokens x 6 query heads);
24 accumulators a thread spilled (2570 us at 32k), so for more than 12 rows two thread groups split the rows and read the same V
(`_da_vhalf!`: 991 us; also 18 rows 943 -> 926 us); (2) the int4 GEMV was launched with 3 columns per pass, so 4 columns read the
weights twice (+17.6 ms); now 4 per pass: verify4 113.6 -> 96.6 ms (verify3 91.5).
27B int4, 128 new tokens, steam-engine prose: ndraft=2 22.5 tok/s decode-only (2.31 tokens/step, 11.9 ms MTP per cycle),
ndraft=3 21.6 (2.54 tokens/step, 16.1 ms MTP): a wash. Julia code prompt (192 tokens): ndraft=2 25.6 tok/s (2.73 tokens/step),
ndraft=3 28.6 (3.41 tokens/step, 95% / 89% / 81% of the first / second / third drafts accepted): +12%. Outputs identical to plain.
So ndraft=3 pays on predictable text and costs ~4% on prose; the default stays 1. Next: choose the draft count per cycle from the
recent acceptance rate (adaptive), and make the MTP chain cheaper (each chain step is a full ~4 ms graph launch).

### Int8 KV cache (done, option `kv_dtype=Int8`; examples `--kv-int8`) -- `src/DecodeAttn.jl`
Per (slot, KV head) row: D signed bytes + 16 bytes of padding whose first word is the Float32 scale (absmax/127), so the cache's
first dimension is head_dim + `DA_Q8_PAD` (shapes via `NN._kv_shapes`; 1 byte + 0.06 byte of scale per element: half the Float16
cache). AMD GPU, Qwen3.5/3.6-style full-attention layers (the split-K kernel: D 128/256, no window, <= 24 rows per KV head; the
chunk path for longer steps; anything else with an Int8 cache is an error, not a silent fallback).
Write: `_da_write_q8_kernel!` (decode/verify steps and chunk steps quantize the new tokens on the way in), `quantize_kv!` (one-shot
prefill's K/V). Read: the K tile goes int8 -> Float16 exactly with the 0x6400 mantissa trick (byte ^ 0x80, OR into 1024.0, minus
1152; dims in the order b0 b2 b1 b3 of each word, the staged queries are permuted the same way) into the same WMMA, the scores are
scaled per key; V's scale is folded into the probabilities after the softmax sum (the V pass reads a byte pair per lane). Chunked
prefill dequantizes the cache prefix to Float16 per layer-chunk and runs the flash kernel over it (`_kv_dequant_kernel!`). The MTP
head's own cache stays Float16 (one layer).
Microbenchmark at 32k keys (S=1/3/4): 722/926/984 us (Float16) -> 525/762/790 us; relative difference to the Float16 kernel on the
same dequantized values 4e-4..6e-4. 27B int4 decode step at 32k context: plain 93.4 -> 90.5 ms, verify3 105.8 -> 103.0 ms (about
3% faster; at 1k no difference); the cache for 32k tokens is 1.1 GB instead of 2.1 GB. Chunked prefill is as fast (308 tok/s @2048,
275 @8192 vs 289 / 271). Quality: greedy 96-token continuations after 3000/1500/700-token prompts of real text match the Float16
cache token for token on two of three prompts and diverge at token 86 on the third (the first token's top-5 identical on all three);
speculative decoding output is identical to plain with either cache. One-shot vs chunked prefill differ more with the int8 cache
(logits rel err 5.6e-3 on 192 tokens, the chunked path reads earlier chunks quantized) than with Float16 (0).
Tests: tests/test_decode_split.jl ("int8 KV cache": decode, verify, 4-token, 2-sequence, chunk steps against the Float16 kernel over
the dequantized cache). No end-to-end tiny-model test (its head_dim is 16, below the kernel's 128/256).
Not done: a perplexity measurement (only greedy-agreement spot checks), int8 for the Gemma / windowed layers and for batched
chunk steps (B > 1), int4 KV.


### Adaptive draft count (done, option `ndraft=0`; `--ndraft=0` in the bench) -- `_spec_loopN!(...; adaptive=true)`
Up to 3 drafts per cycle; after each verify the loop picks 1..3 drafts maximizing expected tokens per ms, from an EMA (0.1) of the
conditional acceptance of each draft position (only cycles that tested it) and fixed costs (verify 86/91.5/96.6 ms for 2/3/4
columns, draft chain 5/12/16 ms). The first draft step is padded to the previous cycle's width, so graphs for every width get
compiled on first use. Output identical to plain decoding. Lighthouse-story prompt, 128 tokens, 27B int4, decode-only tok/s: adaptive 21.1, fixed 3 drafts 20.5,
fixed 1 draft 18.0 (same 69% first-draft acceptance; adaptive spent 805 ms in MTP steps vs 915 ms). So it is about +3% over a fixed 3 here
and never loses to it; the case where fixed 3 loses (low acceptance) was not hit by this prompt, and the starting estimate (0.7) and
EMA rate are untuned.
Cheaper MTP chain: not started.

### Fused MTP draft chain (done, small win) -- `_mtp_chain_graph` / `_mtp_chain!` in `src/Speculative.jl`
Drafts 2..ND now run as ONE graph (n single-pair steps; each step's token is the previous step's device-side argmax, the position
tensor is `pos + j`), instead of n launches with a host readback between them. Used when there is no `draft_hook`
(`LUMINAL_MTP_UNFUSED=1` switches back). Same drafts as before (27B, lighthouse prompt, ndraft=3: 37/22/14 accepted of 54 cycles in
both). But the gain is small: MTP time over 54 cycles 915 -> 852 ms (about 1.2 ms per cycle of ~16.9), so launch and readback overhead
was not what the chain costs; each step is dominated by the work itself (the vocabulary head, 248k rows, is the biggest weight).
Next for a cheaper chain: `draft_vocab` (restricted head, already an option; its acceptance cost is untested at 3 drafts), or
a smaller/int4 head for the draft only.

### 128K context on the 27B (int4, chunk 256) and a prefill profile
Decode step by context, random K/V fill (examples/qwen35_context_bench.jl), plain / verify-3 step in ms, Float16 -> Int8 KV:
1k 100/109 -> 99/107; 32k 94/106 -> 90/102; 65k 105/119 -> 99/113; 100k 116/133 -> 107/126; 131k 126/146 -> 116/137 (7.9 -> 8.7 tok/s plain).
Real 130K-token prompt (repo sources + notes, then "Create a recursive descent parser in C ...", chat format, 384 new tokens, ndraft=0):
Int8 KV: prefill 1160 s = 112 tok/s, decode 15.7 tok/s (144 ms verify + 26 ms MTP per cycle, 2.68 tokens/cycle, first drafts 79% accepted),
min MemAvailable 83.6 GB. Float16 KV: prefill 1224 s = 106 tok/s, decode 15.2 tok/s (156 + 25 ms, 2.76 tokens/cycle), min avail 79.7 GB.
The two outputs diverge at character 37 ("C89/C90/C99" vs "C89/C99") and both stay coherent; a real quality comparison (perplexity or
logit error over many positions) is still not done.
Prefill profile (rocprofv3 --kernel-trace, 32k, Float16 KV): the GPU is 91-94% busy throughout (rocm-smi agrees), so there is no
launch-gap headroom worth chasing. At 2k context: int4 WMMA GEMM (`_wmma_q4_wide_kernel_`, 1.12 ms avg, ~37 TFLOP/s while running) 57%,
delta rule 27% (4.3 ms per linear layer per chunk, flat in context), flash prefill 2%. Early -> late in a 32k prefill: GEMM 55% -> 34%,
flash prefill 6% -> 31% (3 -> 26 ms per call), delta rule 25% -> 16%, broadcast kernels 6% -> 14%.
Found: `chunk_attention!` rebuilt the transposed V prefix per layer and chunk with a strided broadcast (9 ms at 32k keys, ~10% of the late
prefill); now a tiled LDS transpose kernel (`_vtranspose_kernel!`, test in tests/test_decode_split.jl). 32k prefill 222 -> 234.9 tok/s,
chunked logits still identical to one-shot. Remaining ideas: the flash prefill kernel itself is now the main long-context cost (26 ms per call
at 32k, ~4x that at 128k; it reads K/V from L2 per 32-row query block), the delta-rule kernel (16-25%, serial recurrence), the 231 us
3-D broadcast kernels late in the prefill (not yet identified), and keeping a persistent transposed V instead of re-transposing the prefix.

### Flash prefill kernel: what was tried (no kernel change kept)
Standalone microbenchmark (random Float16 cache, D=256, H=24, KVH=4, S=256, chunk attention at cache offset qoff; CPU-reference check;
script not in the repo): baseline 0.98 ms @1k, 39.7-40 ms @32.5k (5.2 TFLOP/s), 178 ms @130.8k (4.6 TFLOP/s). At 130k a layer-call
is ~90 ms on average over the prefill x 16 layers x ~500 chunks, i.e. attention is the majority of a 130K-token prefill (1160 s).
Findings:
- Ablations (each only timing-valid): no K/V streaming 43.9 ms, no WMMA 35.0, no shuffles 37.7, no exp 40.5, no LDS P store 44.0 (NaN),
  no QK loop 22.2, no PV loop 19.0. Removing syncs makes it slower (65 ms). Nothing is one bottleneck.
- Grid sweep: time is a step function of the workgroup count: H=16 (128 WGs) 22 ms, H=20 (160) 26 ms, H=24 (192) 42 ms, H=32 (256) 41 ms,
  H=40 (320) 40 ms. The kernel uses 178-184 VGPRs (occupancy 8 waves/SIMD = 160 WGs resident); a round takes ~16 ms with 32 WGs and
  ~26 ms with 160, so each wave is latency-bound, and the 192-WG grid is just over one round (tail). Qwen3.6-27B has exactly H=24.
- The bounds-checked indexing puts a branch + s_waitcnt vmcnt(0) between every K fragment load and its WMMA; `@inbounds` gives the ideal
  load/WMMA batching in the ISA but was SLOWER (48 ms), so it is not simply load latency.
- Tried and not faster: QK split across the wave pair (each wave 2 of 4 key tiles, partial max/sum exchanged through LDS): 39 ms, correct;
  plus 128-byte-per-lane K loads in blocks of 4 steps: 39 ms; explicit unroll of the 8 softmax rows: 61 ms; PV in two V-fragment groups:
  type-unstable, spills. Both split variants hit 192 VGPRs with scratch spills.
Untried, the most promising: (1) a structure with fewer VGPRs (< 128, 12 waves/SIMD) so the whole grid is resident at once; (2) software
pipelining (issue QK of tile kt+1 before softmax+PV of tile kt; needs the split kernel's smaller score registers); (3) staging the K/V
tile in LDS cooperatively (coalesced loads, shared by the 4 waves and, with GQA, by several heads) like the int4 GEMM does, which also
removes the per-chunk V transpose; (4) ISA-level control of registers/occupancy (waves-per-EU), which AMDGPU.jl does not expose here.
Realistic ceiling from occupancy/tail alone is ~25-30% of attention time; more needs the LDS-staged design.

### Flash prefill attention through LDS (done) -- `_flash_prefill_lds!` in `src/FlashAttn.jl`
Port of the design in gufo's `WmmaCausalAttention` (`~/devel/AI/gufo/src/models/qwen/hip/kernels/attention_wmma.hip`; its header
comments are worth reading: each choice was found by ablation). 256 threads = 8 waves, 64 rows per block (2 query heads of the same
KV head x 32 queries), one 16-key tile per iteration: wave w computes the 16x16 score tile of row block w%4 over dim half w/4 (Q in
registers), the two halves are summed in the softmax (4 threads per row, from LDS), O += P V with wave w owning dim tiles w and w+8.
K tile coalesced into padded LDS, V tile one key per lane and written TRANSPOSED to LDS, the next tile's K/V prefetched into registers
right after the K staging barrier. 16-byte LDS fragment reads from Float16 arrays go through `unsafe_load` on an `LLVMPtr{WMMA_V8H,3}`
(`_lds_frag`). It reads the KV cache directly (no V transpose pass: `_vtranspose_kernel!` stays only for the fallback), also for
`flash_prefill!`; the Int8 cache path dequantizes V in the K layout. Needs D % 128 == 0 and an even number of query heads per KV head;
otherwise the old kernel runs.
Microbenchmark (D=256, H=24, KVH=4, S=256): 32.5k keys 39.7 -> 19.2 ms (10.7 TFLOP/s), 130.8k keys 178 -> 77 ms; rel err vs CPU
2e-4..5e-4. 27B 32k prefill 234.9 -> 256.3 tok/s (136 -> 125 s), chunked logits still identical to one-shot.
What the investigation before it found (kept for reference): the old kernel was latency-bound with 184 VGPRs, tail round of 32 workgroups
(192 > 160 resident), every K/V fragment read from global memory by every wave; splitting QK across the wave pair, `@inbounds`, 128-byte
loads, unrolling and staging K alone in LDS (-11% at 130k, nothing at 32k) did not help; the gufo structure did. Gufo reports ~20 TFLOP/s
for its kernel on this chip; ours is at ~11, so there is probably more (its 16-key tile also avoids the cross-wave max exchange; the
causal-tail exactness fix and `__launch_bounds__(256, 2)` occupancy control are things we do not have).

### Launch bounds from Julia: our AMDGPU.jl branch (`~/devel/AMDGPU.jl`, branch `luminal-launch-bounds`, commit e4fa557)
AMDGPU.jl never sets `amdgpu-flat-work-group-size` / `amdgpu-waves-per-eu`, so LLVM assumes 1024-thread workgroups and caps every kernel at
192 VGPRs (wave32). The branch adds compiler options `flat_work_group_size` and `waves_per_eu` (`@roc ... flat_work_group_size=256 k(...)`),
based on the exact commit Luminal's Manifest used (`simeonschaub/AMDGPU.jl#sds/update_wrappers` @ ba948d9; not pushed anywhere yet).
Use it: `julia --project=. -e 'using Pkg; Pkg.develop(path=expanduser("~/devel/AMDGPU.jl"))'` (the Manifest is untracked; the old one is in
`Manifest.toml.bak-before-amdgpu-fork`). Luminal detects the option with `@static if :flat_work_group_size in AMDGPU.COMPILER_KWARGS`
(`_launch_flash_lds!`), so stock AMDGPU.jl still works. The attribute must be per kernel: applied to all kernels it makes AMDGPU.jl's own
fill/copy kernels (launched with 1024 threads) fault on the device.
Measured (LDS flash prefill kernel, 32.5k / 130.8k keys): 18.2 -> 17.1 ms and 76.7 -> 72.5 ms (about -6%); `waves_per_eu` adds nothing. The VGPR cap
is not the main gap to gufo. The kernel keeps its 64 Q registers in scratch (~290 B) because it needs more than the cap allows; unrolling the
loop so Q stays in registers made it slower even with 256 VGPRs (more live ranges, still 240 B of scratch), so that needs a different register
plan (e.g. Q read from LDS per step, or fewer live accumulators), not just a bigger cap.

### Where a prefill chunk's GPU time goes (27B slice, 8 layers, chunk 256, short context; rocprofv3, GPU 94% busy) and what fusion can buy
`_wmma_q4_wide_kernel_` (int4 GEMM) 56.4% (1.14 ms avg per call), `_delta_rule_reg_kernel_` 26.0% (4.37 ms per call, 48 recurrent layers x 256
tokens), the other int4 matmuls 4.2% (lm_head etc.), then elementwise broadcasts ~6%, copies ~3%, fills ~1%, RMSNorm ~1%. A full 64-layer
chunk is ~800 ms (317 tok/s); gufo's whole chunk is ~390 ms (656 tok/s), so our GEMMs alone (~450 ms) are slower than their entire chunk.
The earlier "16% in small kernels" figure was wrong: LUMINAL_TRACE_TIME synchronizes and charges allocation to each op (an 8-layer slice shows
`Add (5120,256)` at 2.7 ms each; the kernel is 48 us). The biggest elementwise kernel (SwiGLU multiply, 4.4M elements, 235 us) already runs
at ~225 GB/s, so fusion only removes traffic: the realistic gain from fusing every small op (residual+RMSNorm, SwiGLU into the GEMM
epilogue, copies) is ~5-7%. Larger: the delta-rule kernel (26%; gufo fuses conv + recurrence + post-norm/gate + int8 quantization in one
kernel but its recurrence is also token-serial) and the GEMM (gufo ~1.5x faster: int8 WMMA with quantized activations).

## Delta-rule register kernel rewrite (4 lanes per column)

Kernel microbenchmark (27B shape: dk=dv=128, 16 key / 48 value heads, 256 tokens): 4.78 ms -> 0.54 ms per call (8.9x); 8-layer prefill slice 2490 -> 3093 tok/s (+24%).
What mattered, in order found:
- Launch bounds (`flat_work_group_size`, fork): 4.78 -> 2.68 ms (the 192-VGPR cap caused 1.7 KB of scratch).
- Step algebra so only the state tuple is live: kv = dec*S^T k, o = dec*S^T q + delta*(k.q), S = dec*S + k delta^T; k.q is computed once per 16-step chunk in LDS. Explicit `muladd`.
- 4 lanes per column (32 state registers each, no scratch) instead of 2: 2.0 -> 1.5 ms.
- Snapshot code compiled out when snap_t == 0 (`Val{SNAP}`): -16%.
- Interleaved key assignment across the lane quad (lane l holds keys l, l+4, ...): with a contiguous split every lane of a quad hit the same LDS bank (4-way conflict): 1.24 -> 0.54 ms.
- `waves_per_eu=12` (~120 VGPRs) so all 48 heads are resident: ~1.4x. DPP quad swaps instead of ds_bpermute: no measurable change, kept.
Counters before the fixes: waves waited ~80% of the time, VALU ~22% busy.

### Full 64-layer prefill profile after the delta-rule rewrite (chunk 256, 1024 tokens, rocprofv3 kernel trace)

Prefill 317 -> 389 tok/s (2.63 s for 1024 tokens). GPU time by kernel: `_wmma_q4_wide_kernel_` 76.8% (1.14 ms/call), broadcasts 7.6%, copies 4.1%, `_delta_rule_reg_kernel_` 3.7% (0.45 ms/call, was 4.4 ms), RMSNorm 1.6%, other int4 matmul 1.5%, fills 1.2%, causal conv 0.9%, flash attention 1.0%.
The int4 GEMM is now the whole story (gufo's chunk is ~1.5x faster there); fusion of the small kernels is worth ~10% at most.

## int4 GEMM: tile layout experiments (no gain) and where the ceiling is

Measured on ffn gate/up (M=17408, K=5120, N=512), all in one process, best of 15:
- Pure register WMMA loop (no memory): ~51 TFLOP/s on this chip (gufo quotes 55-59).
- Current `_wmma_q4_wide_kernel!` (128x256 block, 16 waves of 32x64): 30.5 TFLOP/s (60% of the 51).
- Ablations of that kernel: LDS fragment reads + WMMA + barrier only (no global loads, no dequant): 39.5; no WMMA: 2.07 ms of the 3.09 ms total; no dequant: 35.7; no weight/activation prefetch: 33.9.
- Variants (scratch, not kept): software-pipelined loop (dequant of tile t+1 issued with the WMMAs of tile t, one barrier per tile): 30.0; 128x256 block with 8 waves of 64x64 (half the LDS fragment traffic per WMMA, 256 VGPRs): 29.9; both together: 23.4 (spills / worse schedule).
- Hyperopt sweep (benchmarks/gemm_sweep.jl) over kernel, prefetch depth, waves_per_eu: the default is the best.
- `flat_work_group_size=512` on the wide kernel is *slower* (25 vs 28-30 TFLOP/s): do not add launch bounds to it.
- int8 WMMA is no faster than f16 on gfx1151 (gufo: 55 vs 59 TOPS ceilings), so W4A8 only saves dequant work (~0.8 ms of 3.1) at a numerics cost.
Gufo's dense GEMMs run at 30-34 TFLOP/s (docs/models/qwen3.8-flash-next/EXPERIMENTS.md), i.e. about where ours is. The prefill gap to gufo is therefore not the GEMM; chunk 512 (N=512) is worth +15% by itself (N=256 underfills the 40 CUs).

## Gufo comparison re-measured on this machine (2026-10-12): there is no 1.4x gap

Earlier I compared our int4 prefill to gufo's *published* Q4_K_XL number (656 tok/s, pp2048, docs/models/qwen3.8-27b/BENCHMARKS.md). On this machine, same prompt length (2048), 27B Q8 weights:
- gufo `bench -p 2048` on Qwen3.8-27B-UD-Q8_K_XL: **288 tok/s** (its published Q8 figure is 485; not reproduced here), tg16 7.25 tok/s.
- Luminal `--int8` (chunk 512) on Qwen3.6-27B-Q8_0: **289 tok/s**.
So for Q8 the two engines are at parity. Gufo cannot run the other 4-bit files here (IQ4_XS: unsupported storage type; no Q4_K_XL file on disk; it also pins the Qwen3.8 chat-template hash, so Qwen3.6 GGUFs are refused), so the Q4 comparison is open. Luminal int4: 467 tok/s (chunk 512); ratio int4/int8 for us 1.62, for gufo's published numbers 1.35 (which would put gufo's Q4 at ~390 tok/s on this machine if the ratio held). The published figures were measured on a different day/configuration; do not use them as a target.

## Float16 GEMM inputs (compile pass) and where the Qwen3.8-27B Q4_K_XL prefill stands vs gufo (2026-10-12)

Measured here, Qwen3.8-27B UD-Q4_K_XL, depth 0, chunk 512 (Luminal int4 re-quantized from the K/IQ-quants; gufo native):
| tokens | Luminal | gufo | gufo/Luminal |
| 512 | 459 | 421 | 0.92 |
| 2048 | 465 | 690 | 1.48 |
| 8192 | 416 | 646 | 1.55 |
| 16384 | 394 | 620 | 1.57 |
| 32768 | 355 | 584 | 1.65 |
Gufo's 4-bit decode is 12.1 tok/s here.
Timed-region trace (2048 tokens, 4.4 s): GPU idle 3.7% (67 ms launch gaps over 12.3k kernels, median 4.9 us; 3 x ~30 ms at the chunk boundaries = the explicit GC.gc(false) + HIP.reclaim). GEMM 73% of kernel time (~32 TFLOP/s; gufo needs ~40), elementwise/copy/fill ~20%.
- `chunk_attention!` no longer reads the position back from the device (CHUNK_POS_HINT): it synchronized the stream in every full-attention layer. Throughput unchanged.
- LUMINAL_CHUNK_REUSE=1 (persistent per-node buffers, no per-chunk GC): +3.5%, 18-20 GB instead of ~13 GB; LUMINAL_CHUNK_CAPTURE=1 is still blocked by other host syncs (hipErrorStreamCaptureUnsupported).
- compile pass `dty`: RMSNorm outputs, SwiGLU / gated-norm multiplies that feed only matmuls (N >= 384, N % 256 == 0) are computed as Float16, so the Float32->Float16 copy before each GEMM is gone (LUMINAL_F16_GEMM_INPUTS=0 disables). Logits differ by 1.8e-5 relative. +2-3% (472 -> 478-483 tok/s).
- Remaining copies: the strided V slice of the conv output (6144x512, ~50 ms) and q/k slices (~16 ms); the L2-norm chain of q and k (12 small kernels per layer, ~100 ms); the residual add after each GEMM (~55 ms, could be in the GEMM epilogue); zero-fills (~45 ms).
- Idea: static buffer planning (lifetime-based offsets into one arena, like gufo's fixed working set of ~0.5 MB/token vs our 32 MB/token) would remove the GC step, the memory growth, and allow capture.

### L2NormOp and the GEMM residual epilogue
- `L2NormOp` (src/Ops.jl, one wave per column): Gated DeltaNet's q/k normalization is one kernel (15 ms per 2048-token prefill) instead of Mul + SumReduce + Recip + Expand + Mul (~100 ms of broadcasts and reductions).
- `wmma_q4_gemm!(...; residual = R)`: the residual add is in the 128-row kernels' store (HR is a compile-time flag: no cost without it). Standalone 5120x17408x512 GEMM: 2.80 ms; + residual in the epilogue 2.94 ms; + a separate add 3.05 ms. Each lane reads the residual with the same strided pattern as its stores; a coalesced store through LDS would make both cheaper.
- Prefill 2048 tokens (Q4_K_XL, chunk 512): 472 -> 483 (Float16 inputs) -> 486-488 tok/s. Per-kernel totals are within ~1.5% run to run; compare wall time, not single-kernel sums.

## Prefill memory: dropping dead buffers promptly (2026-10-12)

A 512-token chunk of the 27B peaked at +11.8 GB over the loaded model (sampled every 2 ms: used 29.95 vs 18.17 GB) and needed a young GC after every chunk (~30 ms); without it the process grew 3 GB per call and was OOM-killed in a long run. Two causes, both in the executor, not in the Julia GC as such:
1. A fused group that reads one buffer several times (SwiGLU: `gate * sigmoid(gate) * up`) decremented the buffer's consumer count once per distinct input, so the count never reached zero and e.g. every layer's 34 MB gate output lived until the next run. Fixed: `build_fused_expr!` counts the uses (`leaf_uses`) and the step decrements by that many.
2. Every derived array (reshape / reinterpret / realized view, e.g. `C2 = reshape(C, M, N)` inside a GEMM wrapper) holds its own reference to the device buffer until its finalizer runs (`GPUArrays.derive` does `copy(x.buf)`), so `unsafe_free!` of the parent freed nothing while a wrapper was alive. Fixed with `AMDGPU.DERIVE_SCOPE` (our AMDGPU.jl branch): each executor step records the arrays derived while it runs (not the aliases it stores) and frees them when it is done (`_scope_begin/_hold/_resume/_end!` in Compiler.jl).
Result (full 27B, chunk 512): peak working set +0.4 GB over the weights (was +11.8 GB); no growth across calls; no per-chunk GC needed (`LUMINAL_CHUNK_GC=none` is now the default when the hook exists; `young` / `full` still available). Prefill 2048 tokens 490 -> 497 tok/s, 8192: 446 -> 455, 32768: 355 -> 381 tok/s; the 32K run never dropped below 88 GB of available memory.
`LUMINAL_TRACE_MEM=1` now also prints per-step hidden allocations, long-lived buffers and the peak.
Still blocking HIP graph capture of the chunk: host syncs in other ops (hipErrorStreamCaptureUnsupported); next candidate for removing the ~12k launch gaps (~1.5%).

## HIP graph capture of the prefill chunk: works, but no gain (and unsafe as is)

With the chunk position passed from the host (`CHUNK_POS_HINT`) the chunk graph no longer synchronizes the stream, so `capture=true` compiles and runs (it needs `free_intermediates=false`, i.e. one persistent buffer per node: ~18 GB). Measured (27B Q4_K_XL, 2048 tokens, chunk 512): captured 478 tok/s vs 497 for the normal executor vs 481 with persistent buffers and no capture. The ~12k launch gaps are only ~1.5% of the time, and persistent per-node buffers lose the cache locality of a reused freed buffer (the 32 MB MALL), which costs about as much. It is also wrong beyond two chunks: the replayed graph keeps the chunk position baked into the attention kernel's launch arguments (the first two chunks are warm-up and capture, so a 2-chunk check passes). Worth revisiting only together with a static buffer plan (small arena, reused) and a device-side position in `chunk_attention!`; the expected gain is <= 1.5%. The switch was removed.

### Vectorized GEMM store (128-row int4 kernels)
`_wmma_store_big!` regroups each 16x16 result tile with one shuffle round between the lanes (r, 0) and (r, 1) so a lane holds 8 consecutive rows, and writes (and reads the residual) as two 16-byte vectors; C is viewed as (M/4, N) 4-float vectors. The shuffles must run for every lane (inside the `col < N` branch they gave wrong rows for ragged N): only the memory accesses are guarded.
Standalone GEMMs (N = 512): gate/up 3.32 -> 2.94 ms, down 2.95 -> 2.78, qkv 2.15 -> 2.05, proj 0.96 -> 0.91 (store-free floor 2.82 / 2.75 / 1.91 / 0.84); with the residual +2.5% instead of +5%. Prefill (Q4_K_XL, chunk 512): 2048 tokens 497 -> 519 tok/s, 8192: 455 -> 474.
Where the time goes now (to choose the next step): attention is ~2% of a 2048-token prefill, ~7% at 8192 and ~23% at 32768 (flash kernel 11-14 TFLOP/s, 1.18 us per key position per 512-token chunk call, 16 layers).

### Flash prefill LDS kernel: unrolled register loops (+30%)
`_flash_prefill_lds!` (D = 256, 24 heads / 4 KV heads, 512-token chunk): 11.6-11.9 -> 15.2-15.7 TFLOP/s at 8K-32K keys (scratch 272 B -> 60 B, 256 VGPRs with the launch bounds). Every loop that indexed a register tuple with its counter kept the tuple in scratch memory: the S = Q K^T loop over the Q fragments (`_qk_dot`, a generated straight-line function), the K / V staging loops over the prefetch tuples and the output epilogue over the accumulators (now `ntuple(Val(n))` over a `let`-rebound copy).
Tried and reverted: three barriers per tile instead of five (stage the next K and a double-buffered V^T at the end of the iteration): correct but 11.4-11.6 TFLOP/s (LDS 33 -> 45 KB drops the occupancy 5 -> 4, and holding two sets of prefetch registers spills).

### 128-row flash prefill kernel (D = 256)
`_flash_prefill_lds2!`: 512 threads = 16 waves, 2 heads x 64 queries per block, three barriers per 16-key tile (the barrier before the K staging and the one after the V staging are redundant: the barrier after the K staging already orders every wave's previous P V before the V restage, and the two softmax barriers publish it). Each K / V tile now serves twice as many rows (half the L2 traffic per FLOP, at 32K keys the 64-row kernel was reading ~6.4 GB per call, ~240 GB/s). 252 VGPRs, no scratch, 45 KB LDS. Kernel: 15.4 -> 18.0 TFLOP/s at 32K keys, 15.3 -> 16.6 at 8K, equal below 2K. Needs the launch-bounds branch (512-thread bounds); `LUMINAL_FLASH_ROWS=64` selects the old kernel. Prefill (Q4_K_XL, chunk 512): 8192 tokens 474 -> 490 tok/s, 32768: 381 -> 431 (the 64-row kernel unrolled alone: 11.7 -> 15.5 TFLOP/s).
What the ablations said about the 64-row kernel (32K keys, 26 ms): without the K/V global loads in the loop -33%, without the S MMAs -44%, without the P V MMAs -38%: latency-bound (overlapping), not tensor-bound; two accumulators for the S chain did not help; a double-buffered V^T (3 barriers, 45 KB LDS) dropped the occupancy and was slower.
Gufo's reported 20 TFLOP/s is the next reference.

### GEMM mainloop: what was measured, and the warp-specialized kernel
Findings on the 128x256 int4 kernel (gate/up, 17408x5120x512, ~30-33 TFLOP/s; a pure register WMMA loop does 51):
- Not memory bound: re-reading one hot tile instead of streaming weights/activations is not faster. Not a compiler register policy issue: `waves_per_eu` alone (2..6) and explicit up-front fragment loads change nothing; `flat_work_group_size=512` is 14% slower.
- VALU instructions are free next to MMAs (8 extra FMA per MMA: no slowdown), so dequantization itself is cheap; what is lost is overlap: all 16 waves staged their share of the next tile and then computed, in lockstep around the barrier (the core alone, LDS fragment reads + MMAs + barrier, runs at 42.6 TFLOP/s).
- An int8 core (iu8 WMMA, half the LDS bytes per fragment) reaches only 44.7 TOPS vs 42.6: LDS traffic is not the limit, so a W4A8 GEMM would not be the lever either.
- Warp specialization (`_wmma_q4_ws_kernel!`, now the default for the 128x256 path, `LUMINAL_WMMA_WS=0` to disable): 16 consumer waves run the MMAs of tile t while 16 producer waves (1024 threads in all) load, dequantize and stage tile t+1 (2 tiles of global loads in flight). With 4 producer waves it was slower (23 TFLOP/s: a wave per SIMD cannot keep the stage fed), 8 about equal, 16 best. Interleaved best-of-40: gate/up 29.5 -> 32.7, down 32.4 -> 34.2, qkv 30.7 -> 31.8, proj 30.3 -> 34.0 TFLOP/s. End to end only +1.3% (525.6 -> 532 tok/s at 2048 tokens, 489 -> 497 at 8192).
Where the time is now (2048 tokens, 3.93 s): GEMM 78.8% (3.04 s), small kernels ~15% (SwiGLU multiply 118 ms, gated-norm multiply 94 ms, RMSNorm 81 ms, copies 75 ms, fills 53 ms), delta rule 4.4%, attention 1.4%, GPU idle 1.7%. Gufo does the same prompt in 2.97 s.
Next candidates: SwiGLU into the gate/up GEMM epilogue (~3%), a fused gated RMSNorm (~3%).

### Gated RMSNorm op and SwiGLU in the gate/up GEMM epilogue
- `GatedRMSNormOp` (one wave per column; the Gated DeltaNet's `rmsnorm(o) * w * silu(z)`; Float16 output when a GEMM reads it): replaces the RMSNorm kernel + the broadcast that multiplied by silu(z) (~120 ms of 2048-token prefill). 532 -> 540 tok/s.
- `SwiGLUOp` (the Qwen3.5 MLP builds `silu(gate) * up` as one node via `Mlp(...; swiglu=true)`); when both inputs are int4 matmuls of the same Float16 activation that nothing else reads (M % 64 == 0, N % 256 == 0, N >= 384) the compiler runs them as one step, `wmma_q4_swiglu!`: a 64-row x 256-column warp-specialized block where each MMA wave computes a 32x32 piece of BOTH the gate and the up result, so the epilogue holds g and u of the same elements and stores `silu(g) * u` as Float16 (no Float32 gate/up results, no elementwise kernel). Standalone 17408x5120x512: 5.87 ms vs 6.09 ms for two GEMMs + the elementwise kernel (-3.6%; a plain GEMM over the 34816 stacked rows is 5.6 ms: the 64-row block is ~5% less efficient than the 128-row one). The tile function reads the two B fragments once and streams the four A fragments (96 VGPRs, occupancy 16: reading all six first needed 120 VGPRs and ran at one block per WGP, 18% slower). End to end 540 -> 549 tok/s at 2048 tokens (LUMINAL_FUSED_SWIGLU=0 disables).
- 2048 / 8192-token prefill (Q4_K_XL, chunk 512): 549 / 510 tok/s (gufo: 690 / 646).
- The warp-specialized kernel's idea would also fit a "gate/up rows interleaved in one weight" layout (then the unmodified 128-row kernel + a pairing epilogue), but that needs a third copy of the FFN weights (5.7 GB) or a different decode path.

## What gufo's prefill does (read from its source, a rocprofv3 trace of `gufo bench -p 2048` and its compiled ISA), 2026-10-12

Qwen3.8-27B UD-Q4_K_XL, pp2048 on this machine: gufo 689 tok/s (2.97 s), us 549 (3.73 s).
- Gufo's GEMM time per 2048-token prefill is ~2.4 s (70% of its kernel time), ours 3.0 s: the GEMM is ~0.65 s of the 0.75 s gap. Its small kernels are fused as ours now are (quantizing norm kernels, SwiGLU/residual in the GEMM epilogues, QK-norm + RoPE + KV write in one kernel).
- GEMM kernel: `HalfPrefillGemmKernel<256, 256, 8, 4, ...>`: 256x256 block, 1024 threads (32 waves, 8 per SIMD), a 32x64 tile per wave (the same as ours), K64 per iteration, ONE 64 KB LDS stage filled from registers between two barriers, grids ordered in groups of 8 row tiles for L2 reuse. Paired gate/up kernel with the SwiGLU epilogue (same idea as `wmma_q4_swiglu!`). FP16 path for batches >= 1024 (the W8A8 int8 route is used below that; int8 gives no matrix-rate gain on gfx1151).
- In the compiled loop the next tile's weight *unpacking* (shifts / masks, ~100 VALU) is interleaved with the WMMAs of the compute phase and the LDS reads are software-pipelined; only the final scale-FMA and the `ds_store`s are between the barriers.
- CORRECTION (supersedes my first reading of the trace): the "57 TFLOP/s" down projection was an averaging artifact. The Q5_K residual kernel instance (same template, same grid) is used for both the K=17408 down projection (~9.6 ms = 38 TFLOP/s) and the K=5120 attention / SSM output projections (~3.5 ms = 31 TFLOP/s); I averaged the two. Gufo's own kernel standalone (harness on synthetic Q4_K / Q5_K weights, 4 s warm-up, batch 512 and 2048) gets 32-38 TFLOP/s on every shape, the same as our kernels (33-34). Data content matters only ~10-20% (zero weights + constant activations: 40.7 TFLOP/s) and the warm-up does not change it.
- Per-run accounting, steady-state 2048-token prefill (kernel traces): gufo 2.97 s, 888 kernels, GPU 99.7% busy: GEMM kernels 2.67 s (38 TFLOP/s average over the 101 TFLOP), everything else 0.29 s (DeltaNet recurrence 92 ms, attention 41, conv 32, norms...). Ours (current build) 3.76 s, 7453 kernels: GEMM 3.08 s (33 TFLOP/s), everything else 0.59 s (delta rule 172 ms, linear copies 75, broadcasts 63, attention 54, zero fills 52, conv 49, rmsnorm 41, gated rmsnorm 24, permutedims 22, l2norm 13). So the 0.76 s gap is ~0.4 s GEMM (their in-model rate is ~14% above their own standalone synthetic rate and above ours) and ~0.3-0.4 s of small kernels (copies, fills, broadcasts, permutes ~0.21 s, delta rule 0.08 s).
- A port of that structure to Julia (benchmarks/gemm_gufo_style.jl, not wired in) reaches 33-34 TFLOP/s on the same synthetic data, i.e. gufo's kernel's standalone rate and 5% above our warp-specialized kernel at N = 512, 20% above it at N = 2048 (33.7 vs 28; chunk 512 stays better for the other kernels). What it took: (1) unrolling the four K16 steps (otherwise a runtime loop with 32 accumulator register copies); (2) a conflict-free LDS layout (a fragment row's two 16-byte halves in separate planes so 16 lanes read 256 contiguous bytes; a 32-byte row stride made every fragment read 2-way bank-conflicted: 32 -> 25 TFLOP/s); (3) a rotated (not XOR) row swizzle so 8 lanes of a store hit 8 distinct bank groups; (4) no 64-bit integer division by the runtime group size in the loop (a shift now, in all the int4 kernels). `iglp_opt` made it slower. Ablation of the final kernel: the core alone (LDS reads + MMAs + barriers) 44 TFLOP/s; loads cost 0.48 ms, decode 0.36, LDS stores 0.59 of its 2.67 ms, partly overlapping.

### Zero fills and the delta-rule staging (2048-token prefill: 549 -> 555 tok/s, 8192: 510 -> 517)
- `_writes_all` now includes CausalConv, DeltaRule, Slice, SumReduce, MaxReduce (their kernels write every element): ~1800 fill kernels per prefill gone (52 ms).
- Delta rule (`_delta_rule_reg_kernel!`): 0.94 -> ~0.83 ms per 512-token call. Counters (rocprofv3 --pmc, 6 calls): waves waited 65% of their time, 27% of it at barriers, 14% on LDS instruction issue, 9% on global loads; the staging loop that loaded and stored one element at a time paid ~8 memory latencies per 16-step chunk. Now every thread issues its q / k / v loads in one batch (`ntuple(Val(NUQ))`, the closure-captured loop variables rebound with `let`: otherwise boxed) and the k.q of each step is a wave-wide reduction instead of 16 threads doing a 128-long dependent chain.
- Tried, no gain (kernel microbenchmark, warmed up): 16-byte vector LDS reads with a padded contiguous per-lane layout (the LDS-bound theory; scalar pairs -> b128: slightly slower, 268 B of scratch when the k and q tuples are live together; reloading k to shorten the live range was worse); prefetching the next chunk's loads into registers across the 16 steps (spills at the 128-VGPR budget of 12 waves/SIMD; at 10 waves no faster); splitting every head over 2 workgroups like gufo's row split (96 blocks of 8 waves, better balance over the 20 WGPs: 0.964 vs 0.967 ms; 4 splits slower). Gufo's recurrence is 1.0 us per token (no LDS, no barriers, k / q read from L1 with float4 loads, 96 blocks), ours 1.6 us.

### The 256 x 256 GEMM is the default (`LUMINAL_WMMA_G=0` for the warp-specialized one)
`_wmma_q4_g_kernel!` (gufo's structure, see "What gufo's prefill does") is used when M % 256 == 0, K % 64 == 0, padded N % 256 == 0 and the group is a power of two. End to end (Q4_K_XL, chunk 512, three alternating runs): 8192 tokens 530.5 / 529.3 vs 518.4 tok/s (+2%); 2048 tokens 568.9 / 551.2 vs 556.8 (+0-2%, noisy).

### Paired gate/up kernel in the 256 x 256 structure: no gain, fusion off by default
`_wmma_q4_gpair_kernel!` (a 256-row block = 128 gate + 128 up rows in interleaved 16-row tiles, so each wave's two A tiles are the gate and up fragments of the same output rows; epilogue silu(g) * u -> Float16) is correct, but against two plain 256x256 GEMMs + the elementwise SwiGLU kernel it is 3% slower in isolation (interleaved best-of-40, 17408x5120x512: 6.16 vs 5.97 ms; the 64-row kernel's apparent 3.6% win was against the older, slower GEMMs) and identical end to end (4 alternating runs, 2048 / 8192 tokens: fused 553-555 / 533-534, unfused 549-555 / 534 tok/s). `LUMINAL_FUSED_SWIGLU=1` turns the fused step on (default off). State at this commit: 554 tok/s at 2048 tokens, 533 at 8192 (gufo 689 / 646).

## Why gufo's prefill is faster: allocation, launch gaps, in-model vs standalone GEMM (2026-10-12)

Three checks against gufo (Qwen3.8-27B Q4_K_XL, chunk 512, 2048 tokens, rocprofv3 kernel traces).

- **Weight placement: no effect on prefill.** gufo keeps weights in anonymous host memory (`madvise(MADV_HUGEPAGE)`, `hipHostRegister(Mapped|ReadOnly)`, zero-copy device pointer); we use `hipMallocAsync`. The VRAM carve-out here is 512 MB, so both are GTT. `benchmarks/gemm_weight_placement.jl` runs our g kernel with weights in a device allocation, in registered host memory, and in registered THP-backed memory (AnonHugePages confirmed to grow): 35.4-35.8 TFLOP/s for all three, for down and gate shapes at N=512 and 2048. gufo's own notes credit huge pages with ~2% on decode only ("comparable PP").
- **Launch gaps: about 0.1 s of the 0.7 s difference.** `benchmarks/trace_gaps.py`: ours 8408 kernels, 2.3% idle (115 ms; 42 ms of it before `gpu_fill` at chunk starts); gufo 1.1% idle. Kernels under 20 us total only 6 ms, so the 9x kernel count is not itself the cost. The mid-size small ops are: `broadcast_cartesian` 237 ms (1611 launches, 147 us avg), delta rule 199 ms, `linear_copy` 101 ms, causal conv 62 ms, rmsnorm 54 ms, permutedims 30 ms.
- **g kernel in-model equals standalone.** Back-to-back, 16 weight copies, same profiler: standalone down (M=5120, K=17408, N=512) 2766 us, gate (M=17408, K=5120) 2673 us; in the model ~2750 us and ~2700 us. So there is no in-model penalty on our side: the sustained rate is ~33-34 TFLOP/s (the 35.5 seen with synchronised timing is a clock effect). gufo's in-model GEMM at ~38 TFLOP/s is therefore a property of its kernel under sustained load, not of how it is called.

Remaining ideas, in order of size: `broadcast_cartesian` (0.24 s: find which ops emit it and fuse or eliminate them), the sustained-rate difference of the GEMM (gufo's kernel is `WKQuantA8BlockedWmmaGEMMKernel`: int8 activations, Q8_1), linear copies and permutes.

### W8A8 int8 is not what makes gufo faster; block swizzle does not help (2026-10-12)
- gufo's own trace (2048-token prompt): 81% of its kernel time is the **FP16** `HalfPrefillGemmKernel<256,256,8,4,...>` (Q4_K, Q5_K, Q6_K, IQ4_XS instantiations); the int8 `WKQuantA8Blocked`/`W8A8Blocked` kernels are 8%. gufo launches each GEMM once at N=2048 (grid x = 8), not in chunks of 512. So an int8-activation path would copy the minor path; with iu8 only 5-10% faster than f16 in the core (44.7 vs 42.6 TOPS) and the scale conversion costing ~75% of the WMMA VALU time, it was not built.
- gufo's sustained in-model rates (median per launch): gate/up (M=34816 fused, K=5120) 38.8 TFLOP/s, down (K=17408) 38.5, other K=5120 shapes ~31 (small M). Ours, same profiler: 30.7-33 at N=512, 31-32 at N=2048; synchronised best-of timing at N=2048: gate 32.0, down 35.6. N does not matter for our kernel (31.7 / 31.1 at 2048 vs 30.7 / 30.8 at 512), so chunk size is not the lever.
- Workgroup swizzle (gufo orders its grid in groups of 8 row tiles for L2 reuse): `benchmarks/gemm_swizzle.jl` with group sizes 0/2/4/8/16: within +-1% everywhere (gate N=2048 32.0 -> 31.9, down 35.6 -> 36.2). Not adopted.
- CORRECTION / epilogue ablation (`benchmarks/gemm_epilogue.jl`, interleaved best-of-10, 8 rotating weight copies): the "K=5120 is 10% slower than K=17408" reading above was run-to-run clock noise (the same configuration measured 32.0 in one process and 35.5 in another); inside one interleaved run gate (K=5120) 35.5 and down (K=17408) 35.8 TFLOP/s at N=2048, so there is no K-dependent fixed cost. Output store: Float32 vs Float16 (one 16-byte store of 8 rows per tile): gate N=2048 35.5 -> 36.7 (+3%), o-proj-sized (5120x5120) 34.1 -> 36.5 (+7%), down 35.8 -> 35.7, N=512 gate 35.1 -> 36.3. A "no store" variant is *slower* (the 64-value reduction that keeps the MMAs alive costs more than the store), so it is not a valid ablation. Absolute rates differ by up to 10% between processes (shader clock), compare within one run only. Possible follow-up: Float16 outputs for the gate/up (and o-proj) GEMMs, which would also halve the SwiGLU kernel's reads; ~2-3% end to end at best.

### Float16 gate/up GEMM outputs and a vectorized SwiGLU (2026-10-12)
The 237 ms of `broadcast_cartesian` per 2048-token prompt was mostly the SwiGLU multiply (`out .= _swiglu.(g, u)`, 8.9M-element launches, 483 us each = 220 GB/s, i.e. at the memory bandwidth reading two Float32 matrices). Two changes:
- A matmul with a 4-bit weight whose only consumer is a `SwiGLUOp` (M % 256, K % 64, power-of-two group, N >= 384 and % 256, not otherwise fused) is computed as Float16 (`dty` pass in `compile`; `LUMINAL_F16_GEMM_OUT=0` disables). `wmma_q4_gemm!` takes a `Float16` C for the 256 x 256 kernel (`_wmma_store_big!` method: one 16-byte store of 8 rows per tile); `_q4_matmul!` routes anything else through a Float32 temporary. `_vec_rows(C)` replaces the hard-coded 4 rows per vector element in the kernel's row clamp (Float16 outputs have 8 per element: without it every row past M/2 read the weights of row M/2 - 1; found by `tests/test_f16_gemm_output.jl`).
- `_swiglu_f16_kernel!`: 8 elements per thread, 16-byte loads and stores, for Float16 gate / up / output. A plain broadcast over Float16 arrays loads 2 bytes at a time: the Float16 inputs alone only took the SwiGLU from 147 to 132 ms; the vector kernel takes it to 68 ms.
Kernel time of a 2048-token prompt (rocprof): 4458 -> 4336 ms (-2.7%): gate/up GEMMs 1594 -> ~1552, SwiGLU 147 -> 68. End to end, interleaved A/B (2 pairs each): 2048 tokens 573 -> 583 tok/s (+1.7%), 8192 tokens 534.7 -> 546.9 (+2.3%). The first A/B with only the Float16 outputs (no vector kernel) was inside the +-3% noise (545/573 vs 529/553). Gufo: 689 / 646.
Remaining broadcast launches (~77 ms): small per-head and conv-input elementwise ops. Absolute rates move +-3% between processes; compare only interleaved runs.

### Row-split delta rule (gufo's structure): 1.43 -> ~1.1 us per token (2026-10-13)
`_delta_rule_rows_kernel!` (Float32 q / k / v, dk = dv = 128, no snapshots; `LUMINAL_DELTA_RS=0` selects the LDS register kernel): a wave owns 8 rows of the state matrix (4 lanes per row, 32 keys each in registers), k and q are read from memory at every step (16-byte loads, L1 hits), no LDS and no barrier, so the 16 waves of a head are independent. Grid (dv / 64, nv, B), 256 threads.
- gufo's own microbenchmark on this machine (`tools/qwen27b/prefill_deltanet_bench.hip` built with hipcc): production kernel 0.545 ms per 512 tokens = 1.06 us/token (1.14 at 2048), its best variant 1.03; ours was 1.41-1.54. The Julia port with 4-D array indexing was 3x *slower* (4.4 us/token): the token loop had ~2000 instructions (64-bit address arithmetic per element, 35 scalar loads) against gufo's 214 (53 v_fmac, 16 `global_load_b128`). With raw pointers (`pointer(k) + offset`, `unsafe_load(ptr, i, Val(16))` for the eight 16-byte loads, constant offsets from a per-step base) it is 1.09-1.14 us/token, matching gufo's. `@inbounds` does not reach into `ntuple` closures (put it inside them) and closures over reassigned loop variables box (rebind with `let`).
- Things that made no difference: precomputing k.q in a separate kernel (the 32 FMAs are not the limit), scaling the state first (gufo's arithmetic order), `waves_per_eu`.
- End to end (interleaved, 2 pairs): 2048 tokens 585 -> 591 tok/s (+1.0%), 8192 547 -> 553 (+1.1%). Gufo: 689 / 646.
- Tests: `tests/test_delta_rule_gpu.jl` compares the row kernel with the global-memory kernel (lengths that do not fill chunks, fresh / continuing state, batch 2, GQA group 2-3).
- Next in this area: q / k / v arrive as slice copies of the conv output (`linear_copy`, ~95 ms per prompt) and the L2 norms / gated norm are separate kernels; gufo's kernel reads the conv output in place.

### DeltaRuleQKV: the recurrence reads q / k / v from the convolution's output (2026-10-13)
`DeltaRuleQKV` (src/Ops.jl, executor in src/Execution.jl, used by `GatedDeltaNet` in src/Qwen35.jl; `LUMINAL_DELTA_QKV=0` builds the old slice / l2norm / DeltaRule chain): one op replaces the three slice copies of the conv output (q, k, v), the two L2Norm kernels and the DeltaRule. `_l2norm_qk_kernel!` normalizes q and k of every head and token in one launch (one wave each, the old kernel's summation order) into two dense scratch arrays; the row-split recurrence reads v in place (token stride C = 2 nk dk + nv dv). Anything the fast path does not take (other head dims, snapshots for speculative decoding, CPU) slices and calls the separate ops: `tests/test_delta_rule_qkv.jl`.
Kernel launches per 2048-token prompt 8448 -> 7488; `linear_copy` launches 1200 -> 480. End to end, interleaved pairs under the same machine load (FPGA place-and-route jobs were running, so absolute rates were ~10% below the quiet-machine ones): 2048 tokens 535.7 / 536.5 vs 518.6 / 527.1 tok/s (+1.7..3.2%), 8192 tokens 555.1 / 555.1 vs 548.4 / 546.6 (+1.2..1.5%).

### GEMM ISA against gufo's `HalfPrefillGemmKernel<256,256,8,4>`: an exec-mask conditional in the loop head (2026-10-13)
Both main loops (K64 per iteration) have the same skeleton: 32 WMMA, 48 `ds_load_b128`, 4 `ds_store`, 2 barriers (`benchmarks/isa_loop.py` classifies a loop's instructions; gufo's from `hipcc -S --offload-arch=gfx1151` on `src/models/qwen/hip/kernels/prefill_fp16.hip`, ours from `AMDGPU.@device_code_gcn`). Differences:
- gufo's loop (361 instructions): 1 branch, 12 scalar instructions, 209 VALU (Q4_K decode, spread over the four K16 steps, between the WMMA groups); the global loads for the next tile are issued at the very end, after the second barrier.
- ours (292): 74 VALU (the int4 decode, all in one block next to the last K16 step) but 95 scalar instructions and ~10 branches, all exec-mask juggling from `q + 1 < nk && (ld = _g_load(...))` (the compiler cannot prove the conditional uniform, so it structurizes it into nested "Flow" blocks).
- Fix: load unconditionally with `min(q + 1, nk - 1)` (the last iteration re-reads the last tile). Interleaved best-of-10, 3 runs: gate 31.3 -> 32.0, down 31.0 -> 33.2 (N=512); gate 32.4 -> 33.4, down 33-35 -> 34-37 (N=2048): +2.5..6%. End to end under the same (loaded) machine: 535.7 -> 552.2 tok/s at 2048 tokens, 555.1 -> 574.2 at 8192 (+3%).
- No further gain from: splitting the decode across the K16 steps (as gufo does; +0-0.5%), turning the `for q in 0:nk-1` loop into a `while` (the remaining ~96 scalar instructions / 7 branches of the loop structure cost nothing measurable).
- `_wmma_q4_gpair_kernel!` (off by default) still has the conditional.

### Removing the remaining branches from the GEMM loop: possible, no gain (2026-10-13)
After the unconditional prefetch the loop still had ~7 branches and ~96 scalar instructions: they are bounds checks on the five global loads of `_g_load` (`v_cmpx_gt_u64` against the array lengths): `@inbounds` in the kernel body does not reach into a helper's own `getindex` calls; the helper needs it itself. With `@inbounds` inside `_g_load` / `_g_ks` the loop is 205 instructions with ONE branch (gufo's is 361 / 1), 135 VGPRs. But:
- the kernel got 5-25% *slower* (down N=512 34.1 -> 26.1 TFLOP/s): with the branches gone the machine scheduler sinks the global loads to the end of the iteration (right before the second barrier, `s_waitcnt vmcnt(0)`), so they no longer overlap the MMAs. The branch structure had been pinning them early.
- adding `llvm.amdgcn.sched.barrier(0)` after the load issue restores parity (33.0 vs 33.1, 34.1 vs 34.1, down N=512 33.5 vs 34.5), not more. A deeper pipeline as in gufo (loads two tiles ahead, decode of the previous tile's data during the MMAs) is also no faster (-1..-3%).
So the checked loop (kept) is as fast as the one with one branch: the scalar work and branches overlap the matrix work. `benchmarks/gemm_branches.jl`.
Measured on a quiet machine (the FPGA jobs finished): the current kernel does 36.6-37.7 TFLOP/s (gufo's in-model: 38.5), and prefill is 617.8 tok/s at 2048 tokens and 575.2 at 8192 (gufo 689 / 646).

### Does a newer LLVM backend help? ROCm's clang (LLVM 24) on our kernel's IR: no (2026-10-13)
Julia 1.12 generates code with LLVM 18.1.7; hipcc here is AMD clang 24.0.0git. `benchmarks/gemm_llvm_backend.jl` dumps the optimized IR of `_wmma_q4_g_kernel!` (`@device_code_llvm raw=true dump_module=true`), compiles it with `$ROCM/lib/llvm/bin/clang -x ir --target=amdgcn-amd-amdhsa -mcpu=gfx1151 -O3`, links with ld.lld, loads the code object with `HIP.HIPModule` and launches it through the same `HIPKernel` wrapper, interleaved with the Julia-compiled kernel in the same process (identical results). TFLOP/s, 8 rotating weight copies, best of 12:
| shape (M K N) | julia / LLVM 18 | clang -O3 | -O2 | -O3 + max-ilp | -O3 + max-occupancy |
| 17408 5120 512 | 36.0 | 35.8 | 35.9 | 36.9 | 35.9 |
| 5120 17408 512 | 37.3 | 37.0 | 37.0 | 37.8 | 36.9 |
| 17408 5120 2048 | 36.6 | 36.6 | 36.4 | 37.1 | 36.9 |
LLVM 24 at -O3 is not faster than LLVM 18 for this kernel (-0.5..0); `-mllvm -amdgpu-sched-strategy=max-ilp` is consistently +1.3..1.4%. So the compiler version is not what separates us from gufo's GEMM (38.5 in-model; ours 36-38 on a quiet machine); an out-of-process LLVM 24 backend or a newer Julia (1.13 nightly bundles LLVM 20) would not pay off for this kernel. The kernel-level wins so far came from source structure (indexing codegen, an exec-masked conditional), not from the backend. Not tested: the other kernels (delta rule etc.).

### ConvDeltaRule: convolution + SiLU + L2 norm of q / k in one kernel, parallel over tokens (2026-10-13)
`ConvDeltaRule` (src/Ops.jl, executor in src/Execution.jl, built by `GatedDeltaNet` unless `LUMINAL_CONV_FUSED=0`) replaces `CausalConv` + `DeltaRuleQKV` for prefill-sized steps (S >= 16, dk = dv = 128, no snapshots, Float32; anything else runs the two separate ops through the same op). `_conv_norm_kernel!`: one wave per (head of the stacked q / k / v rows, chunk of 16 tokens, batch), a lane owns 4 channels (coalesced), a sliding window of K inputs per channel in registers. The old conv kernel was one thread per (channel, batch) looping over all tokens sequentially (a dependent load per step, only 10240 threads); every output depends only on the previous K - 1 inputs, so the chunks are independent (the first chunk reads the old conv state, hence the state is written by a separate tiny kernel `_conv_state_kernel!` afterwards: a chunk waves writing it themselves would race with the first chunk's read). q / k heads are normalized in the same kernel (same summation order as `_l2norm_kernel!`) and written dense; v is written dense. The recurrence then reads dense q / k / v.
Kernel time per 2048-token prompt: conv 60-66 ms + l2norm_qk 19 ms -> `_conv_norm_kernel_` 40.8 ms (170 us per call; 21 MB read + 10 MB written per call is ~145 us at the memory bandwidth); total kernel time 4336 -> 4106 ms across today's changes. End to end, interleaved pairs on a quiet machine: 2048 tokens 622.6 / 620.8 vs 618.5 / 616.8 tok/s (+0.7%), 8192 tokens 578.1 / 579.1 vs 575.1 / 571.8 (+0.7%). Gufo: 689 / 646. `tests/test_conv_delta_rule.jl` compares with the two separate ops (chunk-unaligned lengths, S == 16, fewer tokens than the conv width, lens 0, fresh / continuing state, batch 2).
Remaining (profile, 2048 tokens): delta rule 141 ms (588 us per 512-token call), broadcast_cartesian 78 ms, swiglu 67 ms, flash 64 ms, rmsnorm 51 ms (805 launches), linear_copy 44 ms (480), conv_norm 41 ms, gated rmsnorm 29 ms, permutedims 28 ms.

### AttnPrep / GatedAttnOut: the gated attention layer's q / k preparation and output gate (2026-10-13)
The 480 `linear_copy` launches (43.7 ms per 2048-token prompt) all came from the 16 full-attention layers, 6 per layer per chunk: the query slice and the gate slice of q_proj's output (3.1M elements each, 184 us: 29.5 ms), and the rotary / pass-through slices of partial RoPE (`_rope`: slice, RotaryEmbed, slice, concat) for q (2.36M + 0.79M) and k (0.39M + 0.13M). Around them the layer also ran the q / k RMSNorms, three permutes (28 ms), the rotary kernel and the concat add.
- `AttnPrep` (src/Ops.jl; kernel `_attn_prep_kernel!`): from q_proj's / k_proj's raw output, the (1 + w) norm weight and the RoPE cos / sin tables, one wave per (head, token, batch) does the per-head RMSNorm, the rotate-half on the first `rotary_dim` dims (a lane owns e = lane + 32 i, so the rotated pair e, e + rotary_dim / 2 is in one lane for rotary_dim % 64 == 0) and writes (D, S, H, B) directly. The cos / sin tables are the same graph nodes as before (`_rope_cos_sin`, factored out of `apply_rotary_embeddings`), so the numerics of the rotation are unchanged.
- `GatedAttnOut`: attention output times sigmoid(gate), the gate read in place from q_proj's rows [query; gate] per head (no gate copy, no reshape, two broadcast kernels fewer); writes Float16 when it feeds the o_proj GEMM (added to the `dty` pass).
- Used by `llama_self_attn_cached` for prefill-sized steps (S >= 16) of a gated layer with QK-norm, 0 < rotary_dim < head_dim, rotary_dim % 64 == 0 (`LUMINAL_ATTN_PREP=0` disables); decode / verify steps keep the composite form. CPU and other shapes run a reference loop inside the op. `tests/test_attn_prep.jl`.
- Chunk-64 vs one-shot logits on 8 real layers: rel err 4.4e-5 (was 0: the one-shot path uses the composite ops), first 16 tokens identical (different RMSNorm summation order).
- End to end, interleaved pairs on a quiet machine: 2048 tokens 637.3 / 638.2 vs 624.1 / 620.4 tok/s (+2.1..2.9%), 8192 tokens 593.1 / 594.5 vs 576.2 / 579.4 (+2.6..2.9%). Gufo: 689 / 646.

### GEMM main loop: LDS / WMMA interleaving experiments (2026-10-13)
gufo's loop interleaves ds_loads with WMMAs finely (`L2 w W2 L2 w W2`); ours had the reads of a K16 step in one cluster. Two ways to ask for gufo's order:
- `llvm.amdgcn.sched.group.barrier` patterns (3 LDS reads : 2 WMMA, 2:1, 1:1, 6:4; `benchmarks/gemm_sched_group.jl`): WRONG RESULTS on LLVM 18 at the production K (rel diff 0.10-0.30 vs the unscheduled kernel; small K = 1024 happened to be exact for two of them). Do not use.
- source-level software pipelining of the fragments (read step ks + 1's A / B fragments before issuing step ks's MMAs; `benchmarks/gemm_frag_prefetch.jl`): identical speed (36.1 vs 36.0, 37.6 vs 37.5, 36.5 vs 36.5, 37.0 vs 37.1 TFLOP/s; the compiler already schedules the reads that way).
The kernel stays at 36-37.5 TFLOP/s on the quiet machine (gufo in-model 38.5). Tried and exhausted on the structure side: unconditional prefetch (+3%), split decode, depth-2 load pipeline, bounds-check removal, workgroup swizzle, f16 output store (gate/up only), ROCm's LLVM 24 backend (+1.4% with max-ilp), fragment prefetch, sched.group.barrier.

## Full re-measurement against gufo, idle machine (2026-10-13, commit 14a3718 + this note)
Qwen3.8-27B-UD-Q4_K_XL, both engines run one after the other on the otherwise idle machine (load average ~1, no other GPU user), two rounds each, prompt processing in tok/s from an empty context. gufo: `gufo bench -m <gguf> -p 512,2048,8192,16384,32768 -n 16` (its mixed Q4_K / Q5_K / Q6_K / IQ4_XS weights, 16.35 GiB); Luminal: `examples/qwen35_prefill_bench.jl --int4 --no-check --chunk=512 --lens=... --max-ctx=33500` (every weight re-quantized to uniform int4, group 32).
| prompt tokens | Luminal r1 / r2 | gufo r1 / r2 | Luminal / gufo |
| 512 | 615.8 / 613.3 | 424.0 / 420.0 | 1.46 |
| 2048 | 635.8 / 637.6 | 696.7 / 689.3 | 0.92 |
| 8192 | 592.4 / 590.4 | 650.6 / 649.8 | 0.91 |
| 16384 | 560.2 / 560.0 | 626.4 / 626.0 | 0.89 |
| 32768 | 507.9 / 507.8 | 581.7 / 584.2 | 0.87 |
Decode (batch 1, 512-token context): Luminal 13.3 tok/s (75.5 ms per token, `qwen35_gguf_bench.jl --int4`), gufo `tg16` 12.13 tok/s.
- Run-to-run spread is under 1% on both engines. Starting point of today's work: 465 / 416 tok/s at 2048 / 8192 tokens (this session's earlier window); now 637 / 591 (+37% / +42%); gufo is unchanged (~690 / 650).
- gufo's 512-token number is low because it takes its W8A8 int8 route below 1024 tokens of batch; the FP16 GEMM it uses from 1024 tokens up is the one we compare with.
- The gap grows with context (8% at 2048 -> 13% at 32768), so the part that is not the GEMM is attention: our flash prefill runs at ~18 TFLOP/s against gufo's ~20 and its share rises with the context (flash 31% of a late 32k prefill versus 6% early, see the context section above). Per-token GEMM work is the same at every length.
- Not like for like: Luminal runs every weight as int4 (the GGUF's Q5_K / Q6_K tensors are dequantized and re-quantized), gufo reads the file's own formats (16.35 GiB). That makes our decode read fewer bytes, and the quality of our int4 is lower than gufo's mixed formats; the prefill numbers are compute-bound and comparable, the decode comparison is not.

### Flash prefill: packed, coalesced V staging (2026-10-13)
Measured against gufo's own attention benchmark (`tools/bench/attn_causal_bench.hip`, built with hipcc; `-b 512 -s 32256 -k` = our 512-query chunk at depth 32256, D=256, 24/4 heads, 409 GFLOP): its shipped-style 128-row kernel 24.4 ms (16.8 TFLOP/s), its pipelined variant ("packed V + swizzle + prefetch") 20.05 ms (20.4 TFLOP/s); ours 22.5 ms (18.2). At S = 2048 from an empty context: gufo wmma-128 16.6, pipelined 19.6, ours 16.6.
Ablations of our kernel (`benchmarks/flash_ablations.jl`, S=2048 / depth 32256): no V transpose stores +8% / +2%, P stores +5% / +0.5%, softmax math +5% / +4%, next-tile global loads +5% / +6%: no single dominant cost, the kernel is limited by the per-16-key-tile dependency chain (3 barriers, 4 waves per SIMD), as gufo's own comments say.
Ported the V staging of gufo's pipelined variant into `_flash_prefill_lds2!`: waves 0-7 load the V tile with 4 lanes covering 64 contiguous bytes of one key row (full cache lines) and 8 key pairs per instruction, each lane writes the same 8 dims of two adjacent keys as 8 packed dwords of V^T (instead of 16 two-byte stores), 16 dwords of padding per 8 dims so the 32 lanes hit 32 banks; results bit-identical to the old kernel. S=2048 from empty 16.5 -> 17.4 TFLOP/s, depth 8192 18.6 -> 19.8, depth 32256 19.1 -> 20.3-20.9 (gufo's best variant: 20.4). A packed 8-byte store of the softmax probabilities added nothing at depth (20.4 vs 20.4) and was not kept.
End to end, idle machine (before -> after): 2048 636.7 -> 639.0, 8192 591.4 -> 597.1, 16384 560.1 -> 563.2, 32768 507.9 -> 514.0 tok/s (+0.4..1.2%; flash is 10% of a 16k prefill and ~18% of a 32k one). gufo: 693 / 650 / 626 / 583.
Where the remaining 8-12% gap sits (2048 tokens, per-prompt GPU time, gufo 2.96 s vs ours ~3.2 s): GEMM kernels ~4% above gufo's (2.77 vs 2.67 s), the small kernels ~0.22 s (513 vs 290 ms: delta rule 113 vs 92, flash 51 vs 41, swiglu 53, rmsnorm 41, broadcasts 62, conv/norm 33, attn_prep + gated out 24, gated rmsnorm 23). At 32k: GEMM ~1.7 s, small kernels ~3.5 s, attention ~3 s of the 8.3 s gap. `chunk_attention!` still converts q to Float16, zero-fills and permutes the output per layer (the `permutedims` / `broadcast` launches): ~15-20 ms per 2048-token prompt left to remove.

### chunk_attention!: Float32 query in, (D, H, S, B) output out (2026-10-13)
For the 128-row flash kernel (D = 256, the Qwen3.5 layers) `chunk_attention!` no longer makes a Float16 copy of q (allocation, zero fill, conversion broadcast), zero-fills an output scratch or permutes it afterwards: `_flash_prefill_lds2!` takes a Float32 query (`_q8` converts two 16-byte loads to the Float16 chunk in registers) and writes the output straight into the (D, H, S, B) layout the output projection reads (`Val(HS)`). The 64-row kernel and the Int8 cache keep the old path. `tests/test_flash_prefill.jl` has a chunk_attention! test against a host reference for both kernels (rel err < 3e-3). End to end (idle machine, 2 runs): 2048 tokens 639.0 -> 643.5 / 640.2, 8192 597.1 -> 598.5 / 601.6 tok/s (+0.5..0.6%).

## What does re-quantizing the GGUF to int4 cost? (2026-10-13, `examples/qwen35_quant_quality.jl`)
Qwen3.8-27B, 100 prompts of 256 real English tokens (this repo's README + gufo's docs, random windows, fixed seed), last-position logits of three builds on identical prompts:
- reference: the Q8_K_XL file, weights as int8 (~near-lossless: int8 error is ~1/250 of a group's range)
- "native Q4_K_XL": the Q4_K_XL file's own values (dequantized, kept as int8: this is what gufo's native-format kernels compute, to int8 noise)
- ours: the same Q4_K_XL file re-quantized to uniform int4, group 32 (what Luminal runs now)
| build (against the Q8_K_XL reference) | top-1 agreement | same top-5 set | mean KL(ref ‖ test) | median KL | max KL |
| native Q4_K_XL values | 97 % | 74 % | 0.0056 nats | 0.0040 | 0.026 |
| Luminal int4 g32 | 85 % | 41 % | 0.0613 nats | 0.0485 | 0.341 |
(int4 against the native Q4_K_XL values: 86 %, KL 0.056.) The int4 re-quantization adds ~11x the KL divergence of the file's own quantization and flips the top-1 token in 15 % of positions instead of 3 %. The log-probability of the true next token moves by -0.037 +- 0.042 (perplexity +3.7 %, not significant with 100 positions; the KL and the agreement are). The Q4_K_XL file is mostly Q5_K (42 % of the parameters), IQ4_XS (21 %), Q4_K (20 %) and Q6_K (13 %); int4 g32 (16 levels, f16 scale + min = 5.0 bits per weight) drops 1-1.5 bits on the 55 % that are Q5_K / Q6_K. Reading the file's formats directly (as gufo does) would also cut the load from 137 s (dequantize + re-quantize) to a few seconds (mmap + upload).
Memory traffic is within 3 % (5.0 vs the file's 5.12 bits per weight), so decode speed is not the argument: quality and load time are.

## Native GGUF weights (stage 1: Q4_K and Q5_K)

`load_qwen35_gguf(...; native=true)` (or `LUMINAL_GGUF_NATIVE=1`, off by default for now) reads Q4_K and Q5_K tensors in the file's own
values: `native_weight` (src/NativeWeights.jl) uploads the raw blocks and repacks them on the GPU into `Q4Weight`'s asymmetric group-32 planes
(Float16 scale = d*sc, min = -dmin*m); Q5_K adds a UInt32 fifth-bit plane (`Q5Weight`). Value error against `gguf_tensor`: 1.6e-4..1.8e-4 on
values up to 0.4 (Float16 rounding). Row slices/concats/permutations (value-head untiling) and the `ssm_out` input permutation (multiples of
128 = whole groups) work on the planes. The 256x256 kernel has a 5-bit variant (`_wmma_q5_g_kernel!`): 31.5-32 TFLOP/s vs 33.5 for 4-bit
(ffn_up shape, N=2048). Q5 weights with N <= 16 (decode) still dequantize to Float32 per call (slow, correct); Q5 weights are not yet
fused into swiglu / Float16-output passes. Q6_K, IQ4_XS, IQ4_NL, Q8_0, Q3_K, IQ3_S still go through dequantize + int4.

Quality (100 prompts x 256 tokens vs the Q8_K_XL int8 reference): Q4_K+Q5_K native (62% of parameters), rest int4:
top-1 88%, top-5 set 55%, KL 0.0327 nats (int4 everywhere: 85% / 41% / 0.0613; all native values: 97% / 74% / 0.0056).
Next: Q6_K (group-16 scales), IQ4_XS (table lookup), Q5 GEMV for decode, swiglu fusion for mixed formats.

Stage 2 (Q5 GEMV, Float16 gate/up outputs for 5-bit weights, residual fusion): the Q5 GEMV (`_q5_gemv_kernel!`, one row per workgroup)
is exact (4e-7) and reads 122 GB/s against 149 GB/s for the 4-bit GEMV (ffn_up shape, N=1). Prefill at 2048 tokens with Q4_K+Q5_K native:
612 / 611 tok/s against 643 / 641 for int4 everywhere (interleaved pairs): -4.7%, about what a 5% slower kernel on 62% of the weights
(plus Q5 layers not using the paired gate/up kernel, which is off by default anyway) predicts.

Stage 3 (IQ4_XS, IQ4_NL, Q6_K, Q8_0, lm_head): every format of the Q4_K_XL file except Q3_K / IQ3_S (3 tensors) is read in the file's values.
- IQ4_XS / IQ4_NL (`IQ4Weight`): the 16-entry value table is looked up four codes at a time with `v_perm_b32` (two 8-byte tables + select),
  codes become the Float16 mantissa of 1024 as for the 4-bit path, 1152 (1024 + the table's +128 bias) is subtracted exactly, then the step multiplies.
  GEMM 33.6-36 TFLOP/s (as fast as 4-bit), GEMV 109-113 GB/s.
- Q6_K (`Q6Weight`): low nibbles in the 4-bit layout, a Float16 step per 16 inputs, the two high bits of a K16 step's codes in one UInt32 (rotate + mask
  0x00300030 onto the mantissa); GEMM 31.3 TFLOP/s (K=17408 shape), GEMV 156 GB/s. The kernels are one generic `_wmma_gx_kernel!` over :q5/:iq4/:q6.
- Q8_0: the existing `QuantWeight` (group 32). `lm_head` (Q6_K) and ssm_alpha / beta / attn_v (Q8_0) were still re-quantized to int4 before this.
All repacks match `gguf_tensor` (dequantize err <= 1e-4, IQ4_NL / Q8_0 exact); row / column permutations exact.

Quality, 100 prompts x 256 tokens vs Q8_K_XL: int4 everywhere 85% / 41% / KL 0.0613 -> Q4_K + Q5_K native 88% / 55% / 0.0327 ->
+ IQ4_XS, Q6_K, IQ4_NL 90% / 64% / 0.0178 -> + lm_head 94% / 66% / 0.0106 -> + Q8_0 96% / 75% / 0.0056, the same as the int8 build of the file's values.
Prefill at 2048 tokens: 581 tok/s native vs 643 int4 (-9.7%; interleaving not repeated for this pair). Still off by default (LUMINAL_GGUF_NATIVE=1).
Open: decode speed and load time with native weights not measured; Q3_K / IQ3_S still int4; fused paired SwiGLU kernel and the
residual-add path only for 4-bit weights; the 5-bit / 6-bit / table kernels are 5-10% slower than the 4-bit one.

Native weights, load time and decode (examples/qwen35_gguf_bench.jl --int4, 512-token prompt, 64 new tokens, same session conditions):
load (after the header read to the weights on the GPU) 38 s native vs 142 s int4 (gufo mmaps in 2.7 s); prefill 567 vs 628 tok/s;
decode 117.8 ms/token = 8.5 tok/s vs 77.8 ms = 12.9 tok/s (gufo 12.1). The decode loss (-34%) is the GEMV kernels (one row per workgroup, 109-156 GB/s
against 149 for the 4-bit kernel at N=1, no multi-row sharing of the activation loads), not the bytes: the native layouts hold 5.5-6.1 bits per
weight against 5.0. Next: profile decode per format and give the 5-bit / 6-bit / table GEMVs the 4-bit kernel's structure.
