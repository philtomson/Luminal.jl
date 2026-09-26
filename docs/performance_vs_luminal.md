# Luminal.jl vs. upstream Luminal: performance tracking

This document tracks how Luminal.jl performs against upstream
[luminal](https://github.com/luminal-ai/luminal) (checked out at `~/devel/luminal`),
so we can see where Luminal.jl is ahead, where it is behind, and what to work on.
Update it whenever either side publishes new measurements: add a row to the
[measurement log](#measurement-log), refresh the tables, and re-rank the
[improvement backlog](#improvement-backlog).

## How the comparison works, and its limits

**The two sides never ran on the same hardware.** Upstream's published numbers
come from an NVIDIA H200. Upstream has CUDA and Metal backends and no AMD one,
and ours come from an AMD Radeon 8060S (Strix Halo APU, ROCm 10). Luminal.jl's
CUDA path is untested. Raw latencies therefore compare *systems*, not software.
To compare the software, each side is also measured against its own hardware's
peak:

| | Upstream (H200) | Luminal.jl (Radeon 8060S) | Ratio |
|---|---:|---:|---:|
| Memory bandwidth (peak) | 4,800 GB/s (HBM3e) | 256 GB/s (LPDDR5X-8000, 256-bit) | 18.8× |
| FP32 compute (peak, non-tensor) | 67 TFLOP/s | ≈30 TFLOP/s (40 CUs, 2.9 GHz, dual-issue) | ≈2.2× |
| Best bandwidth observed here | — | ~236 GB/s (Float16 GEMV) | |

Batch-1 decode reads every weight once per token, so its efficiency is
*achieved bandwidth* = bytes read per token / time per token. Prefill is closer
to compute-bound, so its efficiency is *achieved FLOP/s* = 2 × parameters ×
tokens / time. The 8060S's FP32 figure is an estimate from its clocks, not a
vendor number.

**Workload.** Our [`examples/chat_benchmark.jl`](../examples/chat_benchmark.jl)
follows upstream's `llm_chat` benchmark (`examples/llm_chat/validation/benchmark-128-*`):
- batch 1, greedy decoding;
- a chat prompt of exactly 128 tokens, template included, padded with " quiet";
- exactly 128 output tokens, end-of-sequence ignored;
- one warm-up request, then the median of 3.

The protocols differ in two small ways:
- Upstream's TTFT includes one decode step; ours doesn't. Adding one Float32
  decode step (~148 ms) makes ours ~840 ms.
- Upstream prefills in chunks of 8 tokens into a KV cache sized for the
  request. Ours prefills all 128 tokens in one graph.

## Headline: Llama-3-8B-Instruct, 128 in / 128 out, batch 1

Same checkpoint and revision on both sides:
`NousResearch/Meta-Llama-3-8B-Instruct@53346005`.

| | Upstream, FP32, profiled search | Upstream, FP32, heuristic | Luminal.jl, FP32 | Luminal.jl, Float16 weights | Luminal.jl, int8 weights |
|---|---:|---:|---:|---:|---:|
| TTFT | 5,848 ms | 6,418 ms | **691 ms** | 683 ms | 688 ms |
| TPOT (decode) | 221.3 ms/token | 229.0 ms/token | **148.2 ms/token** | 67.2 ms/token | 38.8 ms/token |
| Decode throughput | 4.5 tok/s | 4.4 tok/s | 6.7 tok/s | 14.9 tok/s | 25.8 tok/s |
| Full request (128 + 128) | ≈33.9 s | 35.5 s | **19.5 s** | 9.2 s | 5.6 s |
| Load (weights, graph) | 85.1 s (load + build) | 85.1 s | **30.6 s** (load) | 30.6 s | 30.7 s |
| Search | 252.8 s (8 candidates) | 51.9 s (8 candidates) | none by default; ~12.7 min measured (79 candidates) | | |
| Cold first request | 17.0 s (after load and search) | | 42.9 s (includes compiling every graph) | 48.3 s | 53.9 s |

Upstream's figures are from `benchmark-128-2026-09-21` and
`profiled-128-2026-09-21`, both at upstream commit `65824ef3`. Its profiled full
request is computed as TTFT + 127 × TPOT. Ours were measured on 2026-09-26 at
Luminal.jl commit `373eb97` (plus this benchmark script).

### Normalized to each machine

| | Upstream (H200) | Luminal.jl (8060S) |
|---|---:|---:|
| FP32 decode: bytes per token | ≈30.0 GB (weights excluding the embedding table) | ≈30.0 GB |
| FP32 decode: achieved bandwidth | 136 GB/s = **2.8%** of peak | 202 GB/s = **79%** of peak |
| Float16 / int8 decode: achieved bandwidth | — (FP32 only) | 223 GB/s (87%) / ≈199 GB/s (78%) |
| FP32 prefill: 2 × 7.5 B params × 128 tokens | ≈1.9 TFLOP | ≈1.9 TFLOP |
| FP32 prefill: achieved compute | ≈0.34 TFLOP/s = **0.5%** of peak | ≈2.8 TFLOP/s = **≈9%** of peak |

Upstream's figures here come from its own kernel inventory
(`benchmark-128-*/OPERATOR_INSPECTION.md`), not from a trace of ours:
- **Decode:** upstream selects no fused attention, softmax or RoPE, and runs
  every layer's attention and RoPE as materialised multiply-then-reduce
  products.
- **Prefill:** most dense projections use cuBLASLt, but some still run as
  generic multiply+reduce.

## Where Luminal.jl is ahead

Measured unless marked otherwise.

1. **FP32 decode latency: 1.5× lower** (148 vs 221 ms/token) on hardware with
   1/19 the bandwidth. Normalized, that's 28× the bandwidth efficiency (79% vs
   2.8% of peak). This comes from hand-written GEMV and fused decode kernels
   (`DecodeAttention`, `RotaryEmbed`, `RMSNormOp`) instead of generic
   multiply+reduce, plus HIP graph capture of the whole decode step.
2. **Prefill / TTFT: 8.5× lower** (691 vs 5,848 ms; ~7× with upstream's extra
   decode step added to ours). That's about 18× the compute efficiency.
   rocBLAS runs every projection; the prompt is prefilled in one graph.
3. **Full 128/128 request: 1.7× faster** in FP32 (19.5 vs 33.9 s).
4. **Load time: 2.8× faster** (30.6 vs 85.1 s). Weights are uploaded as stored
   (BF16) and converted on the GPU.
5. **Reduced precision (upstream has none; it runs FP32 only):**
   - Float16 weights: 67 ms/token, 3.3× upstream's FP32.
   - int8 weights: 38.8 ms/token, 5.7× upstream's FP32. Perplexity +0.25% on
     TinyLlama; greedy output identical to Hugging Face on the 8B validation
     prompts.
6. **Batched decode:** upstream publishes batch-1 numbers only.
   - Llama-3-8B, int8: 76 tok/s at batch 4, 110 tok/s at batch 8.
   - Llama-3-8B, Float16: 52 / 77 tok/s at batch 4 / 8.
   - TinyLlama, int8: 564 tok/s at batch 8.
7. **Default time to first result.** With no search (our default), the first
   request is ready in load + cold request ≈ 74 s. Upstream needs load/build +
   search + cold request ≈ 155 s (heuristic) or ≈ 355 s (profiled).
8. **Per-candidate cost in the measured search:** ~9 s per timed candidate
   (compile + verify + 10 interleaved rounds). Upstream reports ~32 s per
   candidate, 60% of it in pinned-host-memory staging
   (`search-timing-2026-09-21`). See also "behind" #3: we time far more
   candidates.

## Where upstream is ahead, or we have no equivalent

These are ordered roughly by how much they limit Luminal.jl today.

1. **Model coverage.** Upstream runs Qwen3-0.6B/4B, Gemma3-4B (text) and
   Llama-3-8B, has a Qwen3-30B-A3B MoE definition, and loads a chat template
   from the checkpoint. We run Llama-family models only:
   - `llama_config` rejects RoPE scaling (Llama-3.1/3.2), tied embeddings and
     biases;
   - there's no Qwen3 Q/K normalisation, no Gemma, no MoE;
   - chat templates are hard-coded for two formats (`chat_prompt`).

   Three of upstream's four benchmark models can't be compared yet.
2. **Multi-turn KV reuse.** Upstream's chat session keeps the KV cache across
   turns and only prefills the new tokens. `LlamaSession.generate` prefills the
   whole prompt on every call, so a follow-up turn costs a full prefill of the
   conversation.
3. **Measured search time for 8B: ~12.7 min** against upstream's 4.2 min. We
   time many more candidates: every int8 variant for every matrix shape, plus
   19 lowering sites, each twice when it passes. Nothing beat the defaults on
   that run. A cheaper search could prune variants the static cost model or
   earlier shapes rule out, stop early when a decision group shows no signal,
   and share results across batch sizes.
4. **Cold compile.** Our first request takes 43–54 s, mostly Julia JIT
   compilation and compiling each graph. Upstream's first request takes 17 s,
   though after an 85 s load/build and its search. Precompiling Luminal.jl's
   hot paths (PrecompileTools) and caching compiled graphs on disk would cut this.
5. **One graph for all shapes.** Upstream compiles one plan with dynamic
   query/context dimensions and search buckets. We compile separately per
   prefill bucket (16/32/…/256, then multiples of 256) and per batch size, each
   costing seconds on first use.
6. **Backends and portability.** Upstream has CUDA (cuBLASLt) and Metal. We run on
   AMD ROCm; the CUDA path is untested; there's no Metal.
7. **Validation breadth.** Upstream checks full-vocabulary logits against Transformers
   for 3 models × 7 chat turns × prefill chunk sizes 1/4/8, with a stated
   tolerance, including KV reuse and resets. Ours is 3 prompts on Llama-3-8B
   (logits at every prompt position, greedy continuations) plus TinyLlama perplexity.
8. **Long context.** Untested beyond a few hundred tokens here. `DecodeAttention`
   caps the context at 8,192 (local-memory score buffer); prefill uses
   materialised `S × S` attention scores, so memory grows quadratically.
9. **Search generality.** Upstream's e-graph can discover implementations (views vs
   copies, GEMM+add) from primitive ops. Our speed comes mostly from hand-written
   kernels that the model code inserts, and fused ops the search can't derive.

## Not yet comparable

| Case | Luminal.jl | Upstream | To make it comparable |
|---|---|---|---|
| Qwen3-0.6B / 4B, Gemma3-4B | not supported | 128/128 benchmark published | implement the architectures (backlog #1) |
| TinyLlama 1.1B | 6.4 ms/token int8, 11 ms Float16 | no current example | run upstream's `llm_chat` with a Llama-2-style config, if its adapter accepts it |
| Whisper (speech) | whisper-tiny: ~35 ms encoder, ~2.5 ms/token decode | no current example (the old Rust example is gone) | none upstream |
| Same hardware | AMD only | NVIDIA/Apple only | run Luminal.jl's CUDA path on an NVIDIA GPU, or upstream on an H200 next to it |
| Batch > 1, low precision | measured | not published | upstream would need batched sessions and FP16/int8 weights |

## Improvement backlog

Ranked by what the comparison says matters. "Gain" is the expected effect on
the metrics above.

| # | Item | Gain | Effort |
|---:|---|---|---|
| 1 | Model coverage: Qwen3 (Q/K norm, tied embeddings), Gemma3, Llama-3.1/3.2 RoPE scaling, templates from `tokenizer_config.json` | Makes 3 more of upstream's benchmarks comparable; basic usability | medium–large |
| 2 | Multi-turn KV reuse in `LlamaSession` (prefill only new tokens) | Follow-up TTFT becomes proportional to the new message, not the conversation | small–medium |
| 3 | Faster measured search: prune dominated variants, early-stop groups, reuse across batch sizes | 12.7 min → minutes on 8B | medium |
| 4 | Cold start: PrecompileTools workload, on-disk graph/kernel cache | 43–54 s first request → seconds | medium |
| 5 | Prefill: fused/flash attention for long prompts; Float16 GEMM by default where tolerable | TTFT at long context; memory | medium |
| 6 | Decode: 4-bit weights (int4 GEMV) | ~1.6–1.8× 8B decode (weights 7.7 GB → ~4 GB) | medium |
| 7 | Same-hardware comparison: CUDA path on an NVIDIA GPU | Separates software from hardware in every number above | small once a GPU is available |
| 8 | Broader validation: multi-turn, chunked prefill, longer contexts, more models | Confidence in the other items | ongoing |

## Measurement log

| Date | Side | Commit | Hardware | Model / precision | TTFT | TPOT | Notes |
|---|---|---|---|---|---:|---:|---|
| 2026-09-21 | upstream | `65824ef3` | H200 | Llama-3-8B / FP32, heuristic | 6,418 ms | 229.0 ms | `benchmark-128-2026-09-21` |
| 2026-09-21 | upstream | `65824ef3` | H200 | Llama-3-8B / FP32, profiled | 5,848 ms | 221.3 ms | `profiled-128-2026-09-21` |
| 2026-09-26 | Luminal.jl | `373eb97` | Radeon 8060S | Llama-3-8B / FP32 | 691 ms | 148.2 ms | `chat_benchmark.jl` |
| 2026-09-26 | Luminal.jl | `373eb97` | Radeon 8060S | Llama-3-8B / Float16 | 683 ms | 67.2 ms | |
| 2026-09-26 | Luminal.jl | `373eb97` | Radeon 8060S | Llama-3-8B / int8 | 688 ms | 38.8 ms | |

## Reproduce

Luminal.jl (checkpoint in `llama3_8b_instruct/`):

```bash
julia --project=. examples/chat_benchmark.jl llama3_8b_instruct f32    # also f16, int8
julia --project=. examples/batched_decode.jl llama3_8b_instruct int8 1,4,8
```

Upstream: see `~/devel/luminal/examples/llm_chat/validation/benchmark-128-2026-09-21/RESULTS.md`
(`target/release/examples/benchmark --model llama3 --input-tokens 128 --output-tokens 128 --prefill-chunk 8 --repetitions 3`).
