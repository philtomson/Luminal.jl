# Porting the DFlash2 block drafter (Qwen3.8-27B) to Luminal

The draft model is `/mnt/LLMs/models/gufo/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (1.1 GB, arch `dflash`,
z-lab/Inco AI). This note records how it works (from its GGUF metadata and from hipfire's implementation,
`~/devel/AI/hipfire/crates/hipfire-runtime/src/dflash.rs` and `kernels/src/dynamic_conv_f32.hip`) and the plan.

## What the draft does

Instead of MTP's chain of one-token drafts, one forward pass proposes a whole block: `block_size = 8` in the file (hipfire runs 8 or 16;
the conv and the selector work for any length).

Inputs per cycle:
- **Block rows** B: row 0 is the *seed* (the last accepted token), rows 1..B-1 are the MASK token (`mask_token_id` 248070).
  Their input vectors are the *target's* token embeddings (the draft has no embedding table of its own).
  RoPE positions: the next B positions after the context.
- **Context**: for each of the last 2048 positions (`sliding_window`, all 5 layers are sliding), the hidden states of the *target*
  at layers `[6, 20, 34, 48, 62]` (`target_layers`), concatenated to 25600 values. `fc` (25600 -> 5120) then `hidden_norm`
  (`enc.output_norm.weight`) turn each position into one 5120 vector. Each draft layer projects these with its own `attn_k` / `attn_v`
  (K gets `attn_k_norm` and RoPE at its absolute position), so the context acts as a K/V cache that grows by the accepted rows each cycle.

Per layer (5 layers, hidden 5120, 32 query heads, 8 KV heads, head dim 128, FFN 17408, rms eps 1e-6, rope base 1e7):
1. `x_norm = rmsnorm(x) * attn_norm`.
2. **Dynamic causal conv ("prepare")**: `attn_conv_proj` (5120 -> 1280 = 2 phases x kernel 2 x 320 groups) gives per-row coefficients
   `dyn[row, phase, off, group]`; `out[r, c] = sum_{off=0,1; r>=off} (base[off, c] + dyn[r, 0, off, c/16]) * x_norm[r-off, c]`
   with `attn_conv_base` ([5120, 2], extra dim of 2 = phase) as `base`. Left-zero padded, over the B block rows only (no history across cycles).
   This is the input to Q, K, V (noise rows).
3. Q (`attn_q`, `q_norm`, RoPE), noise K/V (`attn_k`, `attn_k_norm`, RoPE), then attention of the B query rows over
   [context K/V (last 2048) | block K/V], **not causal** (`attention.causal = false`), GQA 4:1.
4. `attn_output`, then the *finish* conv (phase 1 of the same coefficients) and the residual add (details to confirm in dflash.rs ~l.3790).
5. FFN: the same pattern with `ffn_norm`, `ffn_conv_proj` / `ffn_conv_base` around gate/up/down (SiLU).
After the layers: `output_norm`.

**Candidate selector** (replaces a draft lm_head): the final hidden of block row r is projected with `selector_hidden` (5120 -> 256).
The *target's* `lm_head` gives logits for the B rows; the top 16 candidates per row (`selector_top_k`) are scored as
`logit + sum_d predecessor[prev_token][d] * proj[d] * successor[cand][d]` (`selector_predecessor` / `selector_successor`, 256 x 248320),
chained: `prev` starts as the seed (anchor) and becomes the chosen token of the previous row. Greedy: argmax of the score.
Output: B-1 proposed tokens (rows 1..B-1).

Then the target verifies `[seed, drafts...]` in one batched step (up to 16 rows) and accepts the longest matching prefix (greedy), or by
rejection sampling for sampled generation (hipfire's slot path is greedy only; ours must stay distribution-exact, so the proposal needs to be
a point mass as for MTP, which the argmax selector is).

## What Luminal needs

1. **Target hidden capture**: the hidden state after layers 6, 20, 34, 48, 62 for every prompt and generated token, kept for the last 2048
   positions (about 100 KB per position, 200 MB). The prefill graph and the verify step both have to return them.
2. **Draft forward**: GGUF loading of the `dflash` architecture; a graph for the block pass with the dynamic conv (expressible with a row shift,
   elementwise multiply and a group broadcast; a fused kernel later), non-causal attention over a ring of context K/V (an incremental K/V cache
   per layer, like the MTP head's), the selector chain (host side is fine at 16 candidates x 256 dims x 7 rows).
3. **Loop**: draft block -> verify B rows (the multi-column GEMV costs 143-167 ms at 8 rows; a WMMA small-N native kernel was prototyped in
   `benchmarks/wmma_nat_skinny.jl`) -> accept -> roll back the recurrent states (the verify graph already snapshots per row) -> append the accepted
   rows' hidden states to the context.
4. **Adaptive block length** from the running acceptance, like `_best_ndraft`.

## Cost model (to check against measurements)

- Draft pass: 1.1 GB of weights (about 5 ms at 220 GB/s) + the `lm_head` over B rows (1.1 GB Q6, about 5 ms) + 5 layers of B-row GEMMs: roughly 15 ms,
  against 24-28 ms for a 3-draft MTP chain today.
- Verify at B = 8 rows costs about 40-60 ms more than at 4 rows today with the GEMV kernel (167 vs 131 ms from `_SPEC_COST_VERIFY_NATIVE`).
- Break-even: the block drafter must raise tokens per cycle from about 2.4-3.3 to above about 3.5-4.

## Plan (each stage ends with a go/no-go)

1. **Acceptance harness (the gate).** Load the draft, capture target hiddens on real text (a few opencode-style conversations: reasoning prose,
   code, tool calls), run the draft at every position, and measure the greedy acceptance length distribution against the target's own continuation
   (teacher-forced, no serving integration). A bug in the port shows up as near-zero acceptance. Compare with MTP's measured 2.4-3.3 tokens per cycle.
   Stop here if the mean accepted length is below about 3.5 on the reasoning prose.
2. Fast path: fused conv/attention kernels, device-side selector, the capture inside the prefill/verify graphs.
3. Integration into `_spec_loopN!` (block verify, adaptive block length, sampled verification, the end-of-generation checkpoint).
4. A skinny native WMMA verify kernel if the 8-16 row GEMV verify is the bottleneck.

## Stage 1 result (2026-10-16): go

`benchmarks/dflash2_accept.jl` is a plain-array reference implementation of the draft (Float32, no graph compiler) run on four texts the target
generated greedily (640 tokens each, short contexts of 30-700 tokens), with the target's hidden states after layers 6, 20, 34, 48 and 62
(`Decoding.prefill_hiddens`). 152 blocks of 8 (seed + 7 drafts), positions every 16 tokens; "tokens per cycle" is 1 + the longest accepted prefix.
MTP was measured on the same prompts with `generate_ids_speculative(ndraft = 0)` (adaptive 1-3 drafts).

| Text | DFlash2, B = 8 | MTP greedy | MTP sampled (T 1, top-p .95, top-k 20) |
|------|---------------:|-----------:|-----------------------------------:|
| reasoning prose (sky and sunset) | 4.21 | 2.42 | 2.46 |
| code (merge intervals) | 5.37 | 3.14 | 3.20 |
| technical prose (flaky CI) | 2.34 | 2.32 | 2.25 |
| narrative (Romeo and Juliet) | 2.45 | 2.29 | 2.09 |
| overall | 3.59 | 2.50 | 2.44 |

Accepted-length histogram (0..7 drafts): 33, 24, 33, 20, 12, 4, 6, 20. Conditional acceptance of draft j given 1..j-1: 78, 80, 65, 68, 71, 87, 77%.
The draft's own logits rank the true token first in 80 / 67 / 48 / 39 / 33 / 25 / 24% of rows 1-7 and within the top 16 in 100 / 92 / 88 / 75 / 77 / 62 / 61%.

Settled by the experiment: `target_layers` index the *output* of layer i (offset 0), the block uses rotate-half RoPE, the norm weights are used as stored (no 1 + w),
`conv_base` is `[hidden, off, phase]`, and the selector's unary term is the raw target logit. The first run scored 1.02 tokens per cycle because of a
reshape error in my head split (features of a head must be contiguous); with that fixed everything above matched without further tuning.

What this does and does not show:
- DFlash2 gets 1.7x MTP's tokens per cycle on reasoning prose and code, and no better than MTP on the two plainer texts. Sampled MTP is within 3% of greedy, so
  sampling should not change the comparison much (not measured for DFlash2; the rejection scheme needs the drafter to propose a point mass, which the argmax selector does).
- Contexts here are short. At 100k+ tokens the draft sees a 2048-row window (its training window), so acceptance could differ.
- Cycle cost is not measured. An 8-row verify with the multi-column GEMV is extrapolated at about 167 ms (4 rows: 131 ms), and graph capture fails at 8 rows.
  With 15-20 ms of drafting a cycle would cost about 190-200 ms against 160-185 ms for MTP, so the net decode gain is about +30% overall,
  about +50% on reasoning prose and code, and a loss on plain prose unless the block length adapts to the recent acceptance.

## Stage 2 result (2026-10-17): integrated, correct, and about even with MTP at long context

`--dflash=<draft gguf>` on `examples/qwen35_server.jl` (and `generate_ids_dflash`) runs the drafter in the real pipeline:
- the target's five layer outputs are extracted inside the chunked prefill (only the last 2048 prompt tokens) and the batched verify step;
- the block pass and the context-ring update are fixed kernel sequences on persistent buffers, captured in HIP graphs (draft pass 18 ms, ring update 3-10 ms);
- verify of up to 8 rows uses the split-K decode attention in one pass over the KV cache (each KV group's 6 query heads are split over two adjacent workgroups of 3 heads, so
  that 8 tokens x 3 heads = 24 rows fit the kernel's LDS; the second workgroup's reads hit the cache), capturable in a HIP graph, and the Float16 dot2 GEMV does up to 8 columns per
  pass (bit-identical to 4, 40-45% faster for 5-8 columns: it also speeds up MTP's longer verifies);
- the block length is chosen per cycle from the running acceptance and the measured cost of each length;
- the end-of-generation state checkpoint and the prefix cache work with it (the ring's validity is tracked next to the live K/V: `PrefixCache.live_df`).
Greedy output is identical to plain greedy decoding; a follow-up request resumes from the checkpoint.

Measured (same prompts, both drafters in one process, sampled as the server does where noted):

| | DFlash2 | MTP |
|-|--------:|----:|
| prompt 1 (reasoning prose), short context, greedy | 19.9-21.0 tok/s, 3.6 tokens/cycle | 14.5-18.4 tok/s, 2.6 |
| prompt 2 (code), short context, greedy | 23.1-23.3 tok/s, 4.35 | 18.2-18.4 tok/s, 2.9 |
| tool call (file write), short context | 32-41 tok/s, 5.8-7.3 | about 22 tok/s, 3.6 (earlier server logs) |
| 40k tokens of context, reasoning text, sampled | 18.6-19.4 tok/s, 4.0-4.5 | 16.4-17.1 tok/s, 2.75-3.1 |
| 40k tokens of context, new code, sampled | 17.6-17.8 tok/s, 3.9-4.0 (13.3-16.3 with the two-launch attention) | 15.1-15.9 tok/s, 2.8-2.9 |

A cycle costs about 150 ms (8-row verify) + 18 (draft) + 5 at short context and about 185 + 27 + 6 at 40k tokens, against about 185 ms for MTP, so the extra
tokens per cycle are partly paid for in verify time at long context (the single-pass 8-token attention took the verify at 40k tokens from 195 to 185 ms; per attention layer an
8-token step costs 2.2 ms against 1.5 ms for 4 tokens at 40k tokens of context, and 5.6 against 3.5 ms at 115k). What is left to gain is mostly a faster 8-column weight multiply
(100-130 GB/s on the layer matrices against 215 GB/s on the output matrix).

A deadlock found on the way (it can also happen without DFlash): a finalizer freeing GPU memory (`hipFreeAsync`) on the HTTP thread while the GPU worker launches a captured
graph (`hipGraphLaunch`) hangs inside the HIP runtime. The server now runs with automatic garbage collection off and collects from the GPU worker thread only (`Luminal.hostgc`).

