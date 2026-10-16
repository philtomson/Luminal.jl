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

## Status

Stage 1 not started; the design is now settled from hipfire's code. Resolved:
- The conv coefficients are computed **once per sublayer** from the normed input (`*_conv_proj`, 1280 values per row), and both phases use them:
  phase 0 (offset 0) wraps the *input* of Q/K/V (attention) or of gate/up (FFN); phase 1 (offset 320 x 2) wraps the *output* of `attn_output` / `ffn_down`
  before the residual add: `x = residual + conv1(wo(attn(...)))`. `*_conv_base` is `[phase, off, hidden]`, `dyn` is `[row, phase, off, group]`.
- The block rows are `[seed, MASK x (B-1)]`, embedded with the **target's** token embedding table; RoPE positions for the queries are
  `position .. position + B - 1` where `position` is the seed's position; context keys carry their own absolute positions (the last 2048).
- The final norm is `output_norm`; rows 1..B-1 go through the **target's** `lm_head` (B-1 x 248320 logits) and the selector, anchored on the seed.
  Row 0 (the seed) is not scored.
- Verify feeds `[seed, d_1..d_(B-1)]`; draft d_i is accepted while the target's argmax after the previous fed token equals it.

Still to settle empirically in the acceptance harness (a wrong choice shows as near-zero acceptance):
- whether `target_layers` index the output of layer `i` (HF `hidden_states[i + 1]`) or its input;
- the selector's unary term (raw target logit, as in hipfire, versus a scaled or log-softmax one).
