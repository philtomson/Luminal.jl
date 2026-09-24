# E-Graph Rewrite Layer: Design Sketch and Prototype Findings

**Status (2026-09-24):** prototype in `src/EGraphRewrite.jl`, opt-in, not wired
into `compile()`. Driver and measurements: `examples/egraph_search.jl`.

## Goal

One rewrite layer, on Metatheory.jl, that explores equivalent versions of a
whole model graph and picks the fastest one *by measuring it*: upstream
Luminal's approach (egglog saturation, then a profiled search over extractions)
in Julia. It replaces the two dead paths in the tree today:

- the SymbolicUtils `@rule` set and `optimize()` in `Compiler.jl`, which have not
  run since `compile()` became an executor (commit `a75ca4b`, Feb 2026);
- `MetatheoryBridge.jl` / `compile_with_search`, which unrolls the graph into a
  tree (exponential on a residual network), drops shapes and views, and
  rebuilds inputs as dummy placeholders.

## Where it sits

```
model code ─► Graph ─► [rewrite layer] ─► Graph ─► compile() ─► CompiledGraph
                          │                          (executor: fusion, buffer
            to_egraph ─► saturate ─► extract          reuse, aliasing, HIP capture)
                                        ▲
                         measured search: compile + time candidates
```

`compile()` stays the executor. The rewrite layer only chooses *which*
equivalent graph it executes.

## Representation

| Luminal | E-graph |
|---|---|
| node with op `T` | e-node `xT(params, inputs...)`: head from the type name, `params` = the op's fields frozen into a literal tuple |
| input / weight | `xIn((node_id,))`: distinct per node, never merged |
| op with no inputs (ARange, Constant) | params also carry the output shape (`ARange(32)` ≠ `ARange(1)`) |
| edge read through a non-trivial view | explicit `xView((view_idx,), child)`; rules never match through a view |
| output shape | e-class analysis `Shp(dims)`, computed for ops rules create, **checked** against the graph for bridged nodes |
| outputs to keep | synthetic root `xRoots(...)` |

Hash-consing keeps shared subgraphs shared, so the e-graph is the DAG and not
a tree. Rebuilding is memoized per e-class, so extraction yields a DAG again.

Metatheory 3.0 notes:
- **Head names must not be defined Julia functions.** `pat_expr` stores
  `hash(op)` for a defined function rather than `hash(:op)`, and such patterns
  silently never match. This is still true on the `ale/3.0` head. Heads use an
  `x` prefix, and the old `MetatheoryOps` functions (`LuminalAdd(x, y) = …`) are
  exactly the pattern to avoid.
- In dynamic rules (`=>`), pattern variables are `EClass` objects (`a.data` is
  the analysis value); `p::Tuple` binds a literal. Returning `nothing` means
  don't fire.
- Literals in quoted exprs must be interpolated values (`$((4, 3))`), otherwise
  `(4, 3)` is a `:tuple` sub-expression.

## Rules (prototype)

- **Canonical:** Expand feeding a broadcasting Add/Mul → zero-copy size-1
  reshape; reshape∘reshape collapse; identity reshape.
- **Algebraic:** a scalar multiply after a matmul can move onto either operand.
- **Precision (opt-in, not exact):** a weight matmul may run as `MatMulF16`.
  This is a new op the executor understands; its weight is stored as Float16 at
  compile time.

Rules that should come next: permute∘permute, transpose-aware matmul (fold a
Permute into the GEMM flags), softmax/RMSNorm as recognized fused ops, and
choosing between kernel implementations (GEMV/GEMM/HalfWeight, workgroup sizes).

## Extraction and search

- **Static extraction:** greedy *DAG* extraction. A node's cost is the summed
  own-cost of the set of e-classes it depends on (bitsets), so shared subgraphs
  count once. The own-cost model is bytes moved plus a fixed launch cost.
  *Tree-cost extraction (Metatheory's default) is unusable here:* costs double
  per residual layer, overflow to `Inf`, and every alternative looks equal. The
  first prototype run picked Float32 for 216 of 245 matmuls for this reason.
- **Choice groups:** e-classes with several alternatives are grouped by
  (alternative signatures, shape), so the 22 layers' q-projections are one
  decision. TinyLlama decode: 2,345 nodes → 2,881 e-classes, 13 groups.
- **Measured search:** coordinate descent over groups. Each candidate is
  extracted, compiled with HIP capture, **checked against the original graph's
  outputs** (logits and new K/V over 4 steps), then timed (median of 30 replayed
  steps). A candidate that disagrees is rejected: that means a bug in a rule.

## Findings (TinyLlama 1.1B decode, Radeon 8060S)

| Graph | ms/token |
|---|---|
| original, Float32 weights | 26.4 |
| original, Float16 weights (hand-tuned path) | 16.7 |
| e-graph, canonical + algebraic rules, static pick | 26.5 |
| e-graph, + precision rule, static pick | 17.0 |
| e-graph, + precision rule, measured search | 16.6 |

- Every candidate matched the original graph (relative error ≤ 1e-6).
- **Precision is the only choice with a signal above noise.** Per group, Float32
  vs Float16: q/down projections 21.0 vs 17.0 ms, gate/up 20.6 vs 16.6, lm_head
  17.5 vs 16.7, k/v (256 rows) 17.3 vs 16.9, the smallest gain.
- **Measurement catches bad rewrites:** moving SiLU's `× -1` into the gate weight
  (a per-step multiply over a 23 MB weight) costs 33.9 ms.
- **Canonical and layout rewrites don't matter for decode today:** all within
  ±0.5 ms. With HIP capture, small elementwise kernels are nearly free and the
  step is bound by weight bandwidth. They should matter more for prefill and
  for larger batches, where elementwise traffic scales with tokens.
- **Noise is ~±2%,** the same size as the keep threshold. The search's one "win"
  (16.63 vs 17.02 ms) is noise. The search needs repeated, interleaved timings
  and a significance test before it can trust small differences.
- **Costs:** bridge + saturation 0.4 to 3.3 s; DAG extraction ~3.4 s (unoptimized);
  each candidate compile + verify + time takes seconds. Fine offline, too slow per call.

## Path to production

1. **Search hygiene:** interleaved repeated timings, confidence intervals,
   deterministic ordering (and bump Metatheory to the `ale/3.0` head for
   deterministic extraction), and a cache of results keyed by graph hash and
   device, so a model is searched once.
2. **Rules that pay off:** kernel-level alternatives (GEMV vs GEMM, HalfWeight
   group size, fused QKV and gate/up projections), which needs multi-pattern
   rules (one rewrite spanning several matmuls that share an input). Metatheory
   patterns are single-rooted, so this needs a rule that queries the e-graph for
   sibling e-nodes, or a pre-pass.
3. **Wire in:** `compile(graph; search=:static | :measured)` runs the layer
   before building the executor; `llama_generate` uses `:static` by default.
4. **Delete** the SymbolicUtils rules, `optimize()`, `MetatheoryBridge.jl` and
   friends once the new layer covers their intent.
5. **Search strategy:** coordinate descent is enough for ~13 groups. Upstream
   uses a genetic algorithm over per-e-class choices; that becomes worth it once
   rules create many interacting groups.
