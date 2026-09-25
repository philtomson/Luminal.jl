# E-Graph Rewrite Layer: Design Sketch and Prototype Findings

**Status (2026-09-24):** in `src/EGraphRewrite.jl`, opt-in through
`compile(graph; retain=..., search=:static | :measured, precision=...)` and
`llama_generate(...; search=...)` (`examples/tinyllama_chat.jl --search=measured`).
The old SymbolicUtils rules, `SymbolicIntegration.jl` and the tree-based
`Metatheory*.jl` modules have been removed. Standalone driver:
`examples/egraph_search.jl`; tests: `tests/test_egraph_rewrite.jl`.

## Goal

One rewrite layer, on Metatheory.jl, that explores equivalent versions of a
whole model graph and picks the fastest one *by measuring it*: upstream
Luminal's approach (egglog saturation, then a profiled search over extractions)
in Julia. It replaced two dead paths (now deleted):

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

- **Merged projections (multi-node, `merge_projections!`):** weight matmuls
  sharing an input (q/k/v, gate/up) get the alternative
  `Slice_i(MatMul(concat(W...), x))`. Metatheory patterns are single-rooted, so
  this is a pass over the e-graph. The concat is `Pad` + `Add` over weights,
  which `compile()` constant-folds at compile time, and a row slice of the
  result aliases instead of copying.

- **Kernel alternatives:** `MatMulF16(group)` is offered at 64, 128 and 256 threads
  per output row (precision rule), and `MatMul(Permute(a), b)` with a
  swap-first-two-dims permute becomes `MatMulT(ta, tb)`, which uses BLAS transpose
  flags instead of copying. Choice signatures include op parameters, so these
  are distinct options.

Rules that should come next: permute∘permute, softmax/RMSNorm as recognized
fused ops, a split-K GEMV variant for short-and-wide weights.

## Extraction and search

- **Static extraction:** greedy *DAG* extraction. A node's cost is the summed
  own-cost of the set of e-classes it depends on (bitsets), so shared subgraphs
  count once. The own-cost model is bytes moved plus a fixed launch cost; e-classes
  that depend only on weights are free (constant-folded), and aliasing slices are free.
  Then a **refinement** pass switches classes whenever the total cost of the whole
  extracted DAG drops, first re-picking any classes the switch pulls in for their
  marginal cost. Greedy alone got stuck on SiLU's `gate * -1 * log2e` chain:
  with scale motion, each link could be a separate 23 MB matmul, and switching
  one link at a time never paid off.
  *Tree-cost extraction (Metatheory's default) is unusable here:* costs double
  per residual layer, overflow to `Inf`, and every alternative looks equal. The
  first prototype run picked Float32 for 216 of 245 matmuls for this reason.
- **Choice groups:** e-classes with several alternatives are grouped by
  (alternative signatures, shape), so the 22 layers' q-projections are one
  decision. TinyLlama decode: 2,345 nodes → 2,881 e-classes, 13 groups.
- **Decisions:** per-class choice groups, plus one *joint* decision per merged
  projection shape (all members switch together; one class at a time would
  never try it).
- **Measured search** (`measured_search`): coordinate descent over decisions. Each
  candidate is extracted, compiled as it will run (HIP capture), **checked
  against the original graph's outputs**, then timed against the incumbent in 5
  interleaved rounds. It's kept only if its median is >1% faster and it wins ≥4
  of 5 rounds. A candidate that disagrees is rejected: that means a bug in a rule.
- **Cache:** winners per decision label (labels encode shapes and signatures)
  go in `~/.cache/Luminal.jl/search/`, keyed by the decision set, device and
  compile options. A later compile applies them after one verification.

## Findings (TinyLlama 1.1B decode, Radeon 8060S)

Kernel alternatives (latest): the measured search picked **GEMV group size 256**
for every projection shape (5/5 rounds each), 14.77 ms/token. That fed back
into the default: `MatMulF16()` and the plain `weight_dtype=Float16` path now use
256, which alone takes the non-search path from 16.7 to **15.1 ms/token**.
`MatMulT` and the merges were within noise once group sizes were right.
One lesson: variants the cost model can't tell apart must tie-break toward the
known-good default. Extraction by rule order picked group 64 and started the
search at 17.8 ms.

Latest full search (kernel variants, group size 256 default, merges): **14.54
ms/token** with Float16, vs 14.9 for the plain Float16 path.

**Search crash (fixed):** a measured search crashed with a `DimensionMismatch`
(a merged gate/up matmul given the merged q/k/v Float16 weight). An
instrumented rerun caught the cause. The `HalfWeight` cache was keyed by
`objectid`, and the search's churn of folded, concatenated weights let a new
array take a dead array's objectid before the dead array's entry was removed:
`STALE HIT: cached HalfWeight (2560, 2048) for array (11264, 2048)`. Entries
now keep a `WeakRef` to their source and only hit for the same array, and
folded weights (per-compile) bypass the cache. A same-shape stale hit would
have silently used another layer's weights, so verification protects the
search as well. A separate segfault inside `hipGraphLaunch` was seen once and
not reproduced (no GC or graph destruction during capture in the instrumented
run). As hardening, captures now run a full GC first and disable GC during
capture, and HIP graphs are destroyed at safe points rather than in finalizers.

Merged projections, refined extraction, measured search:

| Graph | ms/token |
|---|---|
| original, Float16 weights (hand-tuned path) | 16.6 to 16.8 |
| e-graph + precision + merges, measured search | **15.7 to 16.0** |

The search kept q taken from a merged q/k/v product and gate/up as slices of one
merged product (each 5/5 rounds). Generation output is unchanged word for word.
First search of the model: ~4 to 5 min; with the cache: one verification.

First prototype run (before merges and refinement):

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

## Done / next

Done: interleaved timing with a win-rate test; Metatheory pinned to the
`ale/3.0` head; result cache; merged projections (with compile-time constant
folding and slice aliasing); `compile(...; search=...)`; old paths deleted.

Next:
1. **Kernel alternatives as rules:** GEMV vs GEMM, HalfWeight workgroup size,
   transpose flags. These are where the measurable wins are for decode.
2. **Prefill:** the rewrites that were neutral for decode (expand elimination,
   layout) should matter with many tokens; `search` needs symbolic-shape
   support (`search_inputs` + `sym_vals`) for the variable-length prefill graph.
3. **Faster search:** extraction (~8 s on 3,100 e-classes) and per-candidate
   compiles dominate; incremental re-extraction and reusing compiled subgraphs.
4. **Search strategy:** coordinate descent is enough for ~18 decisions. Upstream
   uses a genetic algorithm; that becomes worth it with more interacting rules.
