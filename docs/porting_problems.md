# Luminal Julia Porting Problems: A Debugging Summary

> [!NOTE]
> **Status Update (Feb 2026)**: The critical issues with `Metatheory.jl` described below have been **RESOLVED**. We successfully debugged the custom operator matching issue, submitted a fix to the upstream library (PR #190), and have fully integrated `Metatheory.jl` as the search-based optimization engine for Luminal. This document is preserved for historical context on the debugging process.

This document summarizes the challenges encountered during the early stages of porting Luminal's compiler optimizations from Rust (`egglog`) to Julia, specifically detailing the journey from `Metatheory.jl` to `Symbolics.jl` and back.

## Overarching Goal: Replacing `egglog` with a Julia-native E-Graph System

The primary objective is to port Luminal's compiler optimizations, which currently leverage `egglog` (an e-graph based term rewriting system written in Rust), to a Julia-native solution. This involves finding a Julia library that provides similar term rewriting capabilities.

## Journey: `Metatheory.jl` -> `Symbolics.jl` -> `Metatheory.jl` (Fixed)

### 1. Initial Attempt: `Metatheory.jl`

Initially, `Metatheory.jl` was chosen as the Julia-native e-graph and term rewriting system. However, significant problems were encountered.

#### Problem: Custom Unary Operator Failure
The main problem with `Metatheory.jl` v2/v3 was its **inability to correctly recognize or apply rewrite rules for custom unary operators**, specifically `Luminal_ReLU` (and other custom `Luminal_` functions). While it worked well for binary operators, it consistently failed to match patterns for unary functions.

**Debugging Steps:**
1.  **Placeholder Functions:** Introduced placeholder functions to ensure correct `Expr` matching.
2.  **Explicit Imports:** Added `import Luminal: Luminal_ReLU` to avoid scoping issues.
3.  **Isolated Testing:** Created reproduction scripts which confirmed the bug was specific to unary operators in the matching engine.

### 2. Strategic Pivot: `Symbolics.jl`

Due to the impasse, we temporarily pivoted to `Symbolics.jl` (which uses `Metatheory.jl` internally but had different behavior) and `SymbolicUtils.jl` for a rule-based approach. This allowed us to make progress on Phase 6 (Optimization and Compilation) using a greedy rewriting strategy.

#### Challenges with `Symbolics.jl`
- **Eager Evaluation:** `Symbolics.jl` would often aggressively simplify expressions (e.g., specific `max` calls) to concrete values or different forms, making it hard to preserve the exact graph structure needed for Luminal's compiler.
- **Dispatch Issues:** We encountered `MethodError`s with `TermInterface` when trying to inspect `Num` types wrapped around our custom ops.

### 3. The Breakthrough: Fixing `Metatheory.jl`

In Phase 8 (Search-Based Compilation), we returned to `Metatheory.jl` with a determination to fix the root cause. 

**The Fix:**
We identified that the issue lay in how `Metatheory.jl` handled function arity in its pattern matching compiler. We forked the library, implemented a fix that correctly handles custom unary operators, and verified it with a comprehensive test suite.

**Outcome:**
- The fix was submitted upstream.
- Luminal now successfully uses `Metatheory.jl` for global, search-based optimization (`compile_with_search`), enabling advanced features like complex operator fusion that were difficult to implement with the greedy `Symbolics.jl` approach.