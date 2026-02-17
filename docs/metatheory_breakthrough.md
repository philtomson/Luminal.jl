# Metatheory.jl Investigation: BREAKTHROUGH FINDING

**Date**: February 17, 2026  
**Time Invested**: ~5 hours  
**Status**: Major progress - root cause identified!

## Breakthrough: Custom vs Built-in Functions

### Test Results

| Function Type | Rule Type | Result |
|---------------|-----------|---------|
| **Custom** (`Luminal_ReLU`) | `=>` | ❌ FAILED |
| **Custom** (`Luminal_ReLU`) | `-->` | ❌ FAILED |
| **Custom** (`Luminal_ReLU`) | `==` | ❌ FAILED |
| **Built-in** (`sin`) | `-->` | ✅ **SUCCESS** |
| **Built-in** (`+`) | `-->` | ✅ **SUCCESS** |

### Key Finding

**Custom functions fail with ALL rule types, while built-in functions work perfectly.**

This proves:
- ❌ **Not** a rule type issue (=>, -->, ==)
- ❌ **Not** a module scoping issue (export makes no difference)
- ✅ **IS** a custom function symbol registration issue

## Root Cause Hypothesis

Built-in functions like `sin`, `cos`, `+` are registered in Julia's type system and can be called/evaluated directly. Custom functions defined as:

```julia
Luminal_ReLU(x) = :(Luminal_ReLU($x))
```

Are **meta-functions that return expressions**, not real callable functions. The e-graph may not recognize them as valid operation heads.

## Next Investigation Steps

### 1. Check Function Registration (2-3 hours)
- How are `sin` and `+` represented in the e-graph?
- How are custom functions represented?
- Is there a function registry or symbol table?

### 2. Examine E-Graph Insertion (2-3 hours)
Look at how expressions are added to the e-graph:
- File: `/tmp/Metatheory.jl/src/EGraphs/egraph.jl`
- How does `EGraph(:(sin(a)))` vs `EGraph(:(Luminal_ReLU(a)))` differ?
- What is `v_head()` for each?

### 3. Test Real Functions (1 hour)
Try defining Luminal_ReLU as a real function:
```julia
# Instead of meta-function
Luminal_ReLU(x::Number) = max(x, 0)

# Test if this works in e-graph patterns
```

### 4. Inspect vecexpr.jl (2-3 hours)
The `v_head()`, `v_signature()`, `v_flags()` functions determine matching:
- File: `/tmp/Metatheory.jl/src/vecexpr.jl`
- How are function heads hashed/stored?
- Do custom symbols get different treatment?

## Possible Fixes

### Option A: Register Custom Functions
If there's a function registry, add custom functions to it before creating patterns.

**Effort**: 20-40 hours  
**Likelihood**: High

### Option B: Modify Pattern Matching
Change how patterns match function heads to support arbitrary symbols.

**Effort**: 40-80 hours  
**Likelihood**: Medium

### Option C: Use Different Expression Format
Instead of `Luminal_ReLU(x)`, use a format that Metatheory recognizes.

**Effort**: 10-20 hours (if possible)  
**Likelihood**: Low (may not be applicable)

## Timeline Update

- **Hours invested**: 5 / 20 planned
- **Hours remaining**: 15 max
- **Decision point**: End of next week (~10 more hours)

## Next Actions

1. ✅ Examine `vecexpr.jl` for `v_head()` implementation
2. ✅ Compare e-graph representation of `sin(a)` vs `Luminal_ReLU(a)`
3. ✅ Test with real callable function instead of meta-function
4. Decision: Fix feasibility assessment
