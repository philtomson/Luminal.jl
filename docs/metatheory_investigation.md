# Metatheory.jl Investigation Progress

**Date**: February 16, 2026  
**Status**: Phase A - Investigation In Progress  
**Time Invested**: ~2 hours

## Key Files Analyzed

1. **`Syntax.jl`** - Pattern creation and rule macro expansion
2. **`ematch_compiler.jl`** - E-graph pattern matching compilation  
3. **`vecexpr.jl`** - Vector expression representation (likely)

## Initial Findings

### Pattern Creation (`Syntax.jl` lines 101-153)

When creating a pattern from an expression like `Luminal_ReLU(~x)`:

```julia
function makepattern(ex::Expr, pvars, slots, mod = @__MODULE__, splat = false)::Pat
  if iscall(ex)
    op = operation(ex)
    # ... processing ...
    patargs = map(i -> makepattern(i, pvars, slots, mod), args)
    isdef = isdefined_nested(mod, op)  # ← KEY LINE
    op_obj = isdef ? getfield_nested(mod, op) : op
    
    if isdef && op isa Expr || op isa Symbol
      pat_expr(iscall(ex), op_obj, op, patargs)  # ← Qualified function path
    else
      pat_expr(iscall(ex), op_obj, patargs)      # ← Non-qualified path
    end
  end
end
```

**Critical observation**: 
- `isdefined_nested(mod, op)` checks if a function symbol is defined in the current module
- For `Luminal_ReLU`, this likely returns `false` because:
  1. It's defined in our test file, not in Metatheory's module
  2. The pattern is created in `@theory` macro context

### Next Investigation Steps

1. **Test hypothesis**: Does defining `Luminal_ReLU` in `Main` module help?
2. **Check `pat_expr` signature**: What's the difference between the two `pat_expr` calls?
3. **Examine `ematch_compiler.jl`**: How does it use `op_obj` vs `op` for matching?
4. **Look at `bind_expr`** (line 222-253 in `ematch_compiler.jl`): 
   - Uses `v_head(n)`, `v_signature(n)`, `v_flags(n)` for matching
   - What are these for custom functions?

## Hypothesis

**The issue may be in how custom functions are registered/looked up in the e-graph.**

When a custom function like `Luminal_ReLU` is not defined in the module where the pattern is created, it may:
- Not get proper head/signature encoding
- Not be findable during e-graph matching
- Be treated as a literal instead of a function call

## Proposed Test

Create a minimal reproduction case:

```julia
using Metatheory, Metatheory.EGraphs

# Define function in Main module
Luminal_ReLU(x) = :(Luminal_ReLU($x))

# Create theory
theory = @theory begin
    Luminal_ReLU(~x) => Luminal_Max(~x, 0)
end

# Test
expr = :(Luminal_ReLU(a))
eg = EGraph(expr)
saturate!(eg, theory)
result = extract!(eg, astsize)

# Expected: Luminal_Max(a, 0)
# Actual: Luminal_ReLU(a) (no match)
```

If this still fails, try:
1. Exporting `Luminal_ReLU`
2. Defining it before `@theory`
3. Using fully qualified name `Main.Luminal_ReLU`

## Next Steps (Est. 8-18 hours remaining)

### Immediate (2-4 hours)
- [ ] Run hypothesis test with different module definitions
- [ ] Examine `pat_expr` function signature and implementation
- [ ] Check how `v_head()` is computed for custom functions

### Deep Dive (4-8 hours)
- [ ] Trace through e-graph insertion for custom function
- [ ] Compare with built-in function (like `+`)
- [ ] Identify exact point of matching failure

### Prototype Fix (2-6 hours)
- [ ] Modify pattern creation to handle custom functions
- [ ] OR modify e-matching to recognize custom function heads
- [ ] Test fix with our `Luminal_ReLU` case

## Decision Point

**By end of investigation** (est. 10-20 hours total):
- ✅ **Simple fix** (<60h total implementation): Submit PR to Metatheory.jl
- ❌ **Complex fix** (>100h): Fall back to custom e-graph implementation
- ⚠️ **Uncertain**: Extend investigation by 10 hours max, then decide
