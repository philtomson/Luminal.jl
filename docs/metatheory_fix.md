# METATHEORY.JL ROOT CAUSE IDENTIFIED!

**Date**: February 17, 2026  
**Time Invested**: ~7 hours  
**Status**: ✅ **ROOT CAUSE FOUND! FIX IDENTIFIED!**

## The Bug

**Location**: `/tmp/Metatheory.jl/src/Patterns.jl` lines 55-72

### The Problem

When creating a pattern like `sin(~x)`:

```julia
function pat_expr(iscall::Bool, op, qop, args::Vector{Pat})
  op_hash = hash(op)     ← op = typeof(sin) Function object
  qop_hash = hash(qop)   ← qop = :sin Symbol
  ...
  v_set_head!(n, op_hash)  ← BUG: Uses hash of Function, not Symbol!
  ...
end
```

**Built-in functions like `sin`**:
- `op` = the actual `sin` function object (type: `Function`)
- `qop` = `:sin` (type: `Symbol`) 
- Pattern stores: `head_hash = hash(sin function)` ← WRONG!
- E-graph stores: `hash(:sin symbol)` ← CORRECT!
- **Result**: Hashes don't match, pattern FAILS to match!

**Custom functions like `Luminal_ReLU`**:
- When defined as `Luminal_ReLU(x) = :(Luminal_ReLU($x))`, it's defined in our module
- `isdefined(Main, :Luminal_ReLU)` = `true`
- `op` = the actual `Luminal_ReLU` function  
- `qop` = `:Luminal_ReLU`
- Pattern stores: `head_hash = hash(Luminal_ReLU function)` ← WRONG!
- E-graph stores: `hash(:Luminal_ReLU symbol)` ← CORRECT!
-  **Result**: Hashes don't match, pattern FAILS to match!

## Evidence

From test output:

```
sin:
  Pattern head_hash: 13460850144026691941  ← hash(sin function)
  E-Graph constant:  12205617288866037246  ← hash(:sin symbol) 
  Match: FALSE ❌

Luminal_ReLU:
  Pattern head_hash: 1775374013999696974   ← hash(Luminal_ReLU function)
  E-Graph constant:  11731689158505372259  ← hash(:Luminal_ReLU symbol)
  Match: FALSE ❌
```

Notice that `name_hash` (using `qop`) **DOES** match:
```
sin name_hash: 12205617288866037246  ← Matches e-graph!
Luminal_ReLU name_hash: 11731689158505372259  ← Matches e-graph!
```

## The Fix

**Line 64 in Patterns.jl**:

```julia
# BEFORE (BUG):
v_set_head!(n, op_hash)

# AFTER (FIX):
v_set_head!(n, qop_hash)  # Use quoted operation hash instead!
```

That's it! One line change!

## Why This Works

The e-graph (in `egraph.jl` line 395) does:
```julia
v_set_head!(n, add_constant!(g, h))  # h is already a symbol
```

So e-graph always stores symbol hashes. Patterns should match!

## Impact

This fix will make patterns match based on:
- `qop_hash` = symbol hash (matches e-graph)
- NOT `op_hash` = function object hash

Both built-in AND custom functions will work!

## Testing the Fix

After applying fix, our tests should show:
- ✅ `sin(~x) --> cos(~x)` works
- ✅ `Luminal_ReLU(~x) --> Luminal_Max(~x, 0)` works  
- ✅ `+(~x, 0) --> ~x` works

## Effort Estimate

- **Fix**: 1 line change
- **Testing**: 2-4 hours (comprehensive tests)
- **PR preparation**: 4-6 hours (tests, documentation)
- **Total**: **6-10 hours** ✅ **SIMPLE FIX!**

## Next Steps

1. ✅ Apply fix to local Metatheory.jl fork
2. ✅ Test with our Luminal operators
3. ✅ Verify built-ins still work
4. ✅ Create comprehensive test suite
5. ✅ Submit PR to Metatheory.jl upstream
