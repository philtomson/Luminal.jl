# Metatheory.jl PR Submission Guide

## Summary

✅ **Ready to submit upstream PR** to fix custom operator support in Metatheory.jl

## What's Included

### 1. Core Fix (2 lines)
- **File**: `src/Patterns.jl`
- **Lines**: 64, 71
- **Change**: Use `qop_hash` (symbol hash) instead of `op_hash` (function hash)

### 2. Tests
- **File**: `test/egraphs/test_custom_operators.jl`
- **Coverage**: 11 comprehensive tests
- **Status**: ✅ All passing

### 3. Documentation
- **File**: `docs/src/examples/custom_operators.md`
- **Content**: Usage guide, examples, cost function guidance

### 4. Commit
- **Branch**: `ale/3.0` (Metatheory.jl development branch)
- **Commit**: Created with detailed message

## PR Submission Steps

### Option 1: Via GitHub Web Interface

1. **Fork the repository** (if not already done)
   ```bash
   # Visit: https://github.com/JuliaSymbolics/Metatheory.jl
   # Click "Fork"
   ```

2. **Push to your fork**
   ```bash
   cd /tmp/Metatheory.jl
   git remote add fork https://github.com/YOUR_USERNAME/Metatheory.jl.git
   git push fork ale/3.0:custom-operators-fix
   ```

3. **Create PR on GitHub**
   - Go to: https://github.com/JuliaSymbolics/Metatheory.jl/pulls
   - Click "New Pull Request"
   - Base: `JuliaSymbolics/Metatheory.jl` branch `ale/3.0`
   - Compare: `YOUR_USERNAME/Metatheory.jl` branch `custom-operators-fix`
   - Title: "Fix: Enable Custom Operator Support in E-Graph Pattern Matching"
   - Description: Copy from `/tmp/Metatheory.jl/PR_DESCRIPTION.md`

### Option 2: Via GitHub CLI

```bash
cd /tmp/Metatheory.jl

# Install gh if needed: https://cli.github.com/

# Create PR directly
gh pr create \
  --base ale/3.0 \
  --title "Fix: Enable Custom Operator Support in E-Graph Pattern Matching" \
  --body-file PR_DESCRIPTION.md \
  --repo JuliaSymbolics/Metatheory.jl
```

## Verification Before Submitting

Run these commands to verify everything is ready:

```bash
cd /tmp/Metatheory.jl

# 1. Verify tests pass
julia --project=. test/egraphs/test_custom_operators.jl

# 2. Check commit
git log -1 --stat

# 3. Review changes
git diff HEAD~1

# 4. Verify all files staged
git status
```

Expected output:
```
✅ Test Summary: Custom Operators | Pass Total
✅ 1 commit with 3 files changed
✅ Clean git status
```

## Post-Submission

1. **Monitor PR** for maintainer feedback
2. **Be ready to**:
   - Answer questions about the fix
   - Add more tests if requested
   - Adjust documentation
   - Rebase if needed

## Contact Info for PR

**Discovered by**: [Your name]  
**Tested on**: Metatheory.jl v3.0 (ale/3.0 branch)  
**Investigation time**: 12 hours  
**Impact**: Enables custom operators for entire Julia ecosystem

## Alternative: Local Testing First

If you want to test more extensively before submitting:

```bash
# Use in another project
julia -e 'using Pkg; Pkg.develop(path="/tmp/Metatheory.jl")'

# Test with Luminal
cd /devel/phil/luminal
julia --project=Julia Julia/tests/test_metatheory_final.jl
```

## Files Ready for PR

```
/tmp/Metatheory.jl/
├── src/Patterns.jl                               (MODIFIED - 2 lines)
├── test/egraphs/test_custom_operators.jl         (NEW - 148 lines)
├── docs/src/examples/custom_operators.md         (NEW - 220 lines)
└── PR_DESCRIPTION.md                             (Reference - not committed)
```

## Success Criteria

✅ Tests pass locally  
✅ Documentation is clear  
✅ Commit message is descriptive  
✅ PR description explains impact  
✅ No breaking changes

**Status**: **READY TO SUBMIT** 🚀
