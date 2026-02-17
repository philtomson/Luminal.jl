# Julia Test Suite

## Running Tests

Run all tests from the project root (each test must run in a separate Julia process due to Julia 1.12 JIT compilation constraints):

```bash
for f in Julia/tests/test_*.jl Julia/tests/test.jl; do
    echo "=== $f ==="
    julia --project=Julia "$f"
    echo
done
```

Or run individual tests:

```bash
julia --project=Julia Julia/tests/test_optimizer.jl
```

## Test Inventory

| Test | Description | Tests |
|------|-------------|-------|
| `test_symbolic.jl` | Symbolic expression construction and evaluation | 3 |
| `test_shape_tracker.jl` | ShapeTracker dimension operations | 7 |
| `test.jl` | End-to-end graph build + execution (matmul) | 1 |
| `test_optimizer.jl` | ReLU → `max(x, 0)` canonicalization | 6 |
| `test_algebraic_simplification.jl` | `(a * 1) + 0` simplifies to `a` | 3 |
| `test_associativity.jl` | `(a + 2) + 3` simplifies to `a + 5` | 6 |
| `test_commutativity.jl` | `(2 * a) * 3` simplifies to `6a` | 6 |
| `test_lazy.jl` | Lazy graph evaluation — operations stay symbolic | 11 |
| `test_symbolics_integration.jl` | Graph → SymbolicUtils conversion, symbolic max/min | 17 |

**Total: 60 tests across 9 files**

## Requirements

- **Julia 1.12.5**
- **SymbolicUtils.jl** v3.31.0 (dev-installed from `/devel/phil/SymbolicUtils.jl`)
- **TermInterface.jl** v2.0.0
