# Position-Agnostic Graphs in Luminal.jl

Currently, [Luminal.jl](file:///devel/phil/Luminal.jl/src/Luminal.jl) suffers from severe VRAM spikes during text generation because it compiles a completely new execution graph (allocating ~1-2GB of new intermediate `CuArray` buffers) for every single token position. This happens because sequence lengths and positions (`pos`) are hardcoded as integers during the `compile()` phase.

To achieve the memory efficiency and speed of the Rust version of Luminal, [Luminal.jl](file:///devel/phil/Luminal.jl/src/Luminal.jl) needs to transition to **position-agnostic graphs**. 

## What Are Position-Agnostic Graphs?
In a position-agnostic (or dynamic-shape) architecture, the graph is compiled **exactly once**. Dimensions that change during execution (like sequence length `S` or current token position `pos`) are represented as symbolic variables rather than concrete integers. 

During execution, instead of recompiling, the engine simply updates the value of the symbolic variable in a registry and re-executes the pre-compiled graph operations using pre-allocated (or dynamically pooled) memory buffers.

## Significant Architectural Changes Required

Implementing this in Julia requires four major architectural shifts:

### 1. Symbolic Dimensions ([Graph.jl](file:///devel/phil/Luminal.jl/src/Graph.jl) & [ShapeTracker.jl](file:///devel/phil/Luminal.jl/src/ShapeTracker.jl))
*   **Current State:** `ShapeTracker` accepts `Luminal.DimType` (which can theoretically be a `SymbolicUtils.Sym`), but operations immediately attempt to evaluate these to integers using `eval_dim()`.
*   **Change:** We must fully embrace `SymbolicUtils.jl` for dimensions. Graphs must accept variables like `Sym{Int}(:pos)` and propagate these symbols through mathematical operations (e.g., `arange(pos + 1)`).

### 2. Deferred Memory Allocation ([Compiler.jl](file:///devel/phil/Luminal.jl/src/Compiler.jl))
*   **Current State:** [Compiler.jl](file:///devel/phil/Luminal.jl/src/Compiler.jl) iterates through all nodes, resolves their shapes to integers, and immediately allocates `results[node_id] = CUDA.fill(...)` arrays.
*   **Change:** The `compile` function can no longer allocate memory directly because the final shape is unknown. Compilation must instead produce an abstract "Execution Plan".

### 3. Execution-Time Shape Binding ([Execution.jl](file:///devel/phil/Luminal.jl/src/Execution.jl))
*   **Change:** A new execution context is needed. When `run_graph(pos=5)` is called, the engine must:
    1.  Substitute `pos=5` into all symbolic shape trackers.
    2.  Compute the concrete integer dimensions for that specific run.
    3.  Execute the operations sequence.

### 4. Dynamic Memory Pooling (The Hardest Part)
*   **Current State:** Each graph owns its own unique memory arrays.
*   **Change:** Because shapes change dynamically (e.g., an attention matrix grows from `1x1` to `1x4096`), we cannot use statically sized arrays. We must implement a **Memory Pool Allocator**. 
    *   During execution step 1 (pos=0), a buffer of size `X` is requested from the pool.
    *   During execution step 2 (pos=1), if a buffer needs to be size `X+Y`, the pool either resizes the buffer or provisions a larger contiguous block.
    *   *Note: This is exactly what the Rust Luminal allocator does to achieve its extremely low memory footprint.*

## Comparison: Julia vs. Rust

| Feature | [Luminal.jl](file:///devel/phil/Luminal.jl/src/Luminal.jl) (Current) | Rust Luminal (Current) | [Luminal.jl](file:///devel/phil/Luminal.jl/src/Luminal.jl) (Proposed Position-Agnostic) |
| :--- | :--- | :--- | :--- |
| **Compilation** | Once per token position (~1s delay) | Once at startup | Once at startup |
| **VRAM Usage** | Weights + Peak graph buffer sum `~1GB+` | Weights + Tightly pooled minimum buffers | Weights + Tightly pooled minimum buffers |
| **Shapes** | Static Integers (`1024`) | Dynamic (`S`, `S+1`) | Dynamic (`Sym(:seq)`) |
| **Speed** | Slowed by repeated JIT/allocations | Extremely fast | Fast (matches Rust algorithmically) |

## Summary
By moving to position-agnostic graphs, [Luminal.jl](file:///devel/phil/Luminal.jl/src/Luminal.jl) will stop throwing Out-Of-Memory errors on 8GB cards for 1B parameter models. It will eliminate the per-token compilation stutter, moving the framework from a "proof of concept" to a highly performant, production-ready inference engine that rivals its Rust counterpart.
