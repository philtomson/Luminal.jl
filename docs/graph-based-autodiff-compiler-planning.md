# Graph-Based Autodiff Compiler Planning

## What is a Graph-Based Autodiff Compiler?

A **graph-based autodiff compiler** is a system that performs automatic differentiation (calculating the gradients needed for training neural networks) by mutating and expanding a static computation graph *before* any data is executed.

In modern deep learning frameworks, there are generally two ways to do automatic differentiation:
1. **Tape-based / Eager (like PyTorch or Flux.jl):** As the network runs forward on actual data, a "tape" records every operation. During the backward pass, the system walks backward through the tape to compute gradients dynamically.
2. **Graph-based / Ahead-of-Time (like TensorFlow 1.x, XLA, or Luminal):** The entire forward computation is defined as a static dependency graph (like `Graph` and `Node` structures) first. The autodiff system acts as a **compiler pass**. It analyzes the forward graph and physically inserts new nodes representing the derivative math recursively applying the chain rule. The result is a single, massive combined graph containing both the forward pass and the backward pass.

The main advantage of graph-based autodiff is optimization: because the backward pass is generated as a static graph before runtime, the compiler (like `SymbolicUtils.jl` or `Metatheory.jl` pipelines) can globally optimize the mixed forward/backward graph together—fusing operations, aggressively eliminating redundant math, and optimizing memory allocation.

---

## Implementation Requirements for Luminal.jl

Based on the porting plan, `Luminal.jl` already has the core `Graph`, `Node`, and `Op` primitives, as well as the compiler infrastructure. To add a graph-based autodiff compiler, a graph transformation pass is required:

### 1. Derivative Rules for Base Operators
Define the mathematical local derivatives for every fundamental `Op`. For instance, for a `Mul(A, B)` node, the gradient with respect to `A` is `upstream_gradient * B`, and with respect to `B` is `upstream_gradient * A`.

### 2. The Reverse-Mode Traversal Pass
A function (e.g., `build_backward_graph(graph, loss_node)`) that:
- Performs a reverse topological sort from the loss node back to the input weights.
- Looks at the incoming `upstream_gradient` for each node.
- Spawns new `Node`s into the `Graph` that compute the local derivatives.
- Links these newly created gradient nodes to the inputs of the original operation.

### 3. Gradient Accumulation Logic
When a tensor is used in multiple places (e.g., residual connections), gradients from multiple paths must stream into the same source tensor. The autodiff compiler needs logic to detect when a node has multiple consumers and insert `Add` nodes into the graph to sum up the incoming gradients correctly.

### 4. Memory/Activation Management
To compute derivatives during the backward pass, forward activations are often required (e.g., the derivative of ReLU requires knowing if the input was `> 0`). The compiler must ensure that the buffers for these required forward activations are kept alive and not overwritten before the backward nodes read them.

### 5. Integration with the Optimizer
Once gradients are computed in the graph, they must be applied to the model's weights:
- Evaluate the graph, extract calculated gradient arrays, and apply standard Julia optimizers (like `Optimisers.jl`) externally.
- **Or** (more inline with the Luminal philosophy) insert the optimizer steps directly into the graph as explicit nodes, so the entire training step (forward + backward + update) compiles into one optimized GPU kernel execution.

### 6. Integration with the Existing Compiler
The backward graph will likely contain a lot of naive or unoptimized math (like multiplying by `1.0` or adding `0.0`). This combined graph should be funneled back into the existing `SymbolicUtils.jl` or `Metatheory.jl` pipelines so that it can be simplified and hardware-fused before execution.

---

## Leveraging the Julia Ecosystem

The Julia ecosystem has a mature automatic differentiation (AD) landscape. The following packages and ecosystems are highly relevant for implementing a static *graph-based* autodiff compiler in `Luminal.jl`:

### 1. The "Must-Use" Foundation: `ChainRulesCore.jl`
Instead of manually writing the local derivative math for every single operator, you can leverage **`ChainRulesCore.jl`**. 
- **What it is:** An ecosystem-wide repository of custom forward (`frule`) and reverse (`rrule`) derivative definitions for thousands of Julia functions.
- **How to use it:** When the compiler builds the backward graph, instead of hardcoding gradient math for a `Node`, hook into `ChainRulesCore.rrule(op_function, inputs...)`. This returns the exact expressions needed for the backward pass, which can be translated into new `Node`s.

### 2. The Direct Fit: `Symbolics.jl` / `SymbolicUtils.jl`
Since `Luminal.jl` already converts the computation graph into `SymbolicUtils.jl` objects, there is a massive advantage here.
- **What it is:** The symbolic algebra foundation of Julia, with built-in symbolic differentiation.
- **How to use it:** Write an AD compiler pass that takes the forward `Graph`, translates the loss path into a `Symbolic` expression, calls `Symbolics.derivative(expr, weight_vars)`, and translates the optimized symbolic derivative back into the `GraphTensor` format.

### 3. The Closest Architecture Match: `Yota.jl` (and `Ghost.jl`)
If you want to see how another package does exactly this, look at `Yota.jl`.
- **What it is:** A purely **graph-based** reverse-mode AD package. It traces standard Julia code into a static, tape-based graph representation (using `Ghost.jl`), computes the backward graph, applies graph-level optimizations, and compiles it.
- **How to use it:** While `Luminal.jl` has its own `Graph` and `Node` structures, `Yota.jl`'s source code is the perfect architectural reference for a graph-mutating backward pass.

### 4. What about `Zygote.jl` or `Enzyme.jl`?
These are famous, but might **not** be the right fit for this specific compiler architecture:
- **`Zygote.jl`**: Operates on Julia's typed IR and is geared towards eager, dynamic execution (like PyTorch). It doesn't output a static unrolled graph that can be optimized with `Metatheory.jl`.
- **`Enzyme.jl`**: Incredibly fast, but operates at the LLVM IR level. LLVM-level AD is too low-level for optimizing a high-level `Graph` object with fusion rules before hardware execution.

### Summary Recommendation for Luminal.jl
To stick to the static, aggressively-optimizing architecture:
1. Write a custom compiler pass that walks backward through the `Graph`.
2. Query **`ChainRulesCore.jl`** locally at each `Node` to figure out what math nodes to insert into the backward graph.
3. Feed the resulting massive forward+backward combined `Graph` directly into the existing **`Metatheory.jl`** pipeline to optimize it globally.
