# Entry points implemented by the Reactant extension (ext/LuminalReactantExt.jl),
# active once Reactant.jl is loaded alongside Luminal.

"""
    reactant_function(graph, outputs, inputs) -> f

A pure Julia function `f(xs...)` evaluating `graph` with `xs` bound to the input
nodes `inputs` (ids or `GraphTensor`s, in order), returning the value of
`outputs` (one id/tensor, or a vector of them for a tuple). Unlike `execute` it
allocates every result functionally, so Reactant can trace it: `f` is what
`reactant_compile` and `to_stablehlo` compile.

Works on the primitive graph a model builds (before `compile`): elementwise ops,
reductions, views, matmuls, gather, and the fused RMSNorm / softmax / rotary ops
the builders emit. Tensors preloaded into the graph (`load_weights!`) that are
not listed in `inputs` are baked in as constants. Requires Reactant.jl.
"""
function reactant_function end

"""
    reactant_compile(graph, outputs, inputs, args...) -> compiled

Trace `reactant_function(graph, outputs, inputs)` with example arguments `args`
(plain arrays or Reactant arrays) and compile it with XLA on Reactant's default
backend. Call the result with Reactant arrays (`Reactant.to_rarray`) of the same
shapes. Requires Reactant.jl.
"""
function reactant_compile end

"""
    to_stablehlo(graph, outputs, inputs, args...) -> String

The StableHLO module (MLIR text) of `graph`, traced with example arguments
`args` as in `reactant_compile`. Requires Reactant.jl.
"""
function to_stablehlo end
