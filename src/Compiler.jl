
# Compiler.jl — turns a Graph into an executable plan: one step per node, with
# elementwise fusion, buffer reuse and aliasing, compile-time constant folding and
# HIP graph capture. Graph-level rewrites live in EGraphRewrite.jl.
#
# SymbolicUtils builds the scalar expression of each fused elementwise kernel.

using SymbolicUtils
using SymbolicUtils: term, Sym
using SymbolicUtils.Code: toexpr
# Import execution functions for compiled thunk
using Luminal: execute_op, execute_op!, realize_view, to_device, realized_dims, eval_dim, execute_with_capture
using CUDA
using AMDGPU
using GPUArrays
using KernelAbstractions

function fix_gpu_ast!(expr::Expr)
    if expr.head == :call
        if expr.args[1] == :exp2
            expr.args[1] = :(Base.exp2)
        elseif expr.args[1] == :log2
            expr.args[1] = :(Base.log2)
        elseif expr.args[1] == :sin
            expr.args[1] = :(Base.sin)
        elseif expr.args[1] == :cos
            expr.args[1] = :(Base.cos)
        elseif expr.args[1] == :sqrt
            expr.args[1] = :(Base.sqrt)
        elseif expr.args[1] == :max
            expr.args[1] = :(Base.max)
        elseif expr.args[1] == :min
            expr.args[1] = :(Base.min)
        elseif expr.args[1] in (:floor, :ceil, :round, :trunc, :exp)
            expr.args[1] = :(Base.$(expr.args[1]))
        end
        for i in 2:length(expr.args)
            if expr.args[i] isa Expr
                fix_gpu_ast!(expr.args[i])
            end
        end
    end
    return expr
end
function fix_gpu_ast!(val)
    return val 
end

# --- Compiled Graph Structure ---

struct CompiledGraph
    graph::Luminal.Graph
    steps::Vector{Base.Function}
    results::Vector{Any} 
    cache::Dict{Symbol, Any}
    consumer_count::Vector{Int}
end

function (cg::CompiledGraph)(inputs::Dict; sym_vals::Dict{Symbol, Int} = Dict{Symbol, Int}(), device::Luminal.AbstractDevice = Luminal.get_device())
    for (k, v) in inputs
        if Luminal.to_device(v, device) === v
             # Already resident on the target device: alias it rather than copy.
             cg.results[k] = v
        elseif !isassigned(cg.results, k) || cg.results[k] === nothing || length(cg.results[k]) != length(v)
             cg.results[k] = Luminal.to_device(v, device)
        else
             # Host -> device copy into the existing buffer
             copyto!(cg.results[k], v)
        end
    end
    
    # Initialize live consumer counts for this run
    # Nodes in graph.tensors are weights and persist
    live_counts = copy(cg.consumer_count)
    
    function run_graph()
        for step in cg.steps
            step(cg.results, device, sym_vals, live_counts)
        end
    end
    
    if get(cg.cache, :capture, false)
        # Everything a captured replay bakes in: symbolic dims and input buffers
        # (sym_vals only matter if some shape depends on them)
        cg.cache[:key] = (cg.cache[:dynamic] ? copy(sym_vals) : nothing,
                          sort!([(k, objectid(cg.results[k])) for k in keys(inputs)]))
    end
    execute_with_capture(device, run_graph, cg.cache)
    return cg.results
end

# Drop a dead intermediate. Buffers the graph allocated itself (`owned`) are
# returned to the GPU pool right away instead of waiting for a GC finalizer.
function _release!(res, id, owned)
    buf = res[id]
    res[id] = nothing
    if owned[id] && buf isa AnyGPUArray
        GPUArrays.unsafe_free!(buf)
    end
end

# Make res[node_id] a buffer of size `sz` on `dev`. In reuse mode on GPU, buffers
# sized by a dynamic dim (e.g. decode context length) change size every run and
# GPU allocation is slow, so they are served as a reshaped prefix of a backing
# buffer that only regrows (with 50% headroom).
function _prepare_output!(res, node_id, sz, dev, backing, owned, free_intermediates)
    isassigned(res, node_id) && res[node_id] !== nothing && size(res[node_id]) == sz && return
    if free_intermediates || !(dev isa Luminal.AbstractGPUDevice)
        isassigned(res, node_id) && res[node_id] !== nothing && _release!(res, node_id, owned)
        res[node_id] = Luminal.zero_tensor(dev, Float32, sz...)
    else
        n = prod(sz)
        if backing[] === nothing || length(backing[]) < n
            isassigned(res, node_id) && res[node_id] !== nothing && _release!(res, node_id, owned)
            cap = backing[] === nothing ? n : cld(3n, 2)
            backing[] isa AnyGPUArray && GPUArrays.unsafe_free!(backing[])
            backing[] = Luminal.zero_tensor(dev, Float32, cap)
        end
        res[node_id] = Base.reshape(view(backing[], 1:n), sz)
    end
end

# True if a consumer reads node `id` through `st` exactly as produced (no
# permute, slice, pad or broadcast), so `id` can be fused into that consumer.
function _is_trivial_view(st::ShapeTracker, out::ShapeTracker)
    n = length(st.dims)
    st.indexes == 1:n || return false
    any(st.fake) && return false
    all(p -> isequal(p[1], 0) && isequal(p[2], 0), st.padding) || return false
    for i in 1:n
        lo, hi = st.mask[i]
        (isequal(lo, 0) && (isequal(hi, typemax(Int)) || isequal(hi, st.dims[i]))) || return false
    end
    return isequal(Luminal.DimType[st.dims...], realized_dims(out))
end

# The reduced-precision storage a persistent tensor gets, or `nothing`: it must be a
# 2D Float32 GPU array whose every use is as the left operand of a matmul, read
# unchanged, and all uses must want the same storage:
#   MatMulF16(_, :gemv) -> HalfWeight, MatMulF16(_, :gemm_ex) -> HalfWeightN,
#   MatMulQ8 -> QuantWeight, MatMulQ4 -> Q4Weight, MatMul -> per `weight_dtype`
#   (Float16 / Int8 / Luminal.Int4).
# `weight_dtype` is one type for every weight, or a function of the weight's node id
# (a per-tensor policy, e.g. int8 for sensitive tensors and 4-bit for the rest).
_wdtype(weight_dtype, node_id) = weight_dtype isa Type ? weight_dtype : weight_dtype(node_id)

function _weight_storage(graph, node_id, data, consumers, retain, weight_dtype)
    weight_dtype = _wdtype(weight_dtype, node_id)
    (data isa AnyGPUArray && data isa DenseArray && eltype(data) == Float32 && ndims(data) == 2) || return nothing
    node_id in retain && return nothing
    isempty(consumers[node_id]) && return nothing
    storage = nothing
    for (cid, st) in consumers[node_id]
        c = graph.nodes[cid]
        want = c.op isa Luminal.MatMulF16 ? (c.op.impl === :gemm_ex ? Luminal.HalfWeightN : Luminal.HalfWeight) :
               c.op isa Luminal.MatMulQ8  ? Luminal.QuantWeight :
               c.op isa Luminal.MatMulQ4  ? Luminal.Q4Weight :
               (c.op isa Luminal.MatMul && weight_dtype === Float16) ? Luminal.HalfWeight :
               (c.op isa Luminal.MatMul && weight_dtype === Int8)    ? Luminal.QuantWeight :
               (c.op isa Luminal.MatMul && weight_dtype === Luminal.Int4) ? Luminal.Q4Weight : nothing
        want === nothing && return nothing
        storage === nothing || storage === want || return nothing
        storage = want
        (c.inputs[1][1] == node_id && c.inputs[2][1] != node_id) || return nothing
        _is_trivial_view(st, graph.shapes[node_id]) || return nothing
    end
    align = storage === Luminal.Q4Weight ? 32 : storage === Luminal.QuantWeight ? 16 : storage === Luminal.HalfWeight ? 8 : 1
    size(data, 2) % align == 0 || return nothing
    return storage
end

# Fused scalar kernels, keyed by expression: identical groups (e.g. the same
# RMSNorm in every layer) share one function and hence one GPU compilation.
const _FUSED_KERNELS = Dict{String, Any}()

function _fused_kernel(node_expr, n_args)
    key = string(n_args, ":", node_expr)
    get!(_FUSED_KERNELS, key) do
        name = Symbol("global_fused_", length(_FUSED_KERNELS) + 1)
        args = [Expr(:(::), Symbol("in", i), :Real) for i in 1:n_args]
        Core.eval(Luminal, :(function $name($(args...)); return Float32($node_expr); end))
    end
end

# Movement ops that don't reorder data can return a reshaped alias of a
# contiguous input instead of copying it: any Reshape, an Expand that doesn't
# change the element count, and a Permute that only moves size-1 dims.
function _alias_view(op, arg, sz; strided_ok::Bool=false)
    op isa Luminal.Slice && return _alias_slice(op, arg, sz; strided_ok=strided_ok)
    (op isa Luminal.Reshape || op isa Luminal.Permute || op isa Luminal.Expand) || return nothing
    (arg isa DenseArray && length(arg) == prod(sz)) || return nothing
    if op isa Luminal.Permute
        issorted([d for d in op.dims if size(arg, d) != 1]) || return nothing
    end
    return Base.reshape(arg, sz)
end

# A slice is a contiguous block of a column-major array when every dim after the
# first one it cuts has extent 1 -- e.g. a row range of a (rows, 1, 1) matmul
# output. Then it is a reshaped range of the input's memory.
# With `strided_ok` (every consumer is elementwise, and GPU broadcasts read
# strided views), a non-contiguous slice is returned as a SubArray view instead.
function _alias_slice(op, arg, sz; strided_ok::Bool=false)
    (arg isa DenseArray && prod(sz) > 0) || return nothing   # empty slices take the copy path
    n = ndims(arg)
    length(sz) == n || return nothing
    k = findfirst(i -> sz[i] != size(arg, i), 1:n)
    k === nothing && return Base.reshape(arg, sz)
    if !all(sz[i] == 1 for i in k+1:n)
        strided_ok || return nothing
        ranges = ntuple(i -> begin
            lo = i <= length(op.ranges) ? max(0, Int(op.ranges[i][1])) : 0
            (lo + 1):(lo + sz[i])
        end, n)
        return view(arg, ranges...)
    end
    start = ntuple(i -> i <= length(op.ranges) ? max(0, Int(op.ranges[i][1])) + 1 : 1, n)
    off = LinearIndices(arg)[CartesianIndex(start)] - 1
    return Base.reshape(view(vec(arg), off+1:off+prod(sz)), sz)
end

# --- Fusion Helpers ---

function is_elementwise(op)
    # Check if the Op is purely element-wise and supports scalar broadcast
    return op isa Luminal.Add || op isa Luminal.Mul || op isa Luminal.Mod || 
           op isa Luminal.Max || op isa Luminal.FusedMulAdd || 
           op isa Luminal.FusedAddReLU || op isa Luminal.LessThan ||
           op isa Luminal.Log2 || op isa Luminal.Exp2 || op isa Luminal.Sin || 
           op isa Luminal.Cos || op isa Luminal.Sqrt || op isa Luminal.Recip || 
           op isa Luminal.ReLU || op isa Luminal.Constant ||
           op isa Luminal.Floor || op isa Luminal.Ceil || op isa Luminal.Round ||
           op isa Luminal.Trunc || op isa Luminal.Select ||
           op isa Luminal.Div || op isa Luminal.Exp
end

# Helper functions for fused kernels
scalar_less(x, y) = Float32(x < y)

# Robust mapping from Luminal Ops to symbolic expressions
function op_to_sym(op, inputs)
    if op isa Luminal.Add
        return inputs[1] + inputs[2]
    elseif op isa Luminal.Mul
        return inputs[1] * inputs[2]
    elseif op isa Luminal.Mod
        return term(mod, inputs[1], inputs[2]; type=Real)
    elseif op isa Luminal.Log2
        return term(log2, inputs[1]; type=Real)
    elseif op isa Luminal.Exp2
        return term(exp2, inputs[1]; type=Real)
    elseif op isa Luminal.Sin
        return term(sin, inputs[1]; type=Real)
    elseif op isa Luminal.Cos
        return term(cos, inputs[1]; type=Real)
    elseif op isa Luminal.Sqrt
        return term(sqrt, inputs[1]; type=Real)
    elseif op isa Luminal.Recip
        return term(/, 1.0f0, inputs[1]; type=Real)
    elseif op isa Luminal.ReLU
        return term(max, inputs[1], 0.0f0; type=Real)
    elseif op isa Luminal.Max
        return term(max, inputs...; type=Real)
    elseif op isa Luminal.FusedMulAdd
        return inputs[1] * inputs[2] + inputs[3]
    elseif op isa Luminal.FusedAddReLU
        return term(max, inputs[1] + inputs[2], 0.0f0; type=Real)
    elseif op isa Luminal.LessThan
        return term(ifelse, term(<, inputs[1], inputs[2]), 1.0f0, 0.0f0; type=Real)
    elseif op isa Luminal.Div
        return term(/, inputs[1], inputs[2]; type=Real)   # a term, so it stays one division
    elseif op isa Luminal.Exp
        return term(exp, inputs[1]; type=Real)
    elseif op isa Luminal.Floor
        return term(floor, inputs[1]; type=Real)
    elseif op isa Luminal.Ceil
        return term(ceil, inputs[1]; type=Real)
    elseif op isa Luminal.Round
        return term(round, inputs[1]; type=Real)
    elseif op isa Luminal.Trunc
        return term(trunc, inputs[1]; type=Real)
    elseif op isa Luminal.Select
        return term(ifelse, term(!=, inputs[1], 0.0f0), inputs[2], inputs[3]; type=Real)
    elseif op isa Luminal.Constant
        return Float32(op.value)
    else
        error("Unsupported element-wise op for fusion: $(typeof(op))")
    end
end

# Helper to build a symbolic expression for a fusion group recursively
function build_fused_expr!(graph, node_id, consumer_count, group_inputs, fusible_intermediates, sym_cache, current_st)
    node = graph.nodes[node_id]
    op = node.op
    op isa Luminal.Constant && return Float32(op.value)  # inline as a literal

    # If this node is NOT a fusible intermediate, it's a leaf for THIS fusion group
    if !(node_id in fusible_intermediates)
        input_key = (node_id, current_st)
        if haskey(sym_cache, input_key)
            return sym_cache[input_key]
        end
        sym = Sym{Real}(Symbol("in", length(group_inputs) + 1))
        push!(group_inputs, (node_id, current_st, sym))
        sym_cache[input_key] = sym
        return sym
    end

    input_syms = []
    for (in_id, _, in_st) in node.inputs
        push!(input_syms, build_fused_expr!(graph, in_id, consumer_count, group_inputs, fusible_intermediates, sym_cache, in_st))
    end

    return op_to_sym(op, input_syms)
end

function evaluate_op_shapes(op::Luminal.Op, sym_vals::Dict{Symbol, Int})
    if op isa Luminal.Expand
        return Luminal.Expand(op.dim, Luminal.eval_dim(op.size, sym_vals))
    elseif op isa Luminal.Reshape
        return Luminal.Reshape([Luminal.eval_dim(d, sym_vals) for d in op.shape])
    elseif op isa Luminal.Slice
        return Luminal.Slice([(Luminal.eval_dim(s, sym_vals), Luminal.eval_dim(e, sym_vals)) for (s, e) in op.ranges])
    elseif op isa Luminal.Pad
        return Luminal.Pad([(Luminal.eval_dim(s, sym_vals), Luminal.eval_dim(e, sym_vals)) for (s, e) in op.padding])
    elseif op isa Luminal.Unfold
        return Luminal.Unfold(
            [Luminal.eval_dim(d, sym_vals) for d in op.kernel_shape],
            [Luminal.eval_dim(d, sym_vals) for d in op.stride_shape],
            [Luminal.eval_dim(d, sym_vals) for d in op.dilation_shape]
        )
    else
        return op
    end
end

# --- Memory -------------------------------------------------------------------

_storage_bytes(T, n) = T === Luminal.Q4Weight ?
                           cld(n, 2) + (Luminal.DEFAULT_Q4_SYMMETRIC ? 2 : 4) * cld(n, Luminal.DEFAULT_Q4_GROUP) :
                       T === Luminal.QuantWeight ? n + 4 * cld(n, Luminal.DEFAULT_Q8_GROUP) :
                       (T === Luminal.HalfWeight || T === Luminal.HalfWeightN) ? 2n : 4n

"""
    estimate_compile_bytes(graph; retain=Int[], weight_dtype=Float32, fold=true) -> Int

Upper estimate of the device memory `compile(graph; ...)` would newly allocate:
reduced-precision copies of weights not already converted (the conversion cache
shares them between graphs), constant-folded tensors (e.g. concatenated
weights) and their conversions, and every intermediate. Lets a search skip a
candidate that would not fit instead of running the machine out of memory.
"""
function estimate_compile_bytes(graph::Luminal.Graph; retain::Vector{Int}=Int[],
                                weight_dtype=Float32, fold::Bool=true)
    n = length(graph.nodes)
    consumers = [Tuple{Int, ShapeTracker}[] for _ in 1:n]
    for (cid, node) in enumerate(graph.nodes), (id, _, st) in node.inputs
        push!(consumers[id], (cid, st))
    end
    numel(id) = try prod(Int(eval_dim(d)) for d in realized_dims(graph.shapes[id]); init=1) catch; -1 end
    persistent = falses(n)
    bytes = 0
    for (id, node) in enumerate(graph.nodes)
        haskey(graph.tensors, (id, 1)) || continue
        persistent[id] = true
        data = graph.tensors[(id, 1)]
        T = _weight_storage(graph, id, data, consumers, retain, weight_dtype)
        T === nothing && continue
        entry = get(Luminal._HALF_WEIGHTS, hash(T, objectid(data)), nothing)
        (entry !== nothing && entry[1].value === data) || (bytes += _storage_bytes(T, length(data)))
    end
    folded = falses(n)
    for (id, node) in enumerate(graph.nodes)
        persistent[id] && continue
        k = numel(id)
        k < 0 && continue                     # symbolic shape: sized at run time
        if fold && !isempty(node.inputs) && all(persistent[i] for (i, _, _) in node.inputs)
            persistent[id] = folded[id] = true
        else
            bytes += 4k                       # an intermediate (or input) buffer
        end
    end
    # Folded tensors: those some run-time node reads are kept (converted, if only
    # matmuls read them); the rest are transient (see compile's streaming folding),
    # costing at most a few of the largest at once.
    transient = 0
    for id in findall(folded)
        k = numel(id)
        cs = first.(consumers[id])
        if !isempty(cs) && all(c -> folded[c], cs) && !(id in retain)
            transient = max(transient, 4k)
        else
            ops = [graph.nodes[c].op for c in cs]
            wd = _wdtype(weight_dtype, id)
            per = isempty(ops) ? 4.0 :
                  all(op -> op isa Luminal.MatMulQ4 || (op isa Luminal.MatMul && wd === Luminal.Int4), ops) ? 0.63 :
                  all(op -> op isa Luminal.MatMulQ8 || (op isa Luminal.MatMul && wd === Int8), ops) ? 1.04 :
                  all(op -> op isa Luminal.MatMulF16 || (op isa Luminal.MatMul && wd === Float16), ops) ? 2.0 : 4.0
            bytes += ceil(Int, per * k)
            transient = max(transient, 4k)    # its Float32 value, before conversion
        end
    end
    return bytes + 3 * transient
end

"""
    release!(cg::CompiledGraph)

Free the device memory a compiled graph owns -- its intermediates, constant-folded
tensors (and their converted copies) and captured HIP graph -- without waiting
for the garbage collector. Weights it shares with other graphs are untouched.
The compiled graph must not be run afterwards.
"""
function release!(cg::CompiledGraph)
    free!(x) = x isa AnyGPUArray && (try GPUArrays.unsafe_free!(x) catch end)
    for id in get(cg.cache, :freeable, Int[])
        isassigned(cg.results, id) || continue
        x = cg.results[id]
        if x isa Luminal.HalfWeight
            free!(x.t)
        elseif x isa Luminal.HalfWeightN
            free!(x.w)
        elseif x isa Luminal.QuantWeight
            free!(x.q); free!(x.scale)
        elseif x isa Luminal.Q4Weight
            free!(x.p); free!(x.scale); x.mn === nothing || free!(x.mn)
        else
            free!(x)
        end
        cg.results[id] = nothing
    end
    delete!(cg.cache, :graph)                 # a captured HIP graph is destroyed when collected
    return nothing
end

# --- Main Compile Function ---

"""
    compile(graph; device=get_device(), retain=Int[], free_intermediates=true, fuse=true,
            weight_dtype=Float32, capture=false, fold=true,
            search=:none, precision=false, search_inputs=nothing, search_cache=...,
            search_tolerance=1e-3)

Build an executable `CompiledGraph`. With `free_intermediates=true` each
intermediate buffer is released as soon as its last consumer has run (lowest
peak memory, but every run reallocates). With `false` buffers persist across
runs and are reused, reallocated only when a symbolic dimension changes their
size, which suits graphs executed many times such as per-token decode.
Retained outputs are overwritten by the next run in either mode.
`fuse=true` merges chains of elementwise ops into single kernels.
`fold=true` computes nodes that depend only on persistent tensors (weights) once,
at compile time. `weight_dtype=Float16` (or `Int8`: group-wise int8, or `Luminal.Int4`:
group-wise asymmetric 4-bit) stores GPU weights that are only ever used as the left
operand of matmuls in that format (`HalfWeight`, `QuantWeight`, `Q4Weight`); compute
stays Float32. `weight_dtype` may also be a function of a weight's node id returning
its type, for a different format per tensor (`LlamaSession` builds one from names).
`search=:static` or `:measured` first runs the e-graph rewrite layer
(`EGraphRewrite`) and compiles the graph it picks: by the static cost model, or
by timing verified candidates (cached in `search_cache`). `precision=true`
(`:weights`) lets it choose Float16 weights per matmul; `precision=:activations`
also allows Float16 activations (rocBLAS gemm_ex: ~2x faster prefill, ~5e-3
relative logit error). Candidates must match the original graph's outputs within
`search_tolerance` (relative to each output's largest magnitude). `retain` must list the outputs; the result
maps the original input/output node ids. `:measured` needs static shapes.
`capture=true` (AMD GPUs) records a run into a HIP graph and replays it on
later runs with the same symbolic dims and input buffers (see
`execute_with_capture`). Requires `free_intermediates=false`; pass inputs as
the same device arrays each run, or as host arrays copied into place.
"""
function compile(graph::Luminal.Graph; device::Luminal.AbstractDevice=Luminal.get_device(),
                 retain::Vector{Int}=Int[], free_intermediates::Bool=true, fuse::Bool=true,
                 weight_dtype=Float32, capture::Bool=false, fold::Bool=true,
                 search::Symbol=:none, precision=false, search_inputs=nothing,
                 search_tolerance::Real=1e-3,
                 search_cache::Union{Nothing,String}=joinpath(homedir(), ".cache", "Luminal.jl", "search"),
                 lowering::AbstractDict=Dict{String,Bool}())
    capture && free_intermediates && error("capture=true requires free_intermediates=false")
    if search !== :none
        return Luminal.EGraphRewrite.compile_searched(graph; search=search, precision=precision,
            search_inputs=search_inputs, search_cache=search_cache, search_tolerance=search_tolerance,
            device=device, retain=retain,
            free_intermediates=free_intermediates, fuse=fuse, weight_dtype=weight_dtype,
            capture=capture, fold=fold, lowering=lowering)
    end
    # Lowering sites: fusions and zero-copy views this compiler would apply on its
    # own. Each is named by a stable key (kind, ops, shape) shared by every node
    # with the same pattern -- e.g. all layers' residual adds -- so a measured
    # search can time turning it off (`lowering[key] = false`). The keys seen are
    # recorded on the compiled graph (`cache[:lowering_sites]`).
    sites = Set{String}()
    allow(key::String) = (push!(sites, key); get(lowering, key, true))
    _opname(op) = op isa Luminal.Function ? op.name : string(nameof(typeof(op)))
    _dimstr(id) = string(Tuple(realized_dims(graph.shapes[id])))
    # 0. Consumer count
    consumer_count = zeros(Int, length(graph.nodes))
    for node in graph.nodes
        for (id, _, _) in node.inputs
            consumer_count[id] += 1
        end
    end
    for id in retain
        consumer_count[id] += 1
    end

    compile_device = device
    consumers = [Tuple{Int, ShapeTracker}[] for _ in graph.nodes]
    for (cid, node) in enumerate(graph.nodes), (id, _, st) in node.inputs
        push!(consumers[id], (cid, st))
    end

    # 1. Results array: persistent tensors (weights) are pre-populated
    results = Vector{Any}(undef, length(graph.nodes))
    persistent = falses(length(graph.nodes))
    for (node_id, node) in enumerate(graph.nodes)
        if haskey(graph.tensors, (node_id, 1))
            data = graph.tensors[(node_id, 1)]
            storage = _weight_storage(graph, node_id, data, consumers, retain, weight_dtype)
            if storage !== nothing
                data = Luminal.half_weight(data, storage)
            end
            results[node_id] = data
            persistent[node_id] = true
        end
    end

    # 2. Constant folding: a node whose inputs are all persistent (weights, or nodes
    # folded here) and whose shapes are static is computed once, now, and becomes
    # persistent itself -- e.g. weights concatenated by a rewrite.
    # Folding streams, so a model's worth of concatenated weights never sits in
    # memory at once (Llama-3-8B's merged gate/up alone is ~15 GB in Float32, and its
    # Pad intermediates twice that): an intermediate is freed as soon as all its
    # consumers are folded, and a folded weight is converted to its reduced-precision
    # storage (freeing the Float32 value) as soon as it is complete.
    folded = falses(length(graph.nodes))
    pending = [length(consumers[i]) for i in eachindex(graph.nodes)]   # consumers not yet folded
    for (node_id, node) in enumerate(graph.nodes)
        fold || break
        (persistent[node_id] || isempty(node.inputs)) && continue
        all(persistent[id] for (id, _, _) in node.inputs) || continue
        val = try
            none = Dict{Symbol,Int}()
            dims = [eval_dim(d) for d in realized_dims(graph.shapes[node_id])]
            args = [realize_view(results[id], evaluate_shapes(st, none)) for (id, _, st) in node.inputs]
            out = Luminal.zero_tensor(compile_device, Float32, dims...)
            execute_op!(out, evaluate_op_shapes(node.op, none), args...)
            # Views made for the inputs hold references to whole input buffers until
            # the GC runs; drop them now (the inputs themselves keep theirs).
            for (k, (id, _, _)) in enumerate(node.inputs)
                args[k] !== results[id] && Luminal._free_now!(args[k])
            end
            out
        catch
            nothing   # e.g. symbolic shapes: leave it to run time
        end
        val === nothing && continue
        results[node_id] = val
        persistent[node_id] = folded[node_id] = true
        for (id, _, _) in node.inputs
            folded[id] || continue
            pending[id] -= 1
            if pending[id] == 0 && !(id in retain) && results[id] !== nothing
                Luminal._free_now!(results[id])   # every consumer is folded: no longer needed
                results[id] = nothing
            end
        end
        # A tensor used only as a matmul weight feeds no further folding: convert now.
        # Folded values belong to this compile only: converted directly, no shared cache.
        storage = _weight_storage(graph, node_id, val, consumers, retain, weight_dtype)
        if storage !== nothing
            results[node_id] = storage(val)
            Luminal._free_now!(val)
        end
    end
    for node_id in findall(folded)
        if !(node_id in retain) && all(folded[c] for (c, _) in consumers[node_id]) && results[node_id] !== nothing
            Luminal._free_now!(results[node_id])
            results[node_id] = nothing          # only fed other folded nodes
        end
    end

    processed_pads = falses(length(graph.nodes))
    # 3. Concatenations. concat_along builds Add(Pad(a), Pad(b)), the Pads placing a
    # and b in complementary ranges of one axis: 5 kernels (fill + copy per Pad, then
    # the Add). When the Pads feed only that Add, it runs as one step that copies a
    # and b into their ranges of the output; the Pads get no step.
    concats = Dict{Int, Any}()   # Add node => (axis, [(source id, source view, offset)])
    for (node_id, node) in enumerate(graph.nodes)
        (node.op isa Luminal.Add && length(node.inputs) == 2 && !persistent[node_id]) || continue
        parts = Any[]
        for (pid, _, st) in node.inputs
            pn = graph.nodes[pid]
            (pn.op isa Luminal.Pad && consumer_count[pid] == 1 && !persistent[pid] &&
             _is_trivial_view(st, graph.shapes[pid])) || break
            axes = [i for (i, (lo, hi)) in enumerate(pn.op.padding) if !(isequal(lo, 0) && isequal(hi, 0))]
            length(axes) == 1 && all(x -> x isa Integer, pn.op.padding[axes[1]]) || break
            push!(parts, (pid, axes[1], pn.op.padding[axes[1]], pn.inputs[1]))
        end
        length(parts) == 2 && parts[1][2] == parts[2][2] || continue
        axis = parts[1][2]
        dims = try [eval_dim(d) for d in realized_dims(graph.shapes[node_id])] catch; continue end
        all(p -> isequal(Luminal.DimType[eval_dim(d) for d in realized_dims(graph.shapes[p[1]])], dims), parts) || continue
        # the two ranges must tile the axis exactly
        rs = sort([(lo, dims[axis] - hi) for (_, _, (lo, hi), _) in parts])
        (rs[1][1] == 0 && rs[1][2] == rs[2][1] && rs[2][2] == dims[axis]) || continue
        allow("concat $(_dimstr(node_id))") || continue
        concats[node_id] = (axis, [(src, sst, lo) for (_, _, (lo, _), (src, _, sst)) in parts])
        for (pid, _, _, _) in parts; processed_pads[pid] = true; end
    end

    # 4. Identify fusible intermediates. An elementwise node is fused into its
    # consumer when that is its only use, the consumer is elementwise too, and the
    # consumer reads it unchanged.
    fusible_intermediates = Set{Int}()
    for (node_id, node) in enumerate(graph.nodes)
        fuse || break
        (is_elementwise(node.op) && !(node.op isa Luminal.Constant)) || continue
        haskey(concats, node_id) && continue
        consumer_count[node_id] == 1 && length(consumers[node_id]) == 1 || continue
        persistent[node_id] && continue
        cid, st = consumers[node_id][1]
        haskey(concats, cid) && continue
        c_op = graph.nodes[cid].op
        (is_elementwise(c_op) && !(c_op isa Luminal.Constant)) || continue
        _is_trivial_view(st, graph.shapes[node_id]) || continue
        allow("fuse $(_opname(node.op)) -> $(_opname(c_op)) $(_dimstr(node_id))") &&
            push!(fusible_intermediates, node_id)
    end

    # 5. Residual connections into GEMV weights. An Add of a weight matmul's output
    # (W * x, W stored as HalfWeight/QuantWeight, the output used only here) and
    # another tensor r of the same shape runs as one step: the GEMV kernel adds r
    # in its final write (W * x + r), and the matmul gets no step of its own.
    matmul_adds = Dict{Int, Any}()   # Add => (matmul id, (W, view), (x, view), (r, view))
    fused_matmuls = falses(length(graph.nodes))
    for (node_id, node) in enumerate(graph.nodes)
        fuse || break
        (node.op isa Luminal.Add && length(node.inputs) == 2) || continue
        (persistent[node_id] || haskey(concats, node_id) || node_id in fusible_intermediates) && continue
        any(id in fusible_intermediates for (id, _, _) in node.inputs) && continue
        dims = try [eval_dim(d) for d in realized_dims(graph.shapes[node_id])] catch; continue end
        for k in 1:2
            (mid, _, mst) = node.inputs[k]
            (rid, _, rst) = node.inputs[3 - k]
            m = graph.nodes[mid]
            (m.op isa Luminal.MatMul || m.op isa Luminal.MatMulQ8 || m.op isa Luminal.MatMulQ4 ||
             (m.op isa Luminal.MatMulF16 && m.op.impl === :gemv)) || continue
            (persistent[mid] || processed_pads[mid] || mid in retain) && continue
            consumer_count[mid] == 1 && length(consumers[mid]) == 1 || continue
            _is_trivial_view(mst, graph.shapes[mid]) && _is_trivial_view(rst, graph.shapes[rid]) || continue
            (wid, _, wst) = m.inputs[1]
            (persistent[wid] && (results[wid] isa Luminal.HalfWeight || results[wid] isa Luminal.QuantWeight ||
                                 results[wid] isa Luminal.Q4Weight)) || continue
            rdims = try [eval_dim(d) for d in realized_dims(graph.shapes[rid])] catch; continue end
            isequal(rdims, dims) || continue
            allow("residual $(nameof(typeof(results[wid]))) $(_dimstr(node_id))") || continue
            (xid, _, xst) = m.inputs[2]
            matmul_adds[node_id] = (mid, (wid, wst), (xid, xst), (rid, rst))
            fused_matmuls[mid] = true
            break
        end
    end

    steps = Base.Function[]
    processed = folded .| processed_pads .| fused_matmuls   # these nodes need no step of their own
    owned = fill(false, length(graph.nodes))  # buffers allocated by this graph's own steps
    dynamic = false  # true if any shape depends on symbolic dims (sym_vals)

    for (node_id, node) in enumerate(graph.nodes)
        processed[node_id] && continue
        op = node.op

        if haskey(concats, node_id)
            axis, parts = concats[node_id]
            node_shape = graph.shapes[node_id]
            dims = Tuple(eval_dim(d) for d in realized_dims(node_shape))
            src_sts = [evaluate_shapes(sst, Dict{Symbol,Int}()) for (_, sst, _) in parts]
            owned[node_id] = true
            backing = Ref{Any}(nothing)
            push!(steps, (res, dev, sym_vals, live) -> begin
                _prepare_output!(res, node_id, dims, dev, backing, owned, free_intermediates)
                out = res[node_id]
                for (k, (src, _, lo)) in enumerate(parts)
                    a = realize_view(res[src], src_sts[k])
                    rng = ntuple(i -> i == axis ? ((lo + 1):(lo + size(a, axis))) : (1:dims[i]), length(dims))
                    view(out, rng...) .= a
                end
                for (src, _, _) in parts
                    live[src] -= 1
                    if free_intermediates && live[src] == 0 && !persistent[src]
                        _release!(res, src, owned)
                    end
                end
            end)
            processed[node_id] = true
            continue
        end

        if haskey(matmul_adds, node_id)
            mid, (wid, wst), (xid, xst), (rid, rst) = matmul_adds[node_id]
            mop = graph.nodes[mid].op
            dims = Tuple(eval_dim(d) for d in realized_dims(graph.shapes[node_id]))
            sts = [evaluate_shapes(st, Dict{Symbol,Int}()) for st in (xst, rst)]
            owned[node_id] = true
            backing = Ref{Any}(nothing)
            push!(steps, (res, dev, sym_vals, live) -> begin
                _prepare_output!(res, node_id, dims, dev, backing, owned, free_intermediates)
                x = realize_view(res[xid], sts[1])
                r = realize_view(res[rid], sts[2])
                Luminal.matmul_residual!(res[node_id], mop, res[wid], x, r)
                for id in (wid, xid, rid)
                    live[id] -= 1
                    if free_intermediates && live[id] == 0 && !persistent[id]
                        _release!(res, id, owned)
                    end
                end
            end)
            processed[node_id] = true
            continue
        end

        if op isa Luminal.Constant
            val = to_device(op.value, compile_device)
            push!(steps, (res, dev, sym_vals, live) -> begin
                res[node_id] = val
            end)
            processed[node_id] = true
            continue
        elseif op isa Luminal.Function && op.name == "ARange"
            push!(steps, (res, dev, sym_vals, live) -> begin
                node_shape = graph.shapes[node_id]
                dims_int = map(d -> eval_dim(d, sym_vals), realized_dims(node_shape))
                n = dims_int[1]
                if !isassigned(res, node_id) || res[node_id] === nothing || length(res[node_id]) != n
                    isassigned(res, node_id) && res[node_id] !== nothing && _release!(res, node_id, owned)
                    res[node_id] = to_device(Float32.(collect(0:n-1)), dev)
                end
            end)
            owned[node_id] = true
            processed[node_id] = true
            continue
        elseif op isa Luminal.Function && op.name == "InputTensor"
            processed[node_id] = true
            continue
        end

        # Only nodes inside a fusion group (an intermediate, or a terminal with a fused
        # input) take the generated-kernel path; everything else runs op-by-op.
        in_fusion_group = node_id in fusible_intermediates ||
                          any(id in fusible_intermediates for (id, _, _) in node.inputs)
        if !is_elementwise(op) || !in_fusion_group
            input_specs = node.inputs
            node_shape = graph.shapes[node_id]
            is_persistent = haskey(graph.tensors, (node_id, 1))
            dtype = (compile_device isa Luminal.AbstractGPUDevice) ? Float32 : Float32
            owned[node_id] = !is_persistent
            backing = Ref{Any}(nothing)  # reuse mode on GPU: flat buffer with spare capacity
            aliased = Ref(false)         # res[node_id] currently aliases an input's memory
            # An Expand read only by elementwise ops is a broadcastable view (no copy).
            bcast_ok = op isa Luminal.Expand && !isempty(consumers[node_id]) &&
                       all(is_elementwise(graph.nodes[c].op) && !haskey(concats, c) for (c, _) in consumers[node_id]) &&
                       !(node_id in retain)
            # A Slice read only by elementwise ops may be a strided view (no copy). Only
            # when buffers are not freed mid-run: a SubArray does not hold a reference
            # on its parent's GPU buffer the way a reshape does.
            strided_ok = !free_intermediates && op isa Luminal.Slice && !isempty(consumers[node_id]) &&
                         all(is_elementwise(graph.nodes[c].op) for (c, _) in consumers[node_id]) &&
                         !(node_id in retain)
            # Zero-copy views of movement ops are a lowering site too (a Permute
            # only where it can alias: it moves size-1 dims only).
            view_ok = true
            if length(node.inputs) == 1 && !is_persistent &&
               (op isa Luminal.Reshape || op isa Luminal.Expand || op isa Luminal.Slice ||
                (op isa Luminal.Permute && begin
                    ind = realized_dims(node.inputs[1][3])
                    !all(d -> d isa Integer, ind) || issorted([d for d in op.dims if ind[d] != 1])
                end))
                view_ok = allow("view $(_opname(op)) $(_dimstr(node_id))")
            end
            bcast_ok &= view_ok
            strided_ok &= view_ok

            # Shapes free of symbolic dims are evaluated once here instead of every run.
            static = try
                (in_sts = [evaluate_shapes(st, Dict{Symbol,Int}()) for (_, _, st) in input_specs],
                 dims = [eval_dim(d) for d in realized_dims(node_shape)],
                 op = evaluate_op_shapes(op, Dict{Symbol,Int}()))
            catch
                nothing
            end
            static === nothing && (dynamic = true)

            push!(steps, (res, dev, sym_vals, live) -> begin
                in_sts = static === nothing ? [evaluate_shapes(st, sym_vals) for (_, _, st) in input_specs] : static.in_sts
                run_op = static === nothing ? evaluate_op_shapes(op, sym_vals) : static.op

                # 1. Realize input views
                step_args = [realize_view(res[id], in_sts[k]) for (k, (id, _, _)) in enumerate(input_specs)]

                # 2. Alias or allocate the output
                alias = nothing
                if !is_persistent
                    dims_int = static === nothing ? map(d -> eval_dim(d, sym_vals), realized_dims(node_shape)) : static.dims
                    sz = Tuple(dims_int)
                    alias = (view_ok && length(step_args) == 1) ?
                            _alias_view(run_op, step_args[1], sz; strided_ok=strided_ok) : nothing
                    if alias === nothing && bcast_ok && step_args[1] isa AbstractArray
                        a = step_args[1]
                        alias = Luminal.BroadcastView(Base.reshape(a,
                            (size(a)[1:run_op.dim-1]..., 1, size(a)[run_op.dim:end]...)))
                    end
                    if alias !== nothing
                        res[node_id] = alias
                        aliased[] = true
                    elseif aliased[]
                        # Never write through a stale alias into another node's buffer
                        res[node_id] = nothing
                        aliased[] = false
                    end
                    alias === nothing && _prepare_output!(res, node_id, sz, dev, backing, owned, free_intermediates)
                end

                # 3. Execute
                alias === nothing && execute_op!(res[node_id], run_op, step_args...)

                # 4. Free dead inputs
                for (id, _, _) in input_specs
                    live[id] -= 1
                    if free_intermediates && live[id] == 0 && !persistent[id]
                        _release!(res, id, owned)
                    end
                end
            end)
            processed[node_id] = true
        else
            if !(node_id in fusible_intermediates)
                # Terminal of a fusion group: one broadcast kernel for the whole group
                group_inputs = [] # (id, st, sym)
                sym_cache = Dict{Any, Any}()
                input_syms = [build_fused_expr!(graph, in_id, consumer_count, group_inputs, fusible_intermediates, sym_cache, in_st)
                              for (in_id, _, in_st) in node.inputs]
                node_expr = fix_gpu_ast!(toexpr(op_to_sym(op, input_syms)))
                fused_op = Luminal.FusedElementwiseOp("fused_$node_id", _fused_kernel(node_expr, length(group_inputs)))

                node_shape = graph.shapes[node_id]
                target_rank = length(realized_dims(node_shape))
                align_rank(val, rank) = ndims(val) >= rank ? val :
                    Base.reshape(val, size(val)..., ntuple(_ -> 1, rank - ndims(val))...)
                owned[node_id] = true
                backing = Ref{Any}(nothing)
                static = try
                    (in_sts = [evaluate_shapes(st, Dict{Symbol,Int}()) for (_, st, _) in group_inputs],
                     dims = [eval_dim(d) for d in realized_dims(node_shape)])
                catch
                    nothing
                end
                static === nothing && (dynamic = true)

                push!(steps, (res, dev, sym_vals, live) -> begin
                    in_sts = static === nothing ? [evaluate_shapes(st, sym_vals) for (_, st, _) in group_inputs] : static.in_sts
                    args = [align_rank(realize_view(res[id], in_sts[k]), target_rank) for (k, (id, _, _)) in enumerate(group_inputs)]
                    dims_int = static === nothing ? map(d -> eval_dim(d, sym_vals), realized_dims(node_shape)) : static.dims
                    _prepare_output!(res, node_id, Tuple(dims_int), dev, backing, owned, free_intermediates)

                    Base.invokelatest(execute_op!, res[node_id], fused_op, args...)

                    for (id, _, _) in group_inputs
                        live[id] -= 1
                        if free_intermediates && live[id] == 0 && !persistent[id]
                            _release!(res, id, owned)
                        end
                    end
                end)
                processed[node_id] = true
            end
        end
    end

    return CompiledGraph(graph, steps, results,
                         Dict{Symbol, Any}(:capture => capture, :dynamic => dynamic, :lowering_sites => sites,
                                           :freeable => findall(owned .| folded)),
                         consumer_count)
end