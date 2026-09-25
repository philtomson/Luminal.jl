
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

# A persistent tensor to store as a HalfWeight: a 2D Float32 GPU array whose every
# use is as the left operand of a matmul, read unchanged -- all MatMulF16, or all
# MatMul/MatMulF16 when weight_dtype=Float16.
function _is_matmul_weight(graph, node_id, data, consumers, retain, weight_dtype)
    (data isa AnyGPUArray && data isa DenseArray && eltype(data) == Float32 && ndims(data) == 2) || return false
    size(data, 2) % 8 == 0 || return false
    node_id in retain && return false
    isempty(consumers[node_id]) && return false
    for (cid, st) in consumers[node_id]
        c = graph.nodes[cid]
        c.op isa Luminal.MatMulF16 || (weight_dtype === Float16 && c.op isa Luminal.MatMul) || return false
        (c.inputs[1][1] == node_id && c.inputs[2][1] != node_id) || return false
        _is_trivial_view(st, graph.shapes[node_id]) || return false
    end
    return true
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
function _alias_view(op, arg, sz)
    op isa Luminal.Slice && return _alias_slice(op, arg, sz)
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
function _alias_slice(op, arg, sz)
    (arg isa DenseArray && prod(sz) > 0) || return nothing   # empty slices take the copy path
    n = ndims(arg)
    length(sz) == n || return nothing
    k = findfirst(i -> sz[i] != size(arg, i), 1:n)
    k === nothing && return Base.reshape(arg, sz)
    all(sz[i] == 1 for i in k+1:n) || return nothing
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
           op isa Luminal.ReLU || op isa Luminal.Constant
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

# --- Main Compile Function ---

"""
    compile(graph; device=get_device(), retain=Int[], free_intermediates=true, fuse=true,
            weight_dtype=Float32, capture=false, fold=true,
            search=:none, precision=false, search_inputs=nothing, search_cache=...)

Build an executable `CompiledGraph`. With `free_intermediates=true` each
intermediate buffer is released as soon as its last consumer has run (lowest
peak memory, but every run reallocates). With `false` buffers persist across
runs and are reused, reallocated only when a symbolic dimension changes their
size, which suits graphs executed many times such as per-token decode.
Retained outputs are overwritten by the next run in either mode.
`fuse=true` merges chains of elementwise ops into single kernels.
`fold=true` computes nodes that depend only on persistent tensors (weights) once,
at compile time. `weight_dtype=Float16` stores GPU weights that are only ever used as the left
operand of matmuls in Float16 (see `HalfWeight`); compute stays Float32.
`search=:static` or `:measured` first runs the e-graph rewrite layer
(`EGraphRewrite`) and compiles the graph it picks: by the static cost model, or
by timing verified candidates (cached in `search_cache`). `precision=true` lets
it choose Float16 weights per matmul. `retain` must list the outputs; the result
maps the original input/output node ids. `:measured` needs static shapes.
`capture=true` (AMD GPUs) records a run into a HIP graph and replays it on
later runs with the same symbolic dims and input buffers (see
`execute_with_capture`). Requires `free_intermediates=false`; pass inputs as
the same device arrays each run, or as host arrays copied into place.
"""
function compile(graph::Luminal.Graph; device::Luminal.AbstractDevice=Luminal.get_device(),
                 retain::Vector{Int}=Int[], free_intermediates::Bool=true, fuse::Bool=true,
                 weight_dtype::Type=Float32, capture::Bool=false, fold::Bool=true,
                 search::Symbol=:none, precision::Bool=false, search_inputs=nothing,
                 search_cache::Union{Nothing,String}=joinpath(homedir(), ".cache", "Luminal.jl", "search"))
    capture && free_intermediates && error("capture=true requires free_intermediates=false")
    if search !== :none
        return Luminal.EGraphRewrite.compile_searched(graph; search=search, precision=precision,
            search_inputs=search_inputs, search_cache=search_cache, device=device, retain=retain,
            free_intermediates=free_intermediates, fuse=fuse, weight_dtype=weight_dtype,
            capture=capture, fold=fold)
    end
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
            if _is_matmul_weight(graph, node_id, data, consumers, retain, weight_dtype)
                data = Luminal.half_weight(data)
            end
            results[node_id] = data
            persistent[node_id] = true
        end
    end

    # 2. Constant folding: a node whose inputs are all persistent (weights, or nodes
    # folded here) and whose shapes are static is computed once, now, and becomes
    # persistent itself -- e.g. weights concatenated by a rewrite.
    folded = falses(length(graph.nodes))
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
            out
        catch
            nothing   # e.g. symbolic shapes: leave it to run time
        end
        val === nothing && continue
        results[node_id] = val
        persistent[node_id] = folded[node_id] = true
    end
    for node_id in findall(folded)
        if !(node_id in retain) && all(folded[c] for (c, _) in consumers[node_id])
            results[node_id] = nothing          # only fed other folded nodes
        elseif _is_matmul_weight(graph, node_id, results[node_id], consumers, retain, weight_dtype)
            # Folded values belong to this compile only: convert directly, no shared cache
            results[node_id] = Luminal.HalfWeight(results[node_id])
        end
    end

    # 3. Identify fusible intermediates. An elementwise node is fused into its
    # consumer when that is its only use, the consumer is elementwise too, and the
    # consumer reads it unchanged.
    fusible_intermediates = Set{Int}()
    for (node_id, node) in enumerate(graph.nodes)
        fuse || break
        (is_elementwise(node.op) && !(node.op isa Luminal.Constant)) || continue
        consumer_count[node_id] == 1 && length(consumers[node_id]) == 1 || continue
        persistent[node_id] && continue
        cid, st = consumers[node_id][1]
        c_op = graph.nodes[cid].op
        (is_elementwise(c_op) && !(c_op isa Luminal.Constant)) || continue
        _is_trivial_view(st, graph.shapes[node_id]) && push!(fusible_intermediates, node_id)
    end

    steps = Base.Function[]
    processed = copy(folded)   # folded nodes need no step
    owned = fill(false, length(graph.nodes))  # buffers allocated by this graph's own steps
    dynamic = false  # true if any shape depends on symbolic dims (sym_vals)

    for (node_id, node) in enumerate(graph.nodes)
        processed[node_id] && continue
        op = node.op

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
                    alias = length(step_args) == 1 ? _alias_view(run_op, step_args[1], sz) : nothing
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

    return CompiledGraph(graph, steps, results, Dict{Symbol, Any}(:capture => capture, :dynamic => dynamic), consumer_count)
end