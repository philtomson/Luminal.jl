# EGraphRewrite.jl — graph-level rewrite layer on Metatheory.jl e-graphs (prototype).
#
#   Graph --to_egraph--> EGraph --saturate!(rules)--> EGraph --extract/choose--> Graph --> compile()
#
# * Every Luminal node becomes one e-node. The head is the op type (`xMatMul`,
#   `xPermute`, ...), the first child is a literal tuple of the op's fields, the
#   rest are the input e-classes. Inputs are `xIn((node_id,))`, so they are
#   never merged. The e-graph's hash-consing keeps shared subgraphs shared: the
#   graph stays a DAG, it is never unrolled into a tree.
# * An input edge that reads its producer through a non-trivial view (permute,
#   slice, pad, broadcast in the ShapeTracker) becomes an explicit
#   `xView(child, (view_index,))` e-node, so rules never match through a view.
# * Each e-class carries its output dims (analysis `Shp`). Shapes are computed
#   for the ops rules create, and checked against the graph for bridged nodes.
# * Extraction is DAG extraction: one e-node per e-class, rebuilt into a new
#   Graph memoized by e-class. `choices` overrides the pick per e-class, which is
#   how a measured search tries alternatives.
#
# Head names are prefixed with `x` and must not be defined as Julia functions:
# Metatheory 3.0's pattern compiler hashes a defined function object rather
# than the symbol, and such patterns silently never match.

module EGraphRewrite

using Metatheory
using Metatheory.EGraphs
using Metatheory.VecExprModule: v_isexpr, v_head, v_children
import ..Luminal
using ..Luminal: Graph, ShapeTracker, realized_dims, add_op!, Op, DimType

export Rewriter, to_egraph, saturate_graph!, merge_projections!, choice_groups, decisions,
       extract_graph, static_cost, measured_search

# ---------------------------------------------------------------------------
# Shape analysis
# ---------------------------------------------------------------------------

# Output dims of an e-class, and whether it depends only on weights (then compile()
# constant-folds it and it costs nothing at run time).
struct Shp
    dims::Tuple
    isconst::Bool
end
Shp(dims::Tuple) = Shp(dims, false)

# Views and op types are side tables (e-graph literals must hash by value).
mutable struct BridgeCtx
    views::Vector{ShapeTracker}
    view_index::Dict{Any, Int}
    optypes::Dict{Symbol, DataType}
    weights::Set{Int}          # node ids of persistent tensors (graph.tensors)
    joint::Vector{Any}         # (label, classes, signature) from multi-node rewrites
end
const CTX = Ref{BridgeCtx}()   # read by `make`, which has no other way to reach it

_lit(g, id) = get_constant(g, v_head(g[id].nodes[1]))
_dims(g, id) = (d = g[id].data; d === nothing ? nothing : d.dims)
_isone(x) = x isa Integer && x == 1
_isconst(g, id) = (d = g[id].data; d !== nothing && d.isconst)

function _broadcast(a::Tuple, b::Tuple)
    n = max(length(a), length(b))
    pa = (a..., ntuple(_ -> 1, n - length(a))...)
    pb = (b..., ntuple(_ -> 1, n - length(b))...)
    out = Any[]
    for i in 1:n
        x, y = pa[i], pb[i]
        if isequal(x, y) || _isone(y)
            push!(out, x)
        elseif _isone(x)
            push!(out, y)
        else
            return nothing
        end
    end
    return Tuple(out)
end

function EGraphs.make(g::EGraph{Expr,Shp}, n::VecExpr)
    v_isexpr(n) || return nothing
    ch = v_children(n)
    d = _make_dims(g, get_constant(g, v_head(n)), ch)
    d === nothing && return nothing
    return Shp(d.dims, length(ch) > 1 && all(c -> _isconst(g, c), ch[2:end]))
end

function _make_dims(g, h, ch)
    if h === :xView
        return Shp(Tuple(realized_dims(CTX[].views[_lit(g, ch[1])[1]])))
    elseif h === :xPad
        a = _dims(g, ch[2]); a === nothing && return nothing
        p = _lit(g, ch[1])[1]
        all(x -> x isa Integer, a) || return nothing
        return Shp(ntuple(i -> i <= length(p) ? a[i] + p[i][1] + p[i][2] : a[i], length(a)))
    elseif h === :xSlice
        a = _dims(g, ch[2]); a === nothing && return nothing
        r = _lit(g, ch[1])[1]
        all(x -> x isa Integer, a) || return nothing
        return Shp(ntuple(i -> i <= length(r) ? min(r[i][2], a[i]) - max(r[i][1], 0) : a[i], length(a)))
    elseif h === :xReshape
        return Shp(Tuple(_lit(g, ch[1])[1]))
    elseif h === :xExpand
        a = _dims(g, ch[2]); a === nothing && return nothing
        d, k = _lit(g, ch[1])
        return Shp((a[1:d-1]..., k, a[d:end]...))
    elseif h === :xMul || h === :xAdd
        a, b = _dims(g, ch[2]), _dims(g, ch[3])
        (a === nothing || b === nothing) && return nothing
        r = _broadcast(a, b)
        return r === nothing ? nothing : Shp(r)
    elseif h === :xMatMul || h === :xMatMulF16 || h === :xMatMulQ8 || h === :xMatMulQ4 || h === :xMatMulT
        a, b = _dims(g, ch[2]), _dims(g, ch[3])
        (a === nothing || b === nothing) && return nothing
        if h === :xMatMulT
            ta, tb = _lit(g, ch[1])
            ta && (a = (a[2], a[1], a[3:end]...))
            tb && (b = (b[2], b[1], b[3:end]...))
        end
        batch = _broadcast(a[3:end], b[3:end])
        return batch === nothing ? nothing : Shp((a[1], b[2], batch...))
    end
    return nothing
end

function EGraphs.join(a::Shp, b::Shp)
    isequal(a.dims, b.dims) || error("EGraphRewrite: merged e-classes disagree on shape: $(a.dims) vs $(b.dims)")
    return Shp(a.dims, a.isconst || b.isconst)
end

# ---------------------------------------------------------------------------
# Graph -> EGraph
# ---------------------------------------------------------------------------

_freeze(x::AbstractVector) = Tuple(map(_freeze, x))
_freeze(x::Tuple) = map(_freeze, x)
_freeze(x) = x
_head(op::Op) = Symbol("x", nameof(typeof(op)))
_params(op::Op) = ntuple(i -> _freeze(getfield(op, i)), nfields(op))

_thaw(::Type{T}, p) where {T <: AbstractVector} = convert(T, collect(p))
_thaw(::Type, p) = p
_rebuild_op(T::DataType, params) = T((_thaw(fieldtype(T, i), params[i]) for i in 1:fieldcount(T))...)

"""
    Rewriter

An e-graph built from a Luminal graph, with what's needed to turn extracted
e-nodes back into a graph.
"""
struct Rewriter
    g::EGraph{Expr,Shp}
    ctx::BridgeCtx
    graph::Graph
    roots::Vector{Int}          # original node ids to preserve (outputs)
    class_of::Vector{Id}        # original node id -> e-class id
end

function _edge(g, ctx, child_class::Id, st::ShapeTracker, producer_shape::ShapeTracker)
    Luminal._is_trivial_view(st, producer_shape) && return g[child_class]
    key = Luminal._st_key(st)
    idx = get!(ctx.view_index, key) do
        push!(ctx.views, deepcopy(st)); length(ctx.views)
    end
    return g[addexpr!(g, Expr(:call, :xView, (idx,), g[child_class]))]
end

"""
    to_egraph(graph, roots) -> Rewriter

Build an e-graph with one e-node per node of `graph` (shared subgraphs stay
shared) and a synthetic root over `roots`, the node ids that must survive.
"""
function to_egraph(graph::Graph, roots::Vector{Int})
    ctx = BridgeCtx(ShapeTracker[], Dict{Any,Int}(), Dict{Symbol,DataType}(),
                    Set{Int}(first(k) for k in keys(graph.tensors)), Any[])
    CTX[] = ctx
    for T in (Luminal.Reshape, Luminal.Mul, Luminal.Add, Luminal.MatMul, Luminal.MatMulF16, Luminal.MatMulQ8, Luminal.MatMulQ4, Luminal.MatMulT,
              Luminal.Expand, Luminal.Pad, Luminal.Slice)
        ctx.optypes[Symbol("x", nameof(T))] = T
    end
    g = EGraph{Expr,Shp}()
    class_of = Vector{Id}(undef, length(graph.nodes))
    for (nid, node) in enumerate(graph.nodes)
        op = node.op
        dims = Tuple(realized_dims(graph.shapes[nid]))
        if op isa Luminal.Function && op.name == "InputTensor"
            ex = Expr(:call, :xIn, (nid,))
        else
            h = _head(op)
            ctx.optypes[h] = typeof(op)
            kids = [_edge(g, ctx, class_of[id], st, graph.shapes[id]) for (id, _, st) in node.inputs]
            # Ops without inputs (ARange, Constant) are defined by their output shape
            # and dtype too; append them so e.g. ARange(32) and ARange(1), or
            # Constant(1) and Constant(1f0) (equal as values), stay distinct.
            params = isempty(kids) ? (_params(op)..., dims, graph.dtypes[nid]) : _params(op)
            ex = Expr(:call, h, params, kids...)
        end
        id = addexpr!(g, ex)
        ec = g[id]
        if ec.data === nothing
            isconst = nid in ctx.weights ||
                      (!isempty(node.inputs) && all(_isconst(g, class_of[i]) for (i, _, _) in node.inputs))
            ec.data = Shp(dims, isconst)
        elseif !isequal(ec.data.dims, dims)
            error("EGraphRewrite: shape analysis gives $(ec.data.dims) for node $nid ($(typeof(op))) but the graph has $dims")
        end
        class_of[nid] = id
    end
    g.root = addexpr!(g, Expr(:call, :xRoots, (), (g[class_of[r]] for r in roots)...))
    return Rewriter(g, ctx, graph, copy(roots), class_of)
end

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
# Pattern variables bind to e-classes (their `.data` is the shape); `p::Tuple`
# binds an op's literal parameter tuple. A dynamic rule (`=>`) returning
# `nothing` does not fire.

_scalar(d) = d !== nothing && all(_isone, d)
_has_head(g, ec, h) = any(n -> v_isexpr(n) && get_constant(g, v_head(n)) === h, ec.nodes)
function _is_weight(g, ec)
    for n in ec.nodes
        v_isexpr(n) && get_constant(g, v_head(n)) === :xIn || continue
        get_constant(g, v_head(g[v_children(n)[1]].nodes[1]))[1] in CTX[].weights && return true
    end
    return false
end

# Expand feeding a broadcasting Add/Mul: a size-1 dim (a zero-copy reshape) broadcasts
# the same way, so the materialized copy is unnecessary.
function _expand_to_broadcast(h, p, e, a, b)
    ad = a.data; ad === nothing && return nothing
    d, k = e
    bd = b.data; bd === nothing && return nothing
    expanded = (ad.dims[1:d-1]..., k, ad.dims[d:end]...)
    one_dim  = (ad.dims[1:d-1]..., 1, ad.dims[d:end]...)
    isequal(_broadcast(expanded, bd.dims), _broadcast(one_dim, bd.dims)) || return nothing
    return Expr(:call, h, p, Expr(:call, :xReshape, (one_dim,), a), b)
end

const CANONICAL_RULES = @theory p q e a b s begin
    xMul(p::Tuple, xExpand(e::Tuple, a), b) => _expand_to_broadcast(:xMul, p, e, a, b)
    xMul(p::Tuple, b, xExpand(e::Tuple, a)) => _expand_to_broadcast(:xMul, p, e, a, b)
    xAdd(p::Tuple, xExpand(e::Tuple, a), b) => _expand_to_broadcast(:xAdd, p, e, a, b)
    xAdd(p::Tuple, b, xExpand(e::Tuple, a)) => _expand_to_broadcast(:xAdd, p, e, a, b)
    # Consecutive reshapes collapse; a reshape to the same shape is the identity
    xReshape(p::Tuple, xReshape(q::Tuple, a)) --> xReshape(p, a)
    xReshape(p::Tuple, a) => (a.data !== nothing && isequal(Tuple(p[1]), a.data.dims)) ? a : nothing
end

# Scaling a matmul's result by a scalar equals scaling either operand first. Which
# is cheapest depends on the operand sizes, and on whether an operand is a weight.
const ALGEBRAIC_RULES = @theory p q a b s begin
    xMul(p::Tuple, xMatMul(q::Tuple, a, b), s) =>
        _scalar(s.data === nothing ? nothing : s.data.dims) ? :(xMatMul($q, xMul($p, $a, $s), $b)) : nothing
    xMul(p::Tuple, xMatMul(q::Tuple, a, b), s) =>
        _scalar(s.data === nothing ? nothing : s.data.dims) ? :(xMatMul($q, $a, xMul($p, $b, $s))) : nothing
end

# Precision choice (not exact: rounds weights to Float16): a weight matmul may run
# with Float16 weights, with the GEMV kernel at any of its workgroup sizes (the
# best one depends on the matrix shape). Opt-in.
_half_ok(g, w) = _is_weight(g, w) && w.data !== nothing && length(w.data.dims) == 2
const WEIGHT_VARIANTS = ((64, :gemv), (128, :gemv), (256, :gemv))
const PRECISION_RULES = @theory q w x begin
    xMatMul(q::Tuple, w, x) => _half_ok(_egraph, w) ? :(xMatMulF16($((64, :gemv)), $w, $x)) : nothing
    xMatMul(q::Tuple, w, x) => _half_ok(_egraph, w) ? :(xMatMulF16($((128, :gemv)), $w, $x)) : nothing
    xMatMul(q::Tuple, w, x) => _half_ok(_egraph, w) ? :(xMatMulF16($((256, :gemv)), $w, $x)) : nothing
end

# precision=:activations additionally offers rocBLAS's mixed-precision GEMM
# (:gemm_ex), which rounds the *activations* to Float16 too: ~2x Float32 GEMM for
# prefill-sized inputs, but ~5e-3 relative logit error over TinyLlama's 22 layers.
const ACTIVATION_VARIANTS = ((256, :gemm_ex),)
const ACTIVATION_RULES = @theory q w x begin
    xMatMul(q::Tuple, w, x) => _half_ok(_egraph, w) ? :(xMatMulF16($((256, :gemm_ex)), $w, $x)) : nothing
end

# precision=:int8 offers weight-only int8 (group-wise scales) for weight matmuls:
# half the bytes of Float16 for decode, lossy (see docs). Each variant is
# (threads per workgroup, columns per pass), 0 meaning the shape heuristic
# (`_q8_threads`, `Q8_MAX_COLS`); which is fastest depends on the matrix shape,
# the number of columns and the GPU, so a measured search times them.
const INT8_VARIANTS = ((0, 0), (128, 0), (256, 0), (0, 8))
_int8_ok(g, w) = _half_ok(g, w) && w.data.dims[2] % 16 == 0
const INT8_RULES = @theory q w x begin
    xMatMul(q::Tuple, w, x) => _int8_ok(_egraph, w) ? :(xMatMulQ8($((0, 0)), $w, $x)) : nothing
    xMatMul(q::Tuple, w, x) => _int8_ok(_egraph, w) ? :(xMatMulQ8($((128, 0)), $w, $x)) : nothing
    xMatMul(q::Tuple, w, x) => _int8_ok(_egraph, w) ? :(xMatMulQ8($((256, 0)), $w, $x)) : nothing
    xMatMul(q::Tuple, w, x) => _int8_ok(_egraph, w) ? :(xMatMulQ8($((0, 8)), $w, $x)) : nothing
end

# precision=:int4 offers 4-bit weights (Q4Weight: symmetric, groups of 32 by
# default). Variants are (threads per workgroup, columns per pass), 0 = the
# kernel's per-shape defaults. Accuracy is a larger step than int8 (+5-8%
# perplexity on Llama models): choose it deliberately, or per tensor with
# `weight_dtype` policies (see `weight_preset`).
const INT4_VARIANTS = ((0, 0), (128, 0), (256, 0), (0, 1))
_int4_ok(g, w) = _half_ok(g, w) && w.data.dims[2] % 32 == 0
const INT4_RULES = @theory q w x begin
    xMatMul(q::Tuple, w, x) => _int4_ok(_egraph, w) ? :(xMatMulQ4($((0, 0)), $w, $x)) : nothing
    xMatMul(q::Tuple, w, x) => _int4_ok(_egraph, w) ? :(xMatMulQ4($((128, 0)), $w, $x)) : nothing
    xMatMul(q::Tuple, w, x) => _int4_ok(_egraph, w) ? :(xMatMulQ4($((256, 0)), $w, $x)) : nothing
    xMatMul(q::Tuple, w, x) => _int4_ok(_egraph, w) ? :(xMatMulQ4($((0, 1)), $w, $x)) : nothing
end

# Gross verification tolerance for the static extraction with int8 weights (see
# compile_searched: candidates are then checked against it). Against Float32,
# int8 logits differ by ~2% on real inputs and ~12% on the default all-zero
# search inputs (TinyLlama), so the check can only catch broken candidates
# (layout or indexing errors are O(1)), not quantization quality: that is the
# user's opt-in, measured by perplexity (examples/quant_eval.jl).
const INT8_SEARCH_TOLERANCE = 0.25
# int4 logits differ from Float32 by ~0.2 relative on real inputs, and by more than
# 1 on the all-zero default search inputs (TinyLlama decode: near-flat logits), so
# no tolerance separates correct int4 from broken output there. The static
# extraction is only required to be finite; the int4 kernels' correctness is
# covered by exact tests against dequantized weights, and every candidate must
# still match the static extraction within `search_tolerance`.
const INT4_SEARCH_TOLERANCE = Inf

# `precision` enables reduced-precision alternatives: false (exact rewrites only);
# true or :weights (Float16 weights, Float32 activations); :activations (also
# Float16 activations); :int8 (int8 weights); :int4 (4-bit weights); or a
# collection, e.g. (:weights, :int8).
function _precision_features(precision)
    precision === false && return Set{Symbol}()
    precision === true && return Set([:weights])
    precision isa Symbol && return _precision_features((precision,))
    f = Set{Symbol}()
    for p in precision
        p in (:weights, :activations, :int8, :int4) || error("unknown precision $p (use :weights, :activations, :int8, :int4)")
        push!(f, p)
        p === :activations && push!(f, :weights)
    end
    return f
end

# Kernel choice: a matmul whose operand is a Permute swapping the first two dims can
# read the unpermuted tensor through a BLAS transpose flag instead of copying it.
_is_swap12(p) = (d = p[1]; length(d) >= 2 && d[1] == 2 && d[2] == 1 && all(d[i] == i for i in 3:length(d)))
const KERNEL_RULES = @theory q p r a b begin
    xMatMul(q::Tuple, xPermute(p::Tuple, a), b) =>
        _is_swap12(p) ? :(xMatMulT($((true, false)), $a, $b)) : nothing
    xMatMul(q::Tuple, a, xPermute(p::Tuple, b)) =>
        _is_swap12(p) ? :(xMatMulT($((false, true)), $a, $b)) : nothing
    xMatMul(q::Tuple, xPermute(p::Tuple, a), xPermute(r::Tuple, b)) =>
        (_is_swap12(p) && _is_swap12(r)) ? :(xMatMulT($((true, true)), $a, $b)) : nothing
end

"""
    saturate_graph!(rw; precision=false, iterations=8)

Apply the rule sets until saturation (or the iteration/size limits).
"""
function saturate_graph!(rw::Rewriter; precision=false, iterations::Int=16)
    CTX[] = rw.ctx
    f = _precision_features(precision)
    theory = vcat(CANONICAL_RULES, ALGEBRAIC_RULES, KERNEL_RULES,
                  :weights in f ? PRECISION_RULES : RewriteRule[],
                  :activations in f ? ACTIVATION_RULES : RewriteRule[],
                  :int8 in f ? INT8_RULES : RewriteRule[],
                  :int4 in f ? INT4_RULES : RewriteRule[])
    params = SaturationParams(timeout=iterations, eclasslimit=0, enodelimit=0)
    return saturate!(rw.g, theory, params)
end

# ---------------------------------------------------------------------------
# Cost model and extraction
# ---------------------------------------------------------------------------

_numel(d) = d === nothing ? 0 : prod((x isa Integer ? x : 1 for x in d); init=1)
const LAUNCH_BYTES = 40_000   # fixed per-kernel cost, in equivalent bytes moved

"""
    static_cost(g, n) -> Float64

Rough per-e-node cost in bytes moved plus a fixed launch cost. Views, inputs,
reshapes and literals are free (they alias or are resolved at compile time).
"""
function static_cost(g, n::VecExpr)
    v_isexpr(n) || return 0.0
    h = get_constant(g, v_head(n))
    h in (:xIn, :xView, :xReshape, :xRoots) && return 0.0
    ch = v_children(n)
    length(ch) > 1 && all(c -> _isconst(g, c), ch[2:end]) && return 0.0   # constant-folded
    if h === :xSlice
        a = _dims(g, ch[2])
        a === nothing && return LAUNCH_BYTES
        r = _lit(g, ch[1])[1]
        out = ntuple(i -> i <= length(r) ? min(r[i][2], a[i]) - max(r[i][1], 0) : a[i], length(a))
        k = findfirst(i -> out[i] != a[i], 1:length(a))
        (k === nothing || all(out[i] == 1 for i in k+1:length(a))) && return 0.0   # aliases
    end
    ins = [_numel(_dims(g, c)) for c in ch[2:end]]
    # Weight matmul (2D left operand, weight-sized): the Float16 kernel is a GEMV
    # that loops over the N right-hand columns, so its cost grows ~linearly in N,
    # while rocBLAS's Float32 GEMM stays bandwidth-bound far longer. Calibrated on
    # TinyLlama prefill (Radeon 8060S): f16/f32 time is ~0.55 at N=1, ~1 at N=16,
    # ~2.6 at N=64, ~3.3 at N=256.
    wd = h in (:xMatMul, :xMatMulF16, :xMatMulQ8, :xMatMulQ4) ? _dims(g, ch[2]) : nothing
    if wd !== nothing && length(wd) == 2 && all(x -> x isa Integer, wd) && ins[1] >= 1 << 16
        N = max(1, ins[2] ÷ max(1, wd[2]))
        if h === :xMatMulQ4
            # 4-bit GEMV: 0.5625 bytes per weight; instruction-bound with several
            # columns (unpack and convert per weight and column), so each extra
            # column costs relatively more than for int8
            tie = _lit(g, ch[1]) == (0, 0) ? 0.0 : 1.0
            return 0.5625 * ins[1] * (1 + N / 3) + 4.0 * ins[2] + LAUNCH_BYTES + tie
        end
        if h === :xMatMulQ8
            # int8 GEMV: 1 byte per weight, same column scaling as the Float16 GEMV
            tie = _lit(g, ch[1]) == (0, 0) ? 0.0 : 1.0     # the model can't tell variants apart
            return 1.0 * ins[1] * (1 + N / 12) + 4.0 * ins[2] + LAUNCH_BYTES + tie
        end
        if h === :xMatMulF16
            grp, impl = _lit(g, ch[1])
            if impl === :gemm_ex
                # ~2x Float32 GEMM at any N, plus a fixed overhead that loses to the
                # GEMV for one or two columns (measured: 108 vs 53 us at N=1, 2048^2)
                return 2.0 * ins[1] * (1.5 + N / 64) + 4.0 * ins[2] + LAUNCH_BYTES
            end
            # Group sizes cost the same to the model; prefer the default on ties.
            tie = grp == Luminal.DEFAULT_HALF_GROUP ? 0.0 : 1.0
            return 2.0 * ins[1] * (1 + N / 12) + 4.0 * ins[2] + LAUNCH_BYTES + tie
        end
        return 4.0 * ins[1] * (1 + N / 64) + 4.0 * ins[2] + LAUNCH_BYTES
    end
    if h === :xMatMulF16
        tie = _lit(g, ch[1]) == (Luminal.DEFAULT_HALF_GROUP, :gemv) ? 0.0 : 1.0
        return 2.0 * ins[1] + 4.0 * sum(ins[2:end]) + LAUNCH_BYTES + tie
    elseif h === :xMatMul || h === :xMatMulT || h === :xMul || h === :xAdd
        a = (h === :xMatMul || h === :xMatMulT) ? 0 : maximum(ins; init=0)   # elementwise output ~ largest input
        return 4.0 * (sum(ins) + a) + LAUNCH_BYTES
    end
    return 4.0 * (sum(ins; init=0) + maximum(ins; init=0)) + LAUNCH_BYTES
end

# An alternative's identity within its e-class: its head and its children's heads,
# e.g. "xMul(xExpand|xReshape)" -- stable across layers, unlike e-class ids.
function signature(g, n::VecExpr)
    v_isexpr(n) || return "lit"
    ch = v_children(n)
    kids = [join(sort!(unique([string(v_isexpr(c) ? get_constant(g, v_head(c)) : "lit") for c in g[k].nodes])), "|")
            for k in ch[2:end]]
    params = isempty(ch) ? () : _lit(g, ch[1])
    ps = (params === () || params isa Tuple && length(repr(params)) > 24) ? "" : repr(params)
    return string(get_constant(g, v_head(n)), ps, "(", join(kids, ","), ")")
end

"""
    choice_groups(rw) -> Vector

E-classes that hold more than one extractable alternative, grouped by
(alternative signatures, shape) so that e.g. every layer's q-projection shares
one decision. Each group is `(key, class_ids, signatures)`.
"""
function choice_groups(rw::Rewriter)
    g = rw.g
    groups = Dict{Any, Vector{Id}}()
    for (k, ec) in g.classes
        sigs = sort!(unique([signature(g, n) for n in ec.nodes if v_isexpr(n)]))
        length(sigs) > 1 || continue
        key = (Tuple(sigs), ec.data === nothing ? nothing : ec.data.dims)
        push!(get!(groups, key, Id[]), ec.id)
    end
    return [(k, sort!(v), collect(k[1])) for (k, v) in sort!(collect(groups), by = x -> string(x[1]))]
end

# Greedy DAG extraction: a node's cost is the summed own-cost of the *set* of
# e-classes it depends on, so shared subgraphs count once. (Tree costs, which
# re-count every shared subgraph per use, grow exponentially along a residual
# network and overflow, making every alternative look equally bad.) Returns
# class id => (total cost, chosen e-node).
function _dag_extract(g, choices)
    ids = sort!([ec.id for (_, ec) in g.classes])
    index = Dict(id => i for (i, id) in enumerate(ids))
    own = zeros(Float64, length(ids))                 # own cost of each class's current pick
    best = Dict{Id, Tuple{Float64, VecExpr, BitSet}}()
    changed = true
    while changed
        changed = false
        for id in ids
            ec = g[id]
            forced = get(choices, id, nothing)
            for n in ec.nodes
                forced !== nothing && v_isexpr(n) && signature(g, n) != forced && continue
                deps = BitSet()
                ok = true
                if v_isexpr(n)
                    for child in v_children(n)
                        cb = get(best, find(g, child), nothing)
                        cb === nothing && (ok = false; break)
                        union!(deps, cb[3])
                    end
                end
                ok || continue
                i = index[id]
                i in deps && continue                         # would depend on itself (cycle)
                c = static_cost(g, n)
                total = c + sum((own[k] for k in deps); init=0.0)
                cur = get(best, id, nothing)
                if cur === nothing || total < cur[1] - 1e-9
                    push!(deps, i)
                    best[id] = (total, n, deps)
                    own[i] = c
                    changed = true
                end
            end
        end
    end
    return Dict(id => (b[1], b[2]) for (id, b) in best)
end

# Total cost of the DAG reachable from `root` under `pick` (each class counted
# once), or Inf if the picks form a cycle.
function _total_cost(g, pick, root)
    state = Dict{Id, Int8}()          # 1 = on stack, 2 = done
    total = 0.0
    function visit(c)
        c = find(g, c)
        st = get(state, c, Int8(0))
        st == 2 && return true
        st == 1 && return false       # back edge: cycle
        state[c] = 1
        n = pick[c]
        if v_isexpr(n)
            for ch in v_children(n)
                visit(ch) || return false
            end
        end
        total += static_cost(g, n)
        state[c] = 2
        return true
    end
    return visit(root) ? total : Inf
end

function _reachable(g, pick, root)
    seen = Set{Id}()
    stack = [find(g, root)]
    while !isempty(stack)
        c = pop!(stack)
        c in seen && continue
        push!(seen, c)
        n = pick[c]
        v_isexpr(n) && for ch in v_children(n); push!(stack, find(g, ch)); end
    end
    return seen
end

# Greedy extraction decides one class at a time and cannot see that a subgraph
# it avoids is needed anyway by another consumer. Refine: try switching any class
# to any alternative (respecting forced choices). A switch can pull classes into
# the DAG that are currently picked assuming they stand alone (e.g. the links of
# a scaled-matmul chain); re-pick those for their marginal cost given what the
# DAG already contains, then keep the move if the total DAG cost drops.
function _refine!(g, pick, choices, root, est)
    best = _total_cost(g, pick, root)
    S = _reachable(g, pick, root)
    marginal(m, S) = static_cost(g, m) +
        (v_isexpr(m) ? sum((find(g, ch) in S ? 0.0 : get(est, find(g, ch), Inf) for ch in v_children(m)); init=0.0) : 0.0)
    improved = true
    while improved
        improved = false
        for id in sort!(collect(S))
            haskey(choices, id) && continue
            ec = g[id]
            length(ec.nodes) > 1 || continue
            for n in ec.nodes
                (v_isexpr(n) && n !== pick[id]) || continue
                all(c -> haskey(pick, find(g, c)), v_children(n)) || continue
                saved = Dict{Id, VecExpr}(id => pick[id])
                pick[id] = n
                for _ in 1:3                      # complete newly reachable classes
                    S2 = _reachable(g, pick, root)
                    for k in setdiff(S2, S)
                        haskey(choices, k) && continue
                        alts = [m for m in g[k].nodes if v_isexpr(m) &&
                                all(c -> haskey(pick, find(g, c)), v_children(m))]
                        isempty(alts) && continue
                        m = argmin(m -> marginal(m, S2), alts)
                        if m !== pick[k]
                            haskey(saved, k) || (saved[k] = pick[k])
                            pick[k] = m
                        end
                    end
                end
                t = _total_cost(g, pick, root)
                if t < best - 1e-6
                    best, improved = t, true
                    S = _reachable(g, pick, root)
                else
                    for (k, m) in saved; pick[k] = m; end
                end
            end
        end
    end
    return best
end

"""
    extract_graph(rw; choices=Dict{Id,Symbol}()) -> (graph, idmap)

Rebuild a Graph choosing one e-node per e-class: the alternative whose
`signature` is given in `choices` (keyed by canonical e-class id) when present,
else the node with the lowest static tree cost. `idmap` maps original root and input node ids to new ids;
persistent tensors are carried over.
"""
function extract_graph(rw::Rewriter; choices::AbstractDict=Dict{Id,String}())
    g = rw.g
    CTX[] = rw.ctx
    greedy = _dag_extract(g, choices)
    pick = Dict(id => b[2] for (id, b) in greedy)
    _refine!(g, pick, choices, g.root, Dict(id => b[1] for (id, b) in greedy))
    ng = Graph()
    memo = Dict{Id, Int}()
    idmap = Dict{Int, Int}()
    visiting = Set{Id}()
    function build(cid::Id)
        cid = find(g, cid)
        haskey(memo, cid) && return memo[cid]
        cid in visiting && error("EGraphRewrite: extraction chose a cycle")
        push!(visiting, cid)
        n = pick[cid]
        h = get_constant(g, v_head(n))
        ch = v_children(n)
        params = _lit(g, ch[1])
        dims = collect(DimType, g[cid].data.dims)
        if h === :xIn
            old = params[1]
            t = Luminal.tensor(ng, dims; dtype=rw.graph.dtypes[old])
            haskey(rw.graph.tensors, (old, 1)) && (ng.tensors[(t.id, 1)] = rw.graph.tensors[(old, 1)])
            idmap[old] = t.id
            id = t.id
        else
            inputs = Tuple{Int, Int, ShapeTracker}[]
            for c in ch[2:end]
                cn = pick[find(g, c)]
                if get_constant(g, v_head(cn)) === :xView
                    src = build(v_children(cn)[2])
                    push!(inputs, (src, 0, deepcopy(rw.ctx.views[_lit(g, v_children(cn)[1])[1]])))
                else
                    push!(inputs, (build(c), 0, ShapeTracker(collect(DimType, g[find(g, c)].data.dims))))
                end
            end
            op = _rebuild_op(rw.ctx.optypes[h], params)
            id = add_op!(ng, op, inputs, ShapeTracker(dims)).id
        end
        delete!(visiting, cid)
        memo[cid] = id
        return id
    end
    for r in rw.roots
        idmap[r] = build(rw.class_of[r])
    end
    return ng, idmap
end

# ---------------------------------------------------------------------------
# Multi-node rewrite: merged projections
# ---------------------------------------------------------------------------

"""
    merge_projections!(rw; precision=false) -> Int

For weight matmuls that share an input (q/k/v, gate/up), add the alternative
`Slice_i(MatMul(concat(W_1..W_n), x))` to each member's e-class. The concat is
Pad + Add over weights, which compile() constant-folds. Metatheory patterns
are single-rooted, so this runs as a pass over the e-graph. Each merged group
is recorded as one joint decision (all members switch together). Returns the
number of groups merged.
"""
function merge_projections!(rw::Rewriter; precision=false)
    f = _precision_features(precision)
    g = rw.g
    CTX[] = rw.ctx
    groups = Dict{Id, Vector{Tuple{Id, Id}}}()      # input class => [(matmul class, weight class)]
    for (_, ec) in g.classes, n in ec.nodes
        (v_isexpr(n) && get_constant(g, v_head(n)) === :xMatMul) || continue
        ch = v_children(n)
        w, x = find(g, ch[2]), find(g, ch[3])
        wd = _dims(g, w)
        (_is_weight(g, g[w]) && wd !== nothing && length(wd) == 2) || continue
        push!(get!(groups, x, Tuple{Id, Id}[]), (find(g, ec.id), w))
    end
    merged = 0
    for (x, members) in sort!(collect(groups), by = first)
        members = unique(last, sort!(members))
        length(members) >= 2 || continue
        K = _dims(g, members[1][2])[2]
        all(_dims(g, w)[2] == K for (_, w) in members) || continue
        rows = [_dims(g, w)[1] for (_, w) in members]
        total = sum(rows)
        wcat, off = nothing, 0
        for ((_, w), r) in zip(members, rows)
            pad = Expr(:call, :xPad, ([(off, total - off - r), (0, 0)] |> _freeze,), g[w])
            wcat = wcat === nothing ? pad : Expr(:call, :xAdd, (), wcat, pad)
            off += r
        end
        wc = addexpr!(g, wcat)
        y = addexpr!(g, Expr(:call, :xMatMul, (), g[wc], g[x]))
        f16 = (:weights in f ? WEIGHT_VARIANTS : ())..., (:activations in f ? ACTIVATION_VARIANTS : ())...
        for variant in f16
            union!(g, y, addexpr!(g, Expr(:call, :xMatMulF16, variant, g[wc], g[x])))
        end
        if :int8 in f && K % 16 == 0
            for variant in INT8_VARIANTS
                union!(g, y, addexpr!(g, Expr(:call, :xMatMulQ8, variant, g[wc], g[x])))
            end
        end
        if :int4 in f && K % 32 == 0
            for variant in INT4_VARIANTS
                union!(g, y, addexpr!(g, Expr(:call, :xMatMulQ4, variant, g[wc], g[x])))
            end
        end
        rebuild!(g)
        rank = length(_dims(g, y))
        off = 0
        classes = Id[]
        for ((mc, _), r) in zip(members, rows)
            ranges = ntuple(i -> i == 1 ? (off, off + r) : (0, typemax(Int)), rank)
            sl = addexpr!(g, Expr(:call, :xSlice, (ranges,), g[find(g, y)]))
            union!(g, mc, sl)
            push!(classes, mc)
            off += r
        end
        rebuild!(g)
        push!(rw.ctx.joint, (string("merge ", Tuple(rows), " x ", _dims(g, x)), classes))
        merged += 1
    end
    return merged
end

# ---------------------------------------------------------------------------
# Decisions and measured search
# ---------------------------------------------------------------------------

"""
    decisions(rw) -> Vector{(label, options)}

What the search chooses between. Each option is a `choices` dict for
`extract_graph` (the first, empty, option leaves the pick to the cost model).
Per-class choice groups get one option per alternative signature; merged
projections found by `merge_projections!` get one joint option that moves every
member of every same-shaped group (e.g. all 22 layers' q/k/v) to the merged form.
"""
function decisions(rw::Rewriter)
    g = rw.g
    out = Tuple{String, Vector{Dict{Id, String}}}[]
    for (key, cls, sigs) in choice_groups(rw)
        opts = [Dict{Id, String}()]
        for sig in sigs
            push!(opts, Dict{Id, String}(find(g, c) => sig for c in cls))
        end
        push!(out, (string(key[2], ": ", join(sigs, " | ")), opts))
    end
    joint = Dict{String, Vector{Id}}()
    for (label, classes) in rw.ctx.joint
        append!(get!(joint, label, Id[]), classes)
    end
    for (label, classes) in sort!(collect(joint), by = first)
        forced = Dict{Id, String}()
        for c in classes
            sig = [signature(g, n) for n in g[find(g, c)].nodes if v_isexpr(n) && get_constant(g, v_head(n)) === :xSlice]
            isempty(sig) || (forced[find(g, c)] = first(sig))
        end
        push!(out, (label, [Dict{Id, String}(), forced]))
    end
    return out
end

_option_name(opt) = isempty(opt) ? "static" : first(values(opt))

"""
    measured_search(rw, make_runner; rounds=5, steps=10, margin=0.01, log=true,
                    cache_dir=nothing, cache_tag="")

Coordinate descent over `decisions(rw)`. `make_runner(choices)` must build the
candidate graph, check it against a reference, and return a zero-argument
function that runs one synchronized step -- or `nothing` to reject it.

A candidate replaces the incumbent only if, timed in `rounds` interleaved
rounds of `steps` steps each, its median per-step time is lower by more than
`margin` and it wins at least 80% of the rounds -- twice, in two independent
contests; single timings on a GPU vary by a few percent. Returns `(choices, per-step time of the incumbent)`.

With `cache_dir`, the winning option per decision is stored in a file keyed by
the decision set and `cache_tag` (e.g. the device); a later search of the same
graph applies those choices after one verification instead of timing again.

`extra(choices)`, if given, returns further decisions to try after those of the
e-graph, given the choices made so far (compile_searched's lowering sites: they
depend on the extracted graph). Their options may use keys other than e-class ids.
`release(run)` is called on every runner the search discards -- a rejected
candidate, a replaced incumbent -- so its device memory is freed at once rather
than when the garbage collector gets to it.
"""
function measured_search(rw::Rewriter, make_runner; rounds::Int=5, steps::Int=10,
                         round_budget::Float64=0.3, margin::Float64=0.01, log::Bool=true,
                         cache_dir::Union{Nothing,String}=nothing, cache_tag::AbstractString="",
                         extra = nothing, release = (run -> nothing))
    # Steps per timing round: `steps`, reduced so a round takes about `round_budget`
    # seconds (a prefill graph can take ~0.5 s per run).
    nsteps = Ref(steps)
    function timed(run)
        t = time()
        for _ in 1:nsteps[]; run(); end
        return (time() - t) / nsteps[]
    end
    function median_(v)
        s = sort(v); n = length(s)
        return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
    end
    ds = decisions(rw)
    cache_file = cache_dir === nothing ? nothing :
        joinpath(cache_dir, string(hash((cache_tag, [(l, map(_option_name, o)) for (l, o) in ds])), base = 16) * ".txt")
    if cache_file !== nothing && isfile(cache_file)
        kept = Dict{String, String}()
        for line in eachline(cache_file)
            parts = split(line, '\t')
            length(parts) == 2 && (kept[parts[2]] = parts[1])
        end
        cached = Dict{Any, String}()
        for (label, opts) in ds, o in opts
            get(kept, label, nothing) == _option_name(o) && merge!(cached, o)
        end
        if extra !== nothing
            for (label, opts) in extra(cached), o in opts
                get(kept, label, nothing) == _option_name(o) && merge!(cached, o)
            end
        end
        run = make_runner(cached)
        if run !== nothing
            log && println("  using cached search result ", cache_file)
            run(); t1 = (t = time(); run(); time() - t)
            nsteps[] = clamp(round(Int, round_budget / max(t1, 1e-6)), 1, steps)
            t = median_([(run(); timed(run)) for _ in 1:rounds])
            release(run)
            return cached, t
        end
        log && println("  cached choices failed verification; searching again")
    end

    current = Dict{Any, String}()
    kept_labels = Tuple{String, String}[]
    incumbent = make_runner(current)
    incumbent === nothing && error("measured_search: the static extraction failed verification " *
        "(with precision=:activations, consider a looser search_tolerance)")
    for _ in 1:3; incumbent(); end
    t1 = (t = time(); incumbent(); time() - t)
    nsteps[] = clamp(round(Int, round_budget / max(t1, 1e-6)), 1, steps)
    function try_decision(label, opts)
        for opt in opts[2:end]
            trial = merge(current, opt)
            cand = make_runner(trial)
            if cand === nothing
                log && println("  rejected (verification): ", label)
                continue
            end
            for _ in 1:3; cand(); end
            function contest()
                ti, tc = Float64[], Float64[]
                for _ in 1:rounds
                    push!(ti, timed(incumbent)); push!(tc, timed(cand))
                end
                wins = count(tc .< ti)
                mi, mc = median_(ti), median_(tc)
                return mc < mi * (1 - margin) && wins >= ceil(Int, 0.8 * rounds), mi, mc, wins
            end
            keep, mi, mc, wins = contest()
            # A pass must repeat in a second, independent contest: with dozens of
            # decisions and a few percent of timing noise, single contests admit
            # false positives (seen: a 32-element fusion "saving" 2%).
            keep && (keep = first(contest()))
            log && println(string("  ", keep ? "KEEP  " : "      ", rpad(label, 48)[1:min(end, 48)],
                                  " -> ", rpad(_option_name(opt), 34), round(1e3mi, digits=2), " -> ",
                                  round(1e3mc, digits=2), " ms  (", wins, "/", rounds, " rounds)"))
            if keep
                release(incumbent)
                current, incumbent = trial, cand
                filter!(kl -> kl[2] != label, kept_labels)
                push!(kept_labels, (_option_name(opt), label))
            else
                release(cand)
            end
            GC.gc()
        end
    end
    for (label, opts) in ds
        try_decision(label, opts)
    end
    if extra !== nothing
        more = extra(current)
        log && !isempty(more) && println("  lowering choices: ", length(more))
        for (label, opts) in more
            try_decision(label, opts)
        end
    end
    if cache_file !== nothing
        mkpath(dirname(cache_file))
        open(cache_file, "w") do io
            for (opt, label) in kept_labels; println(io, opt, '\t', label); end
        end
    end
    t = median_([timed(incumbent) for _ in 1:rounds])
    release(incumbent)
    return current, t
end

# ---------------------------------------------------------------------------
# compile(...; search=...) entry point
# ---------------------------------------------------------------------------

"""
    RewrittenGraph

A compiled rewritten graph addressed by the *original* graph's node ids:
calling it takes inputs keyed by original ids, and indexing its results with
an original retained id returns that output.
"""
struct RewrittenGraph
    cg::Any                    # Luminal.CompiledGraph of the rewritten graph
    idmap::Dict{Int, Int}      # original node id => rewritten node id
end

struct RemappedResults
    results::Vector{Any}
    idmap::Dict{Int, Int}
end
Base.getindex(r::RemappedResults, id::Int) = r.results[r.idmap[id]]

function (r::RewrittenGraph)(inputs::Dict; kwargs...)
    mapped = Dict{Int, Any}(r.idmap[k] => v for (k, v) in inputs if haskey(r.idmap, k))
    return RemappedResults(r.cg(mapped; kwargs...), r.idmap)
end

_input_ids(graph) = [nid for (nid, n) in enumerate(graph.nodes)
                     if n.op isa Luminal.Function && n.op.name == "InputTensor" && !haskey(graph.tensors, (nid, 1))]

# A measured-search candidate: runs one synchronized step of its compiled graph.
struct _Runner
    r::RewrittenGraph
    inputs::Dict{Int, Any}
    device::Any
end
(x::_Runner)() = (x.r(x.inputs; device=x.device); Luminal.synchronize_device(x.device); nothing)

# Device memory kept free when deciding whether a candidate fits (the estimate is
# rough, and the rest of the system needs memory too).
const SEARCH_MEMORY_HEADROOM = 4 * 2^30

function compile_searched(graph::Graph; search::Symbol, precision, search_inputs, search_cache,
                          search_tolerance::Real=1e-3,
                          device, retain::Vector{Int}, kwargs...)
    search in (:static, :measured) || error("search must be :none, :static or :measured")
    isempty(retain) && error("compile(...; search=$search) needs `retain`: the node ids to keep")
    rw = to_egraph(graph, retain)
    saturate_graph!(rw; precision=precision)
    merge_projections!(rw; precision=precision)
    # A candidate's choices: e-class id => alternative signature, and lowering-site
    # key (a String, see compile) => "off" to disable that fusion or view.
    user_lowering = get(kwargs, :lowering, Dict{String,Bool}())
    compile_kwargs = Base.structdiff(values(kwargs), NamedTuple{(:lowering,)})
    # With `budget`, a candidate whose estimated new allocations would not fit in the
    # device's free memory (less SEARCH_MEMORY_HEADROOM) is not compiled: `nothing`.
    function build(choices; budget::Bool=false)
        ng, m = extract_graph(rw; choices=Dict{Any,String}(k => v for (k, v) in choices if !(k isa String)))
        if budget
            GC.gc()
            avail = Luminal.available_memory(device)
            if avail >= 0
                need = Luminal.estimate_compile_bytes(ng; retain=[m[r] for r in retain if haskey(m, r)],
                                                      weight_dtype=get(kwargs, :weight_dtype, Float32),
                                                      fold=get(kwargs, :fold, true))
                if need > avail * 2^20 - SEARCH_MEMORY_HEADROOM
                    println("  skipped (memory): needs ~", round(need / 2^30, digits=1), " GB, ",
                            round(avail / 2^10, digits=1), " GB free")
                    return nothing
                end
            end
        end
        lowering = Dict{String,Bool}(user_lowering)
        for (k, v) in choices
            k isa String && (lowering[k] = v != "off")
        end
        cg = Luminal.compile(ng; device=device, retain=[m[r] for r in retain if haskey(m, r)],
                             lowering=lowering, compile_kwargs...)
        return RewrittenGraph(cg, m)
    end
    search === :static && return build(Dict{Any, String}())

    # Default inputs: zeros, placed on the device once (host inputs would be copied
    # in on every timed run).
    inputs = search_inputs !== nothing ? search_inputs : Dict{Int, Any}(
        id => Luminal.to_device(zeros(Float32, (Int(Luminal.eval_dim(d)) for d in realized_dims(graph.shapes[id]))...), device)
        for id in _input_ids(graph))
    outs = [r for r in retain if !(r in _input_ids(graph))]
    refcg = Luminal.compile(graph; device=device, retain=retain, kwargs...)
    ref = refcg(inputs; device=device)
    reference = Dict(o => Array{Float32}(ref[o]) for o in outs)
    Luminal.release!(refcg); ref = refcg = nothing; GC.gc()
    relerr(got, want) = maximum(abs.(got .- want); init=0f0) / max(maximum(abs, want; init=0f0), 1f-6)
    # Lossy weight precision (int8 / int4): the static extraction sets the precision.
    # It is checked once against Float32 with a gross tolerance (catching broken
    # output, not quantization quality), and then becomes the reference every
    # candidate must match within `search_tolerance`: kernel variants, lowering
    # choices and exact rewrites differ from it only in rounding.
    feats = _precision_features(precision)
    gross = :int4 in feats ? INT4_SEARCH_TOLERANCE : :int8 in feats ? INT8_SEARCH_TOLERANCE : nothing
    if gross !== nothing
        r0 = build(Dict{Any,String}(); budget=true)
        r0 === nothing && error("compile(...; search): the static extraction does not fit in device memory")
        res0 = r0(inputs; device=device)
        for o in outs
            got = Array{Float32}(res0[o])
            all(isfinite, got) || error("measured_search: the static extraction produced non-finite output")
            e = relerr(got, reference[o])
            e < gross || error("measured_search: the static extraction differs from Float32 by $e (limit $gross)")
        end
        reference = Dict(o => Array{Float32}(res0[o]) for o in outs)
        Luminal.release!(r0.cg); res0 = r0 = nothing
    end
    function make_runner(choices)
        r = build(choices; budget=true)
        r === nothing && return nothing
        res = r(inputs; device=device)
        for o in outs
            got = Array{Float32}(res[o])
            if !(relerr(got, reference[o]) < search_tolerance)
                Luminal.release!(r.cg)
                return nothing
            end
        end
        return _Runner(r, inputs, device)
    end
    # After the e-graph decisions: each fusion / view the compiler applies to the
    # chosen graph, as a decision to turn it off (sites shared across layers).
    function lowering_decisions(choices)
        r = build(choices; budget=true)
        r === nothing && return Tuple{String, Vector{Dict{Any,String}}}[]
        sites = r.cg.cache[:lowering_sites]
        Luminal.release!(r.cg)
        return [("lowering: " * k, [Dict{Any,String}(), Dict{Any,String}(k => "off")])
                for k in sort!(collect(sites)) if get(user_lowering, k, true)]
    end
    choices, _ = measured_search(rw, make_runner; cache_dir=search_cache, extra=lowering_decisions,
                                 release=run -> run isa _Runner && Luminal.release!(run.r.cg),
                                 cache_tag=string(nameof(typeof(device)), precision, search_tolerance, kwargs))
    return build(choices)
end

end # module EGraphRewrite
