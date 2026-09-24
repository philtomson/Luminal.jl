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

export Rewriter, to_egraph, saturate_graph!, choice_groups, extract_graph, static_cost

# ---------------------------------------------------------------------------
# Shape analysis
# ---------------------------------------------------------------------------

struct Shp
    dims::Tuple
end

# Views and op types are side tables (e-graph literals must hash by value).
mutable struct BridgeCtx
    views::Vector{ShapeTracker}
    view_index::Dict{Any, Int}
    optypes::Dict{Symbol, DataType}
    weights::Set{Int}          # node ids of persistent tensors (graph.tensors)
end
const CTX = Ref{BridgeCtx}()   # read by `make`, which has no other way to reach it

_lit(g, id) = get_constant(g, v_head(g[id].nodes[1]))
_dims(g, id) = (d = g[id].data; d === nothing ? nothing : d.dims)
_isone(x) = x isa Integer && x == 1

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
    h = get_constant(g, v_head(n))
    ch = v_children(n)
    if h === :xView
        return Shp(Tuple(realized_dims(CTX[].views[_lit(g, ch[2])[1]])))
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
    elseif h === :xMatMul || h === :xMatMulF16
        a, b = _dims(g, ch[2]), _dims(g, ch[3])
        (a === nothing || b === nothing) && return nothing
        batch = _broadcast(a[3:end], b[3:end])
        return batch === nothing ? nothing : Shp((a[1], b[2], batch...))
    end
    return nothing
end

function EGraphs.join(a::Shp, b::Shp)
    isequal(a.dims, b.dims) || error("EGraphRewrite: merged e-classes disagree on shape: $(a.dims) vs $(b.dims)")
    return a
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
                    Set{Int}(first(k) for k in keys(graph.tensors)))
    CTX[] = ctx
    for T in (Luminal.Reshape, Luminal.Mul, Luminal.Add, Luminal.MatMul, Luminal.MatMulF16, Luminal.Expand)
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
            # too; append it so e.g. ARange(32) and ARange(1) stay distinct.
            params = isempty(kids) ? (_params(op)..., dims) : _params(op)
            ex = Expr(:call, h, params, kids...)
        end
        id = addexpr!(g, ex)
        ec = g[id]
        if ec.data === nothing
            ec.data = Shp(dims)
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
# with Float16 weights. Opt-in.
const PRECISION_RULES = @theory q w x begin
    xMatMul(q::Tuple, w, x) => (_is_weight(_egraph, w) && w.data !== nothing && length(w.data.dims) == 2) ?
                                :(xMatMulF16($q, $w, $x)) : nothing
end

"""
    saturate_graph!(rw; precision=false, iterations=8)

Apply the rule sets until saturation (or the iteration/size limits).
"""
function saturate_graph!(rw::Rewriter; precision::Bool=false, iterations::Int=8)
    CTX[] = rw.ctx
    theory = vcat(CANONICAL_RULES, ALGEBRAIC_RULES, precision ? PRECISION_RULES : RewriteRule[])
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
    ins = [_numel(_dims(g, c)) for c in ch[2:end]]
    if h === :xMatMulF16
        return 2.0 * ins[1] + 4.0 * sum(ins[2:end]) + LAUNCH_BYTES
    elseif h === :xMatMul || h === :xMul || h === :xAdd
        a = h === :xMatMul ? 0 : maximum(ins; init=0)   # elementwise output ~ largest input
        return 4.0 * (sum(ins) + a) + LAUNCH_BYTES
    end
    return 4.0 * (sum(ins; init=0) + maximum(ins; init=0)) + LAUNCH_BYTES
end

# An alternative's identity within its e-class: its head and its children's heads,
# e.g. "xMul(xExpand|xReshape)" -- stable across layers, unlike e-class ids.
function signature(g, n::VecExpr)
    v_isexpr(n) || return "lit"
    kids = [join(sort!(unique([string(v_isexpr(c) ? get_constant(g, v_head(c)) : "lit") for c in g[k].nodes])), "|")
            for k in v_children(n)[2:end]]
    return string(get_constant(g, v_head(n)), "(", join(kids, ","), ")")
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
    cost = _dag_extract(g, choices)
    ng = Graph()
    memo = Dict{Id, Int}()
    idmap = Dict{Int, Int}()
    visiting = Set{Id}()
    function build(cid::Id)
        cid = find(g, cid)
        haskey(memo, cid) && return memo[cid]
        cid in visiting && error("EGraphRewrite: extraction chose a cycle")
        push!(visiting, cid)
        n = cost[cid][2]
        h = get_constant(g, v_head(n))
        ch = v_children(n)
        params = _lit(g, ch[1])
        dims = collect(DimType, g[cid].data.dims)
        if h === :xIn
            old = params[1]
            t = Luminal.tensor(ng, dims)
            haskey(rw.graph.tensors, (old, 1)) && (ng.tensors[(t.id, 1)] = rw.graph.tensors[(old, 1)])
            idmap[old] = t.id
            id = t.id
        else
            inputs = Tuple{Int, Int, ShapeTracker}[]
            for c in ch[2:end]
                cn = cost[find(g, c)][2]
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

end # module EGraphRewrite
