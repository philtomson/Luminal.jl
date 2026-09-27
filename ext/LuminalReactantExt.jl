# Luminal graphs through Reactant.jl: a functional evaluator of the primitive graph
# that Reactant traces into StableHLO, which XLA compiles.
#
# Luminal's own `execute` writes into preallocated buffers, which tracing cannot
# follow; here every node's value is a fresh array expression. Views are realized
# with `realize_view` as in the interpreter (permuted and sliced views made
# concrete). Ops whose functional form in Execution.jl does not trace get
# traced-array methods below.
module LuminalReactantExt

using Luminal
using Reactant
using Reactant: TracedRArray, TracedRNumber

const TA = TracedRArray

# --- functional forms of ops for traced arrays ---------------------------------

# exp2/log2 via exp/log: not every Reactant version defines them for traced numbers
Luminal.execute_op(::Luminal.Exp2, x::TA{T}) where T = exp.(x .* T(log(2)))
Luminal.execute_op(::Luminal.Log2, x::TA{T}) where T = log.(x) .* T(1 / log(2))

Luminal.execute_op(::Luminal.ReLU, x::TA{T}) where T = max.(x, zero(T))

Luminal.execute_op(op::Luminal.RMSNormOp, x::TA, w) =
    x ./ sqrt.(sum(abs2, x; dims=1) ./ size(x, 1) .+ op.epsilon) .* w

function Luminal.execute_op(::Luminal.SoftmaxOp, x::TA)
    e = exp.(x .- maximum(x; dims=1))
    return e ./ sum(e; dims=1)
end

# x (D, S, H, B); tables c, s (D/2, S) or (D/2, S, B)
function Luminal.execute_op(::Luminal.RotaryEmbed, x::TA, c, s)
    x4 = Base.reshape(x, size(x, 1), size(x, 2), size(x, 3), :)
    h = size(x4, 1) ÷ 2
    c4 = Base.reshape(c, h, size(x4, 2), 1, :)
    s4 = Base.reshape(s, h, size(x4, 2), 1, :)
    x1 = x4[1:h, :, :, :]; x2 = x4[h+1:end, :, :, :]
    return Base.reshape(cat(x1 .* c4 .- x2 .* s4, x2 .* c4 .+ x1 .* s4; dims=1), size(x))
end

# Left-aligned: (N, K) x (K, rest...), or batched (M, K, batch...) x (K, N, batch...)
# with trailing batch dims, as one dot_general
function Luminal.execute_op(::Luminal.MatMul, a::TA, b::TA)
    ndims(a) == 2 && return Base.reshape(a * Base.reshape(b, size(b, 1), :), size(a, 1), size(b)[2:end]...)
    nb = ndims(a) - 2
    r = Reactant.Ops.dot_general(a, b; contracting_dimensions=([2], [1]),
                                 batching_dimensions=(collect(3:ndims(a)), collect(3:ndims(b))))
    return permutedims(r, (nb + 1, nb + 2, 1:nb...))     # (batch..., M, N) -> (M, N, batch...)
end

function Luminal.execute_op(op::Luminal.Pad, x::TA{T}) where T
    return Reactant.Ops.pad(x, Reactant.Ops.constant(zero(T));
                            low=[p[1] for p in op.padding], high=[p[2] for p in op.padding])
end

# Coordinate gather / scatter: 1-based linear indices into the flattened data,
# and which coordinates are in range (the rest read 0 / are dropped, as on CPU/GPU)
_mat(x) = Reactant.TracedUtils.materialize_traced_array(x)

function _lin_index(coords, dims)
    ok = nothing; lin = nothing; stride = 1
    for (c, d) in zip(coords, dims)
        ci = Reactant.Ops.convert(TracedRArray{Int64, ndims(c)}, _mat(c))    # truncates, as the kernels
        okk = (ci .>= 0) .& (ci .< d)
        ok = ok === nothing ? okk : ok .& okk
        lin = lin === nothing ? ci .* stride : lin .+ ci .* stride
        stride *= d
    end
    return ok, lin .+ 1
end

function Luminal.execute_op(::Luminal.GatherND, data::TA{T}, coords...) where T
    ok, lin = _lin_index(coords, size(data))
    safe = ifelse.(ok, lin, 1)
    v = Reactant.Ops.gather_getindex(_mat(vec(data)), _mat(Base.reshape(safe, :, 1)))
    return ifelse.(ok, Base.reshape(v, size(safe)), zero(T))
end

function Luminal.execute_op(op::Luminal.ScatterND, init::TA{T}, src, coords...) where T
    ok, lin = _lin_index(coords, size(init))
    L = length(init)
    idx = Base.broadcast((i, _) -> i, ifelse.(ok, lin, L + 1), src)   # out of range: a spare slot
    ext = cat(_mat(vec(init)), Reactant.Ops.constant(zeros(T, 1)); dims=1)
    f = op.mode === :add ? ((a, b) -> a + b) : ((_, b) -> b)
    r = Reactant.Ops.scatter(f, [ext], _mat(Base.reshape(idx, :, 1)), [_mat(vec(src))];
                             update_window_dims=Int64[], inserted_window_dims=Int64[1],
                             input_batching_dims=Int64[], scatter_indices_batching_dims=Int64[],
                             scatter_dims_to_operand_dims=Int64[1], index_vector_dim=Int64(2))[1]
    return Base.reshape(_mat(r[1:L]), size(init))
end

# W[ids, :] as onehot(ids) * W: token ids arrive as Float32 values
function _gather(W, ids)
    V = size(W, 1)
    onehot = Float32.(Base.reshape(ids, :, 1) .== Base.reshape(Float32.(0:V-1), 1, :))
    return Base.reshape(onehot * W, size(ids)..., size(W, 2))
end

# --- evaluator -------------------------------------------------------------------

_perm(::PermutedDimsArray{T,N,perm}) where {T,N,perm} = perm

function _realize(v, st)
    v = Luminal.realize_view(v, st)
    v isa PermutedDimsArray && return permutedims(parent(v), _perm(v))
    v isa SubArray && return copy(v)
    v isa AbstractArray{<:TracedRNumber} && !(v isa TA) &&
        return Reactant.TracedUtils.materialize_traced_array(v)
    return v
end

_const(d::AbstractArray) = Array{Float32}(d)
_const(d) = Array{Float32}(Luminal.dequantize(d))       # a HalfWeight / QuantWeight / Q4Weight

_id(t::Luminal.GraphTensor) = t.id
_id(t::Integer) = Int(t)

# Nodes the outputs depend on, stopping at bound inputs
function _needed(g, outs, bound)
    need = falses(length(g.nodes)); stack = collect(outs)
    while !isempty(stack)
        id = pop!(stack)
        need[id] && continue
        need[id] = true
        id in bound && continue
        for (i, _, _) in g.nodes[id].inputs
            push!(stack, i)
        end
    end
    return need
end

function _evaluate(g::Luminal.Graph, outs::Vector{Int}, ins::Vector{Int}, xs)
    res = Dict{Int,Any}(zip(ins, xs))
    # while tracing, arrays not derived from inputs become traced constants, so no
    # op mixes plain and traced arrays
    tracing = any(x -> x isa TA, xs)
    lift(a) = tracing && a isa Array ? Reactant.Ops.constant(a) : a
    need = _needed(g, outs, Set(ins))
    for ((id, k), d) in g.tensors
        k == 1 && need[id] && !haskey(res, id) && (res[id] = lift(_const(d)))
    end
    for (id, node) in enumerate(g.nodes)
        (need[id] && !haskey(res, id)) || continue
        op = node.op
        if op isa Luminal.Function && op.name == "InputTensor"
            error("input node $id is neither bound nor loaded into the graph")
        end
        vals = [_realize(res[i], st) for (i, _, st) in node.inputs]
        res[id] = if op isa Luminal.Function && op.name == "ARange"
            n = Luminal.eval_dim(Luminal.realized_dims(g.shapes[id])[1])
            lift(Float32.(collect(0:n-1)))
        elseif op isa Luminal.Iota
            lift(Luminal._iota_values(op, Luminal.eval_dim.(Luminal.realized_dims(g.shapes[id]))))
        elseif op isa Luminal.Function && op.name == "Gather"
            _gather(vals...)
        elseif op isa Luminal.Constant
            op.value isa Number ? Float32(op.value) : lift(_const(op.value))
        else
            Luminal.execute_op(op, vals...)
        end
    end
    return [res[i] for i in outs]
end

function Luminal.reactant_function(g::Luminal.Graph, outputs, inputs)
    single = !(outputs isa AbstractVector)
    outs = single ? [_id(outputs)] : _id.(outputs)
    ins = _id.(collect(inputs))
    return function (xs...)
        length(xs) == length(ins) || throw(ArgumentError("expected $(length(ins)) inputs, got $(length(xs))"))
        r = _evaluate(g, outs, ins, xs)
        return single ? r[1] : Tuple(r)
    end
end

_rarray(x) = x isa Reactant.AbstractConcreteArray ? x : Reactant.to_rarray(x)

function Luminal.reactant_compile(g::Luminal.Graph, outputs, inputs, args...)
    f = Luminal.reactant_function(g, outputs, inputs)
    rs = map(_rarray, args)
    return Reactant.@compile f(rs...)
end

function Luminal.to_stablehlo(g::Luminal.Graph, outputs, inputs, args...)
    f = Luminal.reactant_function(g, outputs, inputs)
    rs = map(_rarray, args)
    return string(Reactant.@code_hlo f(rs...))
end

end
