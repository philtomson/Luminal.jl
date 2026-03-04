# Defines the ShapeTracker, which manages tensor shapes through operations.

using SymbolicUtils
using SymbolicUtils: Sym, BasicSymbolic, substitute

export ShapeTracker, permute!, reshape, slice!, pad!, add_dim!, remove_dim!, expand_dim!, contiguous, is_reshaped, realized_dims

mutable struct ShapeTracker
    dims::Vector{Luminal.DimType}
    indexes::Vector{Int}
    fake::Vector{Bool}
    mask::Vector{Tuple{Luminal.DimType, Luminal.DimType}}
    padding::Vector{Tuple{Luminal.DimType, Luminal.DimType}}

    function ShapeTracker(dims::AbstractVector)
        # Convert input to Vector{Luminal.DimType}
        dims_converted = Luminal.DimType[d for d in dims]
        len = length(dims_converted)
        new(dims_converted,
            collect(1:len),
            fill(false, len),
            fill((0, typemax(Int)), len),
            fill((0, 0), len))
    end
end

"""
    add_dim!(st::ShapeTracker, axis::Int, dim_size::Luminal.DimType)

Add a new dimension at the specified axis.
"""
function add_dim!(st::ShapeTracker, axis::Int, dim_size::Luminal.DimType)
    push!(st.dims, dim_size)
    push!(st.fake, false)
    push!(st.mask, (0, typemax(Int)))
    push!(st.padding, (0, 0))
    insert!(st.indexes, axis, length(st.dims))
end

"""
    expand_dim!(st::ShapeTracker, axis::Int, dim_size::Luminal.DimType)

Add a new fake (broadcasted) dimension at the specified axis.
"""
function expand_dim!(st::ShapeTracker, axis::Int, dim_size::Luminal.DimType)
    add_dim!(st, axis, dim_size)
    st.fake[st.indexes[axis]] = true
end

"""
    remove_dim!(st::ShapeTracker, axis::Int)

Remove the dimension at the specified axis and return its size.
"""
function remove_dim!(st::ShapeTracker, axis::Int)
    index = splice!(st.indexes, axis)
    # Note: We don't remove from dims/fake/mask/padding to keep indexes valid
    # In Rust they do remove and update indexes. We can do that too if needed.
    # But for now, let's just mark it as "removed" by not having it in indexes.
    return st.dims[index]
end

"""
    permute!(st::ShapeTracker, axes::Vector{Int})

Permute the dimensions of the ShapeTracker in-place.
"""
function permute!(st::ShapeTracker, axes::Vector{Int})
    @assert length(axes) == length(st.indexes) "Permutation axes must match number of dimensions"
    st.indexes .= st.indexes[axes]
end

"""
    reshape(st::ShapeTracker, new_dims::AbstractVector)

Return a new, contiguous ShapeTracker for the reshaped dimensions.
"""
function reshape(st::ShapeTracker, new_dims::AbstractVector)
    return ShapeTracker(new_dims)
end

"""
    slice!(st::ShapeTracker, new_mask::AbstractVector)

Apply a slice to the ShapeTracker by updating its mask.
"""
function slice!(st::ShapeTracker, new_mask::AbstractVector)
    for (i, (b, t)) in enumerate(new_mask)
        idx = st.indexes[i]
        curr_b, curr_t = st.mask[idx]
        st.mask[idx] = (max(curr_b, b), min(curr_t, t))
    end
end

"""
    pad!(st::ShapeTracker, new_padding::AbstractVector)

Apply padding to the ShapeTracker by updating its padding values.
"""
function pad!(st::ShapeTracker, new_padding::AbstractVector)
    for (i, (s, e)) in enumerate(new_padding)
        idx = st.indexes[i]
        st.padding[idx] = (st.padding[idx][1] + s, st.padding[idx][2] + e)
    end
end

"""
    expand(st::ShapeTracker, dim::Int, new_size::Int)

Return a new ShapeTracker with a new dimension of `size` inserted at `dim`.
"""
function expand(st::ShapeTracker, dim::Int, new_size::Luminal.DimType)
    new_st = deepcopy(st)
    expand_dim!(new_st, dim, new_size)
    return new_st
end

"""
    contiguous(st::ShapeTracker)

Create a new contiguous ShapeTracker from the realized dimensions.
"""
function contiguous(st::ShapeTracker)
    return ShapeTracker(realized_dims(st))
end

"""
    is_reshaped(st::ShapeTracker)

Check if the ShapeTracker has been modified (permuted, sliced, padded, or fake).
"""
function is_reshaped(st::ShapeTracker)
    # Contiguous if indexes are 1:N and no fake, no padding, no mask besides default
    if st.indexes != collect(1:length(st.indexes))
        return true
    end
    for i in st.indexes
        if st.fake[i] || st.padding[i] != (0, 0) || st.mask[i] != (0, typemax(Int))
            return true
        end
    end
    return false
end

"""
    realized_dims(st::ShapeTracker)

Return the actual dimensions of the tensor, taking into account padding and masking.
"""
function realized_dims(st::ShapeTracker)
    return Luminal.DimType[pad_mask_dim(st.dims[i], st.padding[i], st.mask[i]) for i in st.indexes]
end

function evaluate_shapes(st::ShapeTracker, sym_vals::Dict{Symbol, Int})
    # Create a new ShapeTracker with concrete Int dimensions 
    new_dims = Int[]
    for i in 1:length(st.dims)
        push!(new_dims, Luminal.eval_dim(st.dims[i], sym_vals))
    end
    
    new_st = ShapeTracker(new_dims)
    new_st.indexes = copy(st.indexes)
    new_st.fake = copy(st.fake)
    
    # Evaluate mask and padding
    new_st.mask = [(Luminal.eval_dim(first(m), sym_vals), Luminal.eval_dim(last(m), sym_vals)) for m in st.mask]
    new_st.padding = [(Luminal.eval_dim(first(p), sym_vals), Luminal.eval_dim(last(p), sym_vals)) for p in st.padding]
    
    return new_st
end

function _sym_add(a, b)
    if a isa Int && b isa Int
        return a + b
    elseif a isa Int && a == 0 return b
    elseif b isa Int && b == 0 return a
    else
        return SymbolicUtils.Term(+, Any[a, b])
    end
end

function _sym_sub(a, b)
    if a isa Int && b isa Int
        return a - b
    elseif b isa Int && b == 0 return a
    else
        return SymbolicUtils.Term(-, Any[a, b])
    end
end

function _sym_min(a, b)
    if a isa Int && b isa Int
        return min(a, b)
    elseif b isa Int && b == typemax(Int) return a
    elseif a isa Int && a == typemax(Int) return b
    else
        # Simplification: we don't strictly support min for symbolic yet, assume unbound limit
        return a
    end
end

function pad_mask_dim(dim, padding, mask)
    b = _sym_add(_sym_add(padding[1], padding[2]), dim)
    return _sym_sub(_sym_min(b, mask[2]), mask[1])
end