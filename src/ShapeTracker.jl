#!/usr/bin/env julia
# Defines the ShapeTracker, which manages tensor shapes through operations.

using SymbolicUtils
using SymbolicUtils: Sym, BasicSymbolic, substitute
import Base: max, min

export ShapeTracker, permute!, reshape, slice!, pad!

struct ShapeTracker
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
    permute!(st::ShapeTracker, axes::Vector{Int})

Permute the dimensions of the ShapeTracker in-place.
"""
function permute!(st::ShapeTracker, axes::Vector{Int})
    @assert length(axes) == length(st.dims) "Permutation axes must match number of dimensions"
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
        original_index = st.indexes[i]
        current_b, current_t = st.mask[original_index]
        st.mask[original_index] = (max(current_b, b), min(current_t, t))
    end
end

"""
    pad!(st::ShapeTracker, new_padding::AbstractVector)

Apply padding to the ShapeTracker by updating its padding values.
"""
function pad!(st::ShapeTracker, new_padding::AbstractVector)
    for (i, (s, e)) in enumerate(new_padding)
        original_index = st.indexes[i]
        st.padding[original_index] = (st.padding[original_index][1] + s, st.padding[original_index][2] + e)
    end
end