#!/usr/bin/env julia
# Defines the ShapeTracker, which manages tensor shapes through operations.

import Base: max, min # These will eventually need to be handled by Symbolics.jl's symbolic operations

export ShapeTracker, permute!, reshape

struct ShapeTracker
    dims::Vector{Expression}
    indexes::Vector{Int}
    fake::Vector{Bool}
    mask::Vector{Tuple{Expression, Expression}}
    padding::Vector{Tuple{Expression, Expression}}

    function ShapeTracker(dims::Vector{Expression})
        len = length(dims)
        new(dims, 
            collect(1:len), 
            fill(false, len),
            fill((Luminal.Symbolic.Expression([Luminal.Symbolic.Num(0)]), Luminal.Symbolic.Expression([Luminal.Symbolic.Num(typemax(Int))])), len),
            fill((Luminal.Symbolic.Expression([Luminal.Symbolic.Num(0)]), Luminal.Symbolic.Expression([Luminal.Symbolic.Num(0)])), len))
    end
end

"""
    permute!(st::ShapeTracker, axes::Vector{Int})

Permute the dimensions of the ShapeTracker in-place.
"""
function permute!(st::ShapeTracker, axes::Vector{Int})
    # Note: Julia axes are 1-indexed.
    @assert length(axes) == length(st.dims) "Permutation axes must match number of dimensions"
    st.indexes .= st.indexes[axes]
end

"""
    reshape(st::ShapeTracker, new_dims::Vector{Expression})

Return a new, contiguous ShapeTracker for the reshaped dimensions.
"""
function reshape(st::ShapeTracker, new_dims::Vector{Expression})
    # A reshape operation creates a new contiguous view.
    # The complexity is handled by the index expression of the *previous* shape tracker.
    return ShapeTracker(new_dims)
end

"""
    slice(st::ShapeTracker, new_mask::Vector{Tuple{Expression, Expression}})

Apply a slice to the ShapeTracker by updating its mask.
"""
function slice!(st::ShapeTracker, new_mask::Vector{Tuple{Expression, Expression}})
    for (i, (b, t)) in enumerate(new_mask)
        # The index in the original dims array
        original_index = st.indexes[i]
        
        # Current mask values
        current_b, current_t = st.mask[original_index]

        # Update the mask with the new values, taking the max/min
        # Note: We need max/min operations defined for Expressions
        st.mask[original_index] = (max(current_b, b), min(current_t, t))
    end
end

"""
    pad!(st::ShapeTracker, new_padding::Vector{Tuple{Expression, Expression}})

Apply padding to the ShapeTracker by updating its padding values.
"""
function pad!(st::ShapeTracker, new_padding::Vector{Tuple{Expression, Expression}})
    for (i, (s, e)) in enumerate(new_padding)
        # The index in the original dims array
        original_index = st.indexes[i]

        # Add the new padding to the existing padding
        st.padding[original_index] = (st.padding[original_index][1] + s, st.padding[original_index][2] + e)
    end
end