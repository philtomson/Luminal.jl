#!/usr/bin/env julia

# Add the src directory to the Julia load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Luminal
using Test
using SymbolicUtils

@testset "ShapeTracker Operations" begin
    @testset "Slice" begin
        # Initial Shape: (10, 20, 30)
        # Now using plain Ints
        dims = [10, 20, 30]
        st = Luminal.ShapeTracker(dims)

        # Slice to (2:8, 5:15, 0:30)
        slice_mask = [
            (2, 8),
            (5, 15),
            (0, 30) # No change
        ]
        
        slice!(st, slice_mask)

        # Expected mask after slicing
        # Initial mask is (0, typemax(Int))
        # After slice, it becomes max(0, 2) and min(typemax, 8) -> (2, 8)
        @test st.mask[1] == (2, 8)
        @test st.mask[2] == (5, 15)
        @test st.mask[3] == (0, 30)
    end

    @testset "Pad" begin
        # Initial Shape: (10, 20)
        dims = [10, 20]
        st = Luminal.ShapeTracker(dims)

        # Pad with (1, 1) on first dim and (2, 2) on second
        padding = [
            (1, 1),
            (2, 2)
        ]

        pad!(st, padding)

        # Expected padding
        @test st.padding[1] == (1, 1)
        @test st.padding[2] == (2, 2)

        # Pad again
        pad!(st, padding)
        @test st.padding[1] == (2, 2)
        @test st.padding[2] == (4, 4)
    end
    
    @testset "Symbolic Dimensions" begin
        @syms b::Int
        st = Luminal.ShapeTracker([b, 20])
        
        @test st.dims[1] === b
        @test st.dims[2] === 20
        
        # Test algebra on symbolic dims via padding
        pad!(st, [(2, 0), (0, 0)])
        # padding[1][1] began at 0, now it should be 2.
        # (It doesn't "know" about dimension b, it just stores the padding offset)
        @test st.padding[1][1] == 2
    end
end

println("ShapeTracker tests passed!")