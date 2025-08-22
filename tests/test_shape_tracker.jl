#!/usr/bin/env julia

# Add the src directory to the Julia load path
push!(LOAD_PATH, "../src")

using Luminal
using Luminal.Symbolic
using Test

@testset "ShapeTracker Operations" begin
    @testset "Slice" begin
        # Initial Shape: (10, 20, 30)
        dims = [Expression([Num(10)]), Expression([Num(20)]), Expression([Num(30)])]
        st = Luminal.ShapeTracker(dims)

        # Slice to (2:8, 5:15, :)
        slice_mask = [
            (Expression([Num(2)]), Expression([Num(8)])),
            (Expression([Num(5)]), Expression([Num(15)])),
            (Expression([Num(0)]), Expression([Num(30)])) # No change
        ]
        
        Luminal.slice!(st, slice_mask)

        # Expected mask after slicing
        # Initial mask is (0, typemax(Int))
        # After slice, it becomes max(0, 2) and min(typemax, 8) -> (2, 8)
        # And max(0, 5) and min(typemax, 15) -> (5, 15)
        @test st.mask[1] == (Expression([Num(2)]), Expression([Num(8)]))
        @test st.mask[2] == (Expression([Num(5)]), Expression([Num(15)]))
        # The third dimension should be clipped by typemax(Int), not the original dimension size
        @test st.mask[3] == (Expression([Num(0)]), Expression([Num(30)]))
    end

    @testset "Pad" begin
        # Initial Shape: (10, 20)
        dims = [Expression([Num(10)]), Expression([Num(20)])]
        st = Luminal.ShapeTracker(dims)

        # Pad with (1, 1) on first dim and (2, 2) on second
        padding = [
            (Expression([Num(1)]), Expression([Num(1)])),
            (Expression([Num(2)]), Expression([Num(2)]))
        ]

        Luminal.pad!(st, padding)

        # Expected padding
        @test st.padding[1] == (Expression([Num(1)]), Expression([Num(1)]))
        @test st.padding[2] == (Expression([Num(2)]), Expression([Num(2)]))

        # Pad again
        Luminal.pad!(st, padding)
        @test st.padding[1] == (Expression([Num(2)]), Expression([Num(2)]))
        @test st.padding[2] == (Expression([Num(4)]), Expression([Num(4)]))
    end
end

println("ShapeTracker tests passed!")