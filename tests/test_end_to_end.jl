using Test
using Luminal

# Build (a * b) + c, run it through the interpreter and through compile().
@testset "End to end: matmul + add" begin
    g = Luminal.Graph()
    a = Luminal.tensor(g, [2, 2]); b = Luminal.tensor(g, [2, 2]); c = Luminal.tensor(g, [2, 2])
    e = Luminal.matmul(a, b) + c
    da, db, dc = Float32[1 2; 3 4], Float32[5 6; 7 8], Float32[1 1; 1 1]
    inputs = Dict{Int,Any}(a.id => da, b.id => db, c.id => dc)
    expected = da * db + dc
    for dev in unique([CPUDevice(), Luminal.get_device()])
        @test Array(Luminal.execute(g, e.id, inputs, dev)) ≈ expected
        @test Array(compile(g; device=dev, retain=[e.id])(inputs; device=dev)[e.id]) ≈ expected
    end
end
