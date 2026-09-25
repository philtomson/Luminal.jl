using Test
using Luminal
using Luminal.NN

# RMSNorm (with weight, no bias) runs as one RMSNormOp kernel; it must equal the formula.
@testset "RMSNormOp" begin
    for dev in unique([CPUDevice(), get_device()]), dims in ((96, 3), (96, 4, 2))
        g = Graph()
        ln = NN.RMSNorm(96, g; epsilon=1f-5)
        x = Luminal.tensor(g, collect(dims))
        y = ln(x)
        @test any(n -> n.op isa Luminal.RMSNormOp, g.nodes)
        xv, wv = randn(Float32, dims...), randn(Float32, 96)
        g.tensors[(ln.weight.id, 1)] = Luminal.to_device(wv, dev)
        got = Array(compile(g; device=dev, retain=[y.id])(Dict{Int,Any}(x.id => xv); device=dev)[y.id])
        ref = xv ./ sqrt.(sum(abs2, xv; dims=1) ./ 96 .+ 1f-5) .* wv
        @test got ≈ ref rtol = 1e-5
    end
end
