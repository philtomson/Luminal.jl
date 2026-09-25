using Test
using Luminal
using Luminal.NN

# RotaryEmbed (one kernel) must equal the rotate-half formula.
@testset "RotaryEmbed" begin
    D, S, H, B = 8, 5, 3, 2
    for dev in unique([CPUDevice(), get_device()]), prev in (0, 7)
        g = Graph()
        x = Luminal.tensor(g, [D, S, H, B])
        out = NN.apply_rotary_embeddings(x, prev; base=10000f0)
        @test any(n -> n.op isa Luminal.RotaryEmbed, g.nodes)
        xv = randn(Float32, D, S, H, B)
        got = Array(compile(g; device=dev, retain=[out.id])(Dict{Int,Any}(x.id => xv); device=dev)[out.id])
        half = D ÷ 2
        inv = [10000f0^(-2f0 * (i - 1) / D) for i in 1:half]
        θ = [(t - 1 + prev) * inv[i] for i in 1:half, t in 1:S]
        c = Base.reshape(cos.(θ), half, S, 1, 1); s = Base.reshape(sin.(θ), half, S, 1, 1)
        x0, x1 = xv[1:half, :, :, :], xv[half+1:end, :, :, :]
        @test got ≈ cat(x0 .* c .- x1 .* s, x1 .* c .+ x0 .* s; dims=1) rtol = 1e-4
    end
end
