using Test
using Luminal

# DecodeAttention (one kernel) against an explicit softmax attention reference.
function reference(q, pk, pv, kn, vn, pos, scale)
    D, _, H, B = size(q); KVH = size(pk, 3); G = H ÷ KVH
    out = zeros(Float32, D, H, B)
    for b in 1:B, h in 1:H
        kv = (h - 1) ÷ G + 1
        K = hcat(pk[:, 1:pos, kv, b], kn[:, 1, kv, b])
        V = hcat(pv[:, 1:pos, kv, b], vn[:, 1, kv, b])
        s = scale .* (K' * q[:, 1, h, b])
        p = exp.(s .- maximum(s)); p ./= sum(p)
        out[:, h, b] = V * p
    end
    out
end

@testset "DecodeAttention" begin
    D, H, KVH, B, S = 64, 8, 2, 2, 40
    for dev in unique([CPUDevice(), get_device()]), pos in (0, 17, S)
        g = Graph()
        q  = Luminal.tensor(g, [D, 1, H, B]);   pk = Luminal.tensor(g, [D, S, KVH, B])
        pv = Luminal.tensor(g, [D, S, KVH, B]); kn = Luminal.tensor(g, [D, 1, KVH, B])
        vn = Luminal.tensor(g, [D, 1, KVH, B]); p  = Luminal.tensor(g, [1])
        scale = 1f0 / sqrt(Float32(D))
        o = Luminal.add_op!(g, Luminal.DecodeAttention(scale),
                            [(t.id, 0, t.shape) for t in (q, pk, pv, kn, vn, p)], Luminal.ShapeTracker([D, H, B]))
        vals = Dict(q => randn(Float32, D, 1, H, B), pk => randn(Float32, D, S, KVH, B),
                    pv => randn(Float32, D, S, KVH, B), kn => randn(Float32, D, 1, KVH, B),
                    vn => randn(Float32, D, 1, KVH, B), p => Float32[pos])
        ex = compile(g; device=dev, retain=[o.id])
        got = Array(ex(Dict{Int,Any}(t.id => v for (t, v) in vals); device=dev)[o.id])
        @test got ≈ reference(vals[q], vals[pk], vals[pv], vals[kn], vals[vn], pos, scale) rtol = 1e-4
    end
end
