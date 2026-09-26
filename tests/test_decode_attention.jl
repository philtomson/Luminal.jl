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

# write_cache: the op also stores k_new/v_new into slot pos + 1 of the (device-
# resident, aliased) cache, one position per sequence, leaving other slots alone.
@testset "DecodeAttention write_cache" begin
    D, H, KVH, B, S = 64, 8, 2, 3, 20
    positions = [0, 7, S - 1]
    for dev in unique([CPUDevice(), get_device()])
        g = Graph()
        q  = Luminal.tensor(g, [D, 1, H, B]);   pk = Luminal.tensor(g, [D, S, KVH, B])
        pv = Luminal.tensor(g, [D, S, KVH, B]); kn = Luminal.tensor(g, [D, 1, KVH, B])
        vn = Luminal.tensor(g, [D, 1, KVH, B]); p  = Luminal.tensor(g, [B])
        scale = 1f0 / sqrt(Float32(D))
        o = Luminal.add_op!(g, Luminal.DecodeAttention(scale, true),
                            [(t.id, 0, t.shape) for t in (q, pk, pv, kn, vn, p)], Luminal.ShapeTracker([D, H, B]))
        host = Dict(q => randn(Float32, D, 1, H, B), pk => randn(Float32, D, S, KVH, B),
                    pv => randn(Float32, D, S, KVH, B), kn => randn(Float32, D, 1, KVH, B),
                    vn => randn(Float32, D, 1, KVH, B), p => Float32.(positions))
        cache_k = Luminal.to_device(copy(host[pk]), dev)
        cache_v = Luminal.to_device(copy(host[pv]), dev)
        inputs = Dict{Int,Any}(t.id => v for (t, v) in host)
        inputs[pk.id] = cache_k; inputs[pv.id] = cache_v
        ex = compile(g; device=dev, retain=[o.id, pk.id, pv.id], free_intermediates=false)
        got = Array(ex(inputs; device=dev)[o.id])
        # attention itself is unchanged (per-sequence positions)
        for b in 1:B
            ref = reference(host[q][:, :, :, b:b], host[pk][:, :, :, b:b], host[pv][:, :, :, b:b],
                            host[kn][:, :, :, b:b], host[vn][:, :, :, b:b], positions[b], scale)
            @test got[:, :, b] ≈ ref[:, :, 1] rtol = 1e-4
        end
        K, V = Array(cache_k), Array(cache_v)
        expected_k, expected_v = copy(host[pk]), copy(host[pv])
        for b in 1:B
            expected_k[:, positions[b] + 1, :, b] .= host[kn][:, 1, :, b]
            expected_v[:, positions[b] + 1, :, b] .= host[vn][:, 1, :, b]
        end
        @test K == expected_k
        @test V == expected_v
    end
end
