using Test
using Luminal
using Luminal.PagedAttention
using Random

const DEV = get(ENV, "LUMINAL_TEST_GPU", "0") == "1" ? Luminal.get_device() : CPUDevice()
run(g, ids, inputs) = Dict(k => Array(v) for (k, v) in execute(g, ids, inputs, DEV))

# plain-Julia grouped-query attention for one sequence: q (hd*H, nq), k/v (hd*KV, nk), bias (nk, nq)
function ref_gqa(q, k, v, bias, hd, H, KV)
    groups = H ÷ KV; nq, nk = size(q, 2), size(k, 2)
    out = zeros(Float32, hd * H, nq)
    for h in 1:H
        kvh = (h - 1) ÷ groups + 1
        Q = q[(h - 1) * hd + 1:h * hd, :]; K = k[(kvh - 1) * hd + 1:kvh * hd, :]; V = v[(kvh - 1) * hd + 1:kvh * hd, :]
        S = K' * Q ./ sqrt(Float32(hd)) .+ bias
        S .-= maximum(S; dims=1); P = exp.(S); P ./= sum(P; dims=1)
        out[(h - 1) * hd + 1:h * hd, :] = V * P
    end
    return out
end

@testset "row gather / scatter" begin
    dv = Float32.(Base.reshape(1:18, 3, 6)); iv = Float32[4, 0, 5, 2]; sv = -Float32.(Base.reshape(1:12, 3, 4))
    g = Graph(); data = tensor(g, [3, 6]); idx = tensor(g, [4]); src = tensor(g, [3, 4])
    gr = gather_rows(data, idx); sc = scatter_rows(src, idx, data)
    res = run(g, [gr.id, sc.id], Dict(data.id => dv, idx.id => iv, src.id => sv))
    @test res[gr.id] == dv[:, Int.(iv) .+ 1]
    want = copy(dv); want[:, Int.(iv) .+ 1] = sv
    @test res[sc.id] == want
end

@testset "biases" begin
    g = Graph(); qp = tensor(g, [3]); kp = tensor(g, [5]); qs = tensor(g, [3]); ks = tensor(g, [5])
    c = causal_bias(qp, kp); s = sequence_isolation_bias(qs, ks); w = sliding_window_bias(qp, kp, 2)
    qv = Float32[2, 4, 4]; kv = Float32[0, 1, 2, 3, 4]; qsv = Float32[0, 1, 1]; ksv = Float32[0, 1, 0, 1, 1]
    res = run(g, [c.id, s.id, w.id], Dict(qp.id => qv, kp.id => kv, qs.id => qsv, ks.id => ksv))
    @test res[c.id] == [k > q ? MASKED_SCORE : 0f0 for k in kv, q in qv]
    @test res[s.id] == [ks == qs ? 0f0 : MASKED_SCORE for ks in ksv, qs in qsv]
    @test res[w.id] == [(k > q || k < q - 1) ? MASKED_SCORE : 0f0 for k in kv, q in qv]
    g = Graph(); ip = tensor(g, [4]); ids = packed_sequence_ids(ip, 7)
    @test run(g, [ids.id], Dict(ip.id => Float32[0, 3, 5, 7]))[ids.id] == Float32[0, 0, 0, 1, 1, 2, 2]
end

@testset "paged attention: two sequences interleaved in one pool" begin
    rng = MersenneTwister(1)
    hd, H, KV, slots = 4, 4, 2, 16
    geo = AttentionGeometry(H, KV, hd)
    # sequence A has 3 cached tokens in slots 9, 2, 14; sequence B has 2 in slots 5, 0; this step decodes one token each
    A_slots = [9, 2, 14]; B_slots = [5, 0]
    kc = zeros(Float32, KV * hd, slots); vc = zeros(Float32, KV * hd, slots)
    kc[:, [A_slots; B_slots] .+ 1] = randn(rng, Float32, KV * hd, 5); vc[:, [A_slots; B_slots] .+ 1] = randn(rng, Float32, KV * hd, 5)
    q = randn(rng, Float32, H * hd, 2); nk = randn(rng, Float32, KV * hd, 2); nv = randn(rng, Float32, KV * hd, 2)
    write_slots = [3, 7]                                  # new token of A, of B
    read_slots = [A_slots; 3; B_slots; 7]                 # context in an arbitrary mixed order below
    perm = [4, 1, 6, 2, 5, 3, 7]; read_slots = read_slots[perm]
    kpos = Float32[0, 1, 2, 3, 0, 1, 2][perm]; kseq = Float32[0, 0, 0, 0, 1, 1, 1][perm]
    qpos = Float32[3, 2]; qseq = Float32[0, 1]

    g = Graph()
    kcT = tensor(g, [KV * hd, slots]); vcT = tensor(g, [KV * hd, slots])
    qT = tensor(g, [H * hd, 2]); nkT = tensor(g, [KV * hd, 2]); nvT = tensor(g, [KV * hd, 2])
    ws = tensor(g, [2]); rs = tensor(g, [7]); qp = tensor(g, [2]); kp = tensor(g, [7]); qsq = tensor(g, [2]); ksq = tensor(g, [7])
    bias = causal_bias(qp, kp) + sequence_isolation_bias(qsq, ksq)
    r = paged_attention(qT, nkT, nvT, KvCache(kcT, vcT), CacheAccess(ws, rs), bias, geo)
    inputs = Dict(kcT.id => kc, vcT.id => vc, qT.id => q, nkT.id => nk, nvT.id => nv, ws.id => Float32.(write_slots),
                  rs.id => Float32.(read_slots), qp.id => qpos, kp.id => kpos, qsq.id => qseq, ksq.id => kseq)
    res = run(g, [r.output.id, r.cache.keys.id, r.cache.values.id], inputs)

    kc2 = copy(kc); vc2 = copy(vc); kc2[:, write_slots .+ 1] = nk; vc2[:, write_slots .+ 1] = nv
    @test res[r.cache.keys.id] == kc2 && res[r.cache.values.id] == vc2
    want = zeros(Float32, H * hd, 2)
    for (col, sl) in enumerate(([A_slots; 3], [B_slots; 7]))            # each sequence alone, in logical order
        want[:, col] = ref_gqa(q[:, col:col], kc2[:, sl .+ 1], vc2[:, sl .+ 1], zeros(Float32, length(sl), 1), hd, H, KV)
    end
    @test res[r.output.id] ≈ want atol=1e-5
end
