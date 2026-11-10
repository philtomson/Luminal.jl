# Latent Sparse Attention in the decode step (src/LSADecode.jl), on synthetic caches and no model: the prefill hook's index keys and buckets,
# the op (cache write, index key, selection, attention) for one token, the post-step hook building the block a token closes, and the
# inputs the op refuses. Each against the CPU reference.
using Test
using Luminal
using AMDGPU
using Random, LinearAlgebra

if Luminal.get_device() isa CPUDevice
    @info "no GPU: skipping the LSA decode test"
else
    using Luminal: LSAConfig, LSAState, LatentSparse

    mutable struct FakeCache                      # what the hooks read of a LlamaKVCacheState
        positions::Vector{Int}
        self_cache::Vector{Any}
    end
    softmax_cols(L) = (E = exp.(L .- maximum(L; dims = 1)); E ./ sum(E; dims = 1))

    D, H, KVH, MAX_SEQ, LAYER = 256, 24, 4, 4096, 3
    scale = Float32(1 / sqrt(D))
    rng = MersenneTwister(21)
    maps = Dict(LAYER => (randn(rng, Float32, 768, 64) ./ 30, randn(rng, Float32, 768, 64) ./ 30))
    config = LSAConfig(maps = maps, top_k = 192, top_p = 48)
    st = LSAState(config, [LAYER], MAX_SEQ)
    ls = st.layers[LAYER]
    k16, v16 = Float16.(randn(rng, Float32, D, MAX_SEQ, KVH, 1)), Float16.(randn(rng, Float32, D, MAX_SEQ, KVH, 1))
    pk, pv = ROCArray(k16), ROCArray(v16)
    cache = FakeCache([1900], Any[nothing, nothing, nothing, (pk, pv)])
    Luminal.lsa_register_cache!(st, cache)

    cpu_index_keys(k16, range) = last(LatentSparse.index_vectors(zeros(Float32, D, length(range), H, 1),
                                                               Base.reshape(Float32.(k16[:, range, :, 1]), D, length(range), KVH, 1), maps[LAYER]..., 64))

    @testset "prefill hook" begin
        Luminal.lsa_after_prefill!(st, cache, 1900)
        @test st.built_blocks == 3
        got = permutedims(Array(view(ls.index_keys, :, 1:1900)))
        @test maximum(abs.(got .- cpu_index_keys(k16, 1:1900))) < 2e-6
        table = LatentSparse.build_buckets(got[1:1536, :], 4, 8, 16, 512)
        @test permutedims(Array(view(ls.members, :, 1:192))) == table.members
        @test maximum(abs.(Array(view(ls.centroids, 1:192, :)) .- table.centroids)) < 1e-6
    end

    @testset "the op, one token" begin
        p = 1900
        q = randn(rng, Float32, D, 1, H, 1); kn = randn(rng, Float32, D, 1, KVH, 1); vn = randn(rng, Float32, D, 1, KVH, 1)
        out = AMDGPU.zeros(Float32, D, H, 1)
        Luminal.execute_op!(out, Luminal.LSADecodeAttention(scale, ls), ROCArray(q), pk, pv, ROCArray(kn), ROCArray(vn), ROCArray(Float32[p]))
        # The cache took the token's K and V at slot p
        @test Array(pk)[:, p+1, :, 1] == Float16.(kn[:, 1, :, 1]) && Array(pv)[:, p+1, :, 1] == Float16.(vn[:, 1, :, 1])
        # ... and the index key landed in its column
        index_q, index_k = LatentSparse.index_vectors(q, kn, maps[LAYER]..., 64)
        @test Array(ls.index_keys)[:, p+1] ≈ index_k[1, :] atol = 2e-6
        @test Array(ls.qidx) ≈ index_q[1, :] atol = 2e-6
        # The list is the CPU's selection from the same tables
        cent, members = Array(ls.centroids), permutedims(Array(ls.members))
        nb = p ÷ 512 * 64
        table = LatentSparse.BucketTable(cent, members, Int32.(vec(sum(members .>= 0; dims = 2))), 64, 512)
        want = LatentSparse.select_from_scores(cent[1:nb, :] * index_q[1, :], p, table, 48, 192, 4, 512)
        n = Int(Array(ls.lastpos)[1]) + 1
        got = Int32.(Array(ls.ids)[1:n, 1])
        @test sort(got) == sort(want)
        @test length(want) < p + 1                                  # a real subset
        # The output is a softmax over those keys, the new token among them
        kf, vf = Float32.(Array(pk)[:, :, :, 1]), Float32.(Array(pv)[:, :, :, 1])
        slots = Int.(want) .+ 1
        ref = zeros(Float32, D, H)
        for h in 1:KVH
            ref[:, (h-1)*6+1:h*6] = vf[:, slots, h] * softmax_cols(scale .* (kf[:, slots, h]' * q[:, 1, (h-1)*6+1:h*6, 1]))
        end
        @test norm(Array(out)[:, :, 1] - ref) / norm(ref) < 5e-4
    end

    @testset "post-step hook" begin
        Luminal.lsa_index_cache!(ls, config, pk, 1901, 2048)           # the keys the steps up to the block's end would have written
        cache.positions[1] = 2047
        Luminal.lsa_after_step!(cache)
        @test st.built_blocks == 3                                      # block 3 is not closed yet
        cache.positions[1] = 2048
        Luminal.lsa_after_step!(cache)
        @test st.built_blocks == 4
        index_keys = permutedims(Array(view(ls.index_keys, :, 1:2048)))
        table = LatentSparse.build_buckets(index_keys[1537:2048, :], 4, 8, 16, 512)
        @test permutedims(Array(view(ls.members, :, 193:256))) == table.members .+ ifelse.(table.members .>= 0, Int32(1536), Int32(0))
        Luminal.lsa_after_step!(FakeCache([9999], []))                  # a cache with no state is ignored
    end

    @testset "the op refuses what it does not take" begin
        op = Luminal.LSADecodeAttention(scale, ls)
        out = AMDGPU.zeros(Float32, D, H, 1)
        args(S) = (ROCArray(randn(Float32, D, S, H, 1)), pk, pv, ROCArray(randn(Float32, D, S, KVH, 1)), ROCArray(randn(Float32, D, S, KVH, 1)), ROCArray(Float32[100]))
        @test_throws ErrorException Luminal.execute_op!(out, op, args(2)...)                      # two tokens
        q, _, _, kn, vn, pos = args(1)
        @test_throws ErrorException Luminal.execute_op!(out, op, q, ROCArray(randn(Float32, D, MAX_SEQ, KVH, 1)), pv, kn, vn, pos)   # a Float32 cache
        @test_throws ArgumentError LSAState(LSAConfig(maps = maps), [5], MAX_SEQ)                 # no map for layer 5
        @test_throws ArgumentError LSAState(LSAConfig(maps = maps, top_p = 2000), [LAYER], MAX_SEQ)
        @test_throws ArgumentError LSAState(LSAConfig(maps = maps, bucket_size = 7), [LAYER], MAX_SEQ)
    end
end
