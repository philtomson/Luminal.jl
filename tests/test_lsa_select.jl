# Latent Sparse Attention's selection on the GPU (src/LatentSparseSelect.jl) against the CPU reference, src/LatentSparse.jl:
# the pick kernel exactly on tie-heavy scores, the index query against `index_vectors`, and the whole selection against `read_selection`.
using Test
using Luminal
using AMDGPU
using Random, LinearAlgebra

if Luminal.get_device() isa CPUDevice
    @info "no GPU: skipping the LSA selection test"
else
    Base.include(@__MODULE__, joinpath(@__DIR__, "..", "src", "LatentSparse.jl"))
    using .LatentSparse
    const BLOCK, PER_BLOCK, SINK, WINDOW, TOP_K = 512, 64, 4, 512, 4096
    unit_rows(rng, n, d) = (a = randn(rng, Float32, n, d); a ./ sqrt.(sum(abs2, a; dims = 2)))

    rng = MersenneTwister(11)
    keys = unit_rows(rng, 4096, 64)
    table = build_buckets(keys, 4, 8, 16, BLOCK)
    nb = size(table.members, 1)
    members_dev = ROCArray(permutedims(table.members))
    ids = AMDGPU.zeros(Int32, TOP_K + WINDOW + SINK, 1); lastpos = AMDGPU.zeros(Float32, 1)
    posv(p) = ROCArray(Float32[p])
    function gpu_list()
        n = Int(Array(lastpos)[1]) + 1
        return Int32.(Array(ids)[1:n, 1])
    end

    @testset "pick kernel, ties and cuts" begin
        for levels in (2, 4, 1_000_000), (top_p, top_k) in ((16, 64), (100, 4096), (1024, 4096), (7, 5)), p in (0, 3, 511, 512, 700, 1500, 4095)
            scores = round.(randn(rng, Float32, nb) .* levels) ./ levels        # a few distinct values, negative zeros among them
            Luminal.lsa_pick_expand!(ids, lastpos, ROCArray(scores), members_dev, posv(p); top_p, top_k, sink_size = SINK, window_size = WINDOW,
                                     cluster_block_size = BLOCK, block_bucket_count = PER_BLOCK)
            visible = min(p ÷ BLOCK * PER_BLOCK, nb)
            @test gpu_list() == select_from_scores(scores[1:visible], p, table, top_p, top_k, SINK, WINDOW)
        end
    end

    @testset "index query" begin
        q, k = randn(rng, Float32, 256, 1, 24, 1), randn(rng, Float32, 256, 1, 4, 1)
        wq, wk = randn(rng, Float32, 768, 64) ./ 30, randn(rng, Float32, 768, 64) ./ 30
        want, _ = index_vectors(q, k, wq, wk, 64)
        qidx = AMDGPU.zeros(Float32, 64)
        Luminal.lsa_index_query!(qidx, ROCArray(q), ROCArray(permutedims(wq)); rotary_dim = 64)
        @test Array(qidx) ≈ want[1, :] atol = 1f-5
    end

    @testset "the whole selection" begin
        D, H, KVH = 256, 24, 4
        wq = randn(rng, Float32, KVH * (D - 64), 64) ./ 30
        wq_t = ROCArray(permutedims(wq))
        # Index keys are unit rows in a 64-dim space, so draw the query heads and let the projection make the index query
        q = randn(rng, Float32, D, 4096, H, 1)
        index_q, _ = index_vectors(q, randn(rng, Float32, D, 4096, KVH, 1), wq, wq, 64)
        qidx = AMDGPU.zeros(Float32, 64); scores = AMDGPU.zeros(Float32, nb); centroids_dev = ROCArray(table.centroids)
        for p in (0, 511, 512, 700, 2047, 4095)
            Luminal.lsa_select!(ids, lastpos, qidx, scores, ROCArray(q[:, p+1:p+1, :, :]), wq_t, centroids_dev, members_dev, posv(p);
                                rotary_dim = 64, top_p = 1024, top_k = TOP_K, sink_size = SINK, window_size = WINDOW,
                                cluster_block_size = BLOCK, block_bucket_count = PER_BLOCK)
            sel, len = read_selection(index_q[p+1:p+1, :], [p], table, 1024, TOP_K, SINK, WINDOW)
            @test sort(gpu_list()) == sort(sel[1, 1:len[1]])
        end
    end

    @testset "selection refuses bad input" begin
        scores = AMDGPU.zeros(Float32, nb)
        pick(; kw...) = Luminal.lsa_pick_expand!(ids, lastpos, scores, members_dev, posv(0); top_p = 16, top_k = 64, sink_size = SINK,
                                                 window_size = WINDOW, cluster_block_size = BLOCK, block_bucket_count = PER_BLOCK, kw...)
        @test_throws ArgumentError pick(top_p = 1025)                                  # past the workgroup
        @test_throws ArgumentError pick(top_p = 0)
        @test_throws ArgumentError pick(top_k = 5000)                                  # ids too narrow
        @test_throws ArgumentError Luminal.lsa_pick_expand!(ids, lastpos, AMDGPU.zeros(Float32, nb + 1), members_dev, posv(0); top_p = 16,
                                                           top_k = 64, sink_size = SINK, window_size = WINDOW, cluster_block_size = BLOCK,
                                                           block_bucket_count = PER_BLOCK)  # scores and members disagree
        @test_throws ArgumentError Luminal.lsa_index_query!(AMDGPU.zeros(Float32, 32), ROCArray(randn(Float32, 256, 1, 24, 1)),
                                                           ROCArray(randn(Float32, 64, 768)); rotary_dim = 64)                       # index dim not 64
        @test_throws ArgumentError Luminal.lsa_index_query!(AMDGPU.zeros(Float32, 64), ROCArray(randn(Float32, 256, 2, 24, 1)),
                                                           ROCArray(randn(Float32, 64, 768)); rotary_dim = 64)                       # two tokens
    end
end
