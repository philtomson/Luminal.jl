# LatentSparse (D-LSA flat, CPU reference) against its own invariants and the torch oracle's fixtures.
#
# The fixtures come from latent-sparse-attention `make_fixtures.py` (build_buckets and read on random clustered unit keys).
# Point LSA_FIXTURES at their directory; without it only the invariants run.
using Test, Random, LinearAlgebra, SafeTensors
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "src", "LatentSparse.jl"))
using .LatentSparse

# The oracle scores buckets with torch's matmul and this port with a gemv, so two buckets whose scores differ by less than an fp32
# ulp can swap. That reorders the expansion, and only the order - so selection is held to the same key set on every row and to
# exact equality on all but a sliver of them. (On the design fixture 38 of 8192 rows swap, each at a float64 score gap under 4e-8.)
function same_selection(got, want, lengths)
    sets = all(sort(got[i, 1:lengths[i]]) == sort(want[i, 1:lengths[i]]) for i in axes(got, 1))
    exact = count(got[i, :] == want[i, :] for i in axes(got, 1)) / size(got, 1)
    return sets && exact >= 0.99
end

function unit_rows(rng, n, d)
    centers = randn(rng, Float32, max(n ÷ 24, 4), d)
    rows = centers[rand(rng, 1:size(centers, 1), n), :] .+ 0.6f0 .* randn(rng, Float32, n, d)
    return rows ./ sqrt.(sum(abs2, rows; dims = 2))
end

@testset "invariants" begin
    rng = MersenneTwister(0)
    keys = unit_rows(rng, 2048 + 100, 64)
    table = build_buckets(keys, 4, 8, 16, 512)
    @test size(table.members) == (4 * 64, 16)
    held = filter(>=(0), vec(table.members))
    @test sort(held) == 0:2047                       # every closed key in exactly one bucket, the open tail in none
    @test all(1 .<= table.sizes .<= 16)
    @test table.sizes == vec(sum(table.members .>= 0; dims = 2))
    @test all(abs.(sqrt.(sum(abs2, table.centroids; dims = 2)) .- 1) .< 1f-5)

    # Dense anchor - every bucket picked and no cut, so a query reads exactly the keys 0..pos once each
    queries = unit_rows(rng, 2148, 64)
    positions = 0:2147
    selected, lengths = read_selection(queries, positions, table, 10^6, 10^6, 4, 512)
    for pos in (0, 3, 511, 512, 700, 1023, 1024, 2047, 2147)
        row = selected[pos + 1, 1:lengths[pos + 1]]
        @test sort(row) == 0:pos
    end

    # A negative zero ties with a positive zero, and a tie goes to the lower bucket
    tied = LatentSparse.BucketTable(zeros(Float32, 3, 2), Int32[4 5 6 -1; 7 8 -1 -1; 9 10 -1 -1], Int32[3, 2, 2], 3, 512)
    @test select_from_scores(Float32[-0.0, 0.0, -0.0], 1600, tied, 3, 100, 4, 512) == Int32[4, 5, 6, 7, 8, 9, 10, 1089:1600..., 0, 1, 2, 3]

    @test_throws ArgumentError build_buckets(keys, 4, 8, 15, 512)         # cap under twice the bucket size
    @test_throws ArgumentError build_buckets(keys, 3, 8, 16, 512)         # pool does not divide the bucket
    @test_throws ArgumentError build_buckets(2f0 .* keys, 4, 8, 16, 512)  # not unit rows
    @test_throws ArgumentError read_selection(queries, positions[1:5], table, 16, 64, 4, 512)
end

@testset "index_vectors" begin
    rng = MersenneTwister(1)
    D, S, H, KV, rotary, dim = 16, 5, 6, 2, 4, 8
    q, k = randn(rng, Float32, D, S, H, 1), randn(rng, Float32, D, S, KV, 1)
    wq, wk = randn(rng, Float32, KV * (D - rotary), dim), randn(rng, Float32, KV * (D - rotary), dim)
    iq, ik = index_vectors(q, k, wq, wk, rotary)
    for t in 1:S
        # Spelled out - head h reads kv head (h-1) ÷ 3, the rotated dims dropped, the kv heads concatenated
        kpf = vcat((k[rotary+1:D, t, h, 1] for h in 1:KV)...)
        qpf = vcat((sum(q[rotary+1:D, t, (h-1)*3+g, 1] for g in 1:3) for h in 1:KV)...)
        @test ik[t, :] ≈ normalize(wk' * kpf)
        @test iq[t, :] ≈ normalize(wq' * qpf)
    end
    @test all(abs.(sqrt.(sum(abs2, iq; dims = 2)) .- 1) .< 1f-5)
    @test_throws ArgumentError index_vectors(q, k, wq[1:5, :], wk, rotary)
    @test_throws ArgumentError index_vectors(q, randn(rng, Float32, D, S, 4, 1), wq, wk, rotary)   # 6 heads over 4 kv heads
end

fixture_dir = get(ENV, "LSA_FIXTURES", "")
for name in ("small", "design")
    path = joinpath(fixture_dir, "$name.safetensors")
    if isempty(fixture_dir) || !isfile(path)
        @info "skipping fixture $name - set LSA_FIXTURES"
        continue
    end
    @testset "fixture $name" begin
        f = SafeTensors.load_safetensors(path)
        pool, bucket, cap, block, sink, window, top_p, top_k, _ = Int.(f["params"])
        keys, queries, positions = f["index_keys"], f["index_queries"], Int.(f["positions"])

        # The oracle's own buckets, so selection is tested apart from clustering
        oracle = LatentSparse.BucketTable(f["bucket_centroids"], f["bucket_members"], f["bucket_sizes"], block ÷ bucket, block)
        selected, lengths = read_selection(queries, positions, oracle, top_p, top_k, sink, window)
        row_equal = [selected[i, :] == f["selected"][i, :] for i in axes(selected, 1)]
        @info "$name selection from the oracle's buckets" exact_rows = count(row_equal) of = length(row_equal)
        @test lengths == f["lengths"]
        @test same_selection(selected, f["selected"], lengths)

        # Clustering
        table = build_buckets(keys, pool, bucket, cap, block)
        @test table.sizes == f["bucket_sizes"]
        same_rows = [table.members[i, :] == f["bucket_members"][i, :] for i in axes(table.members, 1)]
        @info "$name bucket members from Julia's Ward" exact_buckets = count(same_rows) of = length(same_rows)
        @test table.members == f["bucket_members"]
        @test maximum(abs.(table.centroids .- f["bucket_centroids"])) < 1f-5

        # Both together
        end_to_end, end_lengths = read_selection(queries, positions, table, top_p, top_k, sink, window)
        @test same_selection(end_to_end, f["selected"], f["lengths"])
    end
end
