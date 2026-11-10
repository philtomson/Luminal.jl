# Sparse vs dense attention on a captured layer (CPU), with the D-LSA flat selection from LatentSparse.jl.
#
#   julia -t auto --project=. benchmarks/lsa_eval_layer3.jl <capture.jls> <maps_dir> [n_positions]
#
# CPU only. <maps_dir> holds the D-LSA maps as safetensors (the d-lsa-weights-v1 release, converted from .pt). Compares the sparse
# attention output of the selected keys to dense for all heads, against random, window-only and oracle selections at the same width.
using Serialization, SafeTensors, LinearAlgebra, Random, Statistics, Printf
include(joinpath(@__DIR__, "..", "src", "LatentSparse.jl"))
using .LatentSparse

const POOL, BUCKET, CAP, BLOCK, SINK, WINDOW = 4, 8, 16, 512, 4, 512
const LAYER = 3
const GROUP = 6                       # query heads per kv head

normalize_rows(a) = a ./ max.(sqrt.(sum(abs2, a; dims = 2)), 1f-30)

# k_pf [T, 768] and q_pf [T, 768]: the position-free 192 dims of each kv head side by side, each group's queries summed
function position_free(q, k, rotary)
    D, S, H = size(q, 1), size(q, 2), size(q, 3)
    KV = size(k, 3)
    dims = rotary+1:D
    k_pf = reduce(hcat, [permutedims(k[dims, :, h, 1]) for h in 1:KV])                                  # [S, KV*192]
    q_pf = reduce(hcat, [permutedims(dropdims(sum(q[dims, :, (h-1)*GROUP+1:h*GROUP, 1]; dims = 3); dims = 3)) for h in 1:KV])
    return Float32.(q_pf), Float32.(k_pf)
end

function softmax_cols(L)
    M = maximum(L; dims = 1)
    E = exp.(L .- M)
    return E ./ sum(E; dims = 1)
end

# Per kv head h: dense logits [p+1, 6] and probabilities, then the output through a key subset
function evaluate(q, k, v, scale, positions, selections)
    names = collect(keys(selections))
    recall = Dict(n => Float64[] for n in names)
    relerr = Dict(n => Float64[] for n in names)
    for (r, p) in enumerate(positions)
        for h in 1:size(k, 3)
            Kh = view(k, :, 1:p+1, h, 1); Vh = view(v, :, 1:p+1, h, 1)
            Qg = view(q, :, p + 1, (h-1)*GROUP+1:h*GROUP, 1)
            L = scale .* (Kh' * Qg)                                  # [p+1, 6]
            P = softmax_cols(L)
            dense = Vh * P                                           # [256, 6]
            for n in names
                ids = selections[n][r]
                idx = ids .+ 1
                Ls = L[idx, :]
                Ps = softmax_cols(Ls)
                sparse = view(Vh, :, idx) * Ps
                for g in 1:GROUP
                    push!(recall[n], sum(P[idx, g]))
                    push!(relerr[n], norm(sparse[:, g] - dense[:, g]) / norm(dense[:, g]))
                end
            end
        end
    end
    return recall, relerr
end

function main()
    cap = deserialize(ARGS[1]); maps_dir = ARGS[2]
    n_pos = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 256
    q, k, v, scale, rotary = cap["q"], cap["k"], cap["v"], cap["scale"], cap["rotary_dim"]
    N = size(q, 2)
    positions = round.(Int, range(2048, N - 1, length = n_pos)) |> unique
    println(cap["weights"], ": $N tokens, $(length(positions)) query positions from $(first(positions)) to $(last(positions))")
    q_pf, k_pf = position_free(q, k, rotary)
    println("index input ", size(k_pf), ", ", size(q_pf))

    tables = Dict{String,Any}(); sels = Dict{Tuple{String,Int},Any}()
    budgets = [(4096, 1024), (2048, 512), (1024, 256)]
    out = Dict{String,Any}()
    for (label, file) in (("d2048", "qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.safetensors"),
                          ("pca", "qwen38_27b__long-16k-1M-heavy-key-and-query-pca-step0.safetensors"))
        W = SafeTensors.load_safetensors(joinpath(maps_dir, file))
        K_idx = Float32.(normalize_rows(k_pf * W["$LAYER.W_K_idx"]))
        Q_idx = Float32.(normalize_rows(q_pf * W["$LAYER.W_Q_idx"]))
        t = @elapsed table = build_buckets(K_idx, POOL, BUCKET, CAP, BLOCK)
        @printf("%s: built %d buckets in %.1f s (sizes %d..%d)\n", label, size(table.members, 1), t, minimum(table.sizes), maximum(table.sizes))
        for (top_k, top_p) in budgets
            selected, lengths = read_selection(Q_idx[positions .+ 1, :], positions, table, top_p, top_k, SINK, WINDOW)
            sels[(label, top_k)] = [Int.(selected[i, 1:lengths[i]]) for i in eachindex(positions)]
        end
    end

    rng = MersenneTwister(0)
    for (top_k, top_p) in budgets
        selections = Dict{String,Vector{Vector{Int}}}()
        selections["lsa d2048"] = sels[("d2048", top_k)]
        selections["lsa pca"] = sels[("pca", top_k)]
        base = Vector{Vector{Int}}(); rnd = Vector{Vector{Int}}(); orc = Vector{Vector{Int}}()
        for (r, p) in enumerate(positions)
            ws = max(0, p + 1 - WINDOW)
            window_sinks = vcat(collect(ws:p), collect(0:min(SINK, ws, p + 1)-1))
            eligible = collect(SINK:ws-1)
            push!(base, window_sinks)
            n_exp = length(sels[("d2048", top_k)][r]) - length(window_sinks)
            push!(rnd, vcat(shuffle(rng, eligible)[1:min(n_exp, length(eligible))], window_sinks))
            # Oracle - the head-shared top keys by true attention mass, the best any one list for all 24 heads could do
            mass = zeros(Float64, length(eligible))
            for h in 1:size(k, 3)
                L = scale .* (view(k, :, 1:p+1, h, 1)' * view(q, :, p + 1, (h-1)*GROUP+1:h*GROUP, 1))
                mass .+= vec(sum(softmax_cols(L)[eligible .+ 1, :]; dims = 2))
            end
            push!(orc, vcat(eligible[partialsortperm(mass, 1:min(n_exp, length(eligible)); rev = true)], window_sinks))
        end
        selections["window+sinks only"] = base
        selections["random keys"] = rnd
        selections["oracle top keys"] = orc
        recall, relerr = evaluate(q, k, v, scale, positions, selections)
        width = mean(length.(sels[("d2048", top_k)]))
        println(@sprintf("\ntop_k %d, top_p %d - reads %.0f keys on average, %.1f%% of the causal context", top_k, top_p, width,
                         100 * mean(length.(sels[("d2048", top_k)]) ./ (positions .+ 1))))
        println(@sprintf("  %-20s %12s %12s %12s", "selection", "mass recall", "rel err", "rel err p90"))
        for n in ("lsa d2048", "lsa pca", "oracle top keys", "random keys", "window+sinks only")
            println(@sprintf("  %-20s %12.4f %12.4f %12.4f", n, mean(recall[n]), mean(relerr[n]), quantile(relerr[n], 0.9)))
        end
        out["$top_k"] = (recall = Dict(n => mean(recall[n]) for n in keys(recall)), relerr = Dict(n => mean(relerr[n]) for n in keys(relerr)))
    end
    # Stash the selections for cross-capture comparison
    serialize(replace(ARGS[1], ".jls" => "_sel.jls"), Dict("positions" => positions, "sels" => sels))
end
main()
