# MatMul across ranks, left-aligned: (M, K, batch...) * (K, N, batch...) ->
# (M, N, batch...), batch dims broadcasting (size 1, or missing on one side).
# Interpreter and compile() on CPU and GPU against a Float64 loop, and gradients.
using Test
using Luminal
using Luminal: compile

devices = Any[CPUDevice()]
Luminal.get_device() isa CPUDevice || push!(devices, Luminal.get_device())

# reference: out[:, :, I] = A[:, :, I_A] * B[:, :, I_B], size-1 / missing dims broadcast
function ref_matmul(A, B)
    bA, bB = size(A)[3:end], size(B)[3:end]
    n = max(length(bA), length(bB))
    g(t, k) = k <= length(t) ? t[k] : 1
    batch = ntuple(k -> max(g(bA, k), g(bB, k)), n)
    A = Base.reshape(Float64.(A), size(A, 1), size(A, 2), ntuple(k -> g(bA, k), n)...)
    B = Base.reshape(Float64.(B), size(B, 1), size(B, 2), ntuple(k -> g(bB, k), n)...)
    C = zeros(size(A, 1), size(B, 2), batch...)
    for I in CartesianIndices(batch)
        ia = ntuple(k -> size(A, k + 2) == 1 ? 1 : I[k], n); ib = ntuple(k -> size(B, k + 2) == 1 ? 1 : I[k], n)
        C[:, :, Tuple(I)...] = A[:, :, ia...] * B[:, :, ib...]
    end
    return C
end

const CASES = [
    ([5, 7], [7, 3]),                  # 2D x 2D
    ([5, 7], [7, 3, 4]),               # weight x batched activations
    ([5, 7], [7, 3, 4, 2]),
    ([5, 7, 4], [7, 3]),               # batched x unbatched
    ([5, 7, 4, 2], [7, 3]),
    ([5, 7, 4], [7, 3, 4]),            # equal batches (the case that was broken)
    ([5, 7, 4, 2], [7, 3, 4, 2]),
    ([5, 7, 4], [7, 3, 4, 2]),         # rank mismatch: missing dim broadcasts
    ([5, 7, 1, 2], [7, 3, 4, 2]),      # size-1 batch dim broadcasts
    ([5, 7, 4, 1], [7, 3, 1, 2]),
]

@testset "matmul shapes" begin
    for (da, db) in CASES
        g = Graph(); a = tensor(g, da); b = tensor(g, db)
        c = matmul(a, b)
        av, bv = randn(Float32, da...), randn(Float32, db...)
        ref = ref_matmul(av, bv)
        @test Luminal.realized_dims(c.shape) == collect(size(ref))
        ins = Dict(a.id => av, b.id => bv)
        @test execute(g, c.id, ins, CPUDevice()) ≈ ref rtol=1e-4
        @test Luminal.batch_matmul(av, bv) ≈ ref rtol=1e-4               # functional form
        for dev in devices
            @test Array(compile(g; device=dev, retain=[c.id])(ins; device=dev)[c.id]) ≈ ref rtol=1e-4
        end
    end

    # a non-contiguous operand (a permuted view) takes the general path
    g = Graph(); a = tensor(g, [7, 5, 4]); b = tensor(g, [7, 3, 4])
    c = matmul(Luminal.permute(a, [2, 1, 3]), b)
    av, bv = randn(Float32, 7, 5, 4), randn(Float32, 7, 3, 4)
    ref = ref_matmul(permutedims(av, (2, 1, 3)), bv)
    @test execute(g, c.id, Dict(a.id => av, b.id => bv), CPUDevice()) ≈ ref rtol=1e-4
    for dev in devices
        @test Array(compile(g; device=dev, retain=[c.id])(Dict(a.id => av, b.id => bv); device=dev)[c.id]) ≈ ref rtol=1e-4
    end
end

@testset "gradients, 3D batched" begin
    g = Graph(); a = tensor(g, [3, 4, 2]); b = tensor(g, [4, 5, 2])
    loss = sum(sum(sum(matmul(a, b) * matmul(a, b), 1), 1), 1)
    mark_trainable!(a); mark_trainable!(b); gr = backward(loss)
    av, bv = randn(Float32, 3, 4, 2), randn(Float32, 4, 5, 2)
    r = execute(g, [gr[a.id].id, gr[b.id].id], Dict(a.id => av, b.id => bv), CPUDevice())
    # d/dA sum((AB)^2) = 2 (AB) B', d/dB = 2 A' (AB), per batch
    for k in 1:2
        C = av[:, :, k] * bv[:, :, k]
        @test r[gr[a.id].id][:, :, k] ≈ 2 .* C * bv[:, :, k]' rtol=1e-4
        @test r[gr[b.id].id][:, :, k] ≈ 2 .* av[:, :, k]' * C rtol=1e-4
    end
end

@testset "gradients, rank-broadcast" begin
    # a weight applied to batched activations: its gradient sums over the batch
    g = Graph(); w = tensor(g, [3, 4]); x = tensor(g, [4, 5, 2])
    loss = sum(sum(sum(matmul(w, x), 1), 1), 1)
    mark_trainable!(w); gr = backward(loss)
    xv = randn(Float32, 4, 5, 2)
    @test execute(g, gr[w.id].id, Dict(w.id => randn(Float32, 3, 4), x.id => xv), CPUDevice()) ≈
          repeat(sum(xv; dims=(2, 3))[:, 1, 1]', 3, 1) rtol=1e-4
    # a lower-rank operand of an elementwise op broadcasts along trailing dims
    g = Graph(); b = tensor(g, [3]); y = tensor(g, [3, 4])
    loss = sum(sum((y + b) * (y + b), 2), 1)
    mark_trainable!(b); gr = backward(loss)
    yv, bv = randn(Float32, 3, 4), randn(Float32, 3)
    @test execute(g, gr[b.id].id, Dict(b.id => bv, y.id => yv), CPUDevice()) ≈ vec(sum(2 .* (yv .+ bv); dims=2)) rtol=1e-4
end
