using Test
using Luminal

# Exact scans (cumsum, cumprod, cummax) and prod: left folds in the element type, along any axis.
run1(g, out, inputs, dev=CPUDevice()) = execute(g, [out.id], inputs, dev)[out.id]

@testset "scans: values on every axis" begin
    xv = Float32[3 -1 2 0.5; -2 4 -3 1; 1 1 -1 2]          # (3, 4)
    for (f, ref) in ((cumsum, cumsum), (cumprod, cumprod), (cummax, (a; dims) -> accumulate(max, a; dims)))
        for dim in 1:2
            g = Graph(); x = tensor(g, [3, 4])
            @test run1(g, f(x, dim), Dict(x.id => xv)) ≈ ref(xv; dims=dim)
        end
    end
    x3 = Base.reshape(Float32.(collect(1:24)) .- 12, 2, 3, 4)
    g = Graph(); x = tensor(g, [2, 3, 4])
    @test run1(g, cumsum(x, 2), Dict(x.id => x3)) ≈ cumsum(x3; dims=2)
    g = Graph(); x = tensor(g, [2, 3, 4])
    @test run1(g, cumprod(x, 3), Dict(x.id => x3)) ≈ cumprod(x3; dims=3)
end

@testset "scans: negative and zero elements, integers, prod" begin
    # the old prod was exp(sum(log x)): NaN for negatives, wrong for zeros
    xv = Float32[-2 3 0 5; 2 -1 -4 0.5]
    g = Graph(); x = tensor(g, [2, 4])
    @test run1(g, prod(x, 2), Dict(x.id => xv)) ≈ vec(prod(xv; dims=2))
    g = Graph(); x = tensor(g, [2, 4])
    @test run1(g, prod(x, 1), Dict(x.id => xv)) ≈ vec(prod(xv; dims=1))
    g = Graph(); x = tensor(g, [5]; dtype=Int32)
    iv = Int32[3, -2, 4, -1, 2]
    @test run1(g, cumprod(x, 1), Dict(x.id => iv)) == cumprod(iv)
    g = Graph(); x = tensor(g, [5]; dtype=Int32)
    @test run1(g, cummax(x, 1), Dict(x.id => iv)) == accumulate(max, iv)
    g = Graph(); x = tensor(g, [3, 2])
    @test_throws ArgumentError cumsum(x, 3)
end

@testset "scans: gradients" begin
    fd(f, x; h=1f-2) = [(f(setindex!(copy(x), x[i] + h, i)) - f(setindex!(copy(x), x[i] - h, i))) / (2h) for i in eachindex(x)]
    w = Float32[0.5, -1, 2, 1.5]
    xv = Float32[1.5, -0.5, 2, 0.75]
    g = Graph(); x = tensor(g, [4]); wt = tensor(g, [4])
    loss = sum(cumsum(x, 1) * wt, 1)
    mark_trainable!(x); gr = backward(loss)
    res = execute(g, [gr[x.id].id], Dict(x.id => xv, wt.id => w), CPUDevice())
    @test res[gr[x.id].id] ≈ [sum(w[i:end]) for i in 1:4]
    g = Graph(); x = tensor(g, [4]); wt = tensor(g, [4])
    loss = sum(cumprod(x, 1) * wt, 1)
    mark_trainable!(x); gr = backward(loss)
    res = execute(g, [gr[x.id].id], Dict(x.id => xv, wt.id => w), CPUDevice())
    @test res[gr[x.id].id] ≈ fd(v -> sum(cumprod(v) .* w), xv) atol=2e-2
end

if get(ENV, "LUMINAL_TEST_GPU", "0") == "1"
    @testset "scans: GPU equals CPU" begin
        dev = Luminal.get_device()
        xv = randn(Float32, 33, 129, 5)
        for (f, ref) in ((cumsum, cumsum), (cumprod, cumprod), (cummax, (a; dims) -> accumulate(max, a; dims)))
            for dim in 1:3
                g = Graph(); x = tensor(g, [33, 129, 5])
                a = run1(g, f(x, dim), Dict(x.id => xv .* (f === cumprod ? 0.1f0 : 1f0)), dev)
                @test Array(a) ≈ ref(xv .* (f === cumprod ? 0.1f0 : 1f0); dims=dim) rtol=1e-4
            end
        end
    end
end
