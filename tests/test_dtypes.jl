# Element types: inference and strict checking, every dtype through the
# interpreter and compile() (fused, CPU and GPU), casts, trunc_cast, integer
# division, Float64 constants, integer coordinates, gradients, e-graph search.
using Test
using Luminal
using Luminal: compile

const BF16 = Core.BFloat16
# BFloat16 needs `julia -C native,-avx512bf16` on AVX512-BF16 CPUs (a Julia/LLVM bug)
const HAS_BF16 = Luminal._bf16_host_ok()
HAS_BF16 || @info "BFloat16 cases skipped: start Julia with -C native,-avx512bf16 to run them"
devices = Any[CPUDevice()]
Luminal.get_device() isa CPUDevice || push!(devices, Luminal.get_device())

# run on the interpreter and on each device's compiled graph
function run_all(g, outs, inputs)
    ids = [o.id for o in outs]
    rs = Any[execute(g, ids, inputs, CPUDevice())]
    for dev in devices
        cg = compile(g; device=dev, retain=ids)
        r = cg(inputs; device=dev)
        push!(rs, Dict(id => Array(r[id]) for id in ids))
    end
    return rs
end

@testset "inference and strictness" begin
    g = Graph()
    f = tensor(g, [4]); i = tensor(g, [4]; dtype=Int32); b = f < 0
    @test dtype(f) === Float32 && dtype(i) === Int32 && dtype(b) === Bool
    @test dtype(i + 1) === Int32 && dtype(f * 2) === Float32          # literals take the tensor's type
    @test dtype(i < 2) === Bool && dtype(!(f < 0) & (f > 1)) === Bool
    @test dtype(select(b, i, 0)) === Int32
    @test dtype(cast(i, Float32) + f) === Float32
    @test_throws ArgumentError i + f                                   # strict: no implicit promotion
    @test_throws ArgumentError f * b                                   # masks are cast explicitly
    @test_throws ArgumentError i / i                                   # Div is float-only (trunc_div)
    @test_throws ArgumentError i / 2
    @test_throws ArgumentError exp(i)
    @test_throws ArgumentError cast(f, Int32)                          # float -> int: trunc_cast
    @test_throws ArgumentError trunc_cast(i, Int8)
    @test_throws ArgumentError trunc_div(f, f)
    @test_throws InexactError i + 0.5                                  # the literal can't be Int32
    @test_throws ArgumentError !f
    @test_throws ArgumentError tensor(g, [2]; dtype=UInt16)
    HAS_BF16 || @test_throws ArgumentError tensor(g, [2]; dtype=BF16)   # refused with the workaround, not a hang
    @test dtype(cast(f, Bool)) === Bool                                # the != 0 projection
    @test dtype(iota(g, [3], k -> k; dtype=Int64)) === Int64
    @test dtype(constant(g, 0.1, Float64)) === Float64
    # CSE keeps equal values of different dtypes apart
    @test constant(g, 1, Int32).id != constant(g, 1, Float32).id
end

@testset "every dtype, interpreter and compiled" begin
    for T in (Float32, Float64, Float16, (HAS_BF16 ? (BF16,) : ())..., Int8, Int32, Int64)
        g = Graph()
        a = tensor(g, [6, 5]; dtype=T); b = tensor(g, [6, 5]; dtype=T)
        outs = [a + b, a * b, maximum(a, b), relu(a), select(a < b, a, b * 2), sum(a, 1), max_reduce(b, 2),
                a < b, (a <= b) | (a == b)]
        if T <: AbstractFloat
            append!(outs, [a / (abs(b) + 1), exp(a / 4), floor(a * 3)])
        end
        rng = T <: Integer ? (T === Int8 ? (-60:60) : (-1000:1000)) : nothing
        av = rng === nothing ? T.(randn(Float32, 6, 5)) : T.(rand(rng, 6, 5))
        bv = rng === nothing ? T.(randn(Float32, 6, 5)) : T.(rand(rng, 6, 5))
        refs = Any[av .+ bv, av .* bv, max.(av, bv), max.(av, zero(T)), ifelse.(av .< bv, av, bv .* T(2)),
                   vec(reduce(+, av; dims=1)), vec(maximum(bv; dims=2)), av .< bv, (av .<= bv) .| (av .== bv)]
        if T <: AbstractFloat
            append!(refs, [av ./ (abs.(bv) .+ one(T)), exp.(av ./ T(4)), floor.(av .* T(3))])
        end
        for r in run_all(g, outs, Dict(a.id => av, b.id => bv)), (k, o) in enumerate(outs)
            @test eltype(r[o.id]) === eltype(refs[k])
            if T <: Integer || eltype(refs[k]) === Bool
                @test r[o.id] == refs[k]                                  # exact, Int8 wrapping included
            else
                @test Float64.(r[o.id]) ≈ Float64.(refs[k]) rtol=(T in (Float16, BF16) ? 3e-2 : 1e-5)
            end
        end
    end
end

@testset "matmul per float dtype" begin
    for T in (Float32, Float64, Float16, (HAS_BF16 ? (BF16,) : ())...)
        g = Graph()
        w = tensor(g, [8, 16]; dtype=T); x = tensor(g, [16, 5]; dtype=T)
        a = tensor(g, [4, 16, 3, 2]; dtype=T); b = tensor(g, [16, 6, 3, 2]; dtype=T)
        y = matmul(w, x); z = matmul(a, b)
        wv, xv, av, bv = T.(randn(Float32, 8, 16)), T.(randn(Float32, 16, 5)), T.(randn(Float32, 4, 16, 3, 2)), T.(randn(Float32, 16, 6, 3, 2))
        yr = Float64.(wv) * Float64.(xv)
        zr = [sum(Float64(av[i, k, h, n]) * Float64(bv[k, j, h, n]) for k in 1:16) for i in 1:4, j in 1:6, h in 1:3, n in 1:2]
        tol = T === Float64 ? 1e-12 : T === Float32 ? 1e-5 : 0.05
        for r in run_all(g, [y, z], Dict(w.id => wv, x.id => xv, a.id => av, b.id => bv))
            @test eltype(r[y.id]) === T && eltype(r[z.id]) === T
            @test maximum(abs.(Float64.(r[y.id]) .- yr)) / maximum(abs.(yr)) < tol
            @test maximum(abs.(Float64.(r[z.id]) .- zr)) / maximum(abs.(zr)) < tol
        end
    end
end

@testset "casts" begin
    g = Graph()
    f = tensor(g, [5]); i = tensor(g, [5]; dtype=Int32); b = tensor(g, [5]; dtype=Bool)
    outs = [cast(f, Float16), cast(f, HAS_BF16 ? BF16 : Float16), cast(f, Float64), cast(i, Int8), cast(i, Int64), cast(i, Float32),
            cast(b, Float32), cast(b, Int32), cast(f, Bool), cast(i, Bool),
            cast(cast(i, Int8), Float32) * f]                          # fused chain across dtypes
    fv = Float32[1.00048828125, -2.5, 0, 3.14159, 1f6]; iv = Int32[300, -129, 0, 70000, -1]; bv = Bool[1, 0, 1, 1, 0]
    refs = [Float16.(fv), HAS_BF16 ? BF16.(fv) : Float16.(fv), Float64.(fv), iv .% Int8, Int64.(iv), Float32.(iv),
            Float32.(bv), Int32.(bv), fv .!= 0, iv .!= 0, Float32.(iv .% Int8) .* fv]
    for r in run_all(g, outs, Dict(f.id => fv, i.id => iv, b.id => bv)), (k, o) in enumerate(outs)
        @test eltype(r[o.id]) === eltype(refs[k])
        @test isequal(r[o.id], refs[k])
    end
end

@testset "trunc_cast, trunc_div, trunc_rem" begin
    g = Graph()
    f = tensor(g, [6]); d = tensor(g, [6]; dtype=Float64)
    t32 = trunc_cast(f, Int32); t8 = trunc_cast(d, Int8); t64 = trunc_cast(f, Int64)
    a = tensor(g, [6]; dtype=Int32); b = tensor(g, [6]; dtype=Int32)
    q = trunc_div(a, b); m = trunc_rem(a, b); q2 = trunc_div(a, 3)
    fv = Float32[2.7, -2.7, 0.5, -0.5, 1e9, -3]; dv = [127.9, -128.9, 0.1, -0.1, 5, -5]
    av = Int32[7, -7, 7, -7, typemin(Int32), 0]; bv = Int32[2, 2, -2, -2, -1, 5]
    inputs = Dict(f.id => fv, d.id => dv, a.id => av, b.id => bv)
    for r in run_all(g, [t32, t8, t64, q, m, q2], inputs)
        @test r[t32.id] == Int32[2, -2, 0, 0, 1000000000, -3]
        @test r[t8.id] == Int8[127, -128, 0, 0, 5, -5]
        @test eltype(r[t64.id]) === Int64
        @test r[q.id] == Int32[3, -3, -3, 3, typemin(Int32), 0]            # toward zero; typemin ÷ -1 wraps
        @test r[m.id] == Int32[1, -1, 1, -1, 0, 0]                          # sign of the dividend
        @test r[q2.id] == div.(av, Int32(3))
    end
    # refused loudly: NaN, Inf, out of range, zero divisor -- interpreter and compiled
    for (bad, node) in ((Dict(f.id => Float32[NaN, 0, 0, 0, 0, 0]), t32), (Dict(f.id => Float32[Inf, 0, 0, 0, 0, 0]), t32),
                        (Dict(d.id => [128.0, 0, 0, 0, 0, 0]), t8), (Dict(f.id => Float32[3f9, 0, 0, 0, 0, 0]), t32),
                        (Dict(b.id => Int32[1, 0, 1, 1, 1, 1]), q))
        inp = merge(inputs, bad)
        @test_throws Exception execute(g, node.id, inp, CPUDevice())
        for dev in devices
            @test_throws Exception compile(g; device=dev, retain=[node.id])(inp; device=dev)
        end
    end
end

@testset "Float64 constants, integer coordinates" begin
    g = Graph(); x = tensor(g, [3]; dtype=Float64)
    y = x + constant(g, 0.1, Float64) + 1e-300                         # exact double constants
    xv = [1.0, 2.0, 1e-300]
    for r in run_all(g, [y], Dict(x.id => xv))
        @test r[y.id] == xv .+ 0.1 .+ 1e-300
    end
    g = Graph(); A = tensor(g, [4, 3]); i = tensor(g, [5]; dtype=Int32)
    j = iota(g, [5], k -> k % 3; dtype=Int32)
    y = gather(A, [i, j]); s = scatter(A, y * 2, [i, j]; mode=:add)
    Av = randn(Float32, 4, 3); iv = Int32[0, 3, 1, 7, 2]
    ref = [0 <= iv[k] < 4 ? Av[iv[k] + 1, (k - 1) % 3 + 1] : 0f0 for k in 1:5]
    for r in run_all(g, [y, s], Dict(A.id => Av, i.id => iv))
        @test r[y.id] == ref
        sref = copy(Av); for k in 1:5; 0 <= iv[k] < 4 && (sref[iv[k] + 1, (k - 1) % 3 + 1] += 2ref[k]); end
        @test r[s.id] ≈ sref
    end
end

@testset "gradients through casts" begin
    # a Float64 parameter used in Float32: the gradient comes back as Float64
    g = Graph(); w = tensor(g, [4]; dtype=Float64); x = tensor(g, [4])
    loss = sum(cast(w, Float32) * x * x, 1)
    mark_trainable!(w); gr = backward(loss)
    @test dtype(gr[w.id]) === Float64
    xv = Float32[1, 2, 3, 4]
    @test execute(g, gr[w.id].id, Dict(w.id => ones(4), x.id => xv), CPUDevice()) ≈ Float64.(xv .^ 2)
    @test_throws ArgumentError backward(sum(cast(w, Float32) < x, 1) |> t -> cast(t, Int32))   # integer loss
end

@testset "e-graph search" begin
    g = Graph(); x = tensor(g, [16, 4]); w = tensor(g, [8, 16]); k = tensor(g, [8, 4]; dtype=Int32)
    h = matmul(w, x)
    y = select(trunc_rem(k, 2) == 0, h * 2, cast(k, Float32))
    xv, wv, kv = randn(Float32, 16, 4), randn(Float32, 8, 16), Int32.(rand(-9:9, 8, 4))
    inputs = Dict(x.id => xv, w.id => wv, k.id => kv)
    ref = execute(g, y.id, inputs, CPUDevice())
    for dev in devices
        ex = compile(g; device=dev, retain=[y.id], search=:static)
        @test Array(ex(inputs; device=dev)[y.id]) ≈ ref rtol=1e-4
    end
end
