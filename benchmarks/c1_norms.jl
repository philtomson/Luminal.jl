# c1_norms.jl -- LayerNormOp and GeluOp: are the fused ops (recognised from primitives by the fusion rules) right and faster?
#
# For each shape the same primitive graph is compiled as is (search=:none) and through the static e-graph extraction
# (search=:static, which recognises the fused op). Both run on the GPU in interleaved rounds (median ms); the fused result is
# compared with a Float64 CPU reference, which also checks the LayerNormOp kernel at wide rows.
#
# Usage: julia --project=. benchmarks/c1_norms.jl      (stop the model server first: this uses the GPU)

using Luminal
using AMDGPU, Printf, Statistics

const DEV = get_device()

timed(run, n) = (t = time(); for _ in 1:n; run(); end; (time() - t) / n)
nsteps(run; budget=0.2) = (run(); run(); t = (t0 = time(); run(); time() - t0); clamp(round(Int, budget / max(t, 1e-6)), 1, 200))
function interleaved(runs, rounds)
    ns = [nsteps(r) for r in runs]
    ts = [Float64[] for _ in runs]
    for _ in 1:rounds, (i, r) in enumerate(runs)
        push!(ts[i], timed(r, ns[i]))
    end
    return [1e3 * median(t) for t in ts]
end

function layernorm_ref(x, w, b, eps)
    x = Float64.(x); H = size(x, 1)
    mu = sum(x, dims=1) ./ H
    v = sum((x .- mu) .^ 2, dims=1) ./ H
    return (x .- mu) ./ sqrt.(v .+ eps) .* Float64.(w) .+ Float64.(b)
end
gelu_ref(x) = (x = Float64.(x); 0.5 .* x .* (1 .+ tanh.(sqrt(2 / pi) .* (x .+ 0.044715 .* x .^ 3))))

# `build(g, dims)` -> (output tensor, input tensors); `make(dims)` -> input arrays; `ref(inputs...)`
function bench(name, dims, build, make, ref)
    ins = make(dims)
    res = []
    runs = Base.Function[]; outs = Any[]
    for search in (:none, :static)
        g = Graph()
        out, ts = build(g, dims)
        inputs = Dict{Int,Any}(t.id => Luminal.to_device(a, DEV) for (t, a) in zip(ts, ins))
        exec = search === :none ? compile(g; device=DEV, retain=[out.id], free_intermediates=false) :
               compile(g; device=DEV, retain=[out.id], free_intermediates=false, search=:static, precision=false, search_inputs=inputs)
        r = () -> exec(inputs; device=DEV)
        push!(outs, Array(r()[out.id]))
        push!(runs, () -> (r(); AMDGPU.synchronize()))
        push!(res, length(g.nodes))
    end
    t = interleaved(runs, 7)
    expected = ref(ins...)
    err(o) = maximum(abs.(Float64.(o) .- expected)) / max(maximum(abs.(expected)), 1e-6)
    @printf("%-10s %-14s  prim %8.3f ms   fused %8.3f ms   (%.2fx)   max rel err vs Float64: prim %.1e, fused %.1e\n",
            name, string(dims), t[1], t[2], t[1] / t[2], err(outs[1]), err(outs[2]))
end

for (H, N) in ((2048, 1), (2048, 64), (2048, 512), (4096, 64), (4096, 512), (768, 512), (5120, 16))
    bench("layernorm", (H, N),
          (g, d) -> (x = Luminal.tensor(g, [d[1], d[2]]); w = Luminal.tensor(g, [d[1]]); b = Luminal.tensor(g, [d[1]]);
                     (Luminal.layer_norm(x, 1, 1f-5) * w + b, [x, w, b])),
          d -> (randn(Float32, d...) .* 3f0 .+ 1f0, randn(Float32, d[1]), randn(Float32, d[1])),
          (x, w, b) -> layernorm_ref(x, w, b, 1e-5))
end
for n in ((5632, 1), (5632, 64), (5632, 512), (10240, 512), (10240, 4096))
    bench("gelu", n,
          (g, d) -> (x = Luminal.tensor(g, [d[1], d[2]]); (Luminal.gelu(x), [x])),
          d -> (randn(Float32, d...) .* 2f0,),
          x -> gelu_ref(x))
end
