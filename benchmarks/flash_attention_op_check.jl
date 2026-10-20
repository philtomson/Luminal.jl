using Luminal, AMDGPU   # Usage: julia --project=. benchmarks/flash_attention_op_check.jl (GPU)
const DEV = get_device()
function run_fa(dev, ins, causal)
    D, S, H, B = size(ins[1])
    g = Graph(); ts = [Luminal.tensor(g, [D, S, H, B]) for _ in 1:3]
    o = Luminal.flash_attention(ts...; causal=causal)
    inputs = Dict{Int,Any}(t.id => Luminal.to_device(a, dev) for (t, a) in zip(ts, ins))
    Float64.(Array(compile(g; device=dev, retain=[o.id])(inputs; device=dev)[o.id]))
end
function ref(q, k, v, causal)
    D, S, H, B = size(q); out = zeros(Float64, D, S, H, B)
    for b in 1:B, h in 1:H
        s = (Float64.(k[:, :, h, b])' * Float64.(q[:, :, h, b])) ./ sqrt(D)   # (Sk, Sq)
        causal && (for i in 1:S, j in 1:S; j > i && (s[j, i] = -Inf); end)
        s .-= maximum(s, dims=1); p = exp.(s); p ./= sum(p, dims=1)
        out[:, :, h, b] = Float64.(v[:, :, h, b]) * p
    end
    out
end
rel(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))
for (D, S, H, B) in ((32, 8, 2, 1), (64, 16, 2, 1), (64, 64, 2, 1), (64, 256, 2, 1))
    ins = (randn(Float32, D, S, H, B), randn(Float32, D, S, H, B), randn(Float32, D, S, H, B))
    for causal in (false, true)
        e = ref(ins..., causal)
        println((D, S, H, B), " causal=", causal, ": CPU op vs ref ", round(rel(run_fa(CPUDevice(), ins, causal), e), sigdigits=2), "   GPU op vs ref ", round(rel(run_fa(DEV, ins, causal), e), sigdigits=2))
    end
end
