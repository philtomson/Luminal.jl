using Test
using Luminal

# flash_attention in the (HeadDim, Seq, Head, Batch) layout, against a plain softmax
# attention reference, causal and not, on CPU and GPU.
function reference_attention(q, k, v, scale, causal)
    D, N, H, B = size(q)
    out = similar(q)
    for b in 1:B, h in 1:H, i in 1:N
        js = causal ? (1:i) : (1:N)
        s = [scale * sum(q[:, i, h, b] .* k[:, j, h, b]) for j in js]
        p = exp.(s .- maximum(s)); p ./= sum(p)
        out[:, i, h, b] = sum(p[t] .* v[:, j, h, b] for (t, j) in enumerate(js))
    end
    out
end

@testset "Flash Attention" begin
    D, S, H, B = 16, 8, 2, 2
    q_val, k_val, v_val = randn(Float32, D, S, H, B), randn(Float32, D, S, H, B), randn(Float32, D, S, H, B)
    for dev in unique([CPUDevice(), Luminal.get_device()]), causal in (false, true)
        g = Graph()
        q, k, v = tensor(g, [D, S, H, B]), tensor(g, [D, S, H, B]), tensor(g, [D, S, H, B])
        out = flash_attention(q, k, v; causal=causal)
        res = Luminal.execute(g, out.id, Dict(q.id => q_val, k.id => k_val, v.id => v_val), dev)
        @test Array(res) ≈ reference_attention(q_val, k_val, v_val, 1f0 / sqrt(Float32(D)), causal) rtol = 1e-4
    end
end
