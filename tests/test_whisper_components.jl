using Test
using Luminal
using CUDA

@testset "Unfold 1D CPU" begin
    cx = Graph()
    a = tensor(cx, [1, 2, 7])
    b = unfold(a, [3], [2], [1])
    
    a_data = Float32[1 2 3 4 5 6 7; 8 9 10 11 12 13 14]
    a_data = Base.reshape(a_data, 1, 2, 7)
    
    res = execute(cx, b.id, Dict(a.id => a_data), CPUDevice())
    
    @test size(res) == (1, 2, 3, 3)
    @test res[1, 1, 1, :] == Float32[1, 2, 3]
    @test res[1, 1, 2, :] == Float32[3, 4, 5]
    @test res[1, 1, 3, :] == Float32[5, 6, 7]
    
    @test res[1, 2, 1, :] == Float32[8, 9, 10]
    @test res[1, 2, 2, :] == Float32[10, 11, 12]
    @test res[1, 2, 3, :] == Float32[12, 13, 14]
end

if CUDA.functional()
    @testset "Unfold 1D CUDA" begin
        cx = Graph()
        a = tensor(cx, [1, 2, 7])
        b = unfold(a, [3], [2], [1])
        
        a_data = Float32[1 2 3 4 5 6 7; 8 9 10 11 12 13 14]
        a_data = Base.reshape(a_data, 1, 2, 7)
        
        res = execute(cx, b.id, Dict(a.id => a_data), CUDADevice())
        
        @test size(res) == (1, 2, 3, 3)
        @test res[1, 1, 1, :] == Float32[1, 2, 3]
        @test res[1, 1, 2, :] == Float32[3, 4, 5]
        @test res[1, 1, 3, :] == Float32[5, 6, 7]
        
        @test res[1, 2, 1, :] == Float32[8, 9, 10]
        @test res[1, 2, 2, :] == Float32[10, 11, 12]
        @test res[1, 2, 3, :] == Float32[12, 13, 14]
    end
end

# Direct convolution: x (C_in, L), w (C_out, C_in, K), zero padding.
function conv1d_ref(x, w, b; stride=1, padding=0)
    cout, cin, k = size(w)
    xp = hcat(zeros(Float32, cin, padding), x, zeros(Float32, cin, padding))
    lout = (size(xp, 2) - k) ÷ stride + 1
    return [sum(w[o, c, j] * xp[c, (t - 1) * stride + j] for c in 1:cin, j in 1:k) + b[o]
            for o in 1:cout, t in 1:lout]
end

@testset "Conv1D" begin
    devices = Any[CPUDevice()]
    Luminal.get_device() isa CPUDevice || push!(devices, Luminal.get_device())
    # Odd and even lengths, stride 1 and 2 (the stride-2 odd case over-reads the padding).
    for dev in devices, (L, stride) in [(7, 2), (8, 2), (7, 1)]
        cx = Graph()
        a = tensor(cx, [2, L, 1])                           # (channels, length, batch)
        conv = NN.Conv1D(2, 3, 3, cx; stride=stride, padding=1, bias=true)
        x = Float32.(randn(2, L, 1))
        w = Float32.(randn(3, 2, 3))
        b = Float32[0.5, -0.25, 1.0]
        out = conv(a)
        res = Array(execute(cx, out.id, Dict(a.id => x, conv.weight.id => w, conv.bias.id => b), dev))
        ref = conv1d_ref(x[:, :, 1], w, b; stride=stride, padding=1)
        @test size(res) == (size(ref)..., 1)
        @test res[:, :, 1] ≈ ref rtol=1e-5
    end
end

@testset "exact GELU" begin
    cx = Graph()
    xs = Float32[-3, -1, -0.5, 0, 0.5, 1, 3]
    a = tensor(cx, xs)
    out = Luminal.gelu(a; approximate=false)
    res = execute(cx, out.id, Dict(a.id => xs), CPUDevice())
    # 0.5x(1 + erf(x/√2))
    ref = [-0.004049695, -0.15865525, -0.15426877, 0.0, 0.34573123, 0.8413447, 2.9959502]
    @test res ≈ ref atol=1e-6
end

@testset "AudioEncoder" begin
    cx = Graph()
    encoder = NN.AudioEncoder(cx)
    a = tensor(cx, [80, 50, 1]) # (mels, frames, batch)

    inputs = Dict{Int, AbstractArray}(a.id => Float32.(randn(80, 50, 1)))
    for (id, node) in enumerate(cx.nodes)
        if node.op isa Luminal.Function && node.op.name == "InputTensor" && !haskey(inputs, id)
            inputs[id] = 0.05f0 .* randn(Float32, Tuple(Luminal.realized_dims(cx.shapes[id])))
        end
    end
    out = encoder(a)
    res = execute(cx, out.id, inputs, CPUDevice())
    @test size(res) == (NN.D_MODEL, 25, 1)
    @test all(isfinite, res)
end

@testset "TextDecoder" begin
    cx = Graph()
    decoder = NN.TextDecoder(cx)
    enc_out = tensor(cx, [NN.D_MODEL, 25, 1])
    input = tensor(cx, [10, 1]) # (tokens, batch)

    inputs = Dict{Int, AbstractArray}(enc_out.id => Float32.(randn(NN.D_MODEL, 25, 1)),
                                      input.id => Float32.(ones(10, 1)))
    for (id, node) in enumerate(cx.nodes)
        if node.op isa Luminal.Function && node.op.name == "InputTensor" && !haskey(inputs, id)
            inputs[id] = 0.05f0 .* randn(Float32, Tuple(Luminal.realized_dims(cx.shapes[id])))
        end
    end
    out = decoder(enc_out, input)
    res = execute(cx, out.id, inputs, CPUDevice())
    @test size(res) == (NN.VOCAB_SIZE, 10, 1)
    @test all(isfinite, res)
end
