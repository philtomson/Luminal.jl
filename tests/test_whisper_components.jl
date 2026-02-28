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

@testset "Conv1D CPU" begin
    cx = Graph()
    a = tensor(cx, [1, 2, 7])
    conv = NN.Conv1D(2, 3, 3, cx; stride=2, padding=1, bias=true)

    a_data = Base.reshape(Float32[1 2 3 4 5 6 7; 8 9 10 11 12 13 14], 1, 2, 7)
    w_data = ones(Float32, 3, 6)
    b_data = Float32[0.5, 0.5, 0.5]
    
    out = conv(a)
    
    res = execute(cx, out.id, Dict(a.id => a_data, conv.weight.id => w_data, conv.bias.id => b_data), CPUDevice())
    
    @test size(res) == (1, 3, 4)
end

@testset "AudioEncoder" begin
    cx = Graph()
    encoder = NN.AudioEncoder(cx)
    a = tensor(cx, [1, 80, 50]) # Batch, Channels (80 bins), Seq Length (50 frames)
    out = encoder(a)
    
    # Run the graph
    a_data = Float32.(randn(1, 80, 50))
    
    # Collect all uninitialized parameters for dummy execution
    inputs = Dict{Int, AbstractArray}()
    inputs[a.id] = a_data
    
    for (i, node) in enumerate(cx.nodes)
        id = i
        if node.op isa Luminal.Function && node.op.name == "InputTensor" && !haskey(inputs, id)
            dims = Luminal.realized_dims(cx.shapes[id])
            has_syms = !all(x -> x isa Int, dims)
            if !has_syms
                inputs[id] = randn(Float32, Tuple(dims))
            end
        end
    end
    
    res = execute(cx, out.id, inputs, CPUDevice())
    
    @test size(res) == (1, 25, 384)
end

@testset "TextDecoder" begin
    cx = Graph()
    decoder = NN.TextDecoder(cx)
    enc_out = tensor(cx, [1, 25, 384])
    input = tensor(cx, [1, 10]) # 10 tokens
    out = decoder(enc_out, input)
    
    # Run the graph
    inputs = Dict{Int, AbstractArray}()
    inputs[enc_out.id] = Float32.(randn(1, 25, 384))
    inputs[input.id] = Float32.(ones(1, 10)) # dummy tokens
    
    for (i, node) in enumerate(cx.nodes)
        id = i
        if node.op isa Luminal.Function && node.op.name == "InputTensor" && !haskey(inputs, id)
            dims = Luminal.realized_dims(cx.shapes[id])
            has_syms = !all(x -> x isa Int, dims)
            if !has_syms
                inputs[id] = randn(Float32, Tuple(dims))
            end
        end
    end
    
    res = execute(cx, out.id, inputs, CPUDevice())
    
    @test size(res) == (1, 10, 51864)
end
