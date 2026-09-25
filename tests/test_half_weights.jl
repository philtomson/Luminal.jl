using Test
using Luminal
using Luminal.NN

# Float16 matmul weights (compile(...; weight_dtype=Float16)) must match Float32.
dev = get_device()
if dev isa Luminal.AbstractGPUDevice
    @testset "HalfWeight matmul" begin
        W = Float16.(randn(Float32, 96, 64))      # exactly representable in Float16
        Wd = Luminal.to_device(Float32.(W), dev)
        hw = Luminal.half_weight(Wd)
        @test size(hw) == (96, 64)
        @test Luminal.half_weight(Wd) === hw     # converted once per source array
        for xsz in ((64, 1), (64, 7), (64, 5, 2))
            x = randn(Float32, xsz...)
            out = Luminal.zero_tensor(dev, Float32, 96, xsz[2:end]...)
            Luminal.batch_matmul!(out, hw, Luminal.to_device(x, dev))
            ref = Float32.(W) * Base.reshape(x, 64, :)
            @test Array(out) ≈ Base.reshape(ref, 96, xsz[2:end]...) rtol = 1e-5
        end
    end

    @testset "Linear layer with Float16 weights" begin
        g = Graph()
        lin = NN.Linear(64, 96, g; bias=false)
        x = Luminal.tensor(g, [64, 3, 1])
        y = lin(x)
        Wv = Float32.(Float16.(randn(Float32, 96, 64)))
        g.tensors[(lin.weight.id, 1)] = Luminal.to_device(Wv, dev)
        xv = randn(Float32, 64, 3, 1)
        outs = Dict()
        for wdt in (Float32, Float16)
            ex = compile(g; device=dev, retain=[y.id], weight_dtype=wdt)
            outs[wdt] = Array(ex(Dict{Int,Any}(x.id => xv); device=dev)[y.id])
            @test (ex.results[lin.weight.id] isa HalfWeight) == (wdt === Float16)
        end
        @test outs[Float16] ≈ outs[Float32] rtol = 1e-5
    end
else
    @info "No GPU; HalfWeight is GPU-only, skipping"
end

if get_device() isa Luminal.AbstractGPUDevice
    @testset "HalfWeight cache never returns another array's weight" begin
        # Regression: a freed array's objectid can be reused by a new array while
        # its cache entry is still present (seen in a measured search: a merged
        # gate/up weight got the merged q/k/v HalfWeight). Plant such a stale
        # entry and check it is not returned.
        dev = get_device()
        other = Luminal.to_device(randn(Float32, 40, 64), dev)
        W = Luminal.to_device(randn(Float32, 96, 64), dev)
        lock(Luminal._HALF_WEIGHTS_LOCK) do
            Luminal._HALF_WEIGHTS[objectid(W)] = (WeakRef(other), Luminal.HalfWeight(other))
        end
        hw = Luminal.half_weight(W)
        @test size(hw) == (96, 64)
        @test Array(Float32.(hw.t)) ≈ permutedims(Array(W)) rtol = 1e-3
        @test Luminal.half_weight(W) === hw          # and the entry is now W's
    end
end

if get_device() isa Luminal.AbstractGPUDevice
    @testset "int8 weights (QuantWeight / MatMulQ8)" begin
        dev = get_device()
        W = randn(Float32, 96, 64)
        qw = Luminal.QuantWeight(Luminal.to_device(W, dev))
        @test size(qw) == (96, 64)
        # dequantized weights are within half a quantization step of the original
        scale = Array(qw.scale)
        deq = Float32.(permutedims(Array(qw.q))) .* scale
        @test all(abs.(deq .- W) .<= scale ./ 2 .+ 1f-6)

        x = randn(Float32, 64, 5)
        for grp in (64, 128, 256)
            out = Luminal.zero_tensor(dev, Float32, 96, 5)
            Luminal.execute_op!(out, Luminal.MatMulQ8(grp), qw, Luminal.to_device(x, dev))
            @test Array(out) ≈ deq * x rtol = 1e-5          # exact w.r.t. the int8 weights
        end

        # compile(...; weight_dtype=Int8) quantizes plain MatMul weights
        g = Graph()
        lin = NN.Linear(64, 96, g; bias=false)
        xin = Luminal.tensor(g, [64, 3, 1])
        y = lin(xin)
        g.tensors[(lin.weight.id, 1)] = Luminal.to_device(W, dev)
        ex = compile(g; device=dev, retain=[y.id], weight_dtype=Int8)
        @test ex.results[lin.weight.id] isa Luminal.QuantWeight
        xv = randn(Float32, 64, 3, 1)
        @test Array(ex(Dict{Int,Any}(xin.id => xv); device=dev)[y.id]) ≈
              Base.reshape(deq * Base.reshape(xv, 64, 3), 96, 3, 1) rtol = 1e-5
    end
end
