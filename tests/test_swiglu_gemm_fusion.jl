using Test
using Luminal
using AMDGPU

# gate / up int4 GEMMs + SwiGLUOp + down GEMM: the compiler folds the first three into one kernel (wmma_q4_swiglu!) when the shapes
# fit (M % 64 == 0, N % 256 == 0, N >= 384); the result matches the unfused graph (LUMINAL_FUSED_SWIGLU=0).
if Luminal.get_device() isa Luminal.AMDDevice
    @testset "SwiGLU folded into the gate/up GEMMs" begin
        dev = Luminal.get_device()
        for (M, K, N, K2) in ((256, 256, 512, 128), (192, 128, 768, 96), (128, 96, 384, 64))
            g = Luminal.Graph()
            x = Luminal.tensor(g, [K, N, 1])
            Wg = Luminal.tensor(g, [M, K]); Wu = Luminal.tensor(g, [M, K]); Wd = Luminal.tensor(g, [K2, M])
            gate = Luminal.matmul(Wg, x); up = Luminal.matmul(Wu, x)
            h = Luminal.add_op!(g, Luminal.SwiGLUOp(), [(gate.id, 0, gate.shape), (up.id, 0, up.shape)], Luminal.ShapeTracker(Luminal.realized_dims(gate.shape)))
            y = Luminal.matmul(Wd, h)
            for (t, w) in ((Wg, 0.05f0), (Wu, 0.05f0), (Wd, 0.05f0))
                g.tensors[(t.id, 1)] = Luminal.to_device(randn(Float32, Luminal.realized_dims(t.shape)...) .* w, dev)
            end
            xv = randn(Float32, K, N, 1)
            run(fused) = withenv("LUMINAL_FUSED_SWIGLU" => fused ? "1" : "0") do
                ex = compile(g; device = dev, retain = [y.id], weight_dtype = Luminal.Int4)
                (Array(ex(Dict{Int,Any}(x.id => xv); device = dev)[y.id]), ex)
            end
            (a, exa) = run(true); (b, exb) = run(false)
            if N % 256 == 0 && M % 64 == 0
                @test any(s -> startswith(s, "swiglu gemm"), exa.cache[:lowering_sites])     # the fusion was taken ...
                @test length(exa.steps) < length(exb.steps)                                    # ... and replaced the two matmul steps + the op
            else
                @test !any(s -> startswith(s, "swiglu gemm"), exa.cache[:lowering_sites])   # (384 columns: not a whole number of 256-wide blocks)
            end
            @test maximum(abs.(a .- b)) / maximum(abs.(b)) < 5e-3
        end
    end
end
