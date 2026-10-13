using Test
using Luminal
using AMDGPU

# gate / up int4 GEMMs feeding a SwiGLUOp write their results as Float16 (the 256 x 256 kernel's store converts, the SwiGLU reads half the
# bytes); the result matches the Float32 path (LUMINAL_F16_GEMM_OUT=0) up to the Float16 rounding of the two pre-activations.
if Luminal.get_device() isa Luminal.AMDDevice
    @testset "Float16 gate/up GEMM outputs" begin
        dev = Luminal.get_device()
        for (M, K, N, K2) in ((256, 128, 512, 128), (512, 192, 768, 256), (256, 64, 1024, 64))
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
            function run(f16out)
                out = nothing; log = ""
                withenv("LUMINAL_F16_GEMM_OUT" => f16out ? "1" : "0", "LUMINAL_FUSED_SWIGLU" => "0", "LUMINAL_F16_DEBUG" => "1") do
                    path = tempname()
                    open(path, "w") do io
                        redirect_stdout(io) do
                            ex = compile(g; device = dev, retain = [y.id], weight_dtype = Luminal.Int4)
                            out = Array(ex(Dict{Int,Any}(x.id => xv); device = dev)[y.id])
                        end
                    end
                    log = read(path, String)
                end
                return out, log
            end
            (a, la) = run(true); (b, lb) = run(false)
            @test count("[f16 out]", la) == 2            # gate and up
            @test count("[f16 out]", lb) == 0
            @test maximum(abs.(a .- b)) / maximum(abs.(b)) < 5e-3
        end
    end
end
