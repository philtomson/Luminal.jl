using Test
using Luminal
using AMDGPU

# The register-resident DeltaRule kernel (four lanes per state column; AMD GPUs only) against
# the global-memory kernel it replaces (LUMINAL_DELTA_REG=0), on L2-normalized q/k as in the
# model: they agree to Float32 rounding. Different head dims, lengths per batch entry
# (including ones that do not fill the 16-step chunks), fresh and continuing states.
dev = get_device()
if dev isa Luminal.AMDDevice
    nrm(x) = x ./ sqrt.(sum(abs2, x; dims = 1))
    rel(a, b) = maximum(abs.(Array(a) .- Array(b))) / max(maximum(abs.(Array(a))), 1f-9)
    @testset "DeltaRule register kernel vs global-memory kernel" begin
        for (dk, dv, nk, nv, S, B, lenv, fresh) in ((128, 128, 2, 6, 70, 1, [70], true),
                                                    (128, 128, 2, 4, 33, 2, [33, 5], false),
                                                    (64, 96, 2, 4, 40, 2, [40, 17], true),
                                                    (32, 32, 1, 3, 16, 1, [16], false),
                                                    (16, 16, 2, 2, 9, 2, [9, 0], true))
            q = ROCArray(nrm(randn(Float32, dk, nk, S, B))); k = ROCArray(nrm(randn(Float32, dk, nk, S, B)))
            v = ROCArray(randn(Float32, dv, nv, S, B)); g = ROCArray(-abs.(randn(Float32, nv, S, B)) .* 0.1f0)
            beta = ROCArray(rand(Float32, nv, S, B)); st0 = ROCArray(randn(Float32, dv, dk, nv, B) .* 0.1f0)
            lens = ROCArray(Float32.(lenv)); op = Luminal.DeltaRule(Float32(1 / sqrt(dk)), fresh)
            function run(reg)
                withenv("LUMINAL_DELTA_REG" => reg ? "1" : "0") do
                    out = AMDGPU.fill(7f0, dv, nv, S, B); st = copy(st0)
                    Luminal.execute_op!(out, op, q, k, v, g, beta, st, lens); AMDGPU.synchronize()
                    out, st
                end
            end
            o0, s0 = run(false); o1, s1 = run(true)
            @test rel(o0, o1) < 5e-6
            @test rel(s0, s1) < 5e-6
            # positions past a sequence's length are zero
            for b in 1:B
                @test all(==(0f0), Array(o1)[:, :, lenv[b]+1:S, b])
            end
        end
    end
else
    @info "Skipping DeltaRule register kernel test (no AMD GPU)"
end
