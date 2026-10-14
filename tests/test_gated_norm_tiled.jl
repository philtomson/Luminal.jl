# GatedRMSNormOp with `tiled_nk`: head h = k * G + g of the result lands in slot g * nk + k (llama.cpp's tiled value-head order).
using Test
using Luminal
using AMDGPU

@testset "GatedRMSNormOp tiled output slots" begin
    dev = Luminal.get_device()
    dev isa Luminal.AMDDevice || (@info "no AMD device: skipped"; return)
    D, nk, G, S = 128, 16, 3, 256
    nv = nk * G
    x = randn(Float32, D, nv, S); z = randn(Float32, D, nv, S); w = randn(Float32, D)
    xd, zd, wd = Luminal.to_device(x, dev), Luminal.to_device(z, dev), Luminal.to_device(w, dev)
    o0 = AMDGPU.zeros(Float32, D, nv, S); o1 = AMDGPU.zeros(Float32, D, nv, S)
    Luminal.execute_op!(o0, Luminal.GatedRMSNormOp(1f-6, 0, nv), xd, wd, zd)
    Luminal.execute_op!(o1, Luminal.GatedRMSNormOp(1f-6, nk, nv), xd, wd, zd)
    a = Array(o0); b = Array(o1)
    for h in 0:nv-1, s in 1:S
        k, g = divrem(h, G)
        @test b[:, g * nk + k + 1, s] == a[:, h + 1, s]
    end
    o2 = AMDGPU.zeros(Float32, D, nv * S)
    Luminal.execute_op!(o2, Luminal.GatedRMSNormOp(1f-6, nk, nv), Base.reshape(xd, D, nv * S), wd, Base.reshape(zd, D, nv * S))
    @test Array(o2) == Base.reshape(b, D, nv * S)
    ref = x ./ sqrt.(sum(abs2, x; dims = 1) ./ D .+ 1f-6) .* w .* (z ./ (1f0 .+ exp.(-z)))
    @test maximum(abs.(a .- ref)) < 1e-4
end
