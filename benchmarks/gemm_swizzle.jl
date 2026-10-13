using Luminal, AMDGPU, Printf
const L = Luminal
Base.include(L, joinpath(@__DIR__, "gemm_swizzle_variant.jl"))
NC = 8
for N in (512, 2048), (M, K, name) in ((17408, 5120, "gate"), (5120, 17408, "down"))
    qws = [L.Q4Weight(ROCArray(randn(Float32, M, K)); group=32) for _ in 1:NC]
    X = ROCArray(randn(Float16, K, N)); C = AMDGPU.zeros(Float32, M, N)
    Bv = reinterpret(L.WMMA_V8H, X); Cv = reinterpret(L.WMMA_V4F, C)
    run(q, gs) = begin
        Pw = reinterpret(UInt32, q.v)
        @roc(groupsize=1024, gridsize=(N÷256, M÷256), L._wmma_q4_gs_kernel!(Cv, Pw, q.scale, q.scale, Bv, K, N, 32, Val(true), Cv, Val(false), Val(gs)))
    end
    gss = (0, 2, 4, 8, 16)
    res = Dict(g => Inf for g in gss)
    ref = nothing
    for rep in 1:8, g in gss
        AMDGPU.synchronize()
        t = @elapsed (for q in qws; run(q, g); end; AMDGPU.synchronize())
        res[g] = min(res[g], t / NC)
    end
    run(qws[1], 0); AMDGPU.synchronize(); c0 = Array(C); run(qws[1], 8); AMDGPU.synchronize(); err = maximum(abs.(Array(C) .- c0))
    fl = 2.0 * M * K * N
    println(name, " N=", N, "  ", join([@sprintf("GS=%d %.1f", g, fl / res[g] / 1e12) for g in gss], "  "), "  (swizzle err ", err, ")")
end
