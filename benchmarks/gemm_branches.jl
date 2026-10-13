using Luminal, AMDGPU, Printf
const L = Luminal
Base.include(L, joinpath(@__DIR__, "gemm_branch_variants.jl"))
NC = 8
for N in (512, 2048), (M, K, name) in ((17408, 5120, "gate"), (5120, 17408, "down"))
    qws = [L.Q4Weight(ROCArray(randn(Float32, M, K)); group=32) for _ in 1:NC]
    X = ROCArray(randn(Float16, K, N)); C = AMDGPU.zeros(Float32, M, N)
    Bv = reinterpret(L.WMMA_V8H, X); Cv = reinterpret(L.WMMA_V4F, C)
    run(q, v) = begin
        Pw = reinterpret(UInt32, q.v)
        if v == 0
            @roc groupsize=1024 gridsize=(N÷256, M÷256) L._wmma_q4_gv0_kernel!(Cv, Pw, q.scale, q.scale, Bv, K, N, 32, Val(true), Cv, Val(false))
        elseif v == 3
            @roc groupsize=1024 gridsize=(N÷256, M÷256) L._wmma_q4_gv3_kernel!(Cv, Pw, q.scale, q.scale, Bv, K, N, 32, Val(true), Cv, Val(false))
        elseif v == 2
            @roc groupsize=1024 gridsize=(N÷256, M÷256) L._wmma_q4_gv2_kernel!(Cv, Pw, q.scale, q.scale, Bv, K, N, 32, Val(true), Cv, Val(false))
        else
            @roc groupsize=1024 gridsize=(N÷256, M÷256) L._wmma_q4_gv1_kernel!(Cv, Pw, q.scale, q.scale, Bv, K, N, 32, Val(true), Cv, Val(false))
        end
    end
    vs = (0, 2, 3); res = Dict(v => Inf for v in vs)
    run(qws[1], 0); AMDGPU.synchronize(); c0 = Array(C); for vv in (2, 3); fill!(C, 0); run(qws[1], vv); AMDGPU.synchronize(); @assert Array(C) ≈ c0 rtol=1e-5; end
    for rep in 1:10, v in vs
        AMDGPU.synchronize(); t = @elapsed (for q in qws; run(q, v); end; AMDGPU.synchronize()); res[v] = min(res[v], t / NC)
    end
    fl = 2.0 * M * K * N
    println(rpad(name, 5), " N=", lpad(N, 4), "  ", join([@sprintf("%s %.1f", ("checked", "inbounds", "inb+sched", "depth2")[v+1], fl / res[v] / 1e12) for v in vs], "   "), "  TFLOP/s")
end
