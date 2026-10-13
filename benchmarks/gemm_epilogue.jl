using Luminal, AMDGPU, Printf
const L = Luminal
Base.include(L, joinpath(@__DIR__, "gemm_epilogue_variant.jl"))
NC = 8
for N in (512, 2048), (M, K, name) in ((17408, 5120, "gate"), (5120, 17408, "down"), (5120, 5120, "o5120"))
    qws = [L.Q4Weight(ROCArray(randn(Float32, M, K)); group=32) for _ in 1:NC]
    X = ROCArray(randn(Float16, K, N)); C = AMDGPU.zeros(Float32, M, N); C16 = AMDGPU.zeros(Float16, M, N)
    Bv = reinterpret(L.WMMA_V8H, X); Cv = reinterpret(L.WMMA_V4F, C); C16v = reinterpret(NTuple{8,VecElement{Float16}}, C16)
    run(q, ep) = begin
        Pw = reinterpret(UInt32, q.v)
        @roc(groupsize=1024, gridsize=(N÷256, M÷256), L._wmma_q4_gs_kernel!(Cv, Pw, q.scale, q.scale, Bv, K, N, 32, Val(true), Cv, Val(false), Val(0), Val(ep), C16v))
    end
    eps = (0, 1, 2); names = ("f32 store", "no store", "f16 store")
    res = Dict(e => Inf for e in eps)
    for rep in 1:10, e in eps
        AMDGPU.synchronize()
        t = @elapsed (for q in qws; run(q, e); end; AMDGPU.synchronize())
        res[e] = min(res[e], t / NC)
    end
    fl = 2.0 * M * K * N
    println(rpad(name, 6), " N=", lpad(N, 4), "  ", join([@sprintf("%s %.1f TF (%.2f ms)", names[e+1], fl / res[e] / 1e12, 1e3 * res[e]) for e in eps], "   "))
end
