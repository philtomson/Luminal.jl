using Luminal, AMDGPU, Printf
const L = Luminal
Base.include(L, joinpath(@__DIR__, "flash_ablations.jl"))
D, H, KVH = 256, 24, 4
function bench(f; n=15) f(); AMDGPU.synchronize(); minimum(begin AMDGPU.synchronize(); @elapsed(begin f(); AMDGPU.synchronize() end) end for _ in 1:n) end
for (S, qoff) in ((2048, 0), (512, 32256))
    NK = qoff + S
    q16 = ROCArray(randn(Float16, D, S, H, 1)); kc = ROCArray(randn(Float16, D, NK, KVH, 1)); vc = ROCArray(randn(Float16, D, NK, KVH, 1))
    ob = AMDGPU.zeros(Float32, D, S, H, 1); scale2 = Float32(1 / sqrt(D)) * 1.4426950408889634f0
    Q = reinterpret(L.WMMA_V8H, q16); K = reinterpret(L.WMMA_V8H, kc); V = reinterpret(L.WMMA_V8H, vc)
    run(a) = @roc groupsize=512 gridsize=(cld(S, 64), H ÷ 2) flat_work_group_size=512 L._flash_prefill_ab!(ob, Q, K, V, S, H, KVH, scale2, qoff, Val(8), Val(a))
    keys = S * qoff + S * (S + 1) / 2; fl = 4.0 * D * H * keys
    run(0); AMDGPU.synchronize(); ref = Array(ob)
    res = Dict(a => Inf for a in 0:4)
    for rep in 1:3, a in 0:4; res[a] = min(res[a], bench(() -> run(a))); end
    @printf("S=%5d qoff=%6d  full %.1f | no-V-stores %.1f | 1-of-4 P stores %.1f | no softmax math %.1f | no next-tile global loads %.1f  TFLOP/s\n", S, qoff, [fl / res[a] / 1e12 for a in 0:4]...)
end
