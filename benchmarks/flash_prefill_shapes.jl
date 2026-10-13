using Luminal, AMDGPU, Printf
const L = Luminal
D, H, KVH = 256, 24, 4
function bench(f; n=15) f(); AMDGPU.synchronize(); minimum(begin AMDGPU.synchronize(); @elapsed(begin f(); AMDGPU.synchronize() end) end for _ in 1:n) end
for (S, qoff) in ((2048, 0), (512, 0), (512, 8192), (512, 16384), (512, 32256))
    NK = qoff + S
    q16 = ROCArray(randn(Float16, D, S, H, 1)); kc = ROCArray(randn(Float16, D, NK, KVH, 1)); vc = ROCArray(randn(Float16, D, NK, KVH, 1))
    ob = AMDGPU.zeros(Float32, D, S, H, 1); scale2 = Float32(1 / sqrt(D)) * 1.4426950408889634f0
    f() = L._launch_flash_lds!(ob, reinterpret(L.WMMA_V8H, q16), reinterpret(L.WMMA_V8H, kc), reinterpret(L.WMMA_V8H, vc), S, H, KVH, scale2, qoff, Val(D ÷ 32), H ÷ 2)
    t = bench(f)
    # useful flops: every query attends to qoff + (its index + 1) keys: 4 D per (query, key, head)
    keys = S * qoff + S * (S + 1) / 2
    fl = 4.0 * D * H * keys
    @printf("S=%5d qoff=%6d  %8.3f ms  %5.1f TFLOP/s (useful)\n", S, qoff, 1e3 * t, fl / t / 1e12)
end
