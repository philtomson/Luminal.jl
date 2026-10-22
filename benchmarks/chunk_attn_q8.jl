# chunk_attn_q8.jl -- prefill (chunk) attention over an int8 KV cache against a Float16 cache: one chunk of 256 queries at several cache depths,
# Qwen3.8-27B's attention shape (D = 256, 24 heads, 4 KV heads); ms per call (one layer). The int8 path dequantizes the cached prefix to Float16
# per call (`_kv_dequant_kernel!`) and runs the flash kernel over it.
#
# Result and what was tried: see TODO.md ("Int8 KV cache with the DFlash2 server ...", prefill) and benchmarks/kv_int8_prefill_results.txt.
#
# Usage: julia --project=. benchmarks/chunk_attn_q8.jl      (uses the GPU; stop the model server first)
using Luminal, AMDGPU, Printf, Statistics

const D, H, KVH, S = 256, 24, 4, 256
function timeit(f; reps = 15)
    f(); f(); AMDGPU.synchronize()
    return 1e3 * median([begin AMDGPU.synchronize(); @elapsed(begin f(); AMDGPU.synchronize() end) end for _ in 1:reps])
end
q = ROCArray(randn(Float32, D, S, H, 1)); kn = ROCArray(randn(Float32, D, S, KVH, 1)); vn = ROCArray(randn(Float32, D, S, KVH, 1))
sc = Float32(1 / sqrt(D))
@printf("%8s  %12s  %14s  %s\n", "depth", "Float16 KV", "int8 KV", "(ms per layer, 256 queries)")
for depth in (1024, 8192, 32768, 65536, 131072)
    T = depth + S
    posv = ROCArray(Float32[depth]); Luminal.CHUNK_POS_HINT[] = depth
    xk = ROCArray(Float16.(randn(Float32, D, T, KVH, 1))); xv = ROCArray(Float16.(randn(Float32, D, T, KVH, 1)))
    o = AMDGPU.zeros(Float32, D, H, S, 1)
    t16 = timeit(() -> Luminal.chunk_attention!(o, q, xk, xv, kn, vn, posv, sc))
    pk8 = AMDGPU.zeros(Int8, D + Luminal.DA_Q8_PAD, T, KVH, 1); pv8 = AMDGPU.zeros(Int8, D + Luminal.DA_Q8_PAD, T, KVH, 1)
    Luminal.quantize_kv!(pk8, Float32.(xk), T, 1); Luminal.quantize_kv!(pv8, Float32.(xv), T, 1)
    xk = nothing; xv = nothing
    t8 = timeit(() -> Luminal.chunk_attention!(o, q, pk8, pv8, kn, vn, posv, sc))
    @printf("%8d  %9.3f ms  %11.3f ms   int8 takes %.2fx the Float16 time\n", depth, t16, t8, t8 / t16)
    pk8 = nothing; pv8 = nothing; GC.gc()
end
Luminal.CHUNK_POS_HINT[] = -1
