# int4 GEMM with the weights in a normal device allocation vs host memory (THP or not) registered with HIP, as gufo does.
using Luminal, AMDGPU, Printf
const L = Luminal
const HIP = AMDGPU.HIP
function hostcopy(a::ROCArray{T,N}; thp=true, register=true) where {T,N}
    bytes = sizeof(a); sz = cld(bytes, 2<<20) * (2<<20)
    p = ccall(:mmap, Ptr{Cvoid}, (Ptr{Cvoid}, Csize_t, Cint, Cint, Cint, Clong), C_NULL, sz, 3, 0x22, -1, 0)
    @assert p != Ptr{Cvoid}(-1)
    thp && ccall(:madvise, Cint, (Ptr{Cvoid}, Csize_t, Cint), p, sz, 14)
    h = unsafe_wrap(Array, Ptr{T}(p), size(a)); copyto!(h, Array(a))
    HIP.hipHostRegister(p, sz, HIP.hipHostRegisterMapped | HIP.hipHostRegisterReadOnly)
    r = Ref{Ptr{Cvoid}}(C_NULL); HIP.hipHostGetDevicePointer(r, p, 0)
    unsafe_wrap(ROCArray, Ptr{T}(r[]), size(a))
end
function anon_huge()
    for l in eachline("/proc/meminfo"); startswith(l, "AnonHuge") && return strip(split(l, ':')[2]); end
end
function bench(f; n=20) f(); AMDGPU.synchronize(); minimum(begin AMDGPU.synchronize(); @elapsed(begin f(); AMDGPU.synchronize() end) end for _ in 1:n) end
NC = 6
function mk(q, thp)
    p = hostcopy(q.p; thp)
    L.Q4Weight(p, reinterpret(eltype(q.v), p), hostcopy(q.scale; thp), q.mn === nothing ? nothing : hostcopy(q.mn; thp), q.group)
end
for N in (512, 2048), (M, K, name) in ((5120, 17408, "down"), (17408, 5120, "gate"))
    qws = [L.Q4Weight(ROCArray(randn(Float32, M, K)); group=32) for _ in 1:NC]
    X = ROCArray(randn(Float16, K, N)); C = AMDGPU.zeros(Float32, M, N)
    variants = Dict{String,Any}("device" => qws)
    for (nm, thp) in (("host+thp", true), ("host", false))
        variants[nm] = [mk(q, thp) for q in qws]
    end
    fl = 2.0 * M * K * N
    res = Dict(k => Inf for k in keys(variants))
    for rep in 1:6, (nm, ws) in variants
        t = bench(() -> for q in ws; L.wmma_q4_gemm!(C, q.v, q.scale, q.mn, q.group, X, N); end; n=4) / NC
        res[nm] = min(res[nm], t)
    end
    for nm in ("device", "host+thp", "host"); @printf("%-5s N=%4d %-9s %7.3f ms  %5.1f TFLOP/s\n", name, N, nm, 1e3 * res[nm], fl / res[nm] / 1e12); end
    println("AnonHuge: ", anon_huge())
end
