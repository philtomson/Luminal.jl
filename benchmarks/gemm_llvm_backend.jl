# Compile the Julia-generated LLVM IR of the 256x256 int4 GEMM with ROCm's own clang (LLVM 24) and compare speed in the same process.
using Luminal, AMDGPU, Printf
const L = Luminal
const HIP = AMDGPU.HIP
ROCM = joinpath(ENV["ROCM_PATH"])
CLANG = joinpath(ROCM, "lib/llvm/bin/clang"); LLD = joinpath(ROCM, "lib/llvm/bin/ld.lld")
dir = mktempdir()
M, K, N = parse(Int, get(ENV, "T_M", "17408")), parse(Int, get(ENV, "T_K", "5120")), parse(Int, get(ENV, "T_N", "512"))
NC = 8
qws = [L.Q4Weight(ROCArray(randn(Float32, M, K)); group=32) for _ in 1:NC]
X = ROCArray(randn(Float16, K, N)); C = AMDGPU.zeros(Float32, M, N)
Bv = reinterpret(L.WMMA_V8H, X); Cv = reinterpret(L.WMMA_V4F, C)
Pws = [reinterpret(UInt32, q.v) for q in qws]
args(i) = (Cv, Pws[i], qws[i].scale, qws[i].scale, Bv, K, N, 32, Val(true), Cv, Val(false))
kern0 = @roc launch=false groupsize=1024 gridsize=(N÷256, M÷256) L._wmma_q4_g_kernel!(args(1)...)
irfile = joinpath(dir, "g.ll")
open(irfile, "w") do io
    AMDGPU.@device_code_llvm io=io raw=true dump_module=true @roc launch=false groupsize=1024 gridsize=(N÷256, M÷256) L._wmma_q4_g_kernel!(args(1)...)
end
ir = read(irfile, String)
entry = match(r"define amdgpu_kernel void @(\S+?)\(", ir).captures[1] |> String
println("entry: ", entry, "   IR ", length(ir), " bytes")
function build(tag, flags)
    obj = joinpath(dir, "g_$tag.o"); exe = joinpath(dir, "g_$tag.hsaco")
    run(`$CLANG -x ir $irfile --target=amdgcn-amd-amdhsa -mcpu=gfx1151 -nogpulib -c -o $obj $flags`)
    run(`$LLD -shared -o $exe $obj`)
    fun = HIP.HIPFunction(HIP.HIPModule(read(exe)), entry, Symbol[])
    typeof(kern0)(kern0.f, fun)
end
variants = Dict{String,Any}("julia(LLVM18)" => kern0)
for (tag, flags) in (("O3", `-O3`), ("O2", `-O2`), ("O3_maxilp", `-O3 -mllvm -amdgpu-sched-strategy=max-ilp`), ("O3_maxocc", `-O3 -mllvm -amdgpu-sched-strategy=max-occupancy`))
    try variants[tag] = build(tag, flags) catch e; println("build $tag failed: ", sprint(showerror, e)[1:min(end,300)]) end
end
launch(k, i) = k(args(i)...; groupsize=1024, gridsize=(N÷256, M÷256))
launch(kern0, 1); AMDGPU.synchronize(); ref = Array(C)
for (tag, k) in variants
    fill!(C, 0); launch(k, 1); AMDGPU.synchronize(); err = maximum(abs.(Array(C) .- ref)) / maximum(abs.(ref)); println(rpad(tag, 14), " rel diff vs julia: ", err)
end
fl = 2.0 * M * K * N
res = Dict(t => Inf for t in keys(variants))
for rep in 1:12, (tag, k) in variants
    AMDGPU.synchronize(); t = @elapsed (for i in 1:NC; launch(k, i); end; AMDGPU.synchronize()); res[tag] = min(res[tag], t / NC)
end
for (tag, _) in sort(collect(variants); by = x -> res[x[1]]); @printf("%-14s %7.3f ms  %5.1f TFLOP/s\n", tag, 1e3 * res[tag], fl / res[tag] / 1e12); end
