# Autotuning sweep of the int4 WMMA GEMM launch parameters with Hyperopt.jl, over Qwen3.6-27B's prefill shapes.
#
#   JULIA_LOAD_PATH="@:$PWD/benchmarks/sweep_env:@stdlib" julia -t auto,1 --project=. benchmarks/gemm_sweep.jl [--n=256] [--iters=40]
#
# (Hyperopt lives in benchmarks/sweep_env so the main project does not depend on it.) The search space is
# the kernel (128x256 "wide" or 128x128 "big"), the prefetch depth PD, and waves_per_eu (0 = default; needs the
# AMDGPU.jl launch-bounds branch). Objective: mean TFLOP/s over the shapes. Every candidate is also checked against
# the default configuration's output. Extend `space` and `launch` as new kernel variants appear.
using Luminal, AMDGPU, Hyperopt, Printf, Statistics

arg(name, default) = (i = findfirst(a -> startswith(a, "--$name="), ARGS); i === nothing ? default : split(ARGS[i], "=", limit = 2)[2])
N = parse(Int, arg("n", "256")); iters = parse(Int, arg("iters", "40"))
shapes = ((17408, 5120, "ffn gate/up"), (5120, 17408, "ffn down"), (5120, 5120, "proj"), (12288, 5120, "qkv"))

function timeit(f; reps = 20)
    f(); AMDGPU.synchronize()
    minimum(begin AMDGPU.synchronize(); @elapsed(begin f(); AMDGPU.synchronize() end) end for _ in 1:reps)
end

problems = map(shapes) do (M, K, name)
    qw = Luminal.Q4Weight(ROCArray(randn(Float32, M, K)); group = 32)
    X = ROCArray(randn(Float16, K, cld(N, 256) * 256)); C = AMDGPU.zeros(Float32, M, N)
    ref = AMDGPU.zeros(Float32, M, N)
    Luminal.wmma_q4_gemm!(ref, qw.v, qw.scale, qw.mn, qw.group, X, N)
    AMDGPU.synchronize()
    (; M, K, name, qw, X, C, ref = Array(ref))
end

const _cache = Dict{Any,Float64}()
evaluate(kind, pd, wpe) = get!(() -> _evaluate(kind, pd, wpe), _cache, (kind, pd, wpe))
function _evaluate(kind, pd, wpe)
    tf = Float64[]
    for p in problems
        run() = Luminal.wmma_q4_gemm!(p.C, p.qw.v, p.qw.scale, p.qw.mn, p.qw.group, p.X, N; kind, pd, wpe)
        try
            run(); AMDGPU.synchronize()
            err = maximum(abs.(Array(p.C) .- p.ref)) / maximum(abs.(p.ref))
            err < 1e-3 || return 0.0                       # wrong result: worthless
            push!(tf, 2.0 * p.M * p.K * N / timeit(run) / 1e12)
        catch e
            return 0.0                                      # failed to compile / launch
        end
    end
    return mean(tf)
end

println("N = $N; shapes: ", join((s[3] for s in shapes), ", "))
base = evaluate(:wide, 1, 0)
@printf("default (wide, PD=1, wpe=0): %.2f TFLOP/s\n", base)
ho = @hyperopt for i = iters, sampler = RandomSampler(), kind = [:wide, :big], pd = 1:4, wpe = [0, 2, 3, 4, 6, 8]
    r = evaluate(kind, pd, wpe)
    @printf("%3d  %-5s PD=%d wpe=%d  %.2f TFLOP/s\n", i, kind, pd, wpe, r)
    -r                                                      # Hyperopt minimizes
end
@printf("best: %s  %.2f TFLOP/s (default %.2f)\n", ho.minimizer, -ho.minimum, base)
