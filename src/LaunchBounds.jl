import AMDGPU
using AMDGPU: @roc
# `@roc_bounded nt wpe <@roc options> kernel(args...)`: @roc with launch bounds (`nt` threads per workgroup, `wpe` waves per
# SIMD, 0 = unset) on an AMDGPU.jl that supports them: `maxthreads` (upstream) or `flat_work_group_size` (the luminal-launch-bounds
# fork), and `waves_per_eu` (the fork, or upstream once JuliaGPU/AMDGPU.jl#1163 is in). Without them the kernel is launched unbounded.
const LAUNCH_BOUNDS = :maxthreads in AMDGPU.COMPILER_KWARGS ? :maxthreads :
                       :flat_work_group_size in AMDGPU.COMPILER_KWARGS ? :flat_work_group_size : :none
const HAS_LAUNCH_BOUNDS = LAUNCH_BOUNDS !== :none
const HAS_WAVES_PER_EU = :waves_per_eu in AMDGPU.COMPILER_KWARGS

macro roc_bounded(nt, wpe, ex...)
    opts = Any[ex[1:end-1]...]
    LAUNCH_BOUNDS === :none && return esc(Expr(:macrocall, Symbol("@roc"), __source__, opts..., ex[end]))
    push!(opts, Expr(:(=), LAUNCH_BOUNDS, nt))
    HAS_WAVES_PER_EU && push!(opts, Expr(:(=), :waves_per_eu, wpe))
    esc(Expr(:macrocall, Symbol("@roc"), __source__, opts..., ex[end]))
end
