# Next-Concept Modeling text generation (https://github.com/HazyResearch/ncm) with the TinyStories checkpoint.
#
#   python examples/ncm_convert.py ~/devel/AI/ncm_artifacts/tinystories      # once: .pt -> safetensors
#   julia --project=. examples/ncm_generate.jl ~/devel/AI/ncm_artifacts/tinystories [--n=4] [--seed=1] [--temp=1.0]
#                                              [--top_k=0] [--top_p=1.0] [--cpu]
#
# A sample is 128 BPE tokens produced coarse to fine: 12 levels of codes, one forward pass of the transformer each.
using Luminal, Random
using Luminal.NCM

dir = isempty(ARGS) || startswith(ARGS[1], "--") ? joinpath(homedir(), "devel/AI/ncm_artifacts/tinystories") : ARGS[1]
opt(name, default) = (i = findfirst(a -> startswith(a, "--$name="), ARGS); i === nothing ? default : parse(typeof(default), split(ARGS[i], "=")[2]))
n, seed = opt("n", 4), opt("seed", 1)
temp, top_k, top_p = opt("temp", 1.0), opt("top_k", 0), opt("top_p", 1.0)
device = "--cpu" in ARGS ? Luminal.CPUDevice() : Luminal.get_device()

gen = NCMGenerator(dir; device)
rng = Xoshiro(seed)
generate_text(gen; batch=n, rng=Xoshiro(0))                  # compiles the per-level graphs
t = @elapsed texts = generate_text(gen; batch=n, temperature=temp, top_k, top_p, rng)
for (i, s) in enumerate(texts)
    println("--- sample $i ---\n", s, "\n")
end
println("$n samples of 128 tokens in $(round(t, digits=2)) s on $(typeof(device))")
