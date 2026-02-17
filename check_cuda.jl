
using CUDA
println("CUDA symbols starting with 'graph':")
for s in names(CUDA; all=true)
    if contains(lowercase(string(s)), "graph")
        println(s)
    end
end
println("\nSearching for launch or execute:")
for s in names(CUDA; all=true)
    if contains(lowercase(string(s)), "launch") || contains(lowercase(string(s)), "exec")
        println(s)
    end
end
println("Done.")
