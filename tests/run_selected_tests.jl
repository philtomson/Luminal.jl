using Test

test_files = [
    "test_shape_tracker.jl",
    "test_algebraic_simplification.jl",
    "test_associativity.jl",
    "test_commutativity.jl",
    "test_compilation.jl",
    "test_fusion.jl",
    "test_nn_layers.jl",
    "test_llama.jl",
    "test_metatheory_rules.jl",
    "test_metatheory_cost.jl",
    "test_metatheory_optimizer.jl"
]

println("Running $(length(test_files)) test files...")

failed_tests = []

for f in test_files
    println("\n" * "="^40)
    println("Running $f")
    println("="^40)
    try
        @testset "$f" begin
            include(f)
        end
    catch e
        println("FAILED: $f")
        push!(failed_tests, f)
        # showerror(stdout, e, catch_backtrace())
    end
end

println("\n" * "="^40)
println("Test Summary")
println("="^40)
if isempty(failed_tests)
    println("All $(length(test_files)) test files PASSED! ✅")
else
    println("$(length(failed_tests)) test files FAILED: ❌")
    for f in failed_tests
        println("- $f")
    end
end
