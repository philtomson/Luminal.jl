
using Metatheory
using Metatheory.EGraphs
using Metatheory.VecExprModule: VecExpr, v_head

Luminal_ReLU(x) = :(Luminal_ReLU($x))
Luminal_Max(x, y) = :(Luminal_Max($x, $y))

theory4 = @theory begin
    Luminal_ReLU(~x) == Luminal_Max(~x, 0)
end

expr4 = :(Luminal_ReLU(a))
println("Input: ", expr4)
eg4 = EGraph(expr4)
saturate!(eg4, theory4)

# Debug: Print e-graph content
println("E-Graph content:")
# We can't use show(eg4) easily if it's huge, but it should be small
display(eg4)

# Check if Luminal_Max is in the e-class of the root
root_id = eg4.root
eclass = eg4[root_id]
println("\nRoot E-Class nodes:")
for n in eclass.nodes
    println("  ", n)
    # Try to resolve head
    h = v_head(n)
    op = haskey(eg4.constants, h) ? eg4.constants[h] : "???"
    println("    Head Symbol: ", op)
end

function cost_function(n::VecExpr, data, costs::Vector{Float64})
    cost = 1.0 + sum(costs)
    if v_head(n) == hash(:Luminal_ReLU)
        cost += 100.0
    end
    return cost
end

result4 = extract!(eg4, cost_function)
println("\nOutput: ", result4)
