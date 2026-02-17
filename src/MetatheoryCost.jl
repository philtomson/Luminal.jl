module MetatheoryCost

using Metatheory
using Metatheory.EGraphs
using Metatheory.VecExprModule: v_isexpr, v_head, v_children, VecExpr
using ..MetatheoryOps

export luminal_cost

"""
    luminal_cost(node::VecExpr, op_symbol, child_costs::Vector{Float64})

Luminal-specific cost function for E-Graph extraction.
Prioritizes:
1. Matrix operations (MatMul) over element-wise
2. Fused operations (FusedMulAdd, FusedAddReLU) over separate ops
3. Smaller AST size (generally)
4. Kernel-friendly patterns
"""
function luminal_cost(node::VecExpr, op_symbol, child_costs::Vector{Float64})
    # Literals/Variables have cost 1.0 (base)
    !v_isexpr(node) && return 1.0
    
    # op_symbol is passed directly in Metatheory v3 extract!
    
    # Base cost is sum of children costs
    base_cost = isempty(child_costs) ? 0.0 : sum(child_costs)
    
    # Operation-specific costs
    # Lower is better
    
    op_cost = 1.0 # Default cost
    
    if op_symbol == :LuminalMatMul
        # Matrix multiplication is computationally heavy but efficient on GPU
        # We want to prefer it over manual loops or element-wise expansions if equivalent
        # However, compared to other ops, it's "expensive" in the sense of compute.
        # But in term of "optimality", we usually want to KEEP it if it exists.
        # Cost functions in rewriting usually target "simple" or "fast" code.
        # If we use execution time as metric, MatMul is high cost.
        # If we use code size/complexity, MatMul is low cost (1 op vs many).
        # Here we target efficient kernels, so fused > separate.
        op_cost = 10.0 
        
    elseif op_symbol == :LuminalConv
        op_cost = 10.0
        
    elseif op_symbol == :LuminalFusedAddReLU
        # Fused ops should be cheaper than their components
        # Add (1) + ReLU (1) = 2
        # Fused = 1.5
        op_cost = 1.5
        
    elseif op_symbol == :LuminalMulAdd
        # Mul (1) + Add (1) = 2
        op_cost = 1.5
        
    elseif op_symbol == :LuminalGEMM
        # MatMul (10) + Add (1) = 11
        # GEMM = 10.5
        op_cost = 10.5
        
    elseif op_symbol == :LuminalReshape || op_symbol == :LuminalView
        # View operations are very cheap (metadata only)
        op_cost = 0.1
        
    elseif op_symbol == :LuminalConstant
        op_cost = 0.5 # Cheaper than variables
        
    else
        # Standard ops (Add, Mul, etc.)
        op_cost = 2.0
    end
    
    return base_cost + op_cost
end

end # module
