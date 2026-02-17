module MetatheoryRules

using Metatheory
using ..MetatheoryOps

export luminal_theory

# === Algebraic Simplification ===
algebraic_rules = @theory begin
    # Identity
    LuminalAdd(~x, LuminalZero()) --> ~x
    LuminalAdd(LuminalZero(), ~x) --> ~x
    LuminalMul(~x, LuminalOne()) --> ~x
    LuminalMul(LuminalOne(), ~x) --> ~x
    LuminalMul(~x, LuminalZero()) --> LuminalZero()
    LuminalMul(LuminalZero(), ~x) --> LuminalZero()
    
    # Commutativity (for canonicalization)
    # Using '==' for bidirectional equality, but careful with loops
    # Generally better to have canonical ordering if possible, or leave it to e-graph
    LuminalAdd(~x, ~y) == LuminalAdd(~y, ~x)
    LuminalMul(~x, ~y) == LuminalMul(~y, ~x)
    
    # Associativity  
    LuminalAdd(LuminalAdd(~x, ~y), ~z) == LuminalAdd(~x, LuminalAdd(~y, ~z))
    LuminalMul(LuminalMul(~x, ~y), ~z) == LuminalMul(~x, LuminalMul(~y, ~z))
    
    # Distributivity
    # x * (y + z) -> x*y + x*z ? usage depends on cost
    LuminalMul(~x, LuminalAdd(~y, ~z)) == LuminalAdd(LuminalMul(~x, ~y), LuminalMul(~x, ~z))
    
    # Idempotency
    LuminalMax(~x, ~x) --> ~x
end

# === Fusion Rules ===
fusion_rules = @theory begin
    # Fuse Add + ReLU
    LuminalReLU(LuminalAdd(~x, ~y)) --> LuminalFusedAddReLU(~x, ~y)
    
    # Fuse Mul + Add -> MulAdd (if backend supports it, e.g. FMA)
    LuminalAdd(LuminalMul(~x, ~y), ~z) --> LuminalMulAdd(~x, ~y, ~z)
    
    # GEMM Fusion: (A * B) + C
    LuminalAdd(LuminalMatMul(~x, ~w), ~bias) --> LuminalGEMM(~x, ~w, ~bias)
    
    # Consecutive Reshapes
    # Reshape(Reshape(x, s1), s2) -> Reshape(x, s2)
    # Note: This assumes s2 is compatible with x. In a graph IR, shapes are usually static properties.
    # But here 's' is an expression (e.g. constant or tuple).
    LuminalReshape(LuminalReshape(~x, ~s1), ~s2) --> LuminalReshape(~x, ~s2)
end

# === Memory Layout Optimization ===
layout_rules = @theory begin
    # Transpose cancellation
    LuminalTranspose(LuminalTranspose(~x)) --> ~x
    
    # MatMul with Transpose -> MatMulTN / MatMulNT / MatMulTT
    # We didn't define these specific ops in MetatheoryOps yet, maybe just GEMM flags?
    # For now, let's keep it simple or map to specific custom ops if we had them.
    # LuminalMatMul(LuminalTranspose(~a), ~b) --> LuminalMatMulTN(~a, ~b)
end

# === Constant Folding ===
# Note: In Metatheory 3.0, we can use `where` clauses with pure julia functions
constant_rules = @theory begin
    # Fold constants using dynamic rules (=>) to evaluate the result
    LuminalAdd(LuminalConstant(~a::Number), LuminalConstant(~b::Number)) => LuminalConstant(a + b)
    LuminalMul(LuminalConstant(~a::Number), LuminalConstant(~b::Number)) => LuminalConstant(a * b)
    LuminalSub(LuminalConstant(~a::Number), LuminalConstant(~b::Number)) => LuminalConstant(a - b)
    
    # Zero/One handling
    LuminalConstant(0) --> LuminalZero()
    LuminalConstant(0.0) --> LuminalZero()
    LuminalConstant(1) --> LuminalOne()
    LuminalConstant(1.0) --> LuminalOne()
end

# Combined ruleset
# Vector of all rules
luminal_theory = [algebraic_rules..., fusion_rules..., layout_rules..., constant_rules...]

end # module
