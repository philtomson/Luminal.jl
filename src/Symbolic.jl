#!/usr/bin/env julia
# Defines a symbolic expression system for tensor dimensions.

module Symbolic

import Base: max, min, +, ==

export Expression, SymbolicTerm, Num, Var, BinaryOp, AddOp, MulOp, DivOp, ModOp, MinOp, MaxOp, GteOp, LtOp, execute

# Abstract parent type for all symbolic terms
abstract type SymbolicTerm end

# Represents a numeric constant
struct Num <: SymbolicTerm
    value::Int
end

# Represents a variable, like a batch dimension
struct Var <: SymbolicTerm
    name::Char
end

# Represents a binary operation
abstract type BinaryOp <: SymbolicTerm end
struct AddOp <: BinaryOp end
struct MulOp <: BinaryOp end
struct DivOp <: BinaryOp end
struct ModOp <: BinaryOp end
struct MinOp <: BinaryOp end
struct MaxOp <: BinaryOp end
struct GteOp <: BinaryOp end # Greater than or equal
struct LtOp <: BinaryOp end  # Less than

# The main Expression struct, which is a list of terms in Reverse Polish Notation
struct Expression
    terms::Vector{SymbolicTerm}
end

"""
    execute(expr::Expression, vars::Dict{Char, Int64})

Evaluate a symbolic expression by providing concrete values for its variables.
"""
function execute(expr::Expression, vars::Dict{Char, Int64})
    stack = Vector{Int}()
    for term in expr.terms
        if term isa Num
            push!(stack, term.value)
        elseif term isa Var
            # ToDo: Handle case where var is not in dict
            push!(stack, vars[term.name])
        elseif term isa BinaryOp
            # Pop order is reversed for RPN
            b = pop!(stack)
            a = pop!(stack)
            if term isa AddOp
                push!(stack, a + b)
            elseif term isa MulOp
                push!(stack, a * b)
            elseif term isa DivOp
                # Note: Julia's `div` is integer division
                push!(stack, div(a, b))
            elseif term isa ModOp
                push!(stack, a % b)
            elseif term isa MinOp
                push!(stack, min(a, b))
            elseif term isa MaxOp
                push!(stack, max(a, b))
            elseif term isa GteOp
                push!(stack, a >= b ? 1 : 0)
            elseif term isa LtOp
                push!(stack, a < b ? 1 : 0)
            end
        end
    end
    # The final result is the last item on the stack
    return pop!(stack)
end

# Overload Base functions to create symbolic expressions

function Base.:(==)(a::Expression, b::Expression)
    return a.terms == b.terms
end

function Base.max(a::Expression, b::Expression)
    if length(a.terms) == 1 && a.terms[1] isa Num && length(b.terms) == 1 && b.terms[1] isa Num
        return Expression([Num(max(a.terms[1].value, b.terms[1].value))])
    end
    return Expression([a.terms..., b.terms..., MaxOp()])
end

function Base.min(a::Expression, b::Expression)
    if length(a.terms) == 1 && a.terms[1] isa Num && length(b.terms) == 1 && b.terms[1] isa Num
        return Expression([Num(min(a.terms[1].value, b.terms[1].value))])
    end
    return Expression([a.terms..., b.terms..., MinOp()])
end

function Base.:+(a::Expression, b::Expression)
    if length(a.terms) == 1 && a.terms[1] isa Num && length(b.terms) == 1 && b.terms[1] isa Num
        return Expression([Num(a.terms[1].value + b.terms[1].value)])
    end
    return Expression([a.terms..., b.terms..., AddOp()])
end

end # module Symbolic
