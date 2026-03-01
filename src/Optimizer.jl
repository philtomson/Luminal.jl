module Optimizer

using ..Luminal
using ..Luminal: Graph, GraphTensor, AbstractDevice, get_device, to_device

export AbstractOptimizer, SGD, Adam, step!

abstract type AbstractOptimizer end

# ──────────────────────────────────────────────────────────────────────────────
# SGD
# ──────────────────────────────────────────────────────────────────────────────

"""
    SGD(; lr=0.01)

Stochastic Gradient Descent optimizer.
"""
mutable struct SGD <: AbstractOptimizer
    lr::Float32
    SGD(; lr=0.01f0) = new(lr)
end

function step!(opt::SGD, graph::Graph, gradients::AbstractDict)
    for (param_id, grad_data) in gradients
        # Check if it's a trainable parameter
        if param_id in graph.trainable
            param_data = graph.tensors[(param_id, 1)]
            
            # W = W - lr * G
            @. param_data -= opt.lr * grad_data
            
            # Update graph storage
            graph.tensors[(param_id, 1)] = param_data
        end
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Adam
# ──────────────────────────────────────────────────────────────────────────────

"""
    Adam(; lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)

Adam optimizer.
"""
mutable struct Adam <: AbstractOptimizer
    lr::Float32
    beta1::Float32
    beta2::Float32
    eps::Float32
    t::Int
    m::Dict{Int, Any} # 1st moment
    v::Dict{Int, Any} # 2nd moment

    function Adam(; lr=0.001f0, beta1=0.9f0, beta2=0.999f0, eps=1f-8)
        new(lr, beta1, beta2, eps, 0, Dict{Int, Any}(), Dict{Int, Any}())
    end
end

function step!(opt::Adam, graph::Graph, gradients::AbstractDict)
    opt.t += 1
    
    for (param_id, grad_data) in gradients
        if param_id in graph.trainable
            param_data = graph.tensors[(param_id, 1)]
            
            # Initialize moments if first time
            if !haskey(opt.m, param_id)
                opt.m[param_id] = fill!(similar(grad_data), 0)
                opt.v[param_id] = fill!(similar(grad_data), 0)
            end
            
            m, v = opt.m[param_id], opt.v[param_id]
            
            # Adam algorithm:
            # m = beta1 * m + (1 - beta1) * g
            # v = beta2 * v + (1 - beta2) * g^2
            # m_hat = m / (1 - beta1^t)
            # v_hat = v / (1 - beta2^t)
            # W = W - lr * m_hat / (sqrt(v_hat) + eps)
            
            @. m = opt.beta1 * m + (1 - opt.beta1) * grad_data
            @. v = opt.beta2 * v + (1 - opt.beta2) * (grad_data * grad_data)
            
            m_hat = m / (1 - opt.beta1^opt.t)
            v_hat = v / (1 - opt.beta2^opt.t)
            
            @. param_data -= opt.lr * m_hat / (sqrt(v_hat) + opt.eps)
            
            graph.tensors[(param_id, 1)] = param_data
        end
    end
end

end # module Optimizer
