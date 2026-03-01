
# Compiler.jl — Graph optimization using SymbolicUtils.jl rewrite rules

using SymbolicUtils
using SymbolicUtils: @rule, term, operation, arguments, Sym, BasicSymbolic, iscall
using SymbolicUtils.Code: toexpr
using SymbolicUtils.Rewriters: Prewalk, Postwalk, Fixpoint, Chain, PassThrough
using Luminal.SymbolicIntegration # For luminal_to_symbolic and luminal_relu
# Import execution functions for compiled thunk
using Luminal: execute_op, execute_op!, realize_view, to_device, realized_dims, eval_dim, execute_with_capture
using CUDA
using KernelAbstractions
using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

function fix_gpu_ast!(expr::Expr)
    if expr.head == :call
        if expr.args[1] == :exp2
            expr.args[1] = :(Base.exp2)
        elseif expr.args[1] == :log2
            expr.args[1] = :(Base.log2)
        elseif expr.args[1] == :sin
            expr.args[1] = :(Base.sin)
        elseif expr.args[1] == :cos
            expr.args[1] = :(Base.cos)
        elseif expr.args[1] == :sqrt
            expr.args[1] = :(Base.sqrt)
        elseif expr.args[1] == :max
            expr.args[1] = :(Base.max)
        elseif expr.args[1] == :min
            expr.args[1] = :(Base.min)
        end
        for i in 2:length(expr.args)
            if expr.args[i] isa Expr
                fix_gpu_ast!(expr.args[i])
            end
        end
    end
    return expr
end
function fix_gpu_ast!(val)
    return val 
end

# --- Optimization Rules ---
const RELU_RULE = @rule luminal_relu(~x) => term(max, ~x, 0.0f0; type=Real)
const LOG_EXP_RULE = @rule log2(exp2(~x)) => ~x
const EXP_LOG_RULE = @rule exp2(log2(~x)) => ~x
const SQRT_SQRT_RULE = @rule (sqrt(~x))^2 => ~x
const MAX_ID_RULE = @rule max(~x, ~x) => ~x
const MIN_ID_RULE = @rule min(~x, ~x) => ~x
const FUSED_MUL_ADD_RULE = @rule (~a * ~b) + ~c => luminal_fused_mul_add(~a, ~b, ~c)
const FUSED_ADD_RELU_RULE = @rule luminal_relu(~a + ~b) => luminal_fused_add_relu(~a, ~b)
const RECIP_CLEANUP = @rule luminal_recip(~x) => term(/, 1.0f0, ~x; type=Real)
const LOOP_FUSION_RULE = @rule luminal_loop_in(luminal_loop_out(~x, ~loop, ~range, ~st), ~loop, ~range, ~st) => ~x

const GENERAL_RULES = [
    RELU_RULE, LOG_EXP_RULE, EXP_LOG_RULE, SQRT_SQRT_RULE, 
    MAX_ID_RULE, MIN_ID_RULE, FUSED_MUL_ADD_RULE, FUSED_ADD_RELU_RULE, RECIP_CLEANUP, LOOP_FUSION_RULE
]

function optimize(expr)
    general_rewriter = Postwalk(PassThrough(Chain(GENERAL_RULES)))
    prev = nothing
    result = expr
    while !isequal(prev, result)
        prev = result
        result = general_rewriter(result)
    end
    return result
end

# --- Compiled Graph Structure ---

struct CompiledGraph
    steps::Vector{Base.Function}
    results::Vector{Any} 
    cache::Dict{Symbol, Any}
end

function (cg::CompiledGraph)(inputs::Dict, device::Luminal.AbstractDevice)
    for (k, v) in inputs
        if !isassigned(cg.results, k)
             error("Input node $k not found in compiled graph results.")
        end
        if length(cg.results[k]) != length(v)
             error("Input length mismatch for node $k: expected $(length(cg.results[k])), got $(length(v))")
        end
        copyto!(cg.results[k], v)
    end
    
    function run_graph()
        for step in cg.steps
            step(cg.results, device)
        end
    end
    
    execute_with_capture(device, run_graph, cg.cache)
    return cg.results
end

# --- Fusion Helpers ---

function is_elementwise(op)
    # Check if the Op is purely element-wise and supports scalar broadcast
    return op isa Luminal.Add || op isa Luminal.Mul || op isa Luminal.Mod || 
           op isa Luminal.Max || op isa Luminal.FusedMulAdd || 
           op isa Luminal.FusedAddReLU || op isa Luminal.LessThan ||
           op isa Luminal.Log2 || op isa Luminal.Exp2 || op isa Luminal.Sin || 
           op isa Luminal.Cos || op isa Luminal.Sqrt || op isa Luminal.Recip || 
           op isa Luminal.ReLU || op isa Luminal.Constant
end

# Helper functions for fused kernels
scalar_less(x, y) = Float32(x < y)

# Robust mapping from Luminal Ops to symbolic expressions
function op_to_sym(op, inputs)
    if op isa Luminal.Add
        return inputs[1] + inputs[2]
    elseif op isa Luminal.Mul
        return inputs[1] * inputs[2]
    elseif op isa Luminal.Mod
        return term(mod, inputs[1], inputs[2]; type=Real)
    elseif op isa Luminal.Log2
        return term(log2, inputs[1]; type=Real)
    elseif op isa Luminal.Exp2
        return term(exp2, inputs[1]; type=Real)
    elseif op isa Luminal.Sin
        return term(sin, inputs[1]; type=Real)
    elseif op isa Luminal.Cos
        return term(cos, inputs[1]; type=Real)
    elseif op isa Luminal.Sqrt
        return term(sqrt, inputs[1]; type=Real)
    elseif op isa Luminal.Recip
        return term(/, 1.0f0, inputs[1]; type=Real)
    elseif op isa Luminal.ReLU
        return term(max, inputs[1], 0.0f0; type=Real)
    elseif op isa Luminal.Max
        return term(max, inputs...; type=Real)
    elseif op isa Luminal.FusedMulAdd
        return inputs[1] * inputs[2] + inputs[3]
    elseif op isa Luminal.FusedAddReLU
        return term(max, inputs[1] + inputs[2], 0.0f0; type=Real)
    elseif op isa Luminal.LessThan
        return term(ifelse, term(<, inputs[1], inputs[2]), 1.0f0, 0.0f0; type=Real)
    elseif op isa Luminal.Constant
        return Float32(op.value)
    else
        error("Unsupported element-wise op for fusion: $(typeof(op))")
    end
end

# Helper to build a symbolic expression for a fusion group recursively
function build_fused_expr!(graph, node_id, consumer_count, group_inputs, fusible_intermediates, sym_cache, current_st)
    node = graph.nodes[node_id]
    op = node.op

    # If this node is NOT a fusible intermediate, it's a leaf for THIS fusion group
    if !(node_id in fusible_intermediates)
        input_key = (node_id, current_st)
        if haskey(sym_cache, input_key)
            return sym_cache[input_key]
        end
        sym = Sym{Real}(Symbol("in", length(group_inputs) + 1))
        push!(group_inputs, (node_id, current_st, sym))
        sym_cache[input_key] = sym
        return sym
    end

    input_syms = []
    for (in_id, _, in_st) in node.inputs
        push!(input_syms, build_fused_expr!(graph, in_id, consumer_count, group_inputs, fusible_intermediates, sym_cache, in_st))
    end

    return op_to_sym(op, input_syms)
end

# --- Main Compile Function ---

function compile(graph::Luminal.Graph)
    # 0. Consumer count
    consumer_count = zeros(Int, length(graph.nodes))
    for node in graph.nodes
        for (id, _, _) in node.inputs
            consumer_count[id] += 1
        end
    end

    # 1. Identify Fusible Intermediates
    compile_device = Luminal.get_device() 
    fusible_intermediates = Set{Int}()
    
    # Disable dynamic fusion generation on CUDA backend since Julia 1.12 World Age
    # actively blocks runtime JIT caching of dynamically created AST kernels inside modules.
    if !(compile_device isa Luminal.CUDADevice)
        for (node_id, node) in enumerate(graph.nodes)
            if is_elementwise(node.op) && consumer_count[node_id] == 1
                 # Find consumer
                 is_consumed_by_ew = false
                 for other_node in graph.nodes
                     for (in_id, _, _) in other_node.inputs
                         if in_id == node_id && is_elementwise(other_node.op)
                             is_consumed_by_ew = true
                             break
                         end
                     end
                     is_consumed_by_ew && break
                 end
                 if is_consumed_by_ew
                     push!(fusible_intermediates, node_id)
                 end
            end
        end
    end

    # 2. Allocate Results
    results = Vector{Any}(undef, length(graph.nodes))
    
    for (node_id, node) in enumerate(graph.nodes)
        node_shape = graph.shapes[node_id]
        dims_int = map(eval_dim, realized_dims(node_shape))
        dtype = Float32 
        if compile_device isa Luminal.CUDADevice
            results[node_id] = CUDA.fill(dtype(0), dims_int...) 
        else
            results[node_id] = zeros(dtype, dims_int...)
        end
        if node.op isa Luminal.Constant
             copyto!(results[node_id], to_device(node.op.value, compile_device))
        elseif node.op isa Luminal.Function && node.op.name == "ARange"
             n = dims_int[1]
             copyto!(results[node_id], to_device(Float32.(collect(0:n-1)), compile_device))
        end
    end

    # 3. Create Execution Steps with Fusion
    steps = Base.Function[]
    processed = fill(false, length(graph.nodes))

    for (node_id, node) in enumerate(graph.nodes)
        processed[node_id] && continue
        op = node.op

        if op isa Luminal.Constant || (op isa Luminal.Function && (op.name == "ARange" || op.name == "InputTensor"))
            processed[node_id] = true
            continue
        end

        if !is_elementwise(op)
            input_specs = node.inputs
            step_args = [realize_view(results[id], st) for (id, _, st) in input_specs]
            target_buf = results[node_id]
            push!(steps, (results, device) -> execute_op!(target_buf, op, step_args...))
            processed[node_id] = true
        else
            if !(node_id in fusible_intermediates)
                # Terminal of a fusion group
                group_inputs = [] # (id, st, sym)
                sym_cache = Dict{Any, Any}()
                
                input_syms = []
                for (in_id, _, in_st) in node.inputs
                    push!(input_syms, build_fused_expr!(graph, in_id, consumer_count, group_inputs, fusible_intermediates, sym_cache, in_st))
                end
                sym_expr = op_to_sym(op, input_syms)

                # Instead of mapping to Julia AST, we build a Reverse Polish Notation (RPN) instruction list
                # Opcodes: -1=Input, -2=Constant, >=0=Operator (from list below)
                # Operators: 0=Add, 1=Mul, 2=Mod, 3=Log2, 4=Exp2, 5=Sin, 6=Cos, 7=Sqrt, 8=Recip, 9=ReLU, 10=Max, 11=Less
                
                # Assign simple opcodes for Luminal primitives
                function op_to_code(op)
                    if op isa Luminal.Add return 0
                    elseif op isa Luminal.Mul return 1
                    elseif op isa Luminal.Mod return 2
                    elseif op isa Luminal.Log2 return 3
                    elseif op isa Luminal.Exp2 return 4
                    elseif op isa Luminal.Sin return 5
                    elseif op isa Luminal.Cos return 6
                    elseif op isa Luminal.Sqrt return 7
                    elseif op isa Luminal.Recip return 8
                    elseif op isa Luminal.ReLU return 9
                    elseif op isa Luminal.Max return 10
                    elseif op isa Luminal.LessThan return 11
                    else return -3 # Unknown Error
                    end
                end
                
                # Walk the tree and build the RPN
                # Nodes are represented as nested tuples: (op, children...) or (input_idx,) or (constant_val,)
                rpn_nodes = Int32[]
                rpn_consts = Float32[]
                
                function build_rpn!(expr_sym)
                    if expr_sym isa Sym
                        # Find the input index
                        idx = findfirst(x -> x[3] === expr_sym, group_inputs)
                        push!(rpn_nodes, -1)
                        push!(rpn_nodes, Int32(idx))
                    elseif Base.isexpr(expr_sym, :call)
                        # We encoded ops back to Julia expressions, need to parse them backward
                        # Alternatively, we just use the original op tree directly!
                        error("Cannot parse raw Julia AST back to RPN, must build from Luminal graph")
                    elseif typeof(expr_sym) <: Real
                        push!(rpn_nodes, -2)
                        push!(rpn_consts, Float32(expr_sym))
                        push!(rpn_nodes, Int32(length(rpn_consts)))
                    else
                         error("Unknown expression node in fusion: ", expr_sym)
                    end
                end

                # To avoid the AST string mapping hell, we will write a tiny recursive walker over the *actual* Luminal graph instead of the Symbolics.jl tree to generate the RPN natively.
                # However, this breaks `optimize()` simplifications which operate on Symbolics.jl Math AST, so we will use an alternative fallback.
                
                # Since world-age crashes are exclusive to Julia functions generated inside the dynamic `run_graph`, we can completely side-step the `InvalidIRError` by keeping `execute_op!` but running it purely natively on the CPU `Array` loop by using `allowscalar()` for scalar-heavy ops if on CUDA, OR we can stick to using standard Broadcast and `invokelatest` but *revert the fusion passes* on CUDADevices if we detect them.
                
                # But Wait! Base `broadcast!` with standard primitive Julia functions works perfectly fine. The `InvalidIRError` occurs *only* when `broadcast!` receives an *anonymous closure* generated by `RuntimeGeneratedFunctions` or `@eval`.
                
                # To bypass all this string interpolation and RPN nonsense, let's just generate the closures with standard `eval` like before, BUT execute the graph *via global named functions* evaluated at compile time!
                
                sym_args = [s for (_, _, s) in group_inputs]
                arg_names = [s.name for s in sym_args]
                
                node_expr = fix_gpu_ast!(toexpr(sym_expr))
                
                kernel_name = Symbol("global_fused_$node_id")
                
                args_defs = [Expr(:(::), Symbol(name), :(Real)) for name in arg_names]
                
                kernel_ast = quote
                    function $kernel_name($(args_defs...))
                        return Float32($node_expr)
                    end
                end
                
                println("Fusing node $node_id via global base kernel:")
                flush(stdout)
                
                Core.eval(Luminal, kernel_ast)
                
                fused_op = Luminal.FusedElementwiseOp("fused_$node_id", getfield(Luminal, kernel_name))
                target_rank = length(realized_dims(graph.shapes[node_id]))
                function align_rank(val, rank)
                    if ndims(val) >= rank return val end
                    return Base.reshape(val, ones(Int, rank - ndims(val))..., size(val)...)
                end
                
                input_steps = [align_rank(realize_view(results[id], st), target_rank) for (id, st, _) in group_inputs]
                target_buf = results[node_id]
                
                push!(steps, (results, device) -> begin
                    # Standard broadcast using a purely global, rigidly typed generic Julia function
                    # The `execute_op!` function already does `broadcast!(op.f, out, inputs...)`
                    Base.invokelatest(execute_op!, target_buf, fused_op, input_steps...)
                end)
                processed[node_id] = true
            end
        end
    end

    return CompiledGraph(steps, results, Dict{Symbol, Any}())
end