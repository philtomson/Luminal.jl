# Implements a simple interpreter to execute a computation graph.



using CUDA
using LinearAlgebra


export execute_op, execute_op!, realize_view, execute, eval_dim, FusedElementwiseOp

# Helper for batch matrix multiplication
function batch_matmul(A, B)
    # ... (Keep existing implementation for interpreter) ...
    # A: (..., M, K), B: (..., K, N)
    
    # Handle simple 2D case
    if ndims(A) == 2 && ndims(B) == 2
        return A * B
    end
    
    # Handle 3D * 2D (Linear on batch): (B, S, D) * (D, V) -> (B, S, V)
    if ndims(A) == 3 && ndims(B) == 2
        B1, S, D = size(A)
        res = similar(A, B1, S, size(B, 2))
        for i in 1:B1
            res[i, :, :] = A[i, :, :] * B
        end
        return res
    end
    
    # Handle 3D batch case: (B, M, K) * (B, K, N) -> (B, M, N)
    if ndims(A) == 3 && ndims(B) == 3
        @assert size(A, 1) == size(B, 1) "Batch dimensions (3D) must match: $(size(A, 1)) vs $(size(B, 1))"
        batch_size = size(A, 1)
        res = similar(A, batch_size, size(A, 2), size(B, 3))
        for i in 1:batch_size
            res[i, :, :] = A[i, :, :] * B[i, :, :]
        end
        return res
    end
    
    # Handle 4D (Llama attention): (B, H, S, D) * (B, H, D, S) -> (B, H, S, S)
    if ndims(A) == 4 && ndims(B) == 4
        @assert size(A, 1) == size(B, 1) && size(A, 2) == size(B, 2) "Batch and head dimensions must match: A $(size(A)), B $(size(B))"
        B1, B2 = size(A, 1), size(A, 2)
        res = similar(A, B1, B2, size(A, 3), size(B, 4))
        for i in 1:B1, j in 1:B2
            res[i, j, :, :] = A[i, j, :, :] * B[i, j, :, :]
        end
        return res
    end
    
    error("Batch matmul not implemented for $(ndims(A))D and $(ndims(B))D (Shapes: $(size(A)) and $(size(B)))")
end

function batch_matmul!(C, A, B)
    # In-place batch matmul: C = A * B
    if ndims(A) == 2 && ndims(B) == 2
        mul!(C, A, B)
        return C
    end
    
    if ndims(A) == 3 && ndims(B) == 2
        B1 = size(A, 1)
        for i in 1:B1
            # View or slice? mul! works on views.
            # C[i, :, :] = A[i, :, :] * B
            mul!(view(C, i, :, :), view(A, i, :, :), B)
        end
        return C
    end
    
    if ndims(A) == 3 && ndims(B) == 3
        B1 = size(A, 1)
        for i in 1:B1
            mul!(view(C, i, :, :), view(A, i, :, :), view(B, i, :, :))
        end
        return C
    end
    
    if ndims(A) == 4 && ndims(B) == 4
        B1, B2 = size(A, 1), size(A, 2)
        for i in 1:B1, j in 1:B2
            mul!(view(C, i, j, :, :), view(A, i, j, :, :), view(B, i, j, :, :))
        end
        return C
    end
    
    error("Batch matmul! not implemented for $(ndims(A))D and $(ndims(B))D")
end

function realize_view(data, st::ShapeTracker)
    # If buffer already matches logical size, it's likely already realized (common in interpreter)
    r_dims = realized_dims(st)
    if length(data) == prod(Int.(Luminal.eval_dim.(r_dims))) && length(data) > 1
        return Base.reshape(data, Int.(Luminal.eval_dim.(r_dims))...)
    end

    if length(data) == 1
        return Base.reshape(data, fill(1, length(st.indexes))...)
    end

    # 1. Reshape to physical rank (excluding fake dims)
    physical_dims = Int[]
    for i in 1:length(st.dims)
        if !st.fake[i]
            push!(physical_dims, Int(Luminal.eval_dim(st.dims[i])))
        end
    end
    
    if length(data) != prod(physical_dims)
        # Fallback: check if it matches logical size or something
        if length(data) == prod([Int(Luminal.eval_dim(d)) for d in st.dims])
             arr = Base.reshape(data, [Int(Luminal.eval_dim(d)) for d in st.dims]...)
        else
             return data
        end
    else
        arr = Base.reshape(data, physical_dims...)
    end
    
    # 2. Restore full rank by adding back fake dims as 1s
    full_arr = arr
    for i in 1:length(st.dims)
        if st.fake[i]
            sz = [size(full_arr)...]
            insert!(sz, i, 1)
            full_arr = Base.reshape(full_arr, sz...)
        end
    end
    
    # 3. Apply Indexing (Permutation and Rank selection)
    res = Base.permutedims(full_arr, st.indexes)
    
    # 4. Apply Mask (Slicing)
    r_dims = realized_dims(st)
    if [size(res)...] != r_dims
        ranges = Any[]
        for i in 1:length(st.indexes)
            idx = st.indexes[i]
            s, e = st.mask[idx]
            dim_size = size(res, i)
            start = max(1, Int(s) + 1)
            stop = min(Int(e), dim_size)
            push!(ranges, start:stop)
        end
        res = view(res, ranges...)
    end

    return res
end

function execute_slice(input, ranges)
    res = input
    for (i, (s, e)) in enumerate(ranges)
        start_idx = max(1, s + 1)
        end_idx = min(size(res, i), e)
        res = selectdim(res, i, start_idx:end_idx)
    end
    return copy(res)
end

function execute_slice!(out, input, ranges)
    # In-place slice is tricky: copyto!(out, sliced_view)
    # Construct view
    v = input
    for (i, (s, e)) in enumerate(ranges)
        start_idx = max(1, s + 1)
        end_idx = min(size(v, i), e)
        v = selectdim(v, i, start_idx:end_idx)
    end
    copyto!(out, v)
    return out
end

function execute_pad(input, padding)
    old_size = size(input)
    new_size = [old_size[i] + padding[i][1] + padding[i][2] for i in 1:length(old_size)]
    res = similar(input, new_size...)
    fill!(res, 0)
    dest_ranges = [ (padding[i][1]+1):(padding[i][1]+old_size[i]) for i in 1:length(old_size) ]
    res[dest_ranges...] = input
    return res
end

function execute_pad!(out, input, padding)
    fill!(out, 0)
    old_size = size(input)
    dest_ranges = [ (padding[i][1]+1):(padding[i][1]+old_size[i]) for i in 1:length(old_size) ]
    # Copy input to center
    # out[dest_ranges...] = input # This might allocate?
    # view(out, dest_ranges...) .= input # In-place
    view(out, dest_ranges...) .= input
    return out
end


# --- Dispatchable Execute functions (Functional) ---

function align_broadcast_ranks(a, b)
    if ndims(a) == ndims(b) return a, b end
    if ndims(a) < ndims(b)
        return Base.reshape(a, ones(Int, ndims(b) - ndims(a))..., size(a)...), b
    else
        return a, Base.reshape(b, ones(Int, ndims(a) - ndims(b))..., size(b)...)
    end
end

execute_op(op::Add, a, b) = begin (a_a, b_a) = align_broadcast_ranks(a, b); a_a .+ b_a end
execute_op(op::Mul, a, b) = begin (a_a, b_a) = align_broadcast_ranks(a, b); a_a .* b_a end
execute_op(op::Mod, a, b) = begin (a_a, b_a) = align_broadcast_ranks(a, b); a_a .% b_a end
execute_op(op::LessThan, a, b) = begin (a_a, b_a) = align_broadcast_ranks(a, b); Float32.(a_a .< b_a) end
execute_op(op::FusedMulAdd, a, b, c) = (a .* b) .+ c
execute_op(op::FusedAddReLU, a, b) = Base.max.(a .+ b, 0)
execute_op(op::Log2, a) = log2.(a)
execute_op(op::Exp2, a) = exp2.(a)
execute_op(op::Sin, a) = sin.(a)
execute_op(op::Cos, a) = cos.(a)
execute_op(op::Sqrt, a) = sqrt.(a)
execute_op(op::Recip, a) = 1.0f0 ./ a
execute_op(op::Reshape, a) = (res = Base.reshape(a, op.shape...); println("Reshape: $(size(a)) -> $(size(res))"); res)
execute_op(op::Permute, a) = (res = Base.permutedims(a, op.dims); println("Permute: $(size(a)) -> $(size(res)) with dims $(op.dims)"); res)
execute_op(op::Contiguous, a) = copy(a)
execute_op(op::MatMul, a, b) = (res = batch_matmul(a, b); println("MatMul: $(size(a)) * $(size(b)) -> $(size(res))"); res)
execute_op(op::SumReduce, a) = (res = dropdims(Base.sum(a, dims=op.dim), dims=op.dim); println("SumReduce: $(size(a)) dim $(op.dim) -> $(size(res))"); res)
execute_op(op::MaxReduce, a) = (res = dropdims(Base.maximum(a, dims=op.dim), dims=op.dim); println("MaxReduce: $(size(a)) dim $(op.dim) -> $(size(res))"); res)
execute_op(op::Slice, a) = execute_slice(a, op.ranges)
execute_op(op::Pad, a) = execute_pad(a, op.padding)
execute_op(op::Constant, device) = to_device(op.value, device)

function execute_op(op::Expand, a)
    curr_sz = [size(a)...]
    println("Expand input size: $(size(a)), dim: $(op.dim), size: $(op.size)")
    insert!(curr_sz, op.dim, 1)
    reshaped = Base.reshape(a, curr_sz...)
    repeats = ones(Int, length(curr_sz))
    repeats[op.dim] = op.size
    res = repeat(reshaped, outer=repeats)
    println("Expand result size: $(size(res))")
    return res
end

execute_op(op::FusedElementwiseOp, inputs...) = broadcast(op.f, inputs...)

function execute_op(op::FlashAttentionOp, q, k, v)
    out = similar(q)
    return execute_op!(out, op, q, k, v)
end

function execute_op(op::Function, inputs...)
    if op.name == "InputTensor"
        return nothing 
    elseif op.name == "ARange"
        error("ARange requires context")
    elseif op.name == "Gather"
        indices = Int.(inputs[2]) .+ 1
        return inputs[1][indices, :]
    elseif op.name == "CumSum"
        return cumsum(inputs[1], dims=ndims(inputs[1]))
    else
        error("Function op with name $(op.name) not implemented.")
    end
end

# --- Dispatchable Execute functions (In-Place) ---

execute_op!(out, op::Add, a, b) = broadcast!(+, out, a, b)
execute_op!(out, op::Mul, a, b) = broadcast!(*, out, a, b)
execute_op!(out, op::Mod, a, b) = broadcast!(%, out, a, b)
execute_op!(out, op::LessThan, a, b) = broadcast!((x,y)->Float32(x<y), out, a, b)
execute_op!(out, op::FusedMulAdd, a, b, c) = broadcast!((x,y,z)->x*y+z, out, a, b, c)
execute_op!(out, op::FusedAddReLU, a, b) = broadcast!((x,y)->max(x+y, 0), out, a, b)
execute_op!(out, op::Log2, a) = broadcast!(log2, out, a)
execute_op!(out, op::Exp2, a) = broadcast!(exp2, out, a)
execute_op!(out, op::Sin, a) = broadcast!(sin, out, a)
execute_op!(out, op::Cos, a) = broadcast!(cos, out, a)
execute_op!(out, op::Sqrt, a) = broadcast!(sqrt, out, a)
execute_op!(out, op::Recip, a) = broadcast!(x->1.0f0/x, out, a)

function execute_op!(out, op::Reshape, a)
    # copyto! allows different shapes if length matches? 
    # Usually copyto!(dest, src).
    # ensure 'a' is viewed as matching length.
    if length(out) != length(a)
        error("Length mismatch in Reshape!: $(length(out)) vs $(length(a))")
    end
    copyto!(out, a)
    return out
end

function execute_op!(out, op::Permute, a)
    permutedims!(out, a, op.dims)
    return out
end

execute_op!(out, op::Contiguous, a) = copyto!(out, a)
execute_op!(out, op::MatMul, a, b) = batch_matmul!(out, a, b)

function execute_op!(out, op::SumReduce, a)
    # out has dropped dims.
    # sum! logic: sum!(dest, src) -> dest dims must include the ones being summed over as size 1?
    # Base.sum! docs: "sums elements of A over dimensions... to matching dimensions of R"
    # If op.dim was dropped in `out`, we need to reshape `out` to include it as size 1.
    
    # Calculate expected R shape for sum!
    # It must equal size(a) but with size 1 at op.dim.
    r_shape = [size(out)...]
    # Insert 1 at op.dim (if it was dropped)
    # This depends on how `out` was allocated.
    # If `out` is truly reduced rank (N-1), we need a view.
    if ndims(out) < ndims(a)
        insert!(r_shape, op.dim, 1)
        r_view = Base.reshape(out, r_shape...)
        sum!(r_view, a)
    else
        sum!(out, a)
    end
    return out
end

function execute_op!(out, op::MaxReduce, a)
    # Similar logic for maximum!
    r_shape = [size(out)...]
    if ndims(out) < ndims(a)
        insert!(r_shape, op.dim, 1)
        r_view = Base.reshape(out, r_shape...)
        maximum!(r_view, a)
    else
        maximum!(out, a)
    end
    return out
end

execute_op!(out, op::Slice, a) = execute_slice!(out, a, op.ranges)
execute_op!(out, op::Pad, a) = execute_pad!(out, a, op.padding)
execute_op!(out, op::Constant, device) = copyto!(out, to_device(op.value, device))

function execute_op!(out, op::Expand, a)
    # expand(a) -> repeat to fills out.
    # Just broadcast!?
    # `out .= a` should work if dimensions align or broadcast rules apply.
    # Expand changes defaults.
    # Julia broadcast automatically expands singleton dims.
    # So if `a` has size 1 where `out` has size N, `out .= a` works.
    # BUT `Expand` op might insert a dimension that wasn't there?
    # If `a` is (N,), Expand(dim=2) -> (N, 1) -> (N, M).
    # `a` needs to be reshaped to (N, 1) first if it isn't already compatible.
    
    # Check if we need to reshape `a`
    # The `compile` realize_view logic might handle inputs? 
    # But `Expand` takes the direct input from another node.
    # In `execute_op`: `curr_sz = insert!(...); reshaped = reshape(a, ...)`
    # We should do the same here.
    
    curr_sz = [size(a)...]
    insert!(curr_sz, op.dim, 1)
    reshaped_a = Base.reshape(a, curr_sz...)
    
    # helper for broadcast copy
    # copyto!(out, reshaped_a) -> this only works if sizes match.
    # broadcast!(identity, out, reshaped_a) -> this does expansion!
    broadcast!(identity, out, reshaped_a)
    return out
end

# CUDA Kernel for Flash Attention Forward
function flash_attn_fwd_kernel(O, Q, K, V, B, H, N, d, scale, causal)
    i = blockIdx().x # sequence index (1 to N)
    bh = blockIdx().y # batch*head index (1 to B*H)
    # Correcting for 1-based indexing in Julia
    b = (bh - 1) ÷ H + 1
    h = (bh - 1) % H + 1
    tid = threadIdx().x
    
    # Shared memory for row accumulation
    s_Q = CUDA.@cuDynamicSharedMem(Float32, d)
    s_O = CUDA.@cuDynamicSharedMem(Float32, d, d * sizeof(Float32))
    s_acc = CUDA.@cuDynamicSharedMem(Float32, blockDim().x, 2 * d * sizeof(Float32))

    # Load Q row
    if tid <= d
        s_Q[tid] = Q[b, h, i, tid]
        s_O[tid] = 0.0f0
    end
    
    m_i = -1f32 / 0f32
    l_i = 0.0f0
    
    sync_threads()
    
    for j in 1:N
        if causal && j > i continue end
        
        # 1. Compute S_ij = sum(Q[i, :] * K[j, :]) * scale
        val = 0.0f0
        if tid <= d
            val = s_Q[tid] * K[b, h, j, tid]
        end
        s_acc[tid] = val
        sync_threads()
        
        # Parallel reduction for dot product
        # Using a simple block reduction
        s = 1
        while s < blockDim().x
            s *= 2
        end
        s ÷= 2
        while s >= 1
            if tid <= s && tid + s <= blockDim().x
                s_acc[tid] += s_acc[tid + s]
            end
            sync_threads()
            s ÷= 2
        end
        dot = s_acc[1] * scale
        
        # 2. Update stats (Online Softmax)
        m_curr = dot
        m_next = max(m_i, m_curr)
        p = exp(m_curr - m_next)
        scale_old = exp(m_i - m_next)
        # Avoid NaN if m_i and m_curr are both -Inf (masked)
        if isnan(scale_old) scale_old = 0.0f0 end
        
        l_next = l_i * scale_old + p
        
        # 3. Update Output row (unnormalized)
        if tid <= d
            s_O[tid] = s_O[tid] * scale_old + p * V[b, h, j, tid]
        end
        
        m_i = m_next
        l_i = l_next
        sync_threads()
    end
    
    # Final normalization and write back
    if tid <= d
        O[b, h, i, tid] = s_O[tid] / l_i
    end
    return
end

# CPU Fallback for Flash Attention
function flash_attn_cpu(q, k, v, scale, causal)
    B, H, N, D = size(q)
    out = similar(q)
    for b in 1:B, h in 1:H
        for i in 1:N
            m_i = -Inf32
            l_i = 0.0f0
            o_row = zeros(Float32, D)
            for j in 1:N
                if causal && j > i continue end
                # Dot product
                dot = sum(q[b, h, i, :] .* k[b, h, j, :]) * scale
                # Online softmax
                m_next = max(m_i, dot)
                p = exp(dot - m_next)
                scale_old = exp(m_i - m_next)
                if isnan(scale_old) scale_old = 0.0f0 end
                
                o_row = o_row .* scale_old .+ p .* v[b, h, j, :]
                l_i = l_i * scale_old + p
                m_i = m_next
            end
            out[b, h, i, :] = o_row ./ l_i
        end
    end
    return out
end

execute_op!(out, op::FusedElementwiseOp, inputs...) = broadcast!(op.f, out, inputs...)

function execute_op!(out, op::FlashAttentionOp, q, k, v)
    if q isa CuArray
        B, H, N, d = size(q)
        # We need to ensure out has same shape
        # Grid: (N, B*H)
        # Block: (max(d, 32),)
        threads = max(32, 1 << (31 - leading_zeros(d - 1) + 1)) # Next power of 2
        shmem = (2 * d + threads) * sizeof(Float32)
        @cuda threads=threads blocks=(N, B*H) shmem=shmem flash_attn_fwd_kernel(out, q, k, v, B, H, N, d, op.scale, op.causal)
    else
        copyto!(out, flash_attn_cpu(q, k, v, op.scale, op.causal))
    end
    return out
end

function execute_op!(out, op::Function, inputs...)
    if op.name == "InputTensor"
        return nothing
    elseif op.name == "ARange"
        error("ARange requires context")
    elseif op.name == "Gather"
        # inputs[1][indices, :] -> out
        indices = Int.(inputs[2]) .+ 1
        # In-place gather?
        # out .= inputs[1][indices, :]
        # This allocates the temporary slice?
        # Yes.
        # Efficient gather! kernels exists in CUDA.jl/NNlib?
        # For now:
        copyto!(out, inputs[1][indices, :])
    elseif op.name == "CumSum"
        cumsum!(out, inputs[1], dims=ndims(inputs[1]))
    else
        error("Function op with name $(op.name) not implemented.")
    end
    return out
end

function execute(graph::Graph, output_id::Int, initial_inputs::Dict, device::AbstractDevice=get_device())
    results = to_device(initial_inputs, device)

    for (node_id, node) in enumerate(graph.nodes)
        haskey(results, node_id) && continue

        op = node.op
        
        # Realize each input according to its ShapeTracker
        input_values = []
        for (id, _, st) in node.inputs
            raw_data = results[id]
            push!(input_values, realize_view(raw_data, st))
        end
        
        node_shape = graph.shapes[node_id]

        if op isa Function && op.name == "ARange"
             n = eval_dim(realized_dims(node_shape)[1])
             current_result = to_device(Float32.(collect(0:n-1)), device)
        else
            # For Constant, we need device
            if op isa Constant
                 current_result = execute_op(op, device)
            else
                 current_result = execute_op(op, input_values...)
            end
        end
        
        results[node_id] = current_result
    end

    return from_device(results[output_id])
end

function eval_dim(d)
    if d isa Int
        return d
    elseif d isa BasicSymbolic
        error("Cannot evaluate symbolic dimension $d without context.")
    else
        return Int(d)
    end
end
