# Latent Sparse Attention in the decode step - the glue between the kernels (DecodeAttnGather.jl, LatentSparseSelect.jl), the CPU bucket
# build (LatentSparse.jl) and a model's decode graph.
#
# A full-attention layer with a map gets an `LSADecodeAttention` op in place of `DecodeAttention`. Per decode token the op writes the new
# K/V into the cache, makes the token's index key (`normalize(k_pf . W_K)`) and stores it, selects the keys to read, and attends over
# them - all on the device, with nothing allocated, so it records into the captured HIP graph the decode step is replayed from.
# What the graph cannot do stays on the host, between steps:
#   - `lsa_after_prefill!` makes the index keys of a prompt from the K cache and builds the buckets of its closed 512-key blocks;
#   - `lsa_after_step!` builds a block's buckets (the CPU Ward, ~0.3 ms) the step its last key is written.
# Both are called by the decode and prefill drivers (NN.llama_decode_step!, Decoding._prefill_into_cache!) when a session has LSA enabled
# (`Decoding.lsa_enable!`).
#
# One sequence, one token a step, a Float16 cache, an AMD GPU. A speculative (several-token) step, a batch, an int8 cache and a prefix
# cache are not supported: the decode graph refuses them when LSA is on, or leaves them dense, where the index would go stale.

using AMDGPU

"""
    LSAConfig(maps; rotary_dim = 64, top_k = 4096, top_p = 1024, sink_size = 4, window_size = 512, pool_size = 4, bucket_size = 8, bucket_cap = 16)

The design's sizes and the learned maps. `maps` takes a decoder layer's 0-based index to its `(W_Q_idx, W_K_idx)`, each
`(kv_heads * (head_dim - rotary_dim), 64)` Float32 - see `lsa_load_maps`. The cluster block is `window_size` keys, as in the design.
"""
Base.@kwdef struct LSAConfig
    maps::Dict{Int,Tuple{Matrix{Float32},Matrix{Float32}}}
    rotary_dim::Int = 64
    top_k::Int = 4096
    top_p::Int = 1024
    sink_size::Int = 4
    window_size::Int = 512
    pool_size::Int = 4
    bucket_size::Int = 8
    bucket_cap::Int = 16
end

"""
    lsa_load_maps(path) -> Dict{Int,Tuple{Matrix{Float32},Matrix{Float32}}}

Read a D-LSA map file in safetensors (`<layer>.W_Q_idx`, `<layer>.W_K_idx`, as the release's `.pt` converts) into `LSAConfig`'s `maps`.
"""
function lsa_load_maps(path::AbstractString)
    tensors = SafeTensors.load_safetensors(path)
    layers = sort!(unique(parse(Int, first(split(k, "."))) for k in keys(tensors)))
    isempty(layers) && throw(ArgumentError("$path holds no maps"))
    return Dict(l => (Float32.(tensors["$l.W_Q_idx"]), Float32.(tensors["$l.W_K_idx"])) for l in layers)
end

"One layer's index state - its projections, the index key of every cached token, its buckets, and the scratch the kernels write."
mutable struct LSALayerState
    config::LSAConfig
    layer::Int
    wq_t::ROCArray{Float32,2}                # (64, kv_heads * (head_dim - rotary_dim))
    wk_t::ROCArray{Float32,2}
    index_keys::ROCArray{Float32,2}          # (64, max_seq)
    centroids::ROCArray{Float32,2}           # (max_buckets, 64)
    members::ROCArray{Int32,2}               # (bucket_cap, max_buckets), 0-based, -1 padded
    scores::ROCArray{Float32,1}              # (max_buckets,)
    qidx::ROCArray{Float32,1}                # (64,)
    kidx::ROCArray{Float32,1}                # (64,)
    ids::ROCArray{Int32,2}                   # (top_k + window + sinks, 1)
    lastpos::ROCArray{Float32,1}             # (1,)
end

"A session's LSA state - the design, a state a layer with a map, and how many 512-key blocks have their buckets built."
mutable struct LSAState
    config::LSAConfig
    layers::Dict{Int,LSALayerState}
    max_seq::Int
    built_blocks::Int
end

lsa_blocks(cfg::LSAConfig) = (cfg.window_size, cfg.window_size ÷ cfg.bucket_size)     # keys a block, buckets a block

"""
    LSAState(config, layers, max_seq)

The state for the decoder layers `layers` (0-based indices of the full-attention layers, each of which needs a map) of a session whose cache
holds `max_seq` positions. Everything is allocated here, so the decode step allocates nothing.
"""
function LSAState(config::LSAConfig, layers, max_seq::Int)
    config.window_size % config.bucket_size == 0 || throw(ArgumentError("bucket_size must divide the window - the cluster block - got $(config.bucket_size), $(config.window_size)"))
    config.top_p <= LS_MAX_TOP_P || throw(ArgumentError("top_p is at most $LS_MAX_TOP_P - got $(config.top_p)"))
    missing_maps = [l for l in layers if !haskey(config.maps, l)]
    isempty(missing_maps) || throw(ArgumentError("no map for layers $missing_maps - the file has $(sort(collect(keys(config.maps))))"))
    block, per_block = lsa_blocks(config)
    max_buckets = max_seq ÷ block * per_block
    max_buckets >= per_block || throw(ArgumentError("max_seq $max_seq holds no whole cluster block of $block keys"))
    width = config.top_k + config.window_size + config.sink_size
    states = Dict{Int,LSALayerState}()
    for l in layers
        wq, wk = config.maps[l]
        size(wq) == size(wk) && size(wq, 2) == 64 || throw(ArgumentError("layer $l's maps must both be (n, 64) - got $(size(wq)) and $(size(wk))"))
        states[l] = LSALayerState(config, l, ROCArray(permutedims(wq)), ROCArray(permutedims(wk)), AMDGPU.zeros(Float32, 64, max_seq),
                                  AMDGPU.zeros(Float32, max_buckets, 64), AMDGPU.fill(Int32(-1), config.bucket_cap, max_buckets),
                                  AMDGPU.zeros(Float32, max_buckets), AMDGPU.zeros(Float32, 64), AMDGPU.zeros(Float32, 64),
                                  AMDGPU.zeros(Int32, width, 1), AMDGPU.zeros(Float32, 1))
    end
    return LSAState(config, states, max_seq, 0)
end

# --- what a decode graph build and its drivers consult -----------------------------------------------------------------

const LSA_LAYER = Ref(-1)                       # the decoder layer (0-based) the decode graph is building
const LSA_ACTIVE = Ref{Any}(nothing)            # the LSAState the decode graph being built uses
# Weak, so a session or a decode cache that is dropped (a cache is 16 GiB at 262k positions) is not kept alive by its entry. Both are mutable
# structs, which compare and hash by identity.
const LSA_SESSIONS = WeakKeyDict{Any,LSAState}()    # a session -> its state, set by `Decoding.lsa_enable!`
const LSA_CACHES = WeakKeyDict{Any,LSAState}()      # a decode cache -> its state, for the drivers' hooks

lsa_layer_state(layer::Integer) = (st = LSA_ACTIVE[]; st === nothing ? nothing : get(st.layers, Int(layer), nothing))

"Run `f` with `st` as the state decode graphs are built with (nothing for dense)."
function with_lsa(f, st)
    old = LSA_ACTIVE[]
    LSA_ACTIVE[] = st
    try
        return f()
    finally
        LSA_ACTIVE[] = old
        LSA_LAYER[] = -1
    end
end

# --- the op -------------------------------------------------------------------------------------------------------------

"""
    LSADecodeAttention(scale, state)

`DecodeAttention`'s inputs and output for one token, over the keys Latent Sparse Attention selects rather than the whole cache. Writes the
token's K/V into the cache and its index key into the layer's state. `state` is a `LSALayerState`, kept by reference, with the design's `config`.
"""
struct LSADecodeAttention <: Op
    scale::Float32
    state::LSALayerState        # mutable, so an op hashes and compares by identity, whatever arrays the state holds
end

# Store a 64-vector as the column of the token at posv[1]
function _lsa_store_column_kernel!(dst, src, posv)
    tid = workitemIdx().x - 1
    pos = unsafe_trunc(Int, @inbounds posv[1])
    tid < 64 && (@inbounds dst[tid + 1, pos + 1] = src[tid + 1])
    return
end

function execute_op!(out, op::LSADecodeAttention, q, pk, pv, kn, vn, posv)
    st = op.state
    cfg = st.config
    D, S, H, B = size(q, 1), size(q, 2), size(q, 3), size(q, 4)
    KVH = size(pk, 3)
    (S == 1 && B == 1) || error("LSADecodeAttention: one token of one sequence a step - got S = $S, B = $B")
    (pk isa ROCArray{Float16,4} && pv isa ROCArray{Float16,4} && q isa ROCArray{Float32,4} && kn isa ROCArray{Float32,4} &&
     vn isa ROCArray{Float32,4} && out isa ROCArray{Float32}) ||
        error("LSADecodeAttention: needs an AMD GPU, a Float16 KV cache and Float32 activations")
    # The cache write - the dense op's own kernel
    @roc groupsize=256 gridsize=cld(D * S * KVH * B, 256) _da_write_kernel!(pk, pv, kn, vn, posv, S)
    # The token's index key, from its k as the attention receives it
    lsa_index_query!(st.kidx, kn, st.wk_t; rotary_dim = cfg.rotary_dim)
    @roc groupsize=64 gridsize=1 _lsa_store_column_kernel!(st.index_keys, st.kidx, posv)
    block, per_block = lsa_blocks(cfg)
    lsa_select!(st.ids, st.lastpos, st.qidx, st.scores, q, st.wq_t, st.centroids, st.members, posv; rotary_dim = cfg.rotary_dim,
                top_p = cfg.top_p, top_k = cfg.top_k, sink_size = cfg.sink_size, window_size = cfg.window_size,
                cluster_block_size = block, block_bucket_count = per_block)
    decode_attention_gather!(out, q, pk, pv, st.ids, st.lastpos, op.scale) ||
        error("LSADecodeAttention: the gather kernel declined head_dim $D, $H query heads on $KVH KV heads")
    return out
end

# --- host upkeep, between steps -----------------------------------------------------------------------------------------

# The index keys of positions [first, n) of one layer, from its K cache (D, max_seq, KVH, 1) Float16
function lsa_index_cache!(ls::LSALayerState, cfg::LSAConfig, pk, first::Int, n::Int)
    D, KVH = size(pk, 1), size(pk, 3)
    rot = cfg.rotary_dim
    chunk = 8192
    for a in first:chunk:n-1
        b = min(a + chunk, n)
        k = Float32.(view(pk, rot+1:D, a+1:b, :, 1))                                   # (D - rot, m, KVH)
        kpf = Base.reshape(permutedims(k, (1, 3, 2)), KVH * (D - rot), b - a)           # a token's kv heads side by side
        proj = ls.wk_t * kpf                                                            # (64, m)
        proj ./= max.(sqrt.(sum(abs2, proj; dims = 1)), 1f-30)
        copyto!(view(ls.index_keys, :, a+1:b), proj)
    end
    return ls
end

"Build the buckets of cluster blocks `from` to `to - 1` (0-based) of every layer, from their index keys, and upload them."
function lsa_build_blocks!(st::LSAState, from::Int, to::Int)
    to > from || return st
    cfg = st.config
    block, per_block = lsa_blocks(cfg)
    to * per_block <= size(first(values(st.layers)).centroids, 1) || error("LSA: block $to is past the cache's $(st.max_seq) positions")
    rows = from * per_block + 1:to * per_block
    for ls in values(st.layers)
        index_keys = Array(view(ls.index_keys, :, from * block + 1:to * block))             # (64, m * block), on the host
        centroids = Matrix{Float32}(undef, length(rows), 64)
        members = Matrix{Int32}(undef, cfg.bucket_cap, length(rows))
        for (j, blk) in enumerate(from:to-1)
            table = LatentSparse.build_buckets(permutedims(index_keys[:, (j-1)*block+1:j*block]), cfg.pool_size, cfg.bucket_size,
                                               cfg.bucket_cap, block)
            r = (j-1) * per_block + 1:j * per_block
            centroids[r, :] = table.centroids
            members[:, r] = permutedims(map(m -> m >= 0 ? m + Int32(blk * block) : m, table.members))   # block-local to absolute
        end
        view(ls.centroids, rows, :) .= ROCArray(centroids)
        view(ls.members, :, rows) .= ROCArray(members)
    end
    st.built_blocks = to
    return st
end

"After a prefill of `n` tokens: the index keys of positions [0, n) from the K caches, and the buckets of every closed block."
function lsa_after_prefill!(st::LSAState, cache, n::Int)
    n <= st.max_seq || error("LSA: $n tokens past the cache's $(st.max_seq) positions")
    for ls in values(st.layers)
        lsa_index_cache!(ls, st.config, cache.self_cache[ls.layer + 1][1], 0, n)
    end
    st.built_blocks = 0
    lsa_build_blocks!(st, 0, n ÷ lsa_blocks(st.config)[1])
    AMDGPU.synchronize()
    return st
end

"After a decode step: build a block's buckets the step its last key was written."
function lsa_after_step!(cache)
    st = get(LSA_CACHES, cache, nothing)
    st === nothing && return nothing
    closed = cache.positions[1] ÷ lsa_blocks(st.config)[1]
    closed > st.built_blocks && lsa_build_blocks!(st, st.built_blocks, closed)
    return nothing
end

"Forget every cache registered for `st` - a session drops the decode caches it built for a state it is replacing."
function lsa_forget_caches!(st::LSAState)
    for c in collect(keys(LSA_CACHES))
        get(LSA_CACHES, c, nothing) === st && delete!(LSA_CACHES, c)
    end
    return nothing
end

function lsa_register_cache!(st::LSAState, cache)
    LSA_CACHES[cache] = st
    return cache
end
