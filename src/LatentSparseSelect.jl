# Latent Sparse Attention's selection on the GPU - one decode token's list of keys to read, built without leaving the device.
#
# The CPU reference is `LatentSparse.read_selection` (and the torch oracle behind it). Three kernels, each a stage of it:
#   1. `lsa_index_query!`  the index query: the position-free dims of q, a group's heads summed, projected and normalized.
#   2. `lsa_score!`        every visible bucket's score, centroid . index query - one thread a bucket.
#   3. `lsa_pick_expand!`  one 1024-thread workgroup: the top_p buckets by a radix select on the scores (the threshold key, then the
#                          candidates in index order, so a tie goes to the lower bucket), sorted best first by a bitonic sort, expanded
#                          to key ids by a prefix scan, then the window and the sinks, -1 padded.
# `lsa_select!` runs the three and returns the list `decode_attention_gather!` (DecodeAttnGather.jl) reads.
#
# Positions in tables and id lists are 0-based, as the oracle's and the fixtures' are. `posv` holds the query token's position.
# Device layouts: centroids (buckets, index_dim) Float32; members (bucket_cap, buckets) Int32, -1 padded - the transpose of the
# fixtures' (buckets, bucket_cap), so a bucket's keys share a cache line; the index projection transposed, (index_dim, kv_heads *
# position_free_dims) Float32. One sequence a call.

using AMDGPU
using KernelAbstractions

const LS_MAX_TOP_P = 1024          # the pick kernel's workgroup size
const LS_THREADS = 1024

# An order-preserving map of a Float32 onto a UInt32 - larger score, larger key (a negative zero is made positive, as a comparison sees it)
@inline function _ls_key(s::Float32)
    s = s == 0f0 ? 0f0 : s
    u = reinterpret(UInt32, s)
    return (u & 0x80000000) != 0 ? ~u : (u | 0x80000000)
end

# Candidate record - the key above the bucket's complement, so a descending sort is by score, then by the lower bucket
@inline _ls_record(key::UInt32, bucket::Int) = (UInt64(key) << 32) | UInt64(0xffffffff - UInt32(bucket))

# Exclusive prefix sum of one Int32 a thread across the 1024-thread workgroup, and the total. `scan` is 1024 Int32 of LDS.
@inline function _ls_exclusive_scan!(scan, tid, v::Int32)
    @inbounds scan[tid + 1] = v
    sync_workgroup()
    off = 1
    while off < LS_THREADS
        t = tid >= off ? (@inbounds scan[tid + 1 - off]) : Int32(0)
        sync_workgroup()
        @inbounds scan[tid + 1] += t
        sync_workgroup()
        off *= 2
    end
    incl = @inbounds scan[tid + 1]
    total = @inbounds scan[LS_THREADS]
    sync_workgroup()
    return incl - v, total
end

# 8 weights of one output along the input dim - a helper, as a closure in a kernel body must not capture a reassigned variable
@inline _ls_load8(w, j, i) = ntuple(u -> @inbounds(w[j + 1, i + u]), Val(8))

# --- 1. the index query -----------------------------------------------------------------------------------------------

function _ls_index_query_kernel!(qidx, q, wq_t, rotary::Int, ::Val{G}) where {G}
    tid = workitemIdx().x - 1
    D = size(q, 1)
    width = D - rotary
    n_in = (size(q, 3) ÷ G) * width
    qpf = @ROCStaticLocalArray(Float32, (1024,), false)
    part = @ROCStaticLocalArray(Float32, (256,), false)
    red = @ROCStaticLocalArray(Float32, (64,), false)
    i = tid
    while i < n_in
        kv = i ÷ width; d = rotary + i % width
        s = 0f0
        for g in 0:G-1
            s += @inbounds q[d + 1, 1, kv * G + g + 1, 1]
        end
        @inbounds qpf[i + 1] = s
        i += 256
    end
    sync_workgroup()
    j = tid % 64; chunk = cld(n_in, 4)
    lo = (tid ÷ 64) * chunk; hi = min(lo + chunk, n_in)
    acc = 0f0
    i = lo
    # Eight inputs at a time, so their weight loads are in flight together
    while i + 7 < hi
        ws = _ls_load8(wq_t, j, i)
        for u in 1:8
            acc += (@inbounds qpf[i + u]) * ws[u]
        end
        i += 8
    end
    while i < hi
        acc += (@inbounds qpf[i + 1]) * (@inbounds wq_t[j + 1, i + 1])
        i += 1
    end
    @inbounds part[tid + 1] = acc
    sync_workgroup()
    total = 0f0
    if tid < 64
        total = (@inbounds part[tid + 1]) + (@inbounds part[tid + 65]) + (@inbounds part[tid + 129]) + (@inbounds part[tid + 193])
        @inbounds red[tid + 1] = total * total
    end
    sync_workgroup()
    if tid < 64
        ss = 0f0
        for t in 1:64
            ss += @inbounds red[t]
        end
        @inbounds qidx[tid + 1] = total / max(sqrt(ss), 1f-30)
    end
    return
end

"""
    lsa_index_query!(qidx, q, wq_t; rotary_dim) -> qidx

The index query of one decode token. `qidx` (64,) Float32 out; `q` (D, 1, H, 1) Float32, as the attention op receives it; `wq_t` the query
projection transposed, (64, kv_heads * (D - rotary_dim)) Float32. Needs index_dim 64 and kv_heads * (D - rotary_dim) <= 1024.
"""
function lsa_index_query!(qidx::ROCArray{Float32,1}, q::ROCArray{Float32,4}, wq_t::ROCArray{Float32,2}; rotary_dim::Integer)
    D, S, H, B = size(q)
    (S == 1 && B == 1) || throw(ArgumentError("q must be (D, 1, H, 1) - got $(size(q))"))
    size(wq_t, 1) == 64 && length(qidx) == 64 || throw(ArgumentError("the index dim must be 64 - got projection $(size(wq_t)) and qidx $(length(qidx))"))
    0 <= rotary_dim < D || throw(ArgumentError("rotary_dim must be in 0:$(D - 1) - got $rotary_dim"))
    n_in = size(wq_t, 2)
    KVH, rem = divrem(n_in, D - rotary_dim)
    (rem == 0 && KVH >= 1 && H % KVH == 0) ||
        throw(ArgumentError("the projection reads $n_in dims, which is not whole kv heads of $(D - rotary_dim) over $H query heads"))
    n_in <= 1024 || throw(ArgumentError("at most 1024 input dims - got $n_in"))
    @roc groupsize=256 gridsize=1 _ls_index_query_kernel!(qidx, q, wq_t, Int(rotary_dim), Val(H ÷ KVH))
    return qidx
end

# --- 2. the scores ----------------------------------------------------------------------------------------------------

function _ls_score_kernel!(scores, centroids, qidx, posv, block::Int, per_block::Int, ::Val{ID}) where {ID}
    tid = workitemIdx().x - 1
    qs = @ROCStaticLocalArray(Float32, (ID,), false)
    tid < ID && (@inbounds qs[tid + 1] = qidx[tid + 1])
    sync_workgroup()
    bucket = (workgroupIdx().x - 1) * 256 + tid
    p = unsafe_trunc(Int, @inbounds posv[1])
    bucket < min(p ÷ block * per_block, size(centroids, 1)) || return
    acc = 0f0
    for j in 1:ID
        acc += (@inbounds centroids[bucket + 1, j]) * (@inbounds qs[j])
    end
    @inbounds scores[bucket + 1] = acc
    return
end

"""
    lsa_score!(scores, centroids, qidx, posv; cluster_block_size, block_bucket_count) -> scores

Score every bucket the token at `posv[1]` may see - the buckets of the cluster blocks closed before its own - by `centroid . qidx`.
`scores` (buckets,) Float32 out, written only for the visible ones; `centroids` (buckets, index_dim) Float32; `qidx` (index_dim,); `posv` (1,) Float32.
"""
function lsa_score!(scores::ROCArray{Float32,1}, centroids::ROCArray{Float32,2}, qidx::ROCArray{Float32,1}, posv::ROCArray{Float32,1};
                    cluster_block_size::Integer, block_bucket_count::Integer)
    size(centroids, 1) == length(scores) || throw(ArgumentError("scores $(length(scores)) and centroids $(size(centroids)) disagree on the bucket count"))
    size(centroids, 2) == length(qidx) || throw(ArgumentError("qidx has $(length(qidx)) dims and the centroids $(size(centroids, 2))"))
    length(qidx) <= 256 || throw(ArgumentError("at most 256 index dims - got $(length(qidx))"))
    @roc groupsize=256 gridsize=cld(length(scores), 256) _ls_score_kernel!(scores, centroids, qidx, posv, Int(cluster_block_size),
                                                                         Int(block_bucket_count), Val(length(qidx)))
    return scores
end

# --- 3. pick and expand -----------------------------------------------------------------------------------------------

function _ls_pick_kernel!(ids, lastpos, scores, members, posv, top_p::Int, top_k::Int, sink::Int, window::Int, block::Int, per_block::Int)
    tid = workitemIdx().x - 1
    hist = @ROCStaticLocalArray(Int32, (256,), false)
    cand = @ROCStaticLocalArray(UInt64, (1024,), false)
    scan = @ROCStaticLocalArray(Int32, (1024,), false)
    sc = @ROCStaticLocalArray(Int32, (4,), false)
    p = unsafe_trunc(Int, @inbounds posv[1])
    visible = min(p ÷ block * per_block, length(scores))
    pe = min(top_p, visible)
    cap = size(members, 1)
    ws = max(0, p + 1 - window)

    # The threshold key of the pe-th best score, by a radix select in four 8-bit passes - `remaining` of the keys equal to it are wanted
    prefix = UInt32(0)
    remaining = pe
    if pe > 0
        for pass in 0:3
            shift = 24 - 8 * pass
            tid < 256 && (@inbounds hist[tid + 1] = Int32(0))
            sync_workgroup()
            i = tid
            while i < visible
                key = _ls_key(@inbounds scores[i + 1])
                if pass == 0 || (key >> (shift + 8)) == prefix
                    KernelAbstractions.@atomic hist[((key >> shift) & 0xff) + 1] += Int32(1)
                end
                i += LS_THREADS
            end
            sync_workgroup()
            # Thread j holds digit 255 - j, so an inclusive scan runs from the best digit down
            v = tid < 256 ? (@inbounds hist[256 - tid]) : Int32(0)
            @inbounds scan[tid + 1] = v
            sync_workgroup()
            off = 1
            while off < 256
                t = (tid < 256 && tid >= off) ? (@inbounds scan[tid + 1 - off]) : Int32(0)
                sync_workgroup()
                tid < 256 && (@inbounds scan[tid + 1] += t)
                sync_workgroup()
                off *= 2
            end
            if tid < 256
                incl = @inbounds scan[tid + 1]; excl = incl - v
                if incl >= remaining && excl < remaining
                    @inbounds sc[1] = Int32(255 - tid)
                    @inbounds sc[2] = Int32(remaining) - excl
                end
            end
            sync_workgroup()
            prefix = (prefix << 8) | UInt32(@inbounds sc[1])
            remaining = Int(@inbounds sc[2])
            sync_workgroup()
        end
    end
    n_gt = pe - remaining

    # The candidates in index order - every key above the threshold, then the first `remaining` equal to it
    slice = cld(max(visible, 1), LS_THREADS)
    lo = tid * slice; hi = min(lo + slice, visible)
    cg = Int32(0); ce = Int32(0)
    if pe > 0
        for i in lo:hi-1
            key = _ls_key(@inbounds scores[i + 1])
            cg += Int32(key > prefix); ce += Int32(key == prefix)
        end
    end
    ex_gt, _ = _ls_exclusive_scan!(scan, tid, cg)
    ex_eq, _ = _ls_exclusive_scan!(scan, tid, ce)
    gpos = Int(ex_gt); epos = Int(ex_eq)
    if pe > 0
        for i in lo:hi-1
            key = _ls_key(@inbounds scores[i + 1])
            if key > prefix
                @inbounds cand[gpos + 1] = _ls_record(key, i)
                gpos += 1
            elseif key == prefix
                epos < remaining && (@inbounds cand[n_gt + epos + 1] = _ls_record(key, i))
                epos += 1
            end
        end
    end
    j = pe + tid
    while j < LS_THREADS
        @inbounds cand[j + 1] = UInt64(0)
        j += LS_THREADS
    end
    sync_workgroup()

    # Best first - a bitonic sort, descending
    k = 2
    while k <= LS_THREADS
        jj = k >> 1
        while jj >= 1
            ixj = tid ⊻ jj
            if ixj > tid
                a = @inbounds cand[tid + 1]; b = @inbounds cand[ixj + 1]
                descending = (tid & k) == 0
                if (descending && a < b) || (!descending && a > b)
                    @inbounds cand[tid + 1] = b; @inbounds cand[ixj + 1] = a
                end
            end
            sync_workgroup()
            jj >>= 1
        end
        k <<= 1
    end

    # Expand the picked buckets, best first - the keys outside the sinks and the window, cut at top_k
    nvalid = Int32(0); bucket = 0
    if tid < pe
        bucket = Int(0xffffffff - UInt32(@inbounds(cand[tid + 1]) & 0xffffffff))
        for s in 1:cap
            m = @inbounds members[s, bucket + 1]
            nvalid += Int32(m >= sink && m < ws)
        end
    end
    ex, total = _ls_exclusive_scan!(scan, tid, nvalid)
    n_exp = min(Int(total), top_k)
    if tid < pe
        w = Int(ex)
        for s in 1:cap
            m = @inbounds members[s, bucket + 1]
            if m >= sink && m < ws
                w < top_k && (@inbounds ids[w + 1] = m)
                w += 1
            end
        end
    end
    wlen = p - ws + 1
    slen = min(sink, ws, p + 1)
    j = tid
    while j < wlen
        @inbounds ids[n_exp + j + 1] = Int32(ws + j)
        j += LS_THREADS
    end
    j = tid
    while j < slen
        @inbounds ids[n_exp + wlen + j + 1] = Int32(j)
        j += LS_THREADS
    end
    len = n_exp + wlen + slen
    j = len + tid
    while j < length(ids)
        @inbounds ids[j + 1] = Int32(-1)
        j += LS_THREADS
    end
    tid == 0 && (@inbounds lastpos[1] = Float32(len - 1))
    return
end

"""
    lsa_pick_expand!(ids, lastpos, scores, members, posv; top_p, top_k, sink_size, window_size, cluster_block_size, block_bucket_count)

The keys the token at `posv[1]` reads. Picks the `top_p` best of its visible buckets' `scores` (a tie to the lower bucket), best first, and
expands them to their keys - those past the sinks and before the window, `top_k` at most - then the window, then the sinks. `ids` (width, 1)
Int32 out, 0-based positions, -1 padded, `width` at least `top_k + window_size + sink_size`; `lastpos` (1,) Float32 out, the count of ids less one,
which is `decode_attention_gather!`'s convention; `members` (bucket_cap, buckets) Int32, -1 padded.
"""
function lsa_pick_expand!(ids::ROCArray{Int32,2}, lastpos::ROCArray{Float32,1}, scores::ROCArray{Float32,1}, members::ROCArray{Int32,2},
                          posv::ROCArray{Float32,1}; top_p::Integer, top_k::Integer, sink_size::Integer, window_size::Integer,
                          cluster_block_size::Integer, block_bucket_count::Integer)
    1 <= top_p <= LS_MAX_TOP_P || throw(ArgumentError("top_p must be in 1:$LS_MAX_TOP_P - got $top_p"))
    (top_k >= 1 && window_size >= 1 && sink_size >= 0) || throw(ArgumentError("top_k and window_size must be positive and sink_size non-negative"))
    size(members, 2) == length(scores) || throw(ArgumentError("members $(size(members)) and scores $(length(scores)) disagree on the bucket count"))
    size(ids, 1) >= top_k + window_size + sink_size && size(ids, 2) == 1 ||
        throw(ArgumentError("ids must be (at least $(top_k + window_size + sink_size), 1) - got $(size(ids))"))
    @roc groupsize=LS_THREADS gridsize=1 _ls_pick_kernel!(ids, lastpos, scores, members, posv, Int(top_p), Int(top_k), Int(sink_size),
                                                       Int(window_size), Int(cluster_block_size), Int(block_bucket_count))
    return ids
end

"""
    lsa_select!(ids, lastpos, qidx, scores, q, wq_t, centroids, members, posv; rotary_dim, top_p, top_k, sink_size, window_size,
                cluster_block_size, block_bucket_count) -> ids

One decode token's selection, all on the device: `lsa_index_query!`, `lsa_score!`, `lsa_pick_expand!`. `qidx` and `scores` are scratch.
The result, `ids` and `lastpos`, is what `decode_attention_gather!` reads.
"""
function lsa_select!(ids, lastpos, qidx, scores, q, wq_t, centroids, members, posv; rotary_dim::Integer, top_p::Integer, top_k::Integer,
                     sink_size::Integer, window_size::Integer, cluster_block_size::Integer, block_bucket_count::Integer)
    lsa_index_query!(qidx, q, wq_t; rotary_dim)
    lsa_score!(scores, centroids, qidx, posv; cluster_block_size, block_bucket_count)
    lsa_pick_expand!(ids, lastpos, scores, members, posv; top_p, top_k, sink_size, window_size, cluster_block_size, block_bucket_count)
    return ids
end
