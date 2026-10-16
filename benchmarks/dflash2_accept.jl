# Stage 1 of the DFlash2 port (docs/dflash2_port_plan.md): how many tokens does the block drafter get right?
#
# A reference implementation of the draft model with plain array operations (Float32, no graph compiler), run on texts the target generated itself
# (greedy), with the target's hidden states after layers 6, 20, 34, 48, 62 as the context. For positions p along each text it drafts the block
# [seed = token p, MASK x 7] and counts how many of the 7 proposed tokens equal the tokens the target actually produced at p+1 .. p+7 (the
# longest matching prefix is what verification would accept). Compare the resulting tokens per cycle (1 + accepted) with MTP's 2.4-3.3.
#
# Usage: LUMINAL_GGUF_NATIVE=1 julia -t auto --project=. benchmarks/dflash2_accept.jl [target.gguf] [draft.gguf] [new_tokens=640]
# Needs the GPU to itself: the target (17 GB) and the draft in Float32 (7.6 GB) are both resident.
using Luminal, Luminal.NN, AMDGPU, Printf, Random, LinearAlgebra, Statistics, Serialization
const reshape = Base.reshape
const L = Luminal
const LT = L.LlamaTokenization
include(joinpath(@__DIR__, "..", "examples", "qwen35_server.jl"))        # render_chat (main() does not run on include)

target_path = length(ARGS) >= 1 ? ARGS[1] : "/mnt/LLMs/models/gufo/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf"
draft_path  = length(ARGS) >= 2 ? ARGS[2] : "/mnt/LLMs/models/gufo/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
NEW = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 640
const B = 8                                   # block size: the seed and 7 mask rows
const WIN = 2048                              # context window (sliding_window)
const TARGET_LAYERS = [6, 20, 34, 48, 62]     # 0-based layers whose output the draft reads
ROPE_STYLE = get(ENV, "DFLASH_ROPE", "half")  # "half" (rotate_half, HF) or "interleaved"
LAYER_OFFSET = parse(Int, get(ENV, "DFLASH_LAYER_OFFSET", "0"))   # 0: output of layer i; -1: its input (output of layer i - 1)

# ── the target and its texts (cached: the target stage takes minutes) ─────────
CACHE = get(ENV, "DFLASH_CACHE", "/tmp/dflash2_accept_cache.jls")
f = read_gguf(target_path)
if isfile(CACHE)
    texts, hidden = Serialization.deserialize(CACHE)
    println("loaded the texts and hidden states from $CACHE")
else
    f = read_gguf(target_path); cfg = qwen35_config(f); tok = LlamaTokenizer(f)
    println("loading the target...")
    W = load_qwen35_gguf(target_path, cfg; quantize = _ -> L.Int4)
    sess = LlamaSession(Qwen35(L.Graph(), nothing; cfg...), tok, target_path; max_seq = 4096, weights = W, decode_weights = L.Int4, kv_dtype = Float16)

    prompts = [
        "Explain step by step why the sky is blue during the day and red at sunset.",
        "Write a Python function that merges overlapping intervals, with a docstring and a few tests.",
        "A Julia test suite passes locally but fails intermittently on CI. List the likely causes and how to investigate each one.",
        "Summarize the plot of Romeo and Juliet in two paragraphs, then say what the play is really about.",
    ]
    texts = Vector{Int}[]
    for (k, pr) in enumerate(prompts)
        ids = Int.(LT.encode(tok, render_chat([(role = "user", content = pr)], nothing; thinking = true); bos = false))
        gen = generate_ids(sess, [ids]; max_new_tokens = NEW)[1]
        gen = [g for g in gen if !(g in tok.eos_ids)]
        push!(texts, vcat(ids, gen))
        @printf("text %d: %d prompt + %d generated tokens\n", k, length(ids), length(gen))
    end
    hidden = [L.Decoding.prefill_hiddens(sess, t, TARGET_LAYERS .+ LAYER_OFFSET) for t in texts]     # per text: 5 x (5120, n, 1)
    println("captured the target hidden states")


    hidden = [[Array(h) for h in hs] for hs in hidden]      # to the host for the cache
    Serialization.serialize(CACHE, (texts, hidden))
end
hidden = [[ROCArray(h) for h in hs] for hs in hidden]

# ── the draft model, Float32 on the device ────────────────────────────────────
df = read_gguf(draft_path)
dev(a) = ROCArray(Array{Float32}(a))
gt(name) = gguf_tensor(df, name)
dmat(name) = dev(gt(name))                     # GGUF (in, out): used as x (rows, in) * W
vecw(name) = dev(reshape(gt(name), 1, :))      # a (1, dim) row for broadcasting
NL = 5; H = 5120; NH = 32; NKV = 8; HD = 128; FF = 17408; KS = 2; GS = 16; NG = H ÷ GS; EPS = 1f-6; THETA = 1f7
println("loading the draft...")
fc = dmat("fc.weight"); hnorm = vecw("enc.output_norm.weight"); onorm = vecw("output_norm.weight")
layers = map(0:NL-1) do i
    b = "blk.$i"
    (an = vecw("$b.attn_norm.weight"), fn = vecw("$b.ffn_norm.weight"), qn = vecw("$b.attn_q_norm.weight"), kn = vecw("$b.attn_k_norm.weight"),
     wq = dmat("$b.attn_q.weight"), wk = dmat("$b.attn_k.weight"), wv = dmat("$b.attn_v.weight"), wo = dmat("$b.attn_output.weight"),
     acp = dmat("$b.attn_conv_proj.weight"), acb = dev(gt("$b.attn_conv_base")),             # base (H, off, phase)
     fcp = dmat("$b.ffn_conv_proj.weight"), fcb = dev(gt("$b.ffn_conv_base")),
     wg = dmat("$b.ffn_gate.weight"), wu = dmat("$b.ffn_up.weight"), wd = dmat("$b.ffn_down.weight"))
end
selh = dmat("selector_hidden.weight")
pred = gt("selector_predecessor.weight"); succ = gt("selector_successor.weight")          # (256, vocab), host
fT = read_gguf(target_path)
embed = gguf_tensor(fT, "token_embd.weight")                                               # target's (5120, vocab), host
lm = ROCArray(Array{Float32}(gguf_tensor(fT, haskey(fT.tensors, "output.weight") ? "output.weight" : "token_embd.weight")))   # (5120, vocab)
VOCAB = size(lm, 2); MASK = Int(df.metadata["tokenizer.ggml.mask_token_id"])
println("draft loaded; vocab $VOCAB, mask token $MASK")

dn_rms(x, w) = x ./ sqrt.(sum(abs2, x; dims = 2) ./ size(x, 2) .+ EPS) .* w
function dn_rope!(x::ROCArray{Float32,3}, pos::Vector{Int})              # x (rows, heads, HD), position per row
    n, h, d = size(x); half = d ÷ 2
    inv = Float32[THETA^(-2f0 * (i - 1) / d) for i in 1:half]
    ang = Float32.(pos) * inv'                                          # (rows, half)
    c = ROCArray(cos.(ang)); s = ROCArray(sin.(ang))
    cc = reshape(c, n, 1, half); ss = reshape(s, n, 1, half)
    if ROPE_STYLE == "half"
        x1 = x[:, :, 1:half]; x2 = x[:, :, half+1:end]
        x[:, :, 1:half] .= x1 .* cc .- x2 .* ss
        x[:, :, half+1:end] .= x2 .* cc .+ x1 .* ss
    else
        xe = x[:, :, 1:2:end]; xo = x[:, :, 2:2:end]
        x[:, :, 1:2:end] .= xe .* cc .- xo .* ss
        x[:, :, 2:2:end] .= xo .* cc .+ xe .* ss
    end
    return x
end
dn_headnorm(x3, w) = x3 ./ sqrt.(sum(abs2, x3; dims = 3) ./ size(x3, 3) .+ EPS) .* reshape(w, 1, 1, :)

# dynamic causal conv over the rows of x (R, H): out[r] = (base[:,0] + dyn0[r, g]) * x[r] + (r > 1) (base[:,1] + dyn1[r, g]) * x[r-1]
function dn_conv(x, base, dyn, phase)
    R = size(x, 1)
    grp(off) = (d = dyn[:, phase*KS*NG+off*NG+1:phase*KS*NG+(off+1)*NG]; reshape(repeat(reshape(d, R, 1, NG), 1, GS, 1), R, H))
    w0 = reshape(base[:, 1, phase+1], 1, H) .+ grp(0)
    w1 = reshape(base[:, 2, phase+1], 1, H) .+ grp(1)
    xprev = vcat(AMDGPU.zeros(Float32, 1, H), x[1:end-1, :])
    return w0 .* x .+ w1 .* xprev
end
dn_silu(x) = x ./ (1f0 .+ exp.(-x))
# a projection's (rows, nh * HD) output, head h at features (h - 1) * HD + 1 : h * HD  ->  (rows, nh, HD), and back
heads(x, nh) = permutedims(reshape(x, size(x, 1), HD, nh), (1, 3, 2))
unheads(x) = reshape(permutedims(x, (1, 3, 2)), size(x, 1), :)

# per text: the projected context rows and every layer's K/V for all of them
ctxK = Vector{Any}(undef, length(texts)); ctxV = Vector{Any}(undef, length(texts))
for (t, hs) in enumerate(hidden)
    n = size(hs[1], 2)
    X = hcat((permutedims(Array(hs[j][:, :, 1]), (2, 1)) for j in 1:5)...)         # (n, 25600) host
    proj = dn_rms(ROCArray(X) * fc, hnorm)
    ks = Any[]; vs = Any[]
    for l in layers
        k = heads(proj * l.wk, NKV); k = dn_headnorm(k, vec(l.kn)); dn_rope!(k, collect(0:n-1))
        push!(ks, k); push!(vs, heads(proj * l.wv, NKV))
    end
    ctxK[t] = ks; ctxV[t] = vs
end

function draft_block(t::Int, p::Int, seed::Int)                          # -> the 7 proposed tokens
    lo = max(0, p - WIN); lc = p - lo
    X = ROCArray(permutedims(embed[:, [seed + 1; fill(MASK + 1, B - 1)]], (2, 1)))     # (B, H)
    qpos = collect(p:p+B-1)
    for (li, l) in enumerate(layers)
        res = X
        xn = dn_rms(X, l.an)
        dyn = xn * l.acp
        c = dn_conv(xn, l.acb, dyn, 0)
        q = heads(c * l.wq, NH); q = dn_headnorm(q, vec(l.qn)); dn_rope!(q, qpos)
        k = heads(c * l.wk, NKV); k = dn_headnorm(k, vec(l.kn)); dn_rope!(k, qpos)
        v = heads(c * l.wv, NKV)
        K = vcat(ctxK[t][li][lo+1:p, :, :], k); V = vcat(ctxV[t][li][lo+1:p, :, :], v)       # (lc + B, NKV, HD)
        att = AMDGPU.zeros(Float32, B, NH, HD)
        for h in 1:NH
            g = (h - 1) ÷ (NH ÷ NKV) + 1
            sc = (q[:, h, :] * K[:, g, :]') .* (1f0 / sqrt(Float32(HD)))
            sc = exp.(sc .- maximum(sc; dims = 2)); sc = sc ./ sum(sc; dims = 2)
            att[:, h, :] .= sc * V[:, g, :]
        end
        ao = unheads(att) * l.wo
        X = res .+ dn_conv(ao, l.acb, dyn, 1)
        res = X
        xn = dn_rms(X, l.fn)
        dyn = xn * l.fcp
        c = dn_conv(xn, l.fcb, dyn, 0)
        ff = (dn_silu.(c * l.wg) .* (c * l.wu)) * l.wd
        X = res .+ dn_conv(ff, l.fcb, dyn, 1)
    end
    Hd = dn_rms(X, onorm)[2:end, :]                                       # rows 1..B-1
    logits = Array(Hd * lm)                                                # (7, vocab)
    proj = Array(Hd * selh)                                                # (7, 256)
    if @isdefined(DIAG_TRUTH) && DIAG_TRUTH[] !== nothing
        for r in 1:B-1
            top = partialsortperm(view(logits, r, :), 1:16; rev = true); tr = DIAG_TRUTH[][r] + 1
            DIAG1[r] += (top[1] == tr); DIAG16[r] += (tr in top); DIAGN[r] += 1
        end
    end
    prev = seed; out = Int[]
    for r in 1:B-1
        top = partialsortperm(view(logits, r, :), 1:16; rev = true)
        sc = [logits[r, j] + sum(view(pred, :, prev + 1) .* view(proj, r, :) .* view(succ, :, j)) for j in top]
        prev = top[argmax(sc)] - 1
        push!(out, prev)
    end
    return out
end

const DIAG_TRUTH = Ref{Any}(nothing); const DIAG1 = zeros(Int, B - 1); const DIAG16 = zeros(Int, B - 1); const DIAGN = zeros(Int, B - 1)
hist = zeros(Int, B)                      # hist[a + 1]: blocks with exactly a accepted drafts
cond = zeros(Int, B - 1); condn = zeros(Int, B - 1)
for (t, ids) in enumerate(texts)
    n = length(ids); np = n - NEW
    first_p = max(np, 64)
    acc_t = Int[]
    for p in first_p:16:n-B
        seed = ids[p+1]                                                    # token at (0-based) position p
        truth = ids[p+2:p+B]
        DIAG_TRUTH[] = truth
        d = draft_block(t, p, seed)
        a = 0
        while a < B - 1 && d[a+1] == truth[a+1]; a += 1; end
        hist[a+1] += 1; push!(acc_t, a)
        for j in 1:B-1
            j > a + 1 && break
            condn[j] += 1; j <= a && (cond[j] += 1)
        end
    end
    @printf("text %d: %d blocks, mean accepted drafts %.2f (tokens/cycle %.2f)\n", t, length(acc_t), mean(acc_t), 1 + mean(acc_t))
end
nb = sum(hist); ma = sum((0:B-1) .* hist) / nb
println("blocks: ", nb, "  accepted-length histogram (0..7): ", hist)
@printf("mean accepted drafts %.2f -> %.2f tokens per cycle (MTP today: 2.4-3.3)\n", ma, 1 + ma)
println("conditional acceptance of draft j given 1..j-1 matched: ", join((@sprintf("d%d %.0f%%", j, 100 * cond[j] / max(condn[j], 1)) for j in 1:B-1), ", "))

println("row logits top-1 = truth: ", join((@sprintf("r%d %.0f%%", r, 100 * DIAG1[r] / max(DIAGN[r], 1)) for r in 1:B-1), ", "))
println("row logits top-16 contain truth: ", join((@sprintf("r%d %.0f%%", r, 100 * DIAG16[r] / max(DIAGN[r], 1)) for r in 1:B-1), ", "))
