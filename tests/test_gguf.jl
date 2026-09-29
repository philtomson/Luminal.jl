# GGUF import: the reader and dequantizers, and the Qwen3.5 mapping from
# llama.cpp's conversion back to Hugging Face's. A synthetic GGUF is written from
# the tiny Qwen3.5 model's weights applying llama.cpp's conversion exactly
# (convert_hf_to_gguf.py, Qwen3_5TextModel: norm weights as 1 + w, A_log as
# -exp(A_log), the conv kernel squeezed, value heads reordered from grouped to
# tiled); load_qwen35_gguf must give the original weights back, and a session on
# them the transformers logits. Q8_0 is checked against its definition.
using Test
using Luminal
using Luminal.NN
using JSON3

const REF = joinpath(@__DIR__, "data", "qwen35_tiny")
const IDX = JSON3.read(read(joinpath(REF, "index.json"), String))
refarr(name) = Base.reshape(collect(reinterpret(Float32, read(joinpath(REF, "$name.f32")))),
                            reverse(Int.(IDX.arrays[Symbol(name)]))...)
relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

# ── a minimal GGUF writer (F32 and Q8_0 tensors) ──────────────────────────────
wstr(io, s) = (write(io, UInt64(ncodeunits(s))); write(io, s))
function q8_0(x::Vector{Float32})               # llama.cpp's reference Q8_0 quantizer
    out = UInt8[]
    for blk in Iterators.partition(x, 32)
        amax = maximum(abs, blk); d = amax / 127; id = d == 0 ? 0f0 : 1 / d
        append!(out, reinterpret(UInt8, [Float16(d)]))
        append!(out, reinterpret(UInt8, Int8[round(Int8, v * id) for v in blk]))
    end
    out
end
# tensors: name => (array in GGUF order, i.e. dims (ne1, ne2, ...), :F32 or :Q8_0)
function write_gguf(path, tensors)
    names = collect(keys(tensors))
    blobs = [tensors[n][2] === :Q8_0 ? q8_0(vec(Float32.(tensors[n][1]))) : reinterpret(UInt8, vec(Float32.(tensors[n][1]))) for n in names]
    open(path, "w") do io
        write(io, UInt32(0x46554747), UInt32(3), UInt64(length(names)), UInt64(0))
        off = 0
        for (n, b) in zip(names, blobs)
            a = tensors[n][1]
            wstr(io, n); write(io, UInt32(ndims(a))); foreach(d -> write(io, UInt64(d)), size(a))
            write(io, UInt32(tensors[n][2] === :Q8_0 ? 8 : 0)); write(io, UInt64(off))
            off += cld(length(b), 32) * 32
        end
        write(io, zeros(UInt8, cld(position(io), 32) * 32 - position(io)))
        for b in blobs
            write(io, b); write(io, zeros(UInt8, cld(length(b), 32) * 32 - length(b)))
        end
    end
end

# llama.cpp's conversion of the tiny model, from its HF weights (Julia (out, in))
function to_gguf(W, cfg)
    p = cfg.prefix; nk, nv, dk, dv = cfg.linear_nk, cfg.linear_nv, cfg.linear_dk, cfg.linear_dv
    G = nv ÷ nk
    # _reorder_v_heads: grouped (k, g) -> tiled (g, k) along rows
    function tiled(A, hd)
        n = size(A, 1); B = similar(A)
        for k in 0:nk-1, g in 0:G-1, d in 0:hd-1
            B[(g * nk + k) * hd + d + 1, :] = A[(k * G + g) * hd + d + 1, :]
        end
        B
    end
    T = Dict{String, Any}()
    m(A) = permutedims(A, (2, 1))                           # HF (out, in) -> GGUF (in, out)
    T["token_embd.weight"] = (m(W["$p.embed_tokens.weight"]), :F32)
    T["output.weight"] = (m(W["lm_head.weight"]), :Q8_0)
    T["output_norm.weight"] = (W["$p.norm.weight"] .+ 1f0, :F32)
    for i in 0:cfg.n_layers-1
        b = "blk.$i"; h = "$p.layers.$i"
        T["$b.attn_norm.weight"] = (W["$h.input_layernorm.weight"] .+ 1f0, :F32)
        T["$b.post_attention_norm.weight"] = (W["$h.post_attention_layernorm.weight"] .+ 1f0, :F32)
        T["$b.ffn_gate.weight"] = (m(W["$h.mlp.gate_proj.weight"]), :F32)
        T["$b.ffn_up.weight"] = (m(W["$h.mlp.up_proj.weight"]), :F32)
        T["$b.ffn_down.weight"] = (m(W["$h.mlp.down_proj.weight"]), :F32)
        if cfg.layer_types[i+1] == "linear_attention"
            la = "$h.linear_attn"; qk = 2nk * dk
            qkv = W["$la.in_proj_qkv.weight"]
            T["$b.attn_qkv.weight"] = (m(vcat(qkv[1:qk, :], tiled(qkv[qk+1:end, :], dv))), :F32)
            T["$b.attn_gate.weight"] = (m(tiled(W["$la.in_proj_z.weight"], dv)), :F32)
            T["$b.ssm_alpha.weight"] = (m(tiled(W["$la.in_proj_a.weight"], 1)), :F32)
            T["$b.ssm_beta.weight"] = (m(tiled(W["$la.in_proj_b.weight"], 1)), :F32)
            T["$b.ssm_a"] = (vec(tiled(Base.reshape(-exp.(W["$la.A_log"]), :, 1), 1)), :F32)
            T["$b.ssm_dt.bias"] = (vec(tiled(Base.reshape(W["$la.dt_bias"], :, 1), 1)), :F32)
            cw = W["$la.conv1d.weight"][:, 1, :]                               # (C, K)
            T["$b.ssm_conv1d.weight"] = (m(vcat(cw[1:qk, :], tiled(cw[qk+1:end, :], dv))), :F32)
            T["$b.ssm_norm.weight"] = (W["$la.norm.weight"], :F32)
            T["$b.ssm_out.weight"] = (m(permutedims(tiled(permutedims(W["$la.out_proj.weight"], (2, 1)), dv), (2, 1))), :F32)
        else
            sa = "$h.self_attn"
            for (g, hf) in (("attn_q", "q_proj"), ("attn_k", "k_proj"), ("attn_v", "v_proj"), ("attn_output", "o_proj"))
                T["$b.$g.weight"] = (m(W["$sa.$hf.weight"]), :F32)
            end
            T["$b.attn_q_norm.weight"] = (W["$sa.q_norm.weight"] .+ 1f0, :F32)
            T["$b.attn_k_norm.weight"] = (W["$sa.k_norm.weight"] .+ 1f0, :F32)
        end
    end
    return T
end

@testset "GGUF: Q8_0 blocks" begin
    x = randn(Float32, 256) .* 3
    path = tempname() * ".gguf"
    write_gguf(path, Dict("t" => (Base.reshape(x, 64, 4), :Q8_0)))
    f = read_gguf(path)
    @test f.tensors["t"].dims == [64, 4] && Luminal.ggml_type(f.tensors["t"]) === :Q8_0
    y = vec(gguf_tensor(f, "t"))
    d = [Float32(Float16(maximum(abs, x[i:i+31]) / 127)) for i in 1:32:256]
    @test all(abs.(y .- x) .<= repeat(d, inner=32) .* 0.6f0)             # half a step, plus the f16 rounding of the scale
end

@testset "GGUF: Qwen3.5 import round trip" begin
    cfg = qwen35_config(REF)
    @test cfg.linear_nv ÷ cfg.linear_nk == 2                               # heads really reordered
    W = load_weights_to_dict(REF; device=CPUDevice())
    path = tempname() * ".gguf"
    write_gguf(path, to_gguf(W, cfg))
    Wg = load_qwen35_gguf(path, cfg; device=CPUDevice())
    @test Set(keys(Wg)) == Set(keys(W))
    for (k, v) in W
        k == "lm_head.weight" && continue                                  # stored as Q8_0
        @test size(Wg[k]) == size(v)
        @test isapprox(Wg[k], v; rtol=1e-6, atol=1e-6) || (@info "mismatch" k; false)
    end
    @test relerr(Wg["lm_head.weight"], W["lm_head.weight"]) < 1e-2
    # and the model on them (lm_head through Q8_0) still follows transformers
    W2 = copy(Wg); W2["lm_head.weight"] = W["lm_head.weight"]
    s = LlamaSession(model_template(REF), nothing, REF; max_seq=32, device=CPUDevice(), weights=W2, decode_weights=Float32)
    last = Vector{Float32}[]
    ids = Int.(refarr("ids"))
    generate_ids(s, [ids[:, b] for b in 1:2]; max_new_tokens=1, last_logits=last)
    @test all(relerr(last[b], refarr("logits_0")[:, end, b]) < 1e-4 for b in 1:2)
end
