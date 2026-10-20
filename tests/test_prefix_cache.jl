# PrefixCache (src/Decoding.jl): a prompt resumed from a saved state must continue exactly like the same prompt prefilled from scratch.
#
# A forgotten state component fails silently (the state is "almost right": the recurrent states, the convolution history, the residual
# stream at the last token, the full-attention K/V below the resume point, the MTP head's K/V), so every kind of resume is compared with
# an uncached run of the same prompt: the prompt's logits, and the decode steps that follow, which read all of that state. Kinds:
#   :sys   the end of the system message (shared by conversations with that system prompt)
#   :turn  the start of the last assistant header
#   :ckpt  a recurrent-state checkpoint along the live prompt (`ckpt_every`)
# and, with the MTP head, the speculative loop (whose output must equal plain greedy decoding). Not covered: the block drafter's (DFlash2)
# ring rows (they need a draft model). Chunked prefill needs a GPU (Float16 K/V); the idea comes from gufo's cache notes (TODO.md).
using Test
using Luminal
using Luminal.NN

# A prefill that ends in NaN/Inf logits fails the request and drops the state it wrote (no resume from poisoned K/V, checkpoints or ring rows)
@testset "a prefill with non-finite logits drops its prefix cache state" begin
    pc = PrefixCache(min_tokens=8, ckpt_every=16)
    pc.live = collect(1:40); pc.live_mtp = 40; pc.live_df = 40
    pc.ckpts = Luminal.Decoding.StateCkpt[]
    @test_throws ErrorException Luminal.Decoding._fail_prefill!(pc, nothing, 40)
    @test isempty(pc.live) && pc.live_mtp == 0 && pc.live_df == 0 && isempty(pc.pending) && isempty(pc.ckpts)
    @test_throws ErrorException Luminal.Decoding._fail_prefill!(nothing, nothing, 40)
end

if Luminal.get_device() isa CPUDevice
    @info "no GPU: skipping the prefix cache test"
else
    const REF = joinpath(@__DIR__, "data", "qwen35_tiny")
    const DEV = Luminal.get_device()

    # random `mtp.*` weights shaped as the head's registered tensors (as in test_speculative.jl)
    function random_mtp_weights(model, dev)
        g = Luminal.Graph(); reg = Luminal.NN.WeightRegistry()
        Luminal.NN.Qwen35MTP(g, reg; model.config...)
        W = Dict{String,Any}()
        for (name, id) in reg.mapping
            startswith(name, "mtp.") || continue
            dims = Tuple(Int.(Luminal.realized_dims(g.shapes[id])))
            W[name] = Luminal.to_device((occursin("norm", name) ? 0.1f0 : 0.3f0) .* randn(Float32, dims...), dev)
        end
        return W
    end

    relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

    @testset "a resumed prefill == a prefill from scratch" begin
        model = model_template(REF)
        s = LlamaSession(model, nothing, REF; max_seq=160, decode_weights=Float32, device=DEV, kv_dtype=Float16)
        merge!(s.weights, random_mtp_weights(model, DEV))
        V = model.config.vocab_size
        toks(n, seed) = mod.(seed .* collect(1:n) .+ 3, V)
        NEW = 6

        # one request: the prompt's logits, the generated tokens and the logits of each decode step
        function request(prompt; prefix=nothing, prefix_at=Int[])
            ll = Vector{Float32}[]; sl = Matrix{Float32}[]
            g = only(generate_ids(s, [prompt]; max_new_tokens=NEW, last_logits=ll, step_logits=sl, prefix, prefix_at))
            return (tokens=g, logits=only(ll), steps=sl)
        end
        same(a, b) = relerr(a.logits, b.logits) < 1e-2 && a.tokens[1] == b.tokens[1] &&
                     length(a.steps) == length(b.steps) && all(relerr(vec(x), vec(y)) < 1e-2 for (x, y) in zip(a.steps, b.steps))

        withenv("LUMINAL_PREFILL_CHUNK" => "16") do
            sysp = toks(40, 7)
            first_prompt = vcat(sysp, toks(32, 11))                  # 72 tokens: a system message of 40, a conversation of 32 more

            # `overwritten`: another request goes through the cache between the save and the resume, so the live K/V no longer holds the
            # saved prompt and the entry's own K/V has to be copied back (a checkpoint only exists along the live prompt: no such variant)
            @testset "resume from a $kind entry (live cache overwritten: $overwritten)" for (kind, at, target) in (
                    (:sys,  [:sys => 40],  vcat(sysp, toks(30, 13))),                      # shares the system message only
                    (:turn, [:turn => 64], vcat(first_prompt[1:64], toks(10, 17))),        # shares the conversation up to the last assistant header
                    (:ckpt, Int[],         vcat(first_prompt[1:50], toks(12, 19)))),       # diverges at 50: the checkpoint at 48 of the live prompt
                    overwritten in (false, true)
                kind === :ckpt && overwritten && continue
                ref = request(target)                                                       # from scratch (no cache)
                pc = PrefixCache(min_tokens=8, ckpt_every=16)
                request(first_prompt; prefix=pc, prefix_at=at)                              # saves the entries / checkpoints
                overwritten && request(toks(60, 31); prefix=pc)                             # an unrelated conversation takes over the live caches
                got = request(target; prefix=pc, prefix_at=at)
                @test pc.resumed_kind == kind                                                # it did resume from that kind
                @test same(got, ref)
            end

            @testset "a resumed prompt that the cache saved again" begin
                # the second request saves new entries on top of the resumed ones; a third request resumes from those
                pc = PrefixCache(min_tokens=8, ckpt_every=16)
                request(first_prompt; prefix=pc, prefix_at=[:sys => 40, :turn => 64])
                second = vcat(first_prompt[1:64], toks(20, 23))
                request(second; prefix=pc, prefix_at=[:sys => 40, :turn => 80])
                third = vcat(second[1:80], toks(8, 29))
                ref = request(third)
                got = request(third; prefix=pc, prefix_at=[:sys => 40, :turn => 80])
                @test pc.resumed_kind in (:turn, :ckpt)
                @test same(got, ref)
            end

            @testset "MTP speculative decoding from a resumed prefill" begin
                d = MTPDrafter(s)
                # (the deepest resumable state wins, so each target shares with the first prompt only what its kind covers)
                for (kind, at, target) in ((:sys,  [:sys => 40],  vcat(sysp, toks(30, 13))),
                                           (:turn, [:turn => 64], vcat(first_prompt[1:64], toks(10, 17))))
                    plain = request(target).tokens                                          # plain greedy decoding from scratch
                    # The output of speculative decoding does not depend on the drafts, so a wrong residual stream or MTP K/V would go
                    # unnoticed in it: also compare the drafts the head makes (from the hidden state at the resume point and its K/V).
                    drafts_scratch = Int[]
                    generate_ids_speculative(s, d, target; max_new_tokens=NEW, draft_hook=(n, dr) -> (push!(drafts_scratch, dr); dr))
                    kv_scratch = (Array(d.k)[:, 1:length(target) - 1, :, :], Array(d.v)[:, 1:length(target) - 1, :, :])
                    pc = PrefixCache(min_tokens=8, ckpt_every=16)
                    generate_ids_speculative(s, d, first_prompt; max_new_tokens=NEW, prefix=pc, prefix_at=at)    # saves the entries (with the MTP K/V)
                    st = Dict{Symbol,Any}(); drafts_resumed = Int[]
                    spec = generate_ids_speculative(s, d, target; max_new_tokens=NEW, prefix=pc, prefix_at=at, stats=st,
                                                    draft_hook=(n, dr) -> (push!(drafts_resumed, dr); dr))
                    @test pc.resumed_kind == kind
                    @test st[:prefill_reused] > 0
                    @test spec == plain
                    @test !isempty(drafts_scratch) && drafts_resumed == drafts_scratch
                    # ... and the head's K/V of the prompt's pairs themselves (one wrong row barely moves a draft, but it shows here)
                    np = length(target) - 1
                    @test relerr(Float32.(Array(d.k)[:, 1:np, :, :]), Float32.(kv_scratch[1])) < 1e-2
                    @test relerr(Float32.(Array(d.v)[:, 1:np, :, :]), Float32.(kv_scratch[2])) < 1e-2
                end
            end
        end
    end
end
