using Test
using Luminal
using Luminal.NN
using Luminal.PagedAttention
using Luminal.PagedAttention.PageTables
using Luminal.PagedLlama

const CFG = (vocab_size=50, hidden=64, n_layers=2, n_heads=4, n_kv_heads=2, intermediate=128)
const ROPE = 10000f0
const DEVS = Any[Luminal.CPUDevice()]
get(ENV, "LUMINAL_TEST_GPU", "0") == "1" && push!(DEVS, Luminal.get_device())

function weights()
    g = Graph(); reg = WeightRegistry()
    Llama(g, reg; CFG..., rope_base=ROPE)
    Dict{String,Any}(k => 0.3f0 .* randn(Float32, Tuple(Luminal.realized_dims(g.shapes[id]))...) for (k, id) in reg.mapping)
end

# logits of every position of `ids`, one forward pass (the reference)
function reference(W, ids, dev)
    g = Graph(); reg = WeightRegistry()
    m = Llama(g, reg; CFG..., rope_base=ROPE)
    inp = Luminal.tensor(g, [length(ids), 1])
    out = m(inp, 0)
    load_weights!(g, reg, W; device=dev)
    res = compile(g; device=dev, retain=[out.id])(Dict{Int,Any}(inp.id => Float32.(Base.reshape(ids, :, 1))); device=dev)
    return Array(res[out.id])[:, :, 1]               # (vocab, S)
end

@testset "PageTable" begin
    pt = PageTable(64; page_size=4)                  # 16 pages
    @test free_pages(pt) == 16
    a = reserve!(pt, 1, 6); b = reserve!(pt, 2, 3)
    @test a == [0, 1, 2, 3, 4, 5] && b == [8, 9, 10]            # sequence 1 owns pages 0, 1; sequence 2 page 2
    @test reserve!(pt, 1, 3) == [6, 7, 12]                      # page 1 fills, then a new page (3)
    @test slots_of(pt, 1) == [0:7; 12] && sequence_length(pt, 1) == 9
    truncate!(pt, 1, 4)
    @test free_pages(pt) == 16 - 1 - 1 && slots_of(pt, 1) == [0, 1, 2, 3]
    release!(pt, 2)
    @test !has_sequence(pt, 2) && free_pages(pt) == 15
    @test reserve!(pt, 3, 1) == [4]                             # the lowest free page is reused first
    @test_throws OutOfPages reserve!(pt, 4, 4 * 16)
    @test sequence_length(pt, 4) == 0 && free_pages(pt) == 14   # unchanged by the failed request
    # a plan that does not fit leaves the table as it was
    before = (free_pages(pt), sequence_length(pt, 1))
    @test_throws ArgumentError plan_step(pt, [1 => [1, 2, 3], 5 => [4]]; rows=3, context=64, pool_slots=64)
    @test (free_pages(pt), sequence_length(pt, 1)) == before && !has_sequence(pt, 5)
    p = plan_step(pt, [1 => [7, 8], 3 => [9]]; rows=4, context=10, pool_slots=64)
    @test p.tokens == [7, 8, 9, 0] && p.positions == [4, 5, 1 - 1 + 1, 0] && p.row_seq == [1, 1, 3, -2]
    @test p.write_slots[1:3] == [slots_of(pt, 1)[5], slots_of(pt, 1)[6], slots_of(pt, 3)[2]] && p.write_slots[4] == 64
    @test p.last_row == [2, 3]
    @test count(==(-1), p.key_seq) == 2 && p.read_slots[end] == 64
end

for dev in DEVS
    name = nameof(typeof(dev))
    @testset "paged Llama == full forward per sequence ($name)" begin
        W = weights()
        seqs = Dict(1 => [3, 14, 15, 9, 2, 6, 5], 2 => [7, 1, 12], 3 => [11, 30, 4, 8, 22])
        ref = Dict(s => reference(W, ids, dev) for (s, ids) in seqs)

        rows, slots, context, page = 4, 64, 24, 4
        g = Graph(); reg = WeightRegistry()
        m = Llama(g, reg; CFG..., rope_base=ROPE)
        pg = build_paged_llama_step!(m, g; rows=rows, slots=slots, context=context)
        load_weights!(g, reg, W; device=dev)
        exec = compile(g; device=dev, retain=paged_retain_ids(pg))
        pool = PagedKvPool(m, slots; device=dev)
        pt = PageTable(slots; page_size=page)

        # Each step mixes prefill chunks and decode tokens of different sequences: [(sequence, token positions fed)]
        schedule = [[(1, 1:3), (2, 1:1)], [(1, 4:4), (2, 2:3), (3, 1:1)], [(3, 2:4), (1, 5:5)],
                    [(1, 6:7), (3, 5:5)]]
        fed = Dict(s => 0 for s in keys(seqs))
        for step in schedule
            items = [s => seqs[s][r] for (s, r) in step]
            for (s, r) in step
                @test first(r) == fed[s] + 1; fed[s] = last(r)
            end
            plan = plan_step(pt, items; rows=rows, context=context, pool_slots=slots)
            logits = Array(paged_llama_step!(exec, pg, pool, plan; device=dev))
            row = 0
            for (s, r) in step, p in r
                row += 1
                @test logits[:, row] ≈ ref[s][:, p] rtol=1e-4 atol=1e-5
            end
        end
        @test all(sequence_length(pt, s) == length(seqs[s]) for s in keys(seqs))

        # a finished sequence's pages go back to the pool and are reused by a new one without disturbing the others
        release!(pt, 2)
        new = [21, 5, 9]
        refn = reference(W, new, dev)
        plan = plan_step(pt, [4 => new, 3 => [6]]; rows=rows, context=context, pool_slots=slots)
        logits = Array(paged_llama_step!(exec, pg, pool, plan; device=dev))
        for p in 1:3
            @test logits[:, p] ≈ refn[:, p] rtol=1e-4 atol=1e-5
        end
        ref3 = reference(W, [seqs[3]; 6], dev)
        @test logits[:, 4] ≈ ref3[:, 6] rtol=1e-4 atol=1e-5
    end
end
