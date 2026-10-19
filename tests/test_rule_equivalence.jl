# Per-rule equivalence: for every named rewrite rule, random graphs that trigger it are saturated, and every graph the
# e-graph can then produce -- the static extraction and each alternative of each decision -- must compute what the original
# graph computes. A final test requires every rule to have fired in at least one case (and a deliberately wrong rule must be
# caught), so a new rule cannot be added without a graph that exercises it.
using Test
using Luminal
using Luminal.EGraphRewrite
using Metatheory
using Random

const EGR = Luminal.EGraphRewrite
const DEV = get_device()
const GPU = DEV isa Luminal.AbstractGPUDevice
const ALL_FEATURES = (:weights, :activations, :int8, :int4)

# rule name => number of cases in which it fired (dynamic rules) or matched (static ones), across all graphs below
const FIRED = Dict{String,Int}()
const EXERCISED = Dict{String,Int}()      # rule name => candidates containing a node it created

relerr(a, b) = maximum(abs.(Float32.(a) .- Float32.(b)); init=0f0) / max(maximum(abs, Float32.(b); init=0f0), 1f-6)

function run_graph(g, outs, inputs, dev)
    r = compile(g; device=dev, retain=outs, free_intermediates=false)(inputs; device=dev)
    return [Array(r[o]) for o in outs]
end

# Every candidate graph of the e-graph of `g` against `reference`: returns the worst relative error and the number of candidates.
function worst_candidate_error(g, outs, inputs, reference, dev; precision=false, extra_rules=RewriteRule[])
    rw = to_egraph(g, outs)
    saturate_graph!(rw; precision=precision, extra_rules=extra_rules)
    rep = rule_report(rw)
    for x in rep.rows
        (x.dynamic ? x.fired : x.matches) > 0 && (FIRED[x.name] = get(FIRED, x.name, 0) + 1)
    end
    choicesets = Any[Dict{Any,String}()]
    for (_, opts) in decisions(rw), o in opts
        isempty(o) || push!(choicesets, o)
    end
    worst = 0.0
    for choices in choicesets
        ng, m = extract_graph(rw; choices=choices)
        for name in keys(provenance(rw; choices=choices).by_rule)
            EXERCISED[name] = get(EXERCISED, name, 0) + 1
        end
        mapped = Dict{Int,Any}(m[k] => v for (k, v) in inputs if haskey(m, k))
        got = run_graph(ng, [m[o] for o in outs], mapped, dev)
        worst = max(worst, maximum(relerr(a, b) for (a, b) in zip(got, reference)))
    end
    return worst, length(choicesets)
end

rdim(rng, lo=2, hi=5) = rand(rng, lo:hi)
rdims(rng, n; lo=2, hi=5) = [rdim(rng, lo, hi) for _ in 1:n]

# `check(builder)`: for each seeded random case, builder(g, rng) -> (output tensors, Dict tensor => input array)
function check(builder; cases=6, tol=1e-5, dev=DEV, precision=false, seed=1, extra_rules=RewriteRule[])
    worst_all = 0.0; ncand = 0
    for c in 1:cases
        rng = MersenneTwister(seed * 1000 + c)
        g = Graph()
        outs, ins = builder(g, rng, dev)
        inputs = Dict{Int,Any}(t.id => a for (t, a) in ins)
        reference = run_graph(g, [o.id for o in outs], inputs, dev)
        worst, n = worst_candidate_error(g, [o.id for o in outs], inputs, reference, dev; precision=precision, extra_rules=extra_rules)
        worst_all = max(worst_all, worst); ncand += n
    end
    return worst_all, ncand
end

input!(g, ins, dims) = (t = Luminal.tensor(g, dims); push!(ins, t => randn(Float32, dims...)); t)

# --- canonical rules: a broadcast copy next to a same-shaped operand becomes a reshape-broadcast ---------------------

function expand_case(op, side)
    return (g, rng, dev) -> begin
        ins = Pair{Luminal.GraphTensor,Any}[]
        base = rdims(rng, rand(rng, 1:3))
        axis = rand(rng, 1:length(base) + 1); k = rdim(rng)
        x = input!(g, ins, base)
        full = vcat(base[1:axis-1], k, base[axis:end])
        y = input!(g, ins, full)
        e = Luminal.expand(x, axis, k)
        out = side === :lhs ? op(e, y) : op(y, e)
        return [out], ins
    end
end

@testset "canonical expand rules" begin
    for (op, name) in ((*, "mul"), (+, "add")), side in (:lhs, :rhs)
        w, n = check(expand_case(op, side))
        @test w < 1e-5 && n > 1
        @test get(FIRED, "canonical.$name-expand-$side", 0) > 0
    end
end

@testset "canonical reshape rules" begin
    w, _ = check((g, rng, dev) -> begin
        ins = Pair{Luminal.GraphTensor,Any}[]
        a, b, c = rdims(rng, 3)
        x = input!(g, ins, [a, b, c])
        return [Luminal.reshape(Luminal.reshape(x, [a * b, c]), [a, b * c])], ins
    end)
    @test w < 1e-6 && get(FIRED, "canonical.reshape-chain", 0) > 0
    w, _ = check((g, rng, dev) -> begin
        ins = Pair{Luminal.GraphTensor,Any}[]
        d = rdims(rng, rand(rng, 1:4))
        x = input!(g, ins, d)
        return [Luminal.reshape(x, copy(d)) * 2f0], ins                  # a reshape to the same shape
    end)
    @test w < 1e-6 && get(FIRED, "canonical.reshape-identity", 0) > 0
end

# --- algebra: a scalar scaling moves into either matmul operand -------------------------------------------------------

@testset "algebra: scalar scale of a matmul" begin
    w, n = check((g, rng, dev) -> begin
        ins = Pair{Luminal.GraphTensor,Any}[]
        M, K, N = rdims(rng, 3); batch = rdims(rng, rand(rng, 0:2), hi=3)
        a = input!(g, ins, [M, K, batch...]); b = input!(g, ins, [K, N, batch...])
        return [Luminal.matmul(a, b) * Float32(rand(rng, [0.5, 2.0, -1.5]))], ins
    end)
    @test w < 1e-5 && n > 2
    @test get(FIRED, "algebra.scale-into-lhs", 0) > 0 && get(FIRED, "algebra.scale-into-rhs", 0) > 0
end

# --- kernels: a matmul with a transposed operand reads it through a transpose flag -----------------------------------

@testset "kernel: matmul with a swapped-axes operand" begin
    for (ta, tb, name) in ((true, false, "lhs"), (false, true, "rhs"), (true, true, "both"))
        w, n = check((g, rng, dev) -> begin
            ins = Pair{Luminal.GraphTensor,Any}[]
            M, K, N = rdims(rng, 3); batch = rdims(rng, rand(rng, 0:2), hi=3)
            a = input!(g, ins, ta ? [K, M, batch...] : [M, K, batch...])
            b = input!(g, ins, tb ? [N, K, batch...] : [K, N, batch...])
            sw(t, f) = f ? Luminal.permute(t, [2, 1, (3:length(Luminal.realized_dims(t.shape)))...]) : t
            return [Luminal.matmul(sw(a, ta), sw(b, tb))], ins
        end)
        @test w < 1e-5 && n > 1
        @test get(FIRED, "kernel.matmul-transpose-$name", 0) > 0
    end
end

# --- weight precision: a weight matmul may use Float16 / int8 / int4 weights (GPU kernels) ----------------------------

# A weight matmul y = W x with W (rows, cols) a persistent tensor. `ref(W)` is what the variant is allowed to compute.
function weight_case(rows, cols; round16=true)
    return (g, rng, dev) -> begin
        ins = Pair{Luminal.GraphTensor,Any}[]
        W = randn(Float32, rows, cols); round16 && (W = Float32.(Float16.(W)))
        w = Luminal.tensor(g, [rows, cols])
        g.tensors[(w.id, 1)] = Luminal.to_device(W, dev)
        x = input!(g, ins, [cols, 1, 1])
        return [Luminal.matmul(w, x)], ins
    end
end

if GPU
    @testset "precision rules on $(nameof(typeof(DEV)))" begin
        # Float16 weights hold Float16-rounded values here, so the kernels are exact up to summation order
        for shape in ((64, 64), (128, 64), (256, 128))
            w, n = check(weight_case(shape...); precision=:weights, cases=1, seed=shape[1])
            @test w < 1e-4 && n > 1
        end
        for name in ("f16-gemv-64", "f16-gemv-128", "f16-gemv-256")
            @test get(FIRED, "precision.$name", 0) > 0
        end
        # mixed-precision GEMM rounds the activations too
        w, _ = check(weight_case(128, 64); precision=:activations, cases=1, tol=5e-3)
        @test w < 5e-3 && get(FIRED, "activation.f16-gemm_ex", 0) > 0
    end

    # int8 / int4 are lossy by design: compare each variant with a matmul by the *dequantized* weights
    for (prec, T, nvar) in ((:int8, Luminal.QuantWeight, 4), (:int4, Luminal.Q4Weight, 4))
        @testset "$prec rules: every variant == the dequantized weights" begin
            rows, cols = 128, 64
            W = randn(Float32, rows, cols); x = randn(Float32, cols, 1, 1)
            D = Array(Luminal.dequantize(T(Luminal.to_device(W, DEV))))
            g = Graph(); w = Luminal.tensor(g, [rows, cols]); xt = Luminal.tensor(g, [cols, 1, 1])
            g.tensors[(w.id, 1)] = Luminal.to_device(W, DEV)
            out = Luminal.matmul(w, xt)
            rw = to_egraph(g, [out.id]); saturate_graph!(rw; precision=prec)
            opts = only(filter(d -> occursin(prec === :int8 ? "xMatMulQ8" : "xMatMulQ4", d[1]), decisions(rw)))[2]
            @test length(filter(!isempty, opts)) == nvar + 1             # the plain matmul and each variant
            for opt in filter(!isempty, opts)
                ng, m = extract_graph(rw; choices=opt)
                ex = compile(ng; device=DEV, retain=[m[out.id]], free_intermediates=false)
                got = Array(ex(Dict{Int,Any}(m[xt.id] => x); device=DEV)[m[out.id]])[:, 1, 1]
                want = (occursin("MatMulQ", first(values(opt))) ? D : W) * x[:, 1, 1]
                @test relerr(got, want) < 1e-4
                for name in keys(provenance(rw; choices=opt).by_rule)
                    EXERCISED[name] = get(EXERCISED, name, 0) + 1
                    FIRED[name] = max(get(FIRED, name, 0), 1)
                end
            end
        end
    end
end

# --- coverage: a rule without a test graph fails here ----------------------------------------------------------------------

@testset "every rule is exercised" begin
    expected = rule_names(precision=GPU ? ALL_FEATURES : false)
    missing_rules = [n for n in expected if get(FIRED, n, 0) == 0]
    @test isempty(missing_rules)
    isempty(missing_rules) || @info "rules with no firing test graph" missing_rules
    # rules that create an expression node are seen in at least one verified candidate
    for n in expected
        startswith(n, "canonical.reshape") && continue                    # their results are existing nodes or static
        @test get(EXERCISED, n, 0) > 0
    end
    GPU || @info "precision rules (Float16 / int8 / int4 weights) need a GPU kernel: not exercised on $(nameof(typeof(DEV)))"
end

# --- the harness catches a wrong rule ------------------------------------------------------------------------------------

@testset "a wrong rule is detected" begin
    bad = EGR._named((@theory p a b begin
        xAdd(p::Tuple, a, b) => :(xMul($p, $a, $b))              # a + b rewritten as a * b
    end), "bad", ("add-as-mul",))
    w, _ = check((g, rng, dev) -> begin
        ins = Pair{Luminal.GraphTensor,Any}[]
        d = rdims(rng, 2)
        return [input!(g, ins, d) + input!(g, ins, d)], ins
    end; cases=2, extra_rules=bad)
    @test w > 0.1
end
