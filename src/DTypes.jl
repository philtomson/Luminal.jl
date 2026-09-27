# Element types of graph tensors. Every node has one output dtype, inferred from
# its op and its inputs' dtypes when it is added (graph.dtypes, next to
# graph.shapes). Typing is strict, as upstream's: an op's operands must share a
# dtype, number literals take the tensor's dtype, and every other conversion is
# an explicit cast / trunc_cast. Comparisons produce Bool.

const DTYPES = (Float32, Float64, Float16, Core.BFloat16, Int8, Int32, Int64, Bool)

_isfloat(T) = T <: AbstractFloat
_isint(T) = T <: Signed
_isnum(T) = T !== Bool                       # arithmetic types (Bool is a truth value)

function _check_dtype(T)
    T in DTYPES || throw(ArgumentError("unsupported dtype $T; supported: $(join(DTYPES, ", "))"))
    T === Core.BFloat16 && _check_bf16_host()
    return T
end

# Julia 1.12's LLVM fails to compile vectorized Float32 -> BFloat16 conversions
# for x86 CPUs with AVX512-BF16 ("Cannot select ... v16bf16"): an abort, or a
# JIT deadlock when the compile runs off the main thread. Any host-side BFloat16
# array code can hit it, so refuse up front, naming the workaround, rather than
# hang. Excluding the feature from the JIT target avoids it.
function _bf16_host_ok()
    Sys.ARCH in (:x86_64, :i686) || return true
    CPUID = Base.BinaryPlatforms.CPUID
    isdefined(CPUID, :JL_X86_avx512bf16) && CPUID.test_cpu_feature(CPUID.JL_X86_avx512bf16) || return true
    return occursin("-avx512bf16", unsafe_string(Base.JLOptions().cpu_target))
end
const _BF16_HOST_OK = Ref{Union{Nothing, Bool}}(nothing)
function _check_bf16_host()
    _BF16_HOST_OK[] === nothing && (_BF16_HOST_OK[] = _bf16_host_ok())
    ok = _BF16_HOST_OK[]
    ok || throw(ArgumentError("BFloat16 tensors need Julia started with `-C native,-avx512bf16` on this CPU: " *
                              "Julia's LLVM crashes or hangs compiling BFloat16 conversions for AVX512-BF16"))
    return nothing
end

_opname(op) = nameof(typeof(op))

function _same_dtype(op, ins)
    all(==(ins[1]), ins) && return ins[1]
    throw(ArgumentError("$(_opname(op)): operands have dtypes $(join(ins, ", ")); " *
                        "convert explicitly with cast(x, T) (or trunc_cast for float -> integer)"))
end

function _require(op, T, ok, what)
    ok(T) || throw(ArgumentError("$(_opname(op)) needs $what operands, got $T"))
    return T
end

"""
    output_dtype(op, input_dtypes) -> DataType

The dtype of `op`'s output given its inputs' dtypes, or an `ArgumentError` when
they don't type-check. Ops without a rule (the fused inference kernels, weight
formats) take their first input's dtype.
"""
output_dtype(op::Op, ins) = isempty(ins) ? Float32 : ins[1]

const _FloatUnary = Union{Log2, Exp2, Exp, Sin, Cos, Sqrt, Recip, Floor, Ceil, Round, Trunc}
output_dtype(op::_FloatUnary, ins) = _require(op, ins[1], _isfloat, "floating-point")
output_dtype(op::Union{ReLU}, ins) = _require(op, ins[1], _isnum, "numeric")

const _Arith = Union{Add, Mul, Mod, Max, FusedMulAdd, FusedAddReLU}
output_dtype(op::_Arith, ins) = _require(op, _same_dtype(op, ins), _isnum, "numeric")
output_dtype(op::Div, ins) = _require(op, _same_dtype(op, ins), _isfloat, "floating-point (for integers use trunc_div)")
output_dtype(op::LessThan, ins) = (_require(op, _same_dtype(op, ins), _isnum, "numeric"); Bool)
output_dtype(op::Union{SumReduce, MaxReduce}, ins) = _require(op, ins[1], _isnum, "numeric")
output_dtype(op::Union{MatMul, MatMulT}, ins) = _require(op, _same_dtype(op, ins), _isfloat, "floating-point")

function output_dtype(op::Select, ins)
    ins[1] in (Bool, Float32) || throw(ArgumentError("Select: the condition must be Bool (or a Float32 0/1 mask), got $(ins[1])"))
    return _same_dtype(op, ins[2:3])
end

_coord_ok(T) = T !== Bool
function output_dtype(op::GatherND, ins)
    all(_coord_ok, ins[2:end]) || throw(ArgumentError("GatherND: coordinates must be integers (or Float32), got $(ins[2:end])"))
    return ins[1]
end
function output_dtype(op::ScatterND, ins)
    all(_coord_ok, ins[3:end]) || throw(ArgumentError("ScatterND: coordinates must be integers (or Float32), got $(ins[3:end])"))
    T = _same_dtype(op, ins[1:2])
    op.mode === :add && _require(op, T, _isnum, "numeric (for mode=:add)")
    return T
end

output_dtype(op::Constant, ins) = _check_dtype(op.value isa AbstractArray ? eltype(op.value) : typeof(op.value))
output_dtype(op::Iota, ins) = op.dtype

function output_dtype(op::Function, ins)
    op.name == "ARange" && return Float32
    return isempty(ins) ? Float32 : ins[1]      # InputTensor's dtype is given explicitly; Gather, CumSum
end

output_dtype(op::Cast, ins) = op.dtype
function output_dtype(op::TruncCast, ins)
    _require(op, ins[1], _isfloat, "floating-point")
    return op.dtype
end
output_dtype(op::Union{TruncDiv, TruncRem}, ins) = _require(op, _same_dtype(op, ins), _isint, "integer")

"""
    dtype(t::GraphTensor) -> DataType

The element type of `t`.
"""
dtype(t) = t.graph_ref.dtypes[t.id]
