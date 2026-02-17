module MetatheoryOps

using Metatheory
using Metatheory.EGraphs

export LuminalAdd, LuminalMul, LuminalSub, LuminalDiv, LuminalMax, LuminalMatMul,
       LuminalMod, LuminalLessThan,
       LuminalReLU, LuminalExp, LuminalLog, LuminalSqrt, LuminalNeg,
       LuminalSin, LuminalCos, LuminalLog2, LuminalExp2, LuminalRecip,
       LuminalTranspose, LuminalSoftmax,
       LuminalReshape, LuminalSlice, LuminalPad, LuminalPermute,
       LuminalReduce,
       LuminalConv, LuminalLayerNorm,
       LuminalConstant, LuminalZero, LuminalOne,
       LuminalSimpleInput, # For simple variable inputs
       LuminalFusedAddReLU, LuminalGEMM, LuminalMulAdd

# --- Binary Operations ---
LuminalAdd(x, y) = :(LuminalAdd($x, $y))
LuminalMul(x, y) = :(LuminalMul($x, $y))
LuminalSub(x, y) = :(LuminalSub($x, $y))
LuminalDiv(x, y) = :(LuminalDiv($x, $y))
LuminalMax(x, y) = :(LuminalMax($x, $y))
LuminalMatMul(x, y) = :(LuminalMatMul($x, $y))
LuminalMod(x, y) = :(LuminalMod($x, $y))
LuminalLessThan(x, y) = :(LuminalLessThan($x, $y))

# --- Unary Operations ---
LuminalReLU(x) = :(LuminalReLU($x))
LuminalExp(x) = :(LuminalExp($x))
LuminalLog(x) = :(LuminalLog($x))
LuminalSqrt(x) = :(LuminalSqrt($x))
LuminalNeg(x) = :(LuminalNeg($x))

LuminalSin(x) = :(LuminalSin($x))
LuminalCos(x) = :(LuminalCos($x))
LuminalLog2(x) = :(LuminalLog2($x))
LuminalExp2(x) = :(LuminalExp2($x))
LuminalRecip(x) = :(LuminalRecip($x))

LuminalTranspose(x) = :(LuminalTranspose($x))
LuminalSoftmax(x) = :(LuminalSoftmax($x))

# --- Shape Operations ---
LuminalReshape(x, shape) = :(LuminalReshape($x, $shape))
LuminalSlice(x, ranges) = :(LuminalSlice($x, $ranges))
LuminalPad(x, pads) = :(LuminalPad($x, $pads))
LuminalPermute(x, dims) = :(LuminalPermute($x, $dims))

# --- Reduction ---
LuminalReduce(op, x, dims) = :(LuminalReduce($op, $x, $dims))

# --- High-Level Ops ---
LuminalConv(x, w, bias, stride, padding) = :(LuminalConv($x, $w, $bias, $stride, $padding))
LuminalLayerNorm(x, gamma, beta, eps) = :(LuminalLayerNorm($x, $gamma, $beta, $eps))

# --- Fused Ops ---
LuminalFusedAddReLU(x, y) = :(LuminalFusedAddReLU($x, $y))
LuminalGEMM(x, y, z) = :(LuminalGEMM($x, $y, $z)) 
LuminalMulAdd(x, y, z) = :(LuminalMulAdd($x, $y, $z)) 

# --- Constants & Inputs ---
LuminalConstant(val) = :(LuminalConstant($val))
LuminalZero() = :(LuminalZero())
LuminalOne() = :(LuminalOne())
LuminalSimpleInput(id) = :(LuminalSimpleInput($id))

end # module
