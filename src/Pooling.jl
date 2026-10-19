# 2-D average pooling (after upstream `luminal_nn::pooling`).
#
# Layout: channels first, as `Conv1D`: (C, H, W) or (C, H, W, N). Windows are built from strided slices, one per kernel tap, so no
# 2-D unfold is needed and the result is exact on every device.
module Pooling

using ..Luminal
import ..Luminal: GraphTensor, realized_dims, slice_along, contiguous, reshape

export avg_pool_2d, adaptive_avg_pool_2d

# the taps `x[.., off + stride * j, ..]`, j = 0:n-1 along `dim`
function _strided(x::GraphTensor, dim::Int, off::Int, stride::Int, n::Int)
    stride == 1 && return slice_along(x, dim, off, off + n)
    extra = max(0, off + stride * n - realized_dims(x.shape)[dim])      # the last window may stop short of a full stride
    t = slice_along(extra > 0 ? Luminal.pad_along(x, dim, 0, extra) : x, dim, off, off + stride * n)
    d = realized_dims(t.shape)
    t = reshape(contiguous(t), [d[1:dim-1]..., stride, n, d[dim+1:end]...])
    t = slice_along(t, dim, 0, 1)
    return reshape(contiguous(t), [d[1:dim-1]..., n, d[dim+1:end]...])
end

function _pool(x::GraphTensor, kernel::NTuple{2,Int}, stride::NTuple{2,Int})
    all(>(0), kernel) || throw(ArgumentError("pooling kernel must be positive"))
    all(>(0), stride) || throw(ArgumentError("pooling stride must be positive"))
    nd = length(realized_dims(x.shape))
    nd in (3, 4) || throw(ArgumentError("2D pooling expects (C, H, W) or (C, H, W, N), got rank $nd"))
    _, h, w = realized_dims(x.shape)[1:3]
    (h >= kernel[1] && w >= kernel[2]) || throw(ArgumentError("pooling kernel $kernel is larger than the $h x $w input"))
    oh, ow = (h - kernel[1]) ÷ stride[1] + 1, (w - kernel[2]) ÷ stride[2] + 1
    out = nothing
    for i in 0:kernel[1]-1, j in 0:kernel[2]-1
        t = _strided(_strided(x, 2, i, stride[1], oh), 3, j, stride[2], ow)
        out = out === nothing ? t : out + t
    end
    return out * (1f0 / (kernel[1] * kernel[2]))
end

"""
    avg_pool_2d(x, kernel, stride=kernel)

Average over `kernel = (kh, kw)` windows moving by `stride`, no padding, on `x` (C, H, W[, N]); the output is
(C, (H - kh) ÷ sh + 1, (W - kw) ÷ sw + 1[, N]).
"""
avg_pool_2d(x::GraphTensor, kernel::NTuple{2,Int}, stride::NTuple{2,Int}=kernel) = _pool(x, kernel, stride)

"""
    adaptive_avg_pool_2d(x, output_size)

Pooling to a fixed `(oh, ow)` output, as upstream Luminal does it: stride `H ÷ oh`, kernel `H - stride * (oh - 1)`, so when the sizes do not
divide the windows overlap. (PyTorch's adaptive pooling picks different, non-uniform windows then; the two agree when they divide.)
"""
function adaptive_avg_pool_2d(x::GraphTensor, output_size::NTuple{2,Int})
    all(>(0), output_size) || throw(ArgumentError("adaptive pooling output size must be positive"))
    _, h, w = realized_dims(x.shape)[1:3]
    (h >= output_size[1] && w >= output_size[2]) || throw(ArgumentError("adaptive pooling cannot enlarge a $h x $w input to $output_size"))
    s = (h ÷ output_size[1], w ÷ output_size[2])
    k = (h - s[1] * (output_size[1] - 1), w - s[2] * (output_size[2] - 1))
    return _pool(x, k, s)
end

end # module Pooling
