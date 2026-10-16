# A custom weight format for Strix Halo: what it could buy

Some inference engines for this hardware (hipfire, for one) ship their own weight
format. This note records what such a format could do for Luminal.jl, which reads
the llama.cpp GGUF blocks directly (`LUMINAL_GGUF_NATIVE=1`: Q4_K, Q5_K, Q6_K, IQ4,
Q8_0 blocks stay as stored and are dequantized inside the kernels).

Short version: little for decode, possibly something for the speculation verify
step and for prefill. The plan is to measure that with a gate microbenchmark
before building a converter.

## What already exists: a repack at load time

Luminal already uses its own GPU layout. `native_weight` (src/NativeWeights.jl)
repacks each GGUF tensor on the GPU when the model loads, to the same size as the
file: the low four bits of the codes in a 16-byte-per-32-weights layout, the fifth
or high bits as one word per group (`Q5Weight.hi`, `Q6Weight.meta`) in an order the
kernels can mask out, Float16 steps for Q6, and a Float16 table lookup for IQ4. So a
custom file would save the load-time repack (small), and only the pieces that are
still in the file's packing are left to precompute. Those are the Q5_K (and Q4_K)
6-bit scale and minimum fields (`blk`, decoded by `_kq_scale_min` on every use).
Expanding them to Float16 costs +16 bytes per 256-weight block (+9%) in exchange for
less ALU work, which is a bad trade for the bandwidth-bound decode.

## Where the time goes (Qwen3.8-27B, UD-Q4_K_XL, ~17 GB of weights)

- **Decode** reads the weights (~17 GB) plus the KV cache (7 GB at 110k tokens)
  on every step. The GEMVs reach ~215-220 GB/s of the ~256 GB/s the memory can
  deliver. A repack that keeps the same bits cannot shrink the bytes, so the
  ceiling is about 5-10%. Fewer bytes means a lower bitrate, which is
  requantization and costs quality.
- **Verify** (speculation, 2-4 rows per step): 106 ms for one row, 131 ms for four.
  The extra 25 ms is compute, so a layout that is cheaper to unpack could give
  5-15% here.
- **Prefill** runs at 330-480 tok/s. The native WMMA kernel dequantizes inside the
  tile loop, so most of its non-matrix-core time is unpacking and shuffling.
  Fragment-ordered weights would remove that. An integer WMMA path could go
  further (see below), at a price in quality.

## What can be precomputed offline, without changing the numbers

- Group scales and mins unpacked from the 6-bit packing into Float16.
- The offset-trick terms the dot2 kernels compute per group (Q5: `m - 1024 s`;
  Q6: `d - 1056 xs`; IQ4: `d - 1152 xs`).
- Nibble order permuted to match the dot2 pairs (`_perm`), so no runtime permute.
- The IQ4 lookup table expanded to int8.
- QKV and gate/up weights fused into one matrix, fewer launches.
- Rows padded to the 256-wide WMMA tile, so prefill never pads at runtime.
- A layout that can be memory-mapped, so loading is reading the file. Load time
  is a small part of a cold start today: warm-up and graph compilation take
  longer than reading the file.

## Costs and risks

- Our own file format and converter, and losing direct use of Unsloth's GGUFs.
- The dynamic quants (UD) keep their quality advantage under a lossless repack.
  Requantizing from an already-quantized GGUF adds error, so a converter should
  repack and not requantize.
- Integer WMMA (hipfire's W4A4 with a Walsh-Hadamard rotation) also quantizes the
  activations. That needs a quality test of its own, separate from the layout.

## Plan

1. **Gate microbenchmark.** Give the dot2 GEMV and the WMMA GEMM weights with
   pre-expanded scales and fragment-ordered layout, on a few real matrices, and
   measure verify and prefill time against the GGUF layout.
2. If it gains under ~10%, stop.
3. If prefill gains much more, build the converter (lossless repack, memory-mapped).
4. Treat the int4/int8 WMMA prefill path as a separate question about activation
   quality, to test only after the layout result.

## Results

Gate run on 2026-10-16 (`format_gate.jl`, one layer's tensors from the
UD-Q4_K_XL file, server idle):

| Tensor | N=1 | N=2 | N=4 | Prefill N=512 |
|--------|-----|-----|-----|---------------|
| Q5_K up 17408x5120 (61 MB) | 166 GB/s | 152 | 128 | 31.4 TFLOP/s |
| Q5_K down 5120x17408 | 148 | 137 | 130 | 27.5 |
| IQ4_XS up 17408x5120 | 169 | 163 | 140 | 34.9 |
| Q6_K lm_head (1.1 GB) | 212 | 220 | 214 | not tested |

- **Decode (N=1):** the big lm_head tensor reaches 212-220 GB/s, the memory limit.
  The ~60 MB layer tensors measure 148-169 GB/s, but a tensor that small is about
  0.3 ms, so launch and tail effects are a large part of that. These are not a
  layout problem; the whole-model decode matches the bandwidth estimate.
- **Verify (N=2-4):** the throughput drops to 128-140 GB/s at N=4, 0.48 ms against a
  floor of about 0.27 ms for the bytes. So about 40% of the 4-row time is compute,
  which is the only decode-side place a cheaper layout could help. For the 25 ms
  that four rows add over one, a ceiling of perhaps 10-15 ms (about 8% of the
  verify step) is plausible, and only if the unpack work went away completely.
- **Prefill:** the native kernels run at 27-35 TFLOP/s, faster than the plain
  Float16 WMMA GEMM in the benchmark (16-19 TFLOP/s), because they read half the
  bytes. There is no layout headroom shown by that comparison. The hardware peak
  of this GPU is higher, but the missing gap is the kernel's tiling, not the
  weight format.

**Conclusion:** a custom format is not worth building now. The repack that matters
already happens at load. Decode is at the memory limit, and the verify step has at
most a high-single-digit-percent gain, which a Q5 scale expansion would partly
give back in bytes. The larger prefill gain needs integer WMMA with quantized
activations, which is a quality question and not a format question.

