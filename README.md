# Luminal.jl

[![Julia](https://img.shields.io/badge/Julia-1.12-9558B2?style=for-the-badge&logo=julia&logoColor=white)](https://julialang.org/)
[![Status](https://img.shields.io/badge/Status-Active%20Development-blue?style=for-the-badge)](https://github.com/luminal-ai/luminal)

A Julia port of [Luminal](https://github.com/luminal-ai/luminal), a deep learning
library built on **ahead-of-time compilation of static graphs**. A model is
written as a graph of a few primitive ops. `compile()` then fuses, aliases, folds
and specializes that graph for a device, and can optionally search over
equivalent graphs for the fastest one.

Models validated against Hugging Face `transformers`:

- **Llama-3-8B-Instruct**: logits within 5e-6 of the reference at every prompt
  position. Greedy chat continuations are identical to HF's with Float32,
  Float16 and int8 decode weights.
- **Qwen3-0.6B and Qwen3-4B**: logits within 5e-6 of the reference at every
  prompt position; greedy chat continuations (thinking off) identical to HF's
  with Float32, Float16 and int8 decode weights (4B), or all but one late token
  at int8 (0.6B). Chat prompts tokenize exactly as upstream Luminal's benchmark.
- **Gemma3-4B** (text): logits within 7e-6 of the reference at every prompt
  position, greedy continuations identical in Float32, Float16 and int8; at 1,300
  tokens (past the 1,024-token sliding window) the last-position logits are within
  6e-6 and the next 8 greedy tokens match.
- **Llama-3.2-1B-Instruct** (RoPE scaling, tied head): logits within 6e-6, greedy
  continuations identical in Float32, Float16 and int8.
- **TinyLlama 1.1B**: prefill plus KV-cached decode. Perplexity in Float32
  matches the reference, and int8 weights cost +0.25%.
- **Whisper** (tiny, and other sizes from their `config.json`): transcription
  matches `transformers` token for token.

## Quick start

```julia
using Luminal

g = Graph()
a = tensor(g, [3, 1])
b = tensor(g, [1, 4])
c = matmul(a, b) * 2.0f0                 # nothing runs yet: this builds a graph

device = get_device()                    # AMDDevice, CUDADevice or CPUDevice
exec = compile(g; device=device, retain=[c.id])
result = exec(Dict(a.id => Float32[1; 2; 3;;], b.id => Float32[1 2 3 4]); device=device)
Array(result[c.id])                      # 3×4
```

Tensors are column-major. Activations use the **(Hidden, Seq, Batch)** layout
and attention heads use (HeadDim, Seq, Heads, Batch). `matmul` is batched over
trailing dimensions: (M, K, …) × (K, N, …).

### Transcribe audio (Whisper)

```julia
using Luminal
# git clone https://huggingface.co/openai/whisper-tiny
text, tokens = transcribe("whisper-tiny", "speech.wav")   # one-shot; ≤ 30 s, decoded with ffmpeg

session = WhisperSession("whisper-tiny")                  # load once, reuse compiled graphs
transcribe(session, "a.wav")
transcribe(session, ["a.wav", "b.wav", "c.wav"])          # one batch
```

### Generate text (Llama)

```julia
using Luminal, Luminal.NN
dir = "Meta-Llama-3-8B-Instruct"                     # or TinyLlama-1.1B-Chat, …
tok = LlamaTokenizer(dir)
model = Llama(Graph(), nothing; llama_config(dir)...)  # architecture from config.json
session = LlamaSession(model, tok, dir; decode_weights=Int8)
generate(session, chat_prompt(tok, "Why is the sky blue?"); max_new_tokens=100)  # compiles on first use
generate(session, chat_prompt.(Ref(tok), ["First question", "Second question"]))  # one batch
```

A session loads the weights once and keeps its compiled prefill graphs (per
padded prompt length and batch size) and decode graphs (per batch size), so
later calls cost only the generation itself. `llama_generate(model, tok, prompt, dir)`
is the one-shot form.

```bash
julia --project=. examples/llama_chat.jl path/to/TinyLlama-1.1B-Chat --chat "Why is the sky blue?"
julia --project=. examples/llama_chat.jl path/to/TinyLlama-1.1B-Chat --int8 --search=measured "..."
julia --project=. examples/llama_chat.jl path/to/TinyLlama-1.1B-Chat --chat "First?" "--prompt=Second?"   # one batch
julia --project=. examples/llama_chat.jl path/to/TinyLlama-1.1B-Chat --chat --int8 "Hi" --interactive          # keep chatting
```

## Installation

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Requirements:
- Julia 1.12.
- For AMD GPUs: AMDGPU.jl with a working ROCm. Development uses ROCm 10.0 from
  [TheRock](https://github.com/ROCm/TheRock) on a Radeon 8060S (gfx1151). Point
  `ROCM_PATH` and `LD_LIBRARY_PATH` at the install.
- For NVIDIA GPUs: CUDA.jl.
- Metatheory.jl 3.0 (`ale/3.0` branch, pinned in the Manifest) for the rewrite layer.
- `ffmpeg`, only to decode audio files for Whisper.

## Examples

| Example | What it does |
|---------|--------------|
| `examples/llama_chat.jl` | Text generation from a Llama-family checkpoint (TinyLlama, Llama-3-8B-Instruct): `--chat`, `--int8`, `--search=static\|measured`, extra `--prompt=` flags for a batch, `--interactive` |
| `examples/llama_reference.py` | Dumps Hugging Face reference logits and greedy continuations for a Llama checkpoint |
| `examples/llama_validate.jl` | Compares a checkpoint's logits, greedy output and batched output with that reference, and times decode |
| `examples/batched_decode.jl` | Decode throughput against batch size |
| `examples/whisper.jl` | Speech-to-text with a Hugging Face Whisper checkpoint |
| `examples/quant_eval.jl` | Perplexity of Float32, Float16 and int8 weights on a fixed passage |
| `examples/prefill_search.jl` | Measured search over prefill graphs |
| `examples/egraph_search.jl` | The e-graph rewrite layer on a small graph |
| `examples/llama.jl` | Decode-step benchmark on a random-weight Llama |
| `examples/phi3.jl` | Phi-3 architecture and weight mapping, random weights |
| `examples/linear_regression.jl` | Training: autograd and the Adam optimizer |
| `examples/whisper_reference.py` | Dumps Hugging Face Whisper reference tensors for validation |

## Performance

Llama-3-8B-Instruct on a Radeon 8060S (Strix Halo iGPU, ROCm 10, weights in
system memory through GTT):

| Decode weights | Batch 1 | Batch 4 | Batch 8 |
|----------------|---------|---------|---------|
| Float32 | 149 ms/token (6.7 tok/s) | | |
| Float16 | 67 ms/token (14.9 tok/s) | 52 tok/s | 77 tok/s |
| int8 | 38.3 ms/token (26 tok/s) | 76 tok/s | 109 tok/s |
| `:int4_mixed_plus` (int8 for sensitive tensors) | 33.7 ms/token (29.7 tok/s) | 84 tok/s | 111 tok/s |
| `:int4_mixed` | 29.0 ms/token (34.4 tok/s) | 93 tok/s | 112 tok/s |
| int4 | 23.5 ms/token (42.5 tok/s) | 98 tok/s | 111 tok/s |

WikiText-2 perplexity against Float32 (two texts): int8 0.0%;
`:int4_mixed_plus` −0.7% / +0.4%; `:int4_mixed` +0.8% / +2.4%; int4 +5.2% / +7.9%.
See [docs/performance_vs_luminal.md](docs/performance_vs_luminal.md).

For comparison, upstream Luminal reports 229 ms/token for the same checkpoint
on an NVIDIA H200 with Float32 weights (batch 1; `examples/llm_chat`, September
2026). Float32 prefill of a 22-token chat prompt takes 0.23 s.

TinyLlama 1.1B on the same GPU, batch 1:

| | Weights | Time | |
|-|---------|------|-|
| Decode | int8 (group-wise, weight-only) | **6.4 ms/token** | 157 tok/s |
| Decode | Float16 | 11.0 ms/token | 91 tok/s |
| Prefill, 16 tokens | Float16 GEMM (`search`, `:activations`) | 22 ms | |
| Prefill, 16 tokens | Float32 | 45 ms | |

Batched decode: B sequences share one step, and each weight is read once per step:

| Batch | int8 ms/step | int8 tok/s | Float16 ms/step | Float16 tok/s |
|------:|-------------:|-----------:|----------------:|--------------:|
| 1 | 6.7 | 149 | 11.2 | 89 |
| 2 | 8.2 | 243 | 12.0 | 167 |
| 4 | 9.7 | 412 | 13.7 | 292 |
| 8 | 14.2 | 564 | 20.3 | 394 |

(`examples/batched_decode.jl`; sequences at ~128 tokens of context. The
batch-1 times here include copying the logits to the host for the argmax.)

Perplexity on the evaluation passage: Float32 12.1229, Float16 12.1228, int8 12.1526.

With a `LlamaSession`, a warm call generating 40 tokens for 4 prompts takes
0.47 s end to end, prefill included. Loading TinyLlama's weights takes 1.3 s:
they are uploaded as stored (BF16) and converted on the GPU. The first load in
a process also compiles those kernels, and takes ~16 s.

Decode started this work at 893 ms/token. It is now bound by memory bandwidth:
the int8 GEMVs read weights at ~215-220 GB/s. On the 8B model they are ~34 of
the ~38.5 ms per token; the rest is small kernels and launch gaps inside the
HIP graph (~5.5 µs per kernel).

Whisper tiny, with a warm `WhisperSession`, spends ~17 ms computing the log-mel
spectrogram (CPU) and ~35 ms in the encoder per clip, then ~2.5 ms per token. A
batch of 4 clips decodes in about the time of 2.

## Features

### Graph and compiler
- **Primitive ops**: unary `Log2, Exp2, Exp, Sin, Sqrt, Recip, ReLU`, rounding
  `Floor, Ceil, Round, Trunc`; binary `Add, Mul, Div, Mod, Max, LessThan`; ternary
  `Select` (`select(c, a, b)`); coordinate-form `GatherND` / `ScatterND`
  (`gather(data, [coords...])`, `scatter(init, src, [coords...]; mode=:replace | :add)`,
  one 0-based coordinate tensor per axis, out-of-range reads 0 and writes are
  dropped, `:add` accumulates repeated coordinates atomically); `Iota`
  (`iota(g, dims, f)`, any function of the coordinates); `SumReduce, MaxReduce`; movement ops (`Permute, Expand,
  Reshape, Slice, Pad`); and `MatMul`. Everything else (softmax, norms, GELU,
  attention) is built from these in `HighLevelOps.jl`.
- **Element types**: tensors are `Float32` by default; `tensor(g, dims; dtype=T)`
  makes any of `Float32, Float64, Float16, BFloat16, Int8, Int32, Int64, Bool`.
  Every node's dtype is inferred (`dtype(t)`) and typing is strict, as upstream's:
  operands share a dtype, number literals take the tensor's, and conversions are
  explicit: `cast(x, T)` (float rounding, integer wrapping, never float -> integer;
  `cast(x, Bool)` is `x != 0`), `trunc_cast(x, T)` (float -> integer toward zero,
  refusing NaN, Inf and out-of-range values), `trunc_div` / `trunc_rem` (refusing a
  zero divisor), and `constant(g, v, Float64)` for exact double constants.
  Comparisons return `Bool` (`!`, `&`, `|` combine them; cast a mask to multiply
  with it). Buffers and fused kernels are typed on CPU and GPU. BFloat16 on x86
  CPUs with AVX512-BF16 needs `julia -C native,-avx512bf16`: Julia 1.12's LLVM
  fails on its vectorized conversions there, and Luminal says so rather than hang.
- **Symbolic shapes**: dimensions may be symbols, for example a decode position,
  resolved at run time.
- **`compile()`** performs:
  - elementwise fusion into generated kernels
  - buffer reuse
  - zero-copy views for reshapes, size-1 permutes, contiguous and strided slices,
    broadcasts, and concatenation
  - common-subexpression elimination
  - constant folding (`fold=true`)
- **HIP graph capture** (`capture=true`): a shape-static step, such as the
  decode step, is recorded once and replayed with a single launch.
- **Weight storage** (`weight_dtype`): `Float16` (transposed, with a GEMV
  kernel), `Int8` (group-wise scales) or `Luminal.Int4` (4-bit, symmetric, groups
  of 32, 0.5625 bytes per weight). Compute stays in Float32.
  - **Per-tensor mixes:** a function of the weight chooses each tensor's format,
    with named presets `:int4_mixed` and `:int4_mixed_plus`.
  - **Choosing a mix:** `examples/quant_sensitivity.jl` measures each tensor type's
    and layer's sensitivity, and `examples/quant_formats.jl` compares formats.
- **Search** (`compile(g; search=:static | :measured)`): an e-graph rewrite layer
  on Metatheory.jl explores equivalent graphs: expand elimination, scale motion,
  merged Q/K/V and gate/up projections, transposed matmuls, and a precision per
  matmul (Float16 or int8 GEMV, Float16 GEMM). It picks a candidate with a DAG
  cost model or by timing verified candidates on the device. Measured results
  are cached in `~/.cache/Luminal.jl/search`. The fusions and zero-copy views
  `compile()` applies on its own (elementwise fusion, views, concatenation, residual
  adds in GEMV epilogues) are named lowering sites the measured search also times
  turning off, and int8 GEMV kernel parameters (threads, columns per pass) are search
  variants. The search skips candidates that would not fit in free device memory
  and frees rejected ones immediately. See
  [docs/egraph_rewrite_layer.md](docs/egraph_rewrite_layer.md).
- **Fused kernels** for the decode hot path: `RotaryEmbed`, `DecodeAttention`
  (single-token attention over the KV cache, GQA-aware), `RMSNormOp`.

### Models and layers
- Layers: `Linear`, `Conv1D`, `Embedding`, `LayerNorm`, `RMSNorm`, `Mlp`,
  `SelfAttention` (GQA, RoPE), `TransformerBlock`.
- **Llama / TinyLlama / Llama-3 / 3.1 / 3.2 / Qwen3**: models built from `config.json`
  (`llama_config`), prefill, and a device-resident KV cache with one compiled
  decode graph for every position. `LlamaSession`/`generate` reuse them, and
  `chat_prompt` produces the model's chat format (Llama-3, Zephyr, ChatML).
  Qwen3's differences are options of the same model: a head size independent of
  the hidden size, QK-norm (an RMSNorm per head of q and k before RoPE), RMSNorm
  epsilon, and tied embeddings, where the output head is its own node loaded
  from the embedding's array (`tie_weight!`), so it can still be stored in
  reduced precision. RoPE scaling (`llama3`, `linear`) sets the frequencies
  (`rope_inv_freqs`).
- **Gemma3** (`Gemma3`, `gemma3_config`): scaled embeddings, (1 + w) RMSNorms
  around both attention and MLP, QK-norm, GELU-tanh MLP, and interleaved local
  (sliding-window, in prefill masks and the decode kernel) and global (scaled
  RoPE) layers. `model_template(dir)` picks Llama or Gemma3 from `config.json`, and
  `generate_ids` generates from token ids.
- **Qwen3.5 / Qwen3.6 (dense; in progress)**: `Qwen35` (`qwen35_config`,
  `model_template`) matches transformers on a tiny random model through
  `LlamaSession`, prefill and decode, batched. Its Gated DeltaNet layers
  (`GatedDeltaNet`; the `CausalConv` and `DeltaRule` ops keep the convolution and
  recurrent states in place, consuming each sequence up to its `lens`) and gated
  full attention (`qwen35_attention`: output gate, RoPE on a quarter of each head,
  (1 + w) QK-norm) share one decode step with a hybrid cache: K/V for the full
  layers, states for the linear ones. Next: loading the 27B checkpoint straight
  into int4.
- **Batched generation**: `llama_generate(model, tok, prompts::Vector{String}, dir)`
  runs one right-padded prefill for all prompts. It then decodes them together,
  each sequence at its own position, and stops each one independently. Its
  output is identical to generating each prompt alone.
- **Phi-3**: architecture and weight mapping.
- **Whisper**:
  - log-mel frontend matching `WhisperFeatureExtractor`
  - audio encoder and text decoder
  - cached greedy decoding (`greedy_decode`, `transcribe`), batched over clips,
    with `WhisperSession` keeping weights and compiled graphs between calls
- **Weights**: safetensors (F32/F16/BF16) mapped by Hugging Face key through a
  `WeightRegistry`. They are converted to Float32 and transposed on the device.
- **Tokenizers**: SentencePiece-style BPE (Llama-2, TinyLlama, Phi-3), byte-level BPE
  (Llama-3, Qwen3, Whisper). Llama tokenizers match Hugging Face token for token,
  added tokens included (special or not, e.g. Qwen3's `<think>`).

### Training
Reverse-mode autodiff (`backward`, `gradients`) with gradient rules for all 12 of
Luminal's original primitives:
- Log2, Exp2, Sin, Sqrt, Recip, Add, Mul, Mod;
- SumReduce and MaxReduce (ties share the gradient);
- Contiguous;
- LessThan (zero gradient).

It also covers ReLU, MatMul and the movement ops (Permute, Reshape, Expand, Slice,
Pad), with broadcasting. `SGD` and `Adam` optimizers are included. The fused
inference kernels (`DecodeAttention`, `RMSNormOp`, `RotaryEmbed`, `SoftmaxOp`) and
the reduced-precision weight formats have no gradients.

### Devices
- **AMD ROCm** (AMDGPU.jl) is the primary, tested target.
- **CPU** runs everything, with plain-loop fallbacks for the fused kernels.
- **CUDA** (CUDA.jl) has backends for the generic paths but has not been
  exercised in recent work.

### XLA via Reactant.jl (optional)
With [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) loaded, a Luminal graph
(as a model builds it, before `compile`) can be traced into StableHLO and
compiled by XLA, which opens it to the Reactant ecosystem (Enzyme, Lux, sharding,
XLA's CPU/GPU/TPU backends):

```julia
using Luminal, Reactant
f  = reactant_function(g, out, [inp])      # pure Julia function Reactant can trace
fc = reactant_compile(g, out, [inp], toks)  # XLA executable; call with Reactant.to_rarray args
to_stablehlo(g, out, [inp], toks)           # the MLIR text
```

Weights preloaded into the graph are baked in as constants; list them in the
inputs to pass them as arguments instead. Covered: the primitive ops, gather,
and the RMSNorm / softmax / rotary ops the model builders emit (Llama prefill
matches the interpreter). Not covered: the compiled graph's kernels
(reduced-precision weights, `DecodeAttention` with its in-place cache update).
Reactant is a weak dependency (`ext/LuminalReactantExt.jl`); its tests have their
own environment, `tests/reactant`.

This is an interoperability route, not a faster one here: Reactant bundles its
own ROCm, which cannot share a process with AMDGPU.jl's, and its XLA build has
no GEMM kernels for gfx1151 (Radeon 8060S), so on this machine it runs on XLA's
CPU backend.

### Not yet implemented
- Multi-GPU or distributed execution
- Tensor-core or matrix-core kernels, and generated (rather than hand-written) kernels
- Other upstream models (for example YOLO)

## Testing

```bash
julia --project=. tests/runtests.jl                  # every tests/test_*.jl
julia --project=. tests/runtests.jl whisper kv_cache # files whose name contains a pattern
julia --project=. tests/test_llama.jl                # a single file
```

GPU tests run when `get_device()` finds a GPU. `test_greedy_decode.jl` also
checks a real transcription against Hugging Face when `whisper_tiny/`
(openai/whisper-tiny) and its reference are present. To produce the reference
(requires `transformers`), run:

```bash
python3 examples/whisper_reference.py whisper_tiny whisper_tiny/ref/audio.f32 whisper_tiny/ref
```

The Reactant extension is tested in its own environment (see
[tests/README.md](tests/README.md)):

```bash
julia --project=tests/reactant -e 'using Pkg; Pkg.instantiate()'
julia --project=tests/reactant tests/reactant/runtests.jl
```

See [tests/README.md](tests/README.md).

## Layout

```
src/
├── Luminal.jl          # module, exports
├── Ops.jl              # primitive and fused op types
├── Graph.jl            # graph, tensors, CSE
├── ShapeTracker.jl     # symbolic shapes and views
├── HighLevelOps.jl     # op library built from the primitives
├── Compiler.jl         # compile(): fusion, buffers, views, folding, capture
├── EGraphRewrite.jl    # e-graph rewrite layer and measured search
├── Execution.jl        # interpreter and kernels (GEMV, attention, norms, …)
├── Device.jl           # devices, transfers, HIP graph capture
├── NN.jl               # layers, Llama, Phi-3, KV-cached decode
├── Whisper.jl          # audio frontend, encoder, decoder, cached decode
├── Decoding.jl         # llama_generate, greedy_decode, transcribe
├── Weights.jl          # safetensors loading, WeightRegistry
├── Autograd.jl, Optimizer.jl
└── LlamaTokenizer.jl, WhisperTokenizer.jl
```

## License

Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
or the MIT license (http://opensource.org/licenses/MIT), at your option.
