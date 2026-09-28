# Test Suite

```bash
julia --project=. tests/runtests.jl                  # every test_*.jl, each in its own module
julia --project=. tests/runtests.jl whisper kv_cache # files whose name contains a pattern
julia --project=. tests/test_rotary.jl               # a single file
```

Tests that have GPU variants run them when `get_device()` finds a GPU, next to the CPU run.

## Core

| Test | Covers |
|------|--------|
| `test_symbolic.jl` | Symbolic expressions |
| `test_shape_tracker.jl` | ShapeTracker views and dimensions |
| `test_symbolic_slice.jl` | Slices with symbolic bounds, CPU and GPU |
| `test_lazy.jl` | Graphs compute nothing until compiled and run |
| `test_end_to_end.jl` | Build, compile and run a small graph |

## Compiler

| Test | Covers |
|------|--------|
| `test_compilation.jl` | `compile()` against the interpreter |
| `test_fusion.jl` | Elementwise fusion |
| `test_gated_deltanet.jl` | Gated DeltaNet (Qwen3.5/3.6 linear attention) against transformers on the tiny model in `data/qwen35_tiny` (from `examples/qwen35_reference.py`): output, convolution and recurrent state after a batch-2 prefill and 3 decode steps; interpreter and `compile()`, CPU and GPU |
| `test_qwen35_attention.jl` | Qwen3.5/3.6 gated full attention (output gate, partial RoPE, (1 + w) QK-norm) against transformers on the tiny model: prefill and 3 cached decode steps, CPU and GPU |
| `test_gemma3.jl` | Gemma3 architecture on a tiny random model with a 3-token window: prefill against a plain-Julia reference, windowed cached decode against it (CPU and GPU), config parsing; the chat prompt's tokens when `gemma3_4b/` is present |
| `test_rope_scaling.jl` | `llama3` / `linear` RoPE scaling: frequencies against transformers' values, config parsing, a scaled model's decode against its prefill |
| `test_qwen3.jl` | Qwen3 architecture on a tiny random model (decoupled head size, QK-norm, tied head, eps 1e-6): prefill against a plain-Julia reference, cached decode against prefill (CPU and GPU), config parsing; the Qwen3 chat prompt's tokens when `qwen3_0.6b/` is present |
| `test_matmul_shapes.jl` | `matmul` across ranks (2D to 4D, batch broadcasting, rank mismatch, non-contiguous operands): interpreter, functional form, `compile()` CPU and GPU; batched and rank-broadcast gradients |
| `test_dtypes.jl` | Dtype inference and strict checks; every dtype through the interpreter and fused `compile()` (CPU and GPU); `cast`, `trunc_cast`, `trunc_div` / `trunc_rem` and their run-time refusals; Float64 constants; integer coordinates; gradients through casts; e-graph search. BFloat16 cases need `julia -C native,-avx512bf16` on AVX512-BF16 CPUs |
| `test_gather_scatter.jl` | Coordinate gather, scatter (`:replace`, atomic `:add`, out of range), `iota`: interpreter, `compile()` CPU and GPU, gradients, e-graph search |
| `test_elementwise_ops.jl` | Rounding, `select`, exact `Div` and `Exp`: interpreter, `compile()` (fused, CPU and GPU), e-graph search, gradients |
| `test_concat_views.jl` | Concatenation and strided-slice views |
| `test_half_weights.jl` | Float16 and int8 weight storage, the weight cache |
| `test_egraph_rewrite.jl` | Graph ⇄ e-graph, rewrite rules, kernel and precision variants, `compile(...; search=...)` |

## Kernels and devices

| Test | Covers |
|------|--------|
| `test_gpu_detection.jl` | Device detection and transfers |
| `test_gpu_execution.jl` | Graph execution on the GPU |
| `test_attention.jl` | Flash attention against a reference, CPU and GPU |
| `test_rotary.jl` | Fused `RotaryEmbed` |
| `test_decode_attention.jl` | Fused single-token `DecodeAttention` |
| `test_rmsnorm.jl` | Fused `RMSNormOp` |

## Layers and models

| Test | Covers |
|------|--------|
| `test_nn_layers.jl` | `Linear`, `Embedding`, `LayerNorm` values |
| `test_llama.jl` | Llama components against reference values |
| `test_llama_compiled.jl` | Llama through `compile()` |
| `test_batched_decode.jl` | Batched decode at different positions per sequence equals batch-1 decode, including with HIP capture; right-padded batched prefill equals individual prefills. With the TinyLlama checkpoint in `tinyllama_chat/`, `LlamaSession` reuses its graphs and batched output equals single-prompt output |
| `test_phi3_loading.jl` | Phi-3 weight mapping |
| `test_weight_loading.jl` | `WeightRegistry` and safetensors; Whisper keys and shapes equal openai/whisper-tiny's |

## Whisper

| Test | Covers |
|------|--------|
| `test_mel_spectrogram.jl` | Slaney mel filterbank (against `WhisperFeatureExtractor` values), STFT, log-mel |
| `test_whisper_components.jl` | `Conv1D` against a direct convolution, exact GELU, encoder and decoder shapes |
| `test_whisper_tokenizer.jl` | Byte-level BPE, special tokens |
| `test_kv_cache.jl` | Cached decoding reproduces the full decoder's logits, CPU and GPU |
| `test_greedy_decode.jl` | `greedy_decode` with mock weights. If `whisper_tiny/ref` exists, also checks that `transcribe` matches Hugging Face token for token (see `examples/whisper_reference.py`) |

## Training

| Test | Covers |
|------|--------|
| `test_autograd.jl` | Gradients of arithmetic, broadcasting, matmul, unary ops |
| `test_optimizer.jl` | SGD and Adam |

## Reactant extension

`reactant/runtests.jl` compiles graphs through Reactant.jl / XLA
(`ext/LuminalReactantExt.jl`) and compares them with the interpreter: matmul and
softmax, elementwise ops and reductions, matmul across ranks, rounding, `select`, `Div` and `Exp`, gather / scatter / `iota`, views (permute,
slice, pad, concat), and a two-layer Llama prefill with weights as arguments and baked in as constants,
plus its StableHLO. It has its own environment, so Reactant never enters
Luminal's:

```bash
julia --project=tests/reactant -e 'using Pkg; Pkg.instantiate()'
julia --project=tests/reactant tests/reactant/runtests.jl   # LUMINAL_REACTANT_BACKEND=gpu to try XLA's GPU backend
```

On a ROCm machine whose kernel the bundled runtime does not support (seen with
gfx1151), Reactant needs the system HSA runtime preloaded, and its
librocm_sysdeps libraries on the library path, e.g.
`env -u LD_LIBRARY_PATH LD_PRELOAD=$ROCM/lib/libhsa-runtime64.so.1 LD_LIBRARY_PATH=$ROCM/lib/rocm_sysdeps/lib julia ...`.
