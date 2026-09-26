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
