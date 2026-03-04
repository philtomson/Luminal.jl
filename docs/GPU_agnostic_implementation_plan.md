# GPU-Agnostic Implementation Plan

This plan outlines the steps to migrate `Luminal.jl` from a CUDA-centric model to a GPU-agnostic architecture that supports both Nvidia (CUDA) and AMD (ROCm) hardware using the internal Julia GPU ecosystem.

## Goals
- Support Nvidia and AMD GPUs with a single codebase.
- Leverage `GPUArrays.jl` for generic array operations.
- Use `KernelAbstractions.jl` (KA) for high-performance kernels.
- Abstract device-specific memory and execution management.

## User Review Required

> [!IMPORTANT]
> This migration will involve replacing many `CUDA.CuArray` references with `AbstractGPUArray`. While `GPUArrays.jl` handles most operations, certain low-level optimizations (like manual memory reclaim) will be abstracted through a unified interface.

## Proposed Changes

### [src/Device.jl]
- Introduce a unified `GPUDevice` structure (or keep `CUDADevice`/`AMDDevice` as thin wrappers over a common backend-agnostic base).
- Add `GPUArrays` and `KernelAbstractions` as primary interface dependencies.
- Implement backend-agnostic `to_device` and `from_device` using `AbstractGPUArray`.

### [src/Execution.jl]
- **Kernel Refactoring**:
    - [NEW] Port `FlashAttention` kernel to `KernelAbstractions.jl`.
    - [NEW] Port `_gather_kernel!` to `KernelAbstractions.jl`.
- **Operator Abstraction**:
    - Update `execute_op!` to use `GPUArrays` functions where possible.
    - Ensure `batch_matmul!` uses backend-specific BLAS via `LinearAlgebra` dispatch on GPU types.

### [src/Compiler.jl]
- Generalize fusion logic. Currently, fusion is limited or disabled on CUDA due to World Age issues. This needs to be evaluated for `AMDGPU.jl`.
- Use a unified `device_is_gpu()` helper instead of explicit `isa CUDADevice`.

### [src/Decoding.jl]
- Replace `CUDA.reclaim()` with a new `Luminal.reclaim!(device)` interface.
- Replace `CUDA.available_memory()` with `Luminal.available_memory(device)`.

## Verification Plan

### Automated Tests
- Run `debug_prefill.jl` on an Nvidia GPU (existing).
- Run `debug_prefill.jl` on an AMD GPU (if available in the environment).
- Add a new test suite specifically for `KernelAbstractions` kernel validation.

### Manual Verification
- Verify that `examples/tinyllama_chat.jl` runs without modification on both CUDA and AMD backends.
