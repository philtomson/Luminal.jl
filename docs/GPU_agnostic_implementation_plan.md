# GPU-Agnostic Implementation Plan

This plan outlines the migration of `Luminal.jl` from a CUDA-centric model to a GPU-agnostic architecture supporting both NVIDIA (CUDA) and AMD (ROCm) hardware.

## Status: Phase 1 Complete (CUDA/KA)

Phase 1 has been successfully implemented and merged into `main`. The architecture now uses `KernelAbstractions.jl` for core kernels and a unified `AbstractGPUDevice` hierarchy.

### Completed Milestones
- [x] **Device Abstraction**: `CUDADevice` and `AMDDevice` inherit from `AbstractGPUDevice`.
- [x] **Memory Management**: Unified `reclaim!(device)` and `available_memory(device)` interfaces.
- [x] **Kernel Porting**: `Gather` and `FlashAttention` kernels migrated to `KernelAbstractions.jl`.
- [x] **Numerical Parity**: Verified that output logits on `CUDADevice` match `CPUDevice` precisely in `debug_prefill.jl`.
- [x] **Fixes**: Resolved `RMSNorm` regression and layout inconsistencies in `SelfAttention`.

---

## Phase 2: AMD GPU Validation

The goal for Phase 2 is to verify the same codebase on AMD hardware with minimal to no changes.

### 1. Environment Setup
- Ensure `AMDGPU.jl` is installed and functional on the target machine.
- Verify that `ROCm` drivers are correctly configured.
- `Luminal.jl` will automatically detect the backend, but you can force it by ensuring `AMDGPU.jl` is loaded.

### 2. Device Dispatch Verification
- Verify that `Luminal.get_device()` returns an `AMDDevice()` when running on the AMD PC.
- Check `src/Device.jl` to ensure the `to_device` and `from_device` methods dispatch correctly to `AMDGPU.ROCArray`.

### 3. Kernel Validation
- Run the `KernelAbstractions` based kernels on the `AMDGPU` backend.
- **Critical Test**: Run `julia --project debug_prefill.jl` and compare results against the CPU/CUDA baselines.

### 4. Verification Steps for LLM on AMD PC
1. **Load Environment**: Activate the `Luminal.jl` project.
2. **Device Check**: Run `using Luminal; Luminal.get_device()` to confirm `AMDDevice` detection.
3. **Prefill Test**: Execute `debug_prefill.jl`.
   - Logits should exhibit similar stats to the CUDA baseline: `mean ≈ -3.41`, `std ≈ 2.71`, `max ≈ 14.49`.
4. **End-to-End Chat**: Run `examples/tinyllama_chat.jl` to verify text generation.

## Future Considerations
- **Performance Tuning**: Optimize KA kernel tiling sizes specifically for AMD's CDNA/RDNA architectures if necessary.
- **Fusion**: Evaluate `AMDGPU` support for the fused elementwise engine in `Compiler.jl`.
