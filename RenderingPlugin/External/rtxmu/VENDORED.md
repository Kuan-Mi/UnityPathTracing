# Vendored RTXMU (NVIDIA RTX Memory Utility)

D3D12 subset of https://github.com/NVIDIA-RTX/RTXMU, vendored from the copy that
ships inside nvrhi (RTXPT's renderer backend):

- Source:  `nvrhi/rtxmu` submodule, commit `0c9ce1177000d5923e2cc6a35ae9cb7ff03748d2` (2024-11-26)
- Files:   `include/rtxmu/{AccelStructManager,D3D12AccelStructManager,D3D12Suballocator,Suballocator,Logger}.h`,
           `src/{D3D12AccelStructManager,D3D12Suballocator,Logger}.cpp` (Vulkan files omitted)
- Local change: `Suballocator.h` adds `#include <string>` (upstream relies on a
  transitive include that our build does not provide).

Used by NativeRenderPlugin's `AccelerationStructure` for BLAS result/scratch/
update-scratch suballocation and batched compaction, replacing per-BLAS
committed resources — the same role it plays in nvrhi/RTXPT.
