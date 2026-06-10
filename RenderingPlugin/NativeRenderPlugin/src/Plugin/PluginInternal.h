#pragma once
#include "DescriptorHeapAllocator.h"
#include "TransientDescriptorRing.h"
#include "SharedUploadPool.h"
#include <cstdint>
#include <functional>

// ---------------------------------------------------------------------------
// Internal helpers shared between Plugin.cpp and the resource classes.
// Not part of the public plugin API.
// ---------------------------------------------------------------------------

// Enqueue a descriptor heap range free to be executed after a GPU fence delay
// (kDeleteDelay frames, matching NR_DestroyBindlessTexture / NR_DestroyBindlessBuffer).
// Safe to call from any thread after the renderer is initialised.
// If the deletion fence is not yet available (e.g. early init), the range is
// freed immediately.
void NR_EnqueueDescriptorRangeFree(DescriptorHeapAllocator* alloc,
                                   uint32_t                 base,
                                   uint32_t                 count);

// Enqueue an arbitrary cleanup task to run after a GPU fence delay (kDeleteDelay
// frames). Used by resource classes to defer-release transient D3D12 resources
// (e.g. per-flush upload buffers) until the GPU is guaranteed done with them.
void EnqueueCleanup(std::function<void()>&& cleanupTask);

// Defer-delete a plugin-owned heap object until the GPU has finished any work that
// may still reference it (the delete runs once the retire-bucket fence completes).
// Type-safe replacement for the old void*+DeferredType dispatch: each caller already
// knows the concrete type, and `delete p` resolves it via the caller's include.
template<class T>
inline void RetireObject(T* p) { if (p) EnqueueCleanup([p]{ delete p; }); }

// ---------------------------------------------------------------------------
// g_frameIndex
//   Global triple-buffer frame index shared by all subsystems (AccelerationStructure,
//   ComputeDescriptorSet, etc.).  Advanced exactly once per frame by
//   AccelerationStructure::BuildOrUpdate().  Cycles 0..2.
// ---------------------------------------------------------------------------
static constexpr uint32_t kGlobalNumFrames = 3;
extern uint32_t g_frameIndex;

// ---------------------------------------------------------------------------
// g_frameSerial
//   Monotonic frame counter advanced alongside g_frameIndex at each frame tick
//   (never wraps in practice). Used by per-frame dedup logic that needs to ask
//   "did X already happen this frame?" — e.g. the bindless SRV-array resource-
//   state sweep in DescriptorSetBase, which only needs to run once per frame
//   per array (the donut equivalent is setPermanentTextureState on loaded
//   textures; here Unity owns the state tracker, so once-per-frame is the
//   safe adaptation).
// ---------------------------------------------------------------------------
extern uint64_t g_frameSerial;

// ---------------------------------------------------------------------------
// g_transientRing
//   Shared transient-descriptor ring used by ComputeDescriptorSet and
//   RayTraceDescriptorSet for per-dispatch SRV/UAV table allocation.  Reserves
//   a contiguous sub-range of s_DescHeap at renderer init; allocations are
//   reclaimed by the frame fence at each NR_FrameTick.
// ---------------------------------------------------------------------------
extern TransientDescriptorRing g_transientRing;

// ---------------------------------------------------------------------------
// g_uploadPool
//   Shared UPLOAD-heap chunk pool used to stage CPU writes into DEFAULT-heap
//   buffers (e.g. NativeBuffer::UploadSnapshot). Initialised at renderer
//   init; chunks are reclaimed by the frame fence at each NR_FrameTick.
// ---------------------------------------------------------------------------
extern SharedUploadPool g_uploadPool;

// ---------------------------------------------------------------------------
// g_scratchPool
//   Shared DEFAULT-heap, UAV-capable scratch pool used as transient build scratch
//   for all D3D12 acceleration-structure builds (BLAS, OMM array, TLAS). The
//   scratch-mode counterpart of g_uploadPool — modelled on nvrhi's m_DxrScratchManager.
//   Chunks are reclaimed by the frame fence at each NR_FrameTick.
// ---------------------------------------------------------------------------
extern SharedUploadPool g_scratchPool;
