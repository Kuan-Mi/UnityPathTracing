#pragma once
#include "DescriptorHeapAllocator.h"
#include "TransientDescriptorRing.h"
#include <cstdint>

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

enum class DeferredType {
    BindlessTexture,
    BindlessBuffer,
    BindlessUAVTexture,
    AccelStruct,
    RayTraceShader,
    RayTraceDescriptorSet,
    ComputeShader,
    ComputeDescriptorSet,
    AccelStructBlas,
    NativeBuffer,
    NativeStructuredBuffer,
    NativeGpuBuffer,
};

// ---------------------------------------------------------------------------
// g_frameIndex
//   Global triple-buffer frame index shared by all subsystems (AccelerationStructure,
//   ComputeDescriptorSet, etc.).  Advanced exactly once per frame by
//   AccelerationStructure::BuildOrUpdate().  Cycles 0..2.
// ---------------------------------------------------------------------------
static constexpr uint32_t kGlobalNumFrames = 3;
extern uint32_t g_frameIndex;

// ---------------------------------------------------------------------------
// g_transientRing
//   Shared transient-descriptor ring used by ComputeDescriptorSet and
//   RayTraceDescriptorSet for per-dispatch SRV/UAV table allocation.  Reserves
//   a contiguous sub-range of s_DescHeap at renderer init; allocations are
//   reclaimed by the frame fence at each NR_FrameTick.
// ---------------------------------------------------------------------------
extern TransientDescriptorRing g_transientRing;
