#pragma once
#include "DescriptorHeapAllocator.h"
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
    AccelStruct,
    RayTraceShader,
    ComputeShader,
    ComputeDescriptorSet,
    AccelStructBlas,
};
