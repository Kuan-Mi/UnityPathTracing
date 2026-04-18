#pragma once
#include <d3d12.h>
#include <vector>
#include "DescriptorHeapAllocator.h"
#include "IUnityLog.h"

// ---------------------------------------------------------------------------
// BindlessBuffer
//   A contiguous block of GPU-visible ByteAddressBuffer SRV descriptors
//   allocated from a shared DescriptorHeapAllocator.
//
//   Mirrors BindlessTexture but targets buffer resources (ByteAddressBuffer /
//   StructuredBuffer viewed as raw).  Each slot is written as a raw SRV
//   (DXGI_FORMAT_R32_TYPELESS + D3D12_BUFFER_SRV_FLAG_RAW).
//
//   Lifecycle:
//     auto* bb = new BindlessBuffer();
//     bb->Initialize(device, &globalAllocator, capacity, log);
//     bb->SetBuffer(0, myBuffer);
//     // ... bind to shader each dispatch via SetBindlessBuffer ...
//     delete bb;   // automatically frees descriptor slots
//
//   Resize() frees the old descriptor range and allocates a new one. After a
//   resize the caller MUST re-bind the object to any shader that references it,
//   since the GPU handle changes.
// ---------------------------------------------------------------------------
class BindlessBuffer
{
public:
    BindlessBuffer() = default;
    ~BindlessBuffer();

    // Must be called exactly once before any other method.
    // |capacity| is the initial number of buffer slots.
    bool Initialize(ID3D12Device*            device,
                    DescriptorHeapAllocator* allocator,
                    uint32_t                 capacity,
                    IUnityLog*               log = nullptr);

    // Set the buffer at |index|. Writes the raw SRV immediately into the heap.
    // Pass nullptr to write a null/fallback SRV.
    // |index| must be < Capacity().
    void SetBuffer(uint32_t index, ID3D12Resource* resource);

    // Resize the array.
    //   Grow: new slots are filled with null SRVs.
    //   Shrink: slots beyond newCapacity are discarded.
    // After resize the GPU handle changes – re-bind to all shaders that use it.
    void Resize(uint32_t newCapacity);

    uint32_t Capacity()  const { return m_capacity;  }
    uint32_t AllocBase() const { return m_allocBase; }

    // GPU handle for the start of the descriptor range.
    // Pass this to SetComputeRootDescriptorTable() for the matching root param.
    D3D12_GPU_DESCRIPTOR_HANDLE GetGPUHandle() const;

private:
    void WriteDescriptor(uint32_t index, ID3D12Resource* resource);
    void WriteNullDescriptor(uint32_t index);

    void Log(UnityLogType type, const char* msg) const;

    ID3D12Device*            m_device    = nullptr;
    DescriptorHeapAllocator* m_allocator = nullptr;
    IUnityLog*               m_log       = nullptr;

    std::vector<ID3D12Resource*> m_buffers;    // non-owning; nullptr = null slot

    uint32_t m_capacity  = 0;
    uint32_t m_allocBase = 0;
    bool     m_initialized = false;
};
