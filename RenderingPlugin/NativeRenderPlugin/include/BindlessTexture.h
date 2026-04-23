#pragma once
#include <d3d12.h>
#include <vector>
#include "DescriptorHeapAllocator.h"
#include "IUnityLog.h"

// ---------------------------------------------------------------------------
// BindlessTexture
//   A contiguous block of GPU-visible Texture2D SRV descriptors allocated from
//   a shared DescriptorHeapAllocator.
//
//   The object is completely independent of any particular shader type. Callers
//   bind it to a pipeline by passing its GPU descriptor handle (GetGPUHandle())
//   to a root descriptor table parameter that expects an unbounded SRV array.
//
//   Lifecycle:
//     auto* bt = new BindlessTexture();
//     bt->Initialize(device, &globalAllocator, capacity, log);
//     bt->SetTexture(0, myTex);
//     // ... bind to shader each dispatch via SetBindlessTexture ...
//     delete bt;   // automatically frees descriptor slots
//
//   Resize() frees the old descriptor range and allocates a new one. After a
//   resize the caller MUST re-bind the object to any shader that references it,
//   since the GPU handle changes.
// ---------------------------------------------------------------------------
class BindlessTexture
{
public:
    BindlessTexture() = default;
    ~BindlessTexture();

    // Must be called exactly once before any other method.
    // |capacity| is the initial number of texture slots.
    bool Initialize(ID3D12Device*            device,
                    DescriptorHeapAllocator* allocator,
                    uint32_t                 capacity,
                    IUnityLog*               log = nullptr);

    // Set the texture at |index|. Writes the SRV immediately into the heap.
    // Pass nullptr to write a null/fallback SRV.
    // |index| must be < Capacity().
    void SetTexture(uint32_t index, ID3D12Resource* resource);

    // Resize the array.
    //   Grow: new slots are filled with null SRVs.
    //   Shrink: slots beyond newCapacity are discarded.
    // After resize the GPU handle changes – re-bind to all shaders that use it.
    void Resize(uint32_t newCapacity);

    uint32_t Capacity()  const { return m_capacity;  }
    uint32_t AllocBase() const { return m_allocBase; }

    // Non-owning pointer to the resource at |index|, or nullptr if the slot is empty.
    ID3D12Resource* GetTexture(uint32_t index) const
    {
        return (index < m_capacity) ? m_textures[index] : nullptr;
    }

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

    std::vector<ID3D12Resource*> m_textures;   // non-owning; nullptr = null slot

    uint32_t m_capacity  = 0;
    uint32_t m_allocBase = 0;
    bool     m_initialized = false;
};
