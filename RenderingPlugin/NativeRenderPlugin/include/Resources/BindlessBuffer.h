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

    // Highest slot index ever assigned a non-null resource, plus one. All slots at
    // or beyond this are guaranteed null, so per-dispatch resource-state walks can
    // stop here instead of scanning the full capacity.
    uint32_t UsedCount() const { return m_usedCount; }

    // Per-frame resource-state sweep dedup — see BindlessTexture.h for the full
    // rationale (donut's setPermanentTextureState adapted to Unity-owned state
    // tracking). Valid for buffers too: anything that rewrites an element between
    // our dispatches (e.g. Unity skinning) runs before the SRP passes, and the
    // new frame serial re-arms the sweep.
    bool NeedsStateSweep(uint64_t frameSerial, uint32_t state) const
    {
        return m_sweptEpoch != m_contentEpoch || m_sweptFrame != frameSerial ||
               (m_sweptStates & state) != state;
    }
    void MarkStateSweep(uint64_t frameSerial, uint32_t state)
    {
        if (m_sweptFrame != frameSerial || m_sweptEpoch != m_contentEpoch)
            m_sweptStates = 0;
        m_sweptEpoch  = m_contentEpoch;
        m_sweptFrame  = frameSerial;
        m_sweptStates |= state;
    }

    // Non-owning pointer to the resource at |index|, or nullptr if the slot is empty.
    ID3D12Resource* GetBuffer(uint32_t index) const
    {
        return (index < m_capacity) ? m_buffers[index] : nullptr;
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

    std::vector<ID3D12Resource*> m_buffers;    // non-owning; nullptr = null slot

    uint32_t m_capacity  = 0;
    uint32_t m_allocBase = 0;
    uint32_t m_usedCount = 0;   // see UsedCount(); grow-only except on shrinking Resize
    bool     m_initialized = false;

    // Sweep-dedup state (see NeedsStateSweep). m_contentEpoch bumps on every
    // SetBuffer / Resize so a content change forces the next dispatch to re-sweep.
    uint64_t m_contentEpoch = 1;
    uint64_t m_sweptEpoch   = 0;
    uint64_t m_sweptFrame   = ~0ull;
    uint32_t m_sweptStates  = 0;
};
