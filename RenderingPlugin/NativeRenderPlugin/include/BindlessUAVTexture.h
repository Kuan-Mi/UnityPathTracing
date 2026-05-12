#pragma once
#include <d3d12.h>
#include <vector>
#include "DescriptorHeapAllocator.h"
#include "IUnityLog.h"

// ---------------------------------------------------------------------------
// BindlessUAVTexture
//   A contiguous block of GPU-visible RWTexture2D UAV descriptors allocated
//   from a shared DescriptorHeapAllocator.
//
//   Mirrors BindlessTexture but targets UAV (unordered-access) descriptors so
//   that it can be bound to an unbounded RWTexture2D[] array in HLSL.
//   Each slot is written with a per-mip-slice UAV, enabling shaders such as
//   GenerateMips / PreprocessEnvironmentMap to write individual mip levels
//   via a single descriptor table.
//
//   Lifecycle:
//     auto* uav = new BindlessUAVTexture();
//     uav->Initialize(device, &globalAllocator, mipCount, log);
//     // slot 0 → mip 1, slot 1 → mip 2, ...
//     uav->SetTexture(0, myTex, 1, DXGI_FORMAT_R32_FLOAT);
//     uav->SetTexture(1, myTex, 2, DXGI_FORMAT_R32_FLOAT);
//     // bind to shader each dispatch via SetBindlessRWTexture(...)
//     delete uav;   // automatically frees descriptor slots
//
//   After Resize() the GPU handle changes — re-bind to all shaders.
// ---------------------------------------------------------------------------
class BindlessUAVTexture
{
public:
    BindlessUAVTexture() = default;
    ~BindlessUAVTexture();

    // Must be called exactly once before any other method.
    // |capacity| is the initial number of UAV descriptor slots.
    bool Initialize(ID3D12Device*            device,
                    DescriptorHeapAllocator* allocator,
                    uint32_t                 capacity,
                    IUnityLog*               log = nullptr);

    // Set the descriptor at |index|.
    //   |resource|  – the D3D12 resource to create a UAV view for (nullptr = null descriptor)
    //   |mipSlice|  – which mip level to expose as the UAV (0-based)
    //   |format|    – DXGI_FORMAT for the UAV view; pass DXGI_FORMAT_UNKNOWN to derive from resource
    // |index| must be < Capacity().
    void SetTexture(uint32_t index, ID3D12Resource* resource,
                    uint32_t mipSlice = 0, DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN);

    // Resize the array.
    //   Grow: new slots are filled with null UAVs.
    //   Shrink: slots beyond newCapacity are discarded.
    // After resize the GPU handle changes – re-bind to all shaders that use it.
    void Resize(uint32_t newCapacity);

    uint32_t Capacity()  const { return m_capacity;  }
    uint32_t AllocBase() const { return m_allocBase; }

    // GPU handle for the start of the descriptor range.
    // Pass this to SetComputeRootDescriptorTable() for the matching root param.
    D3D12_GPU_DESCRIPTOR_HANDLE GetGPUHandle() const;

private:
    struct SlotInfo
    {
        ID3D12Resource* resource  = nullptr;  // non-owning
        uint32_t        mipSlice  = 0;
        DXGI_FORMAT     format    = DXGI_FORMAT_UNKNOWN;
    };

    void WriteDescriptor(uint32_t index);
    void WriteNullDescriptor(uint32_t index);

    void Log(UnityLogType type, const char* msg) const;

    ID3D12Device*            m_device    = nullptr;
    DescriptorHeapAllocator* m_allocator = nullptr;
    IUnityLog*               m_log       = nullptr;

    std::vector<SlotInfo> m_slots;

    uint32_t m_capacity  = 0;
    uint32_t m_allocBase = 0;
    bool     m_initialized = false;
};
