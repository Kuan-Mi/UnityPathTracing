#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <cstdint>

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// NativeGpuBuffer
//   Single-slot GPU-resident (DEFAULT heap) buffer.
//   Created with D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS so it can be
//   bound as either a typed/structured SRV or a typed/structured UAV.
//
//   Typical usage:
//     - GPU writes via UAV (e.g. compute shader histogram pass)
//     - GPU reads via SRV in a subsequent pass
//     - No CPU upload; size is fixed at creation time.
//
//   Binding:
//     Pass the NativeGpuBuffer* as objectPtr with
//     objectKind = NativeGpuBuffer (7) in a BindingSlot.
//     DescriptorSetBase resolves GetResource() to create the SRV/UAV.
// ---------------------------------------------------------------------------
class NativeGpuBuffer
{
public:
    NativeGpuBuffer()  = default;
    ~NativeGpuBuffer() = default;

    // Allocates a DEFAULT-heap buffer of |sizeInBytes| with ALLOW_UNORDERED_ACCESS.
    // sizeInBytes is used as-is (caller is responsible for alignment if needed).
    bool Initialize(ID3D12Device* device, uint32_t sizeInBytes);

    // Return the underlying ID3D12Resource*.
    ID3D12Resource* GetResource() const { return m_buffer.Get(); }

    // Return the GPU virtual address.
    D3D12_GPU_VIRTUAL_ADDRESS GetGPUVA() const;

    uint32_t SizeInBytes() const { return m_sizeInBytes; }

private:
    ComPtr<ID3D12Resource> m_buffer;
    uint32_t               m_sizeInBytes = 0;
};
