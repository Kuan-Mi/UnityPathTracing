#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <cstdint>

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// NativeBuffer
//   Triple-buffered D3D12 upload-heap buffer, persistently mapped.
//   Used to replace Unity GraphicsBuffer for constant-buffer data that is
//   written on the main thread and read by the GPU on the render thread.
//
//   Frame indexing:
//     CPU writes slot g_frameIndex each frame.
//     GetResource() / GetGPUVA() return the same slot so the render thread
//     reads the data written by the same frame's AddRenderPasses call.
//
//   Binding:
//     Pass the NativeBuffer* as objectPtr with objectKind = NativeBuffer(5)
//     in a CS_BindingSlot.  ComputeDescriptorSet::Dispatch resolves the
//     current frame's ID3D12Resource* dynamically, exactly like TLAS.
// ---------------------------------------------------------------------------
class NativeBuffer
{
public:
    NativeBuffer() = default;
    ~NativeBuffer();

    // Allocate three upload-heap buffers of |sizeInBytes| and Map them all.
    // sizeInBytes is rounded up to D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT (256).
    bool Initialize(ID3D12Device* device, uint32_t sizeInBytes);

    // Copy |bytes| of data into the current frame's mapped slot (g_frameIndex).
    // bytes must be <= m_sizeInBytes.
    void Upload(const void* data, uint32_t bytes);

    // Return the ID3D12Resource* for the current frame slot.
    ID3D12Resource* GetResource() const;

    // Return the GPU virtual address for the current frame slot.
    D3D12_GPU_VIRTUAL_ADDRESS GetGPUVA() const;

    uint32_t SizeInBytes() const { return m_sizeInBytes; }

private:
    static constexpr uint32_t kFrames = 3;

    ComPtr<ID3D12Resource> m_buffers[kFrames];
    void*                  m_mapped[kFrames] = {};
    uint32_t               m_sizeInBytes     = 0;
};
