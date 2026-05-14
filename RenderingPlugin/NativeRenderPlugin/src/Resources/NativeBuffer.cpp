#include "NativeBuffer.h"
#include "PluginInternal.h"
#include <cstring>
#include <cassert>

// ---------------------------------------------------------------------------
// NativeBuffer implementation
// ---------------------------------------------------------------------------

NativeBuffer::~NativeBuffer()
{
    for (uint32_t i = 0; i < kFrames; ++i)
    {
        if (m_mapped[i] && m_buffers[i])
        {
            m_buffers[i]->Unmap(0, nullptr);
            m_mapped[i] = nullptr;
        }
    }
}

bool NativeBuffer::Initialize(ID3D12Device* device, uint32_t sizeInBytes)
{
    // Round up to CBV alignment requirement
    constexpr uint32_t kAlign = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT; // 256
    sizeInBytes = (sizeInBytes + kAlign - 1) & ~(kAlign - 1);
    m_sizeInBytes = sizeInBytes;

    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width            = sizeInBytes;
    desc.Height           = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels        = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags            = D3D12_RESOURCE_FLAG_NONE;

    D3D12_RANGE readRange = { 0, 0 };

    for (uint32_t i = 0; i < kFrames; ++i)
    {
        HRESULT hr = device->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&m_buffers[i]));

        if (FAILED(hr)) return false;

        hr = m_buffers[i]->Map(0, &readRange, &m_mapped[i]);
        if (FAILED(hr)) return false;
    }

    return true;
}

void NativeBuffer::Upload(const void* data, uint32_t bytes)
{
    assert(bytes <= m_sizeInBytes);
    if (!data || !m_mapped[g_frameIndex]) return;
    memcpy(m_mapped[g_frameIndex], data, bytes);
}

ID3D12Resource* NativeBuffer::GetResource() const
{
    return m_buffers[g_frameIndex].Get();
}

D3D12_GPU_VIRTUAL_ADDRESS NativeBuffer::GetGPUVA() const
{
    ID3D12Resource* r = m_buffers[g_frameIndex].Get();
    return r ? r->GetGPUVirtualAddress() : 0;
}
