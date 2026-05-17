#include "NativeGpuBuffer.h"

// ---------------------------------------------------------------------------
// NativeGpuBuffer implementation
// ---------------------------------------------------------------------------

bool NativeGpuBuffer::Initialize(ID3D12Device* device, uint32_t sizeInBytes)
{
    if (!device || sizeInBytes == 0) return false;

    m_sizeInBytes = sizeInBytes;

    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width            = sizeInBytes;
    desc.Height           = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels        = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    HRESULT hr = device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&m_buffer));

    return SUCCEEDED(hr);
}

D3D12_GPU_VIRTUAL_ADDRESS NativeGpuBuffer::GetGPUVA() const
{
    return m_buffer ? m_buffer->GetGPUVirtualAddress() : 0;
}
