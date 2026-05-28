#include "NativeStructuredBuffer.h"
#include "PluginInternal.h"
#include <cstring>
#include <cassert>
#include <cstdio>
#include <utility>
#include <windows.h>

// ---------------------------------------------------------------------------
// NativeStructuredBuffer implementation
//
// Fixed-capacity DEFAULT-heap buffer. CPU writes are buffered on the C# side and
// arrive as a self-describing snapshot blob; UploadSnapshot stages them to the GPU
// through a transient UPLOAD buffer, mirroring nvrhi's CommandList::writeBuffer.
// ---------------------------------------------------------------------------

NativeStructuredBuffer::~NativeStructuredBuffer() = default;

bool NativeStructuredBuffer::AllocBuffer(uint32_t capacity)
{
    const uint64_t size = static_cast<uint64_t>(capacity) * m_stride;

    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width            = size;
    desc.Height           = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels        = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags            = D3D12_RESOURCE_FLAG_NONE;

    HRESULT hr = m_device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COMMON, // safe start state; promotes to SRV, transitions to COPY_DEST
        nullptr,
        IID_PPV_ARGS(&m_buffer));
    if (FAILED(hr)) return false;

    m_capacity = capacity;
    return true;
}

bool NativeStructuredBuffer::Initialize(ID3D12Device* device, uint32_t capacity, uint32_t elementStride, IUnityLog* log)
{
    m_device = device;
    m_stride = elementStride;
    m_log    = log;
    return AllocBuffer(capacity);
}

void NativeStructuredBuffer::Log(const char* msg) const
{
    return;
    if (m_log)
        m_log->Log(kUnityLogTypeLog, msg, __FILE__, __LINE__);
    else
        OutputDebugStringA(msg);
}

void NativeStructuredBuffer::UploadSnapshot(ID3D12GraphicsCommandList* cmdList,
                                            const NsbFlushRange*       ranges,
                                            uint32_t                   rangeCount,
                                            const uint8_t*             payload)
{
    assert(cmdList);
    if (!cmdList || !ranges || rangeCount == 0 || !payload) return;

    // The packed payload spans [0, totalBytes); copy it through one transient UPLOAD buffer.
    uint64_t totalBytes = 0;
    for (uint32_t i = 0; i < rangeCount; ++i)
        totalBytes += static_cast<uint64_t>(ranges[i].elementCount) * m_stride;
    if (totalBytes == 0) return;

    D3D12_HEAP_PROPERTIES uploadHeap = {};
    uploadHeap.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC uploadDesc = {};
    uploadDesc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    uploadDesc.Width            = totalBytes;
    uploadDesc.Height           = 1;
    uploadDesc.DepthOrArraySize = 1;
    uploadDesc.MipLevels        = 1;
    uploadDesc.SampleDesc.Count = 1;
    uploadDesc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    uploadDesc.Flags            = D3D12_RESOURCE_FLAG_NONE;

    ComPtr<ID3D12Resource> upload;
    HRESULT hr = m_device->CreateCommittedResource(
        &uploadHeap,
        D3D12_HEAP_FLAG_NONE,
        &uploadDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&upload));
    if (FAILED(hr))
    {
        Log("[NativeStructuredBuffer::UploadSnapshot] transient upload allocation failed");
        return;
    }

    void* mapped = nullptr;
    const D3D12_RANGE readRange = { 0, 0 };
    hr = upload->Map(0, &readRange, &mapped);
    if (FAILED(hr)) return;

    // COMMON -> COPY_DEST. The buffer starts in COMMON and decays back to COMMON at each
    // ExecuteCommandLists boundary, so this transition is always valid here.
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource   = m_buffer.Get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
    cmdList->ResourceBarrier(1, &barrier);

    for (uint32_t i = 0; i < rangeCount; ++i)
    {
        const NsbFlushRange& r = ranges[i];
        const uint64_t bytes      = static_cast<uint64_t>(r.elementCount) * m_stride;
        const uint64_t destOffset = static_cast<uint64_t>(r.elementOffset) * m_stride;

        memcpy(reinterpret_cast<uint8_t*>(mapped) + r.payloadByteOffset,
               payload + r.payloadByteOffset,
               static_cast<size_t>(bytes));

        // Each range copies only its own slice; untouched regions of the DEFAULT
        // buffer keep their previously-uploaded contents.
        cmdList->CopyBufferRegion(m_buffer.Get(), destOffset, upload.Get(), r.payloadByteOffset, bytes);
    }

    upload->Unmap(0, nullptr);

    // COPY_DEST -> COMMON. Returning to COMMON lets the resource be implicitly promoted
    // to NON_PIXEL_SHADER_RESOURCE when shaders read it later in the same command list.
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_COMMON;
    cmdList->ResourceBarrier(1, &barrier);

    // Keep the transient upload buffer alive until the GPU has consumed the copy.
    EnqueueCleanup([res = std::move(upload)] {});
}
