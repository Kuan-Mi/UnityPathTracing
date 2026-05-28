#include "NativeStructuredBuffer.h"
#include "PluginInternal.h"
#include <cstring>
#include <cassert>
#include <cstdio>
#include <windows.h>

// ---------------------------------------------------------------------------
// NativeStructuredBuffer implementation
//
// Fixed-capacity DEFAULT-heap buffer. CPU writes are buffered on the C# side and
// arrive as a self-describing snapshot blob; UploadSnapshot stages them to the GPU
// through the shared UPLOAD chunk pool (g_uploadPool), mirroring nvrhi's
// CommandList::writeBuffer + UploadManager.
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
    desc.Flags            = m_allowUAV ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                                       : D3D12_RESOURCE_FLAG_NONE;

    HRESULT hr = m_device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COMMON, // safe start state; promotes to SRV/UAV, transitions to COPY_DEST
        nullptr,
        IID_PPV_ARGS(&m_buffer));
    if (FAILED(hr)) return false;

    // Name the resource so PIX captures and D3D12 debug-layer messages are readable.
    wchar_t name[80];
    swprintf_s(name, L"NativeStructuredBuffer(n=%u,stride=%u%s)", capacity, m_stride,
               m_allowUAV ? L",uav" : L"");
    m_buffer->SetName(name);

    m_capacity = capacity;
    return true;
}

bool NativeStructuredBuffer::Initialize(ID3D12Device* device, uint32_t capacity, uint32_t elementStride,
                                        IUnityLog* log, bool allowUAV)
{
    m_device   = device;
    m_stride   = elementStride;
    m_log      = log;
    m_allowUAV = allowUAV;
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
                                            const ResourceStateTracker& tracker,
                                            const NsbFlushRange*       ranges,
                                            uint32_t                   rangeCount,
                                            const uint8_t*             payload)
{
    assert(cmdList);
    if (!cmdList || !ranges || rangeCount == 0 || !payload) return;

    // The packed payload spans [0, totalBytes); stage it through one shared-pool allocation.
    uint64_t totalBytes = 0;
    for (uint32_t i = 0; i < rangeCount; ++i)
        totalBytes += static_cast<uint64_t>(ranges[i].elementCount) * m_stride;
    if (totalBytes == 0) return;

    // Suballocate from the shared UPLOAD chunk pool (recycled by frame fence) instead of
    // creating a committed resource every flush.
    SharedUploadPool::Allocation alloc = g_uploadPool.Allocate(totalBytes, 256);
    if (!alloc.IsValid())
    {
        Log("[NativeStructuredBuffer::UploadSnapshot] shared upload allocation failed");
        return;
    }

    // The payload is contiguous; copy it once into the suballocation.
    memcpy(alloc.cpu, payload, static_cast<size_t>(totalBytes));

    // Transition into COPY_DEST. When Unity's tracker is available we go through it so
    // the transition starts from whatever state surrounding passes left the buffer in
    // (e.g. NON_PIXEL_SHADER_RESOURCE) and composes correctly. Otherwise we fall back to
    // a manual barrier, relying on the COMMON decay at the ExecuteCommandLists boundary.
    const bool tracked = tracker.Valid();
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource   = m_buffer.Get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    if (tracked)
    {
        tracker.Require(m_buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST);
    }
    else
    {
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
        cmdList->ResourceBarrier(1, &barrier);
    }

    for (uint32_t i = 0; i < rangeCount; ++i)
    {
        const NsbFlushRange& r = ranges[i];
        const uint64_t bytes      = static_cast<uint64_t>(r.elementCount) * m_stride;
        const uint64_t destOffset = static_cast<uint64_t>(r.elementOffset) * m_stride;

        // Each range copies only its own slice; untouched regions of the DEFAULT
        // buffer keep their previously-uploaded contents.
        cmdList->CopyBufferRegion(m_buffer.Get(), destOffset,
                                  alloc.resource, alloc.offset + r.payloadByteOffset, bytes);
    }

    if (tracked)
    {
        // Record the post-copy state; the next consumer's Require transitions out of it.
        tracker.Notify(m_buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST);
    }
    else
    {
        // COPY_DEST -> COMMON. Returning to COMMON lets the resource be implicitly promoted
        // to NON_PIXEL_SHADER_RESOURCE when shaders read it later in the same command list.
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_COMMON;
        cmdList->ResourceBarrier(1, &barrier);
    }
}
