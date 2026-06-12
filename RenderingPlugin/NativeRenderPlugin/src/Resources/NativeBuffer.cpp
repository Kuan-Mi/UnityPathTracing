#include "NativeBuffer.h"
#include "PluginInternal.h"
#include <cstring>
#include <cassert>
#include <cstdio>
#include <windows.h>

// ---------------------------------------------------------------------------
// NativeBuffer implementation
//
// One class covering the three former buffer wrappers, selected by
// NativeBufferDesc (nvrhi BufferDesc analog):
//   * isVolatile        -> no persistent resource; each Upload() suballocates a
//                          fresh slice from the shared UPLOAD pool (g_uploadPool)
//                          and records its VA, bound as a root CBV by that VA.
//                          Mirrors nvrhi CommandList::writeBuffer for volatile CBs.
//   * DEFAULT + stride  -> GPU-resident, CPU writes staged via UploadSnapshot()
//                          through g_uploadPool, mirroring nvrhi's writeBuffer
//                          for non-volatile buffers.
//   * canHaveUAVs       -> ALLOW_UNORDERED_ACCESS on the DEFAULT resource.
// GPU->CPU readback (RequestReadback/TryReadback) is available on DEFAULT buffers.
// ---------------------------------------------------------------------------

NativeBuffer::~NativeBuffer() = default;

void NativeBuffer::Log(const char* msg) const
{
    return;
    if (m_log)
        m_log->Log(kUnityLogTypeLog, msg, __FILE__, __LINE__);
    else
        OutputDebugStringA(msg);
}

// ---------------------------------------------------------------------------
// DEFAULT-heap single resource (the old NativeStructuredBuffer / NativeGpuBuffer).
// ---------------------------------------------------------------------------
bool NativeBuffer::AllocDefault()
{
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width            = m_desc.byteSize;
    desc.Height           = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels        = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags            = m_desc.canHaveUAVs ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                                               : D3D12_RESOURCE_FLAG_NONE;

    HRESULT hr = m_device->CreateCommittedResource(
        &heapProps, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_COMMON, // safe start state; promotes to SRV/UAV, transitions to COPY_DEST
        nullptr, IID_PPV_ARGS(&m_buffer));
    if (FAILED(hr)) return false;

    // Name the resource so PIX captures and D3D12 debug-layer messages are readable.
    // When the caller supplied a debug name we use it verbatim (so the resource matches
    // the original RTXPT name in PIX); otherwise fall back to an auto-generated label.
    if (!m_desc.debugName.empty())
    {
        const int wlen = MultiByteToWideChar(CP_UTF8, 0, m_desc.debugName.c_str(), -1, nullptr, 0);
        if (wlen > 0)
        {
            std::wstring wname(static_cast<size_t>(wlen), L'\0');
            MultiByteToWideChar(CP_UTF8, 0, m_desc.debugName.c_str(), -1, wname.data(), wlen);
            m_buffer->SetName(wname.c_str());
        }
    }
    else
    {
        wchar_t name[96];
        swprintf_s(name, L"NativeBuffer(bytes=%llu,stride=%u%s)",
                   static_cast<unsigned long long>(m_desc.byteSize), m_desc.structStride,
                   m_desc.canHaveUAVs ? L",uav" : L"");
        m_buffer->SetName(name);
    }
    return true;
}

bool NativeBuffer::Initialize(ID3D12Device* device, const NativeBufferDesc& desc, IUnityLog* log)
{
    if (!device || desc.byteSize == 0) return false;

    m_device = device;
    m_desc   = desc;
    m_log    = log;

    // Round up to the CBV alignment when used as a constant buffer (or as a volatile
    // upload buffer, which the constant-buffer path requires) — matches the old NativeBuffer.
    if (m_desc.isConstantBuffer || m_desc.isVolatile)
    {
        constexpr uint64_t kAlign = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT; // 256
        m_desc.byteSize = (m_desc.byteSize + kAlign - 1) & ~(kAlign - 1);
    }

    // Volatile buffers allocate nothing up front — each Upload() pulls a fresh
    // slice from the shared upload pool (nvrhi's volatile model).
    return m_desc.isVolatile ? true : AllocDefault();
}

ID3D12Resource* NativeBuffer::GetResource() const
{
    // Volatile buffers have no persistent resource; they are bound by VA.
    return m_desc.isVolatile ? nullptr : m_buffer.Get();
}

D3D12_GPU_VIRTUAL_ADDRESS NativeBuffer::GetGpuVirtualAddress() const
{
    if (m_desc.isVolatile) return m_volatileGpuVA;
    return m_buffer ? m_buffer->GetGPUVirtualAddress() : 0;
}

void NativeBuffer::Upload(const void* data, uint32_t bytes)
{
    assert(bytes <= m_desc.byteSize);
    if (!data || bytes == 0 || !m_desc.isVolatile) return;

    // Suballocate a fresh slice from the shared UPLOAD pool, aligned for a CBV,
    // and record its offset-correct GPU VA for the next bind. The chunk lifetime
    // is frame-fence-gated by the pool, so the VA stays valid until this frame's
    // GPU work completes (the buffer must be re-uploaded each frame before use).
    SharedUploadPool::Allocation alloc =
        g_uploadPool.Allocate(bytes, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    if (!alloc.IsValid())
    {
        Log("[NativeBuffer::Upload] shared upload allocation failed");
        return;
    }

    memcpy(alloc.cpu, data, bytes);
    m_volatileGpuVA = alloc.resource->GetGPUVirtualAddress() + alloc.offset;
}

void NativeBuffer::UploadSnapshot(ID3D12GraphicsCommandList* cmdList,
                                  const ResourceStateTracker& tracker,
                                  const NsbFlushRange*       ranges,
                                  uint32_t                   rangeCount,
                                  const uint8_t*             payload)
{
    assert(cmdList);
    if (!cmdList || !ranges || rangeCount == 0 || !payload) return;

    ID3D12Resource* buffer = m_buffer.Get();
    if (!buffer) return;

    const uint32_t stride = m_desc.structStride;

    // The packed payload spans [0, totalBytes); stage it through one shared-pool allocation.
    uint64_t totalBytes = 0;
    for (uint32_t i = 0; i < rangeCount; ++i)
        totalBytes += static_cast<uint64_t>(ranges[i].elementCount) * stride;
    if (totalBytes == 0) return;

    // Suballocate from the shared UPLOAD chunk pool (recycled by frame fence) instead of
    // creating a committed resource every flush.
    SharedUploadPool::Allocation alloc = g_uploadPool.Allocate(totalBytes, 256);
    if (!alloc.IsValid())
    {
        Log("[NativeBuffer::UploadSnapshot] shared upload allocation failed");
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
    barrier.Transition.pResource   = buffer;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    if (tracked)
    {
        tracker.Require(buffer, D3D12_RESOURCE_STATE_COPY_DEST);
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
        const uint64_t bytes      = static_cast<uint64_t>(r.elementCount) * stride;
        const uint64_t destOffset = static_cast<uint64_t>(r.elementOffset) * stride;

        // Each range copies only its own slice; untouched regions of the DEFAULT
        // buffer keep their previously-uploaded contents.
        cmdList->CopyBufferRegion(buffer, destOffset,
                                  alloc.resource, alloc.offset + r.payloadByteOffset, bytes);
    }

    if (tracked)
    {
        // Record the post-copy state; the next consumer's Require transitions out of it.
        tracker.Notify(buffer, D3D12_RESOURCE_STATE_COPY_DEST);
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

// Called with m_readbackMutex held.
bool NativeBuffer::EnsureReadbackCapacity(uint64_t bytes)
{
    if (bytes == 0) return false;
    if (m_readbackBuffer && m_readbackSlotCapacity >= bytes) return true;

    // Round the per-slot capacity up so a sequence of slowly growing requests doesn't
    // reallocate every time.
    const uint64_t slotCapacity = (bytes + 255) & ~255ull;

    // (Re)allocate the READBACK-heap staging ring. READBACK resources must be
    // created in (and remain in) COPY_DEST, so no state transition is ever needed.
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_READBACK;

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width            = slotCapacity * kReadbackLag;
    desc.Height           = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels        = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags            = D3D12_RESOURCE_FLAG_NONE;

    ComPtr<ID3D12Resource> staging;
    HRESULT hr = m_device->CreateCommittedResource(
        &heapProps, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&staging));
    if (FAILED(hr))
    {
        Log("[NativeBuffer::EnsureReadbackCapacity] readback alloc failed");
        return false;
    }

    // The old staging resource may still be the destination of in-flight GPU copies —
    // release it through the fence-gated cleanup queue, and retire any unconsumed
    // requests since their data lives in the old resource.
    if (m_readbackBuffer)
        EnqueueCleanup([old = std::move(m_readbackBuffer)] {});
    m_readbackBuffer       = staging;
    m_readbackSlotCapacity = slotCapacity;
    m_readbackConsumed     = m_readbackIssued;
    return true;
}

void NativeBuffer::RequestReadback(ID3D12GraphicsCommandList* cmdList,
                                   const ResourceStateTracker& tracker,
                                   uint64_t srcByteOffset,
                                   uint64_t bytes,
                                   uint64_t fenceTarget)
{
    assert(cmdList);
    if (!cmdList || bytes == 0) return;

    ID3D12Resource* buffer = m_buffer.Get();
    if (!buffer) return;

    // Clamp to the buffer's byte size.
    const uint64_t bufBytes = m_desc.byteSize;
    if (srcByteOffset >= bufBytes) return;
    if (srcByteOffset + bytes > bufBytes) bytes = bufBytes - srcByteOffset;

    // Transition the source to COPY_SOURCE through Unity's tracker so the barrier composes
    // with whatever state the buffer was last left in (UAV/SRV/COPY_DEST). When the tracker
    // is unavailable we cannot safely barrier here, so skip rather than risk a state mismatch.
    if (!tracker.Valid())
    {
        Log("[NativeBuffer::RequestReadback] no state tracker; readback skipped");
        return;
    }

    std::lock_guard<std::mutex> lock(m_readbackMutex);
    if (!EnsureReadbackCapacity(bytes)) return;

    // Next ring slot. With kReadbackLag slots a request from <lag frames ago is never
    // overwritten before TryReadback had a chance to harvest it.
    const uint64_t slot       = m_readbackIssued % kReadbackLag;
    const uint64_t slotOffset = slot * m_readbackSlotCapacity;

    tracker.Require(buffer, D3D12_RESOURCE_STATE_COPY_SOURCE);

    cmdList->CopyBufferRegion(m_readbackBuffer.Get(), slotOffset, buffer, srcByteOffset, bytes);

    tracker.Notify(buffer, D3D12_RESOURCE_STATE_COPY_SOURCE);

    m_readbackSlots[slot].bytes       = bytes;
    m_readbackSlots[slot].fenceTarget = fenceTarget;
    ++m_readbackIssued;
}

bool NativeBuffer::TryReadback(uint64_t completedFenceValue, void* dst, uint64_t dstBytes)
{
    if (!dst) return false;

    std::lock_guard<std::mutex> lock(m_readbackMutex);
    if (!m_readbackBuffer || m_readbackIssued == m_readbackConsumed) return false;

    // Scan unconsumed requests newest-first; fence targets are monotonic, so the first
    // completed one found is the newest. Requests older than (issued - lag) have had their
    // slot reused by a newer request and are excluded.
    const uint64_t oldestValid = (m_readbackIssued > kReadbackLag) ? m_readbackIssued - kReadbackLag : 0;
    const uint64_t scanFloor   = (m_readbackConsumed > oldestValid) ? m_readbackConsumed : oldestValid;

    for (uint64_t idx = m_readbackIssued; idx > scanFloor; --idx)
    {
        const uint64_t      slot = (idx - 1) % kReadbackLag;
        const ReadbackSlot& s    = m_readbackSlots[slot];
        if (completedFenceValue < s.fenceTarget) continue; // still in flight; try an older one

        const uint64_t copyBytes  = (dstBytes < s.bytes) ? dstBytes : s.bytes;
        const uint64_t slotOffset = slot * m_readbackSlotCapacity;

        void*       mapped    = nullptr;
        D3D12_RANGE readRange = { static_cast<SIZE_T>(slotOffset),
                                  static_cast<SIZE_T>(slotOffset + copyBytes) };
        if (FAILED(m_readbackBuffer->Map(0, &readRange, &mapped)) || !mapped)
        {
            Log("[NativeBuffer::TryReadback] Map failed");
            return false;
        }
        memcpy(dst, static_cast<const uint8_t*>(mapped) + slotOffset, static_cast<size_t>(copyBytes));

        D3D12_RANGE writeRange = { 0, 0 }; // CPU did not write
        m_readbackBuffer->Unmap(0, &writeRange);

        m_readbackConsumed = idx; // retire this request and everything older
        return true;
    }
    return false; // all unconsumed requests still in flight
}
