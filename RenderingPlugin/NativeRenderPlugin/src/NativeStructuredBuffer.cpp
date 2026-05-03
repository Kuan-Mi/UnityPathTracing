#include "NativeStructuredBuffer.h"
#include "PluginInternal.h"
#include <cstring>
#include <cassert>
#include <algorithm>
#include <cstdio>
#include <windows.h>

#undef min
#undef max

// ---------------------------------------------------------------------------
// NativeStructuredBuffer implementation
// ---------------------------------------------------------------------------

NativeStructuredBuffer::~NativeStructuredBuffer()
{
    FreeStaging();
}

void NativeStructuredBuffer::FreeStaging(GrowResult* outOld)
{
    for (uint32_t i = 0; i < kFrames; ++i)
    {
        if (m_mappedStaging[i] && m_staging[i])
        {
            m_staging[i]->Unmap(0, nullptr);
            m_mappedStaging[i] = nullptr;
        }
        if (outOld)
            outOld->oldStaging[i] = std::move(m_staging[i]);
        else
            m_staging[i] = nullptr;
    }
}

bool NativeStructuredBuffer::AllocStaging(uint32_t capacity)
{
    const uint64_t size = static_cast<uint64_t>(capacity) * m_stride;

    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width            = size;
    desc.Height           = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels        = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags            = D3D12_RESOURCE_FLAG_NONE;

    const D3D12_RANGE readRange = { 0, 0 };

    for (uint32_t i = 0; i < kFrames; ++i)
    {
        HRESULT hr = m_device->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&m_staging[i]));
        if (FAILED(hr)) return false;

        hr = m_staging[i]->Map(0, &readRange, &m_mappedStaging[i]);
        if (FAILED(hr)) return false;
    }
    return true;
}

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

    ComPtr<ID3D12Resource> newBuf;
    HRESULT hr = m_device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COMMON, // safe start state; promotes to SRV, transitions to COPY_DEST
        nullptr,
        IID_PPV_ARGS(&newBuf));
    if (FAILED(hr)) return false;

    m_buffer   = std::move(newBuf);
    m_capacity = capacity;

    // Reset all per-frame dirty regions
    for (uint32_t i = 0; i < kFrames; ++i)
    {
        m_dirtyMin[i]    = 0;
        m_dirtyMax[i]    = 0;
        m_hasPending[i]  = false;
    }
    return true;
}

bool NativeStructuredBuffer::Initialize(ID3D12Device* device, uint32_t capacity, uint32_t elementStride, IUnityLog* log)
{
    m_device = device;
    m_stride = elementStride;
    m_log    = log;
    if (!AllocStaging(capacity)) return false;
    if (!AllocBuffer(capacity))  return false;
    return true;
}

void NativeStructuredBuffer::Log(const char* msg) const
{
    return;
    if (m_log)
        m_log->Log(kUnityLogTypeLog, msg, __FILE__, __LINE__);
    else
        OutputDebugStringA(msg);
}

void NativeStructuredBuffer::UploadRange(const void* data, uint32_t elementOffset, uint32_t elementCount)
{
    assert(data);
    assert(elementOffset + elementCount <= m_capacity);
    if (!data) return;

    const uint32_t fi = g_frameIndex;
    assert(m_mappedStaging[fi]);
    if (!m_mappedStaging[fi]) return;

    // On the first partial write of this frame, copy the previous frame's staging
    // as a baseline so that gap regions between disjoint UploadRange calls are
    // never left as zeros when FlushPendingCopies copies the merged dirty span.
    if (!m_hasPending[fi])
    {
        const uint32_t prevFi = (fi + kFrames - 1) % kFrames;
        if (m_mappedStaging[prevFi])
        {
            memcpy(m_mappedStaging[fi], m_mappedStaging[prevFi],
                   static_cast<size_t>(m_capacity) * m_stride);
        }
    }

    uint8_t* dst = reinterpret_cast<uint8_t*>(m_mappedStaging[fi])
                 + static_cast<uint64_t>(elementOffset) * m_stride;
    memcpy(dst, data, static_cast<size_t>(elementCount) * m_stride);

    // Expand dirty region for this frame
    const uint32_t rangeMax = elementOffset + elementCount;
    if (!m_hasPending[fi])
    {
        m_dirtyMin[fi]   = elementOffset;
        m_dirtyMax[fi]   = rangeMax;
        m_hasPending[fi] = true;
    }
    else
    {
        m_dirtyMin[fi] = std::min(m_dirtyMin[fi], elementOffset);
        m_dirtyMax[fi] = std::max(m_dirtyMax[fi], rangeMax);
    }

    // {
    //     char _buf[256];
    //     snprintf(_buf, sizeof(_buf),
    //         "[NativeStructuredBuffer::UploadRange] frame=%u  offset=%u  count=%u  bytes=%zu  dirtyRange=[%u,%u)",
    //         fi, elementOffset, elementCount,
    //         static_cast<size_t>(elementCount) * m_stride,
    //         m_dirtyMin[fi], m_dirtyMax[fi]);
    //     Log(_buf);
    // }
}

void NativeStructuredBuffer::FlushPendingCopies(ID3D12GraphicsCommandList* cmdList)
{
    assert(cmdList);
    if (!cmdList) return;

    const uint32_t fi = g_frameIndex;
    if (!m_hasPending[fi])
    {
        // char _buf[128];
        // snprintf(_buf, sizeof(_buf),
        //     "[NativeStructuredBuffer::FlushPendingCopies] frame=%u  no pending upload, skipped", fi);
        // Log(_buf);
        return;
    }

    const uint64_t byteOffset = static_cast<uint64_t>(m_dirtyMin[fi]) * m_stride;
    const uint64_t byteCount  = static_cast<uint64_t>(m_dirtyMax[fi] - m_dirtyMin[fi]) * m_stride;

    // {
    //     char _buf[256];
    //     snprintf(_buf, sizeof(_buf),
    //         "[NativeStructuredBuffer::FlushPendingCopies] frame=%u  dirtyElems=[%u,%u)  byteOffset=%llu  byteCount=%llu",
    //         fi, m_dirtyMin[fi], m_dirtyMax[fi],
    //         static_cast<unsigned long long>(byteOffset),
    //         static_cast<unsigned long long>(byteCount));
    //     Log(_buf);
    // }

    // COMMON → COPY_DEST
    // The buffer starts in COMMON state and decays back to COMMON at each
    // ExecuteCommandLists boundary, so this transition is always valid here.
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource   = m_buffer.Get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
    cmdList->ResourceBarrier(1, &barrier);

    // Copy from the current frame's staging slice to the GPU buffer
    cmdList->CopyBufferRegion(
        m_buffer.Get(),    byteOffset,
        m_staging[fi].Get(), byteOffset,
        byteCount);

    // COPY_DEST → COMMON
    // Returning to COMMON lets the resource be implicitly promoted to
    // NON_PIXEL_SHADER_RESOURCE when shaders read it later in the same command list,
    // and guarantees a clean known state at the next ExecuteCommandLists boundary.
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_COMMON;
    cmdList->ResourceBarrier(1, &barrier);

    m_hasPending[fi] = false;
}

bool NativeStructuredBuffer::Grow(uint32_t newCapacity, GrowResult& out)
{
    if (newCapacity <= m_capacity) return true;

    // Move old GPU buffer to output for deferred deletion by the caller
    out.oldBuffer = std::move(m_buffer);

    // Move old staging to output; re-allocate fresh staging for the new capacity
    FreeStaging(&out);

    if (!AllocStaging(newCapacity)) return false;
    if (!AllocBuffer(newCapacity))  return false;
    return true;
}

