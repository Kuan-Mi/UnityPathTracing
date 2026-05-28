#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <cstdint>
#include "IUnityLog.h"
#include "INativeResource.h"
#include "ResourceStateTracker.h"

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// Flush snapshot blob layout (shared contract with the C# NativeStructuredBuffer).
//
//   [ NsbFlushHeader ]                       (16 bytes)
//   [ NsbFlushRange  ] x header.rangeCount   (12 bytes each)
//   [ packed payload bytes ]                 (each range's data at payloadByteOffset)
//
// The blob is allocated by NR_NSB_AllocFlushBuffer, filled on the C# main thread,
// passed as the data pointer of IssuePluginEventAndData, then parsed and freed by
// the render-thread flush callback. payloadByteOffset is relative to the start of
// the packed payload region.
// ---------------------------------------------------------------------------
struct NsbFlushHeader
{
    uint64_t bufferHandle; // NativeStructuredBuffer*
    uint32_t stride;
    uint32_t rangeCount;
};

struct NsbFlushRange
{
    uint32_t elementOffset;
    uint32_t elementCount;
    uint32_t payloadByteOffset;
};

/// <summary>
/// GPU-resident (DEFAULT heap) structured buffer of fixed capacity.
///
/// The allocation is immutable in size: to change capacity the owner disposes
/// this instance and constructs a new one (deferred-deleted via the plugin's
/// fence-gated cleanup queue), mirroring nvrhi's immutable Buffer model.
///
/// CPU writes are buffered entirely on the C# side; this class never holds a
/// pending-upload queue. The render thread receives a self-describing snapshot
/// blob and stages it through one transient UPLOAD buffer (UploadSnapshot),
/// recording the CopyBufferRegion(s) into the DEFAULT buffer, then defer-releases
/// the transient buffer. Mirrors nvrhi's CommandList::writeBuffer.
/// </summary>
class NativeStructuredBuffer : public INativeResource
{
public:
    // allowUAV: create the DEFAULT-heap resource with ALLOW_UNORDERED_ACCESS so it can
    // be bound as a RWStructuredBuffer/RWBuffer (GPU writes) in addition to the CPU
    // upload path — the nvrhi canHaveUAVs analog. Mixed CPU-upload + GPU-UAV buffers
    // (e.g. RTXPT's lightsBuffer) require this.
    bool Initialize(ID3D12Device* device, uint32_t capacity, uint32_t elementStride,
                    IUnityLog* log = nullptr, bool allowUAV = false);
    ~NativeStructuredBuffer() override;

    /// <summary>
    /// Render-thread: stages the packed payload through one suballocation from the shared
    /// UPLOAD chunk pool and records a CopyBufferRegion per range. State transitions go
    /// through Unity's resource-state tracker (|tracker|) so the COPY_DEST transition
    /// composes with the SRV/UAV barriers of surrounding passes; when the tracker is
    /// unavailable it falls back to manual COMMON->COPY_DEST->COMMON barriers.
    /// Each range copies only its own slice; untouched regions of the DEFAULT buffer keep
    /// their previously-uploaded contents. The upload memory is recycled by the frame fence.
    /// </summary>
    void UploadSnapshot(ID3D12GraphicsCommandList* cmdList,
                        const ResourceStateTracker& tracker,
                        const NsbFlushRange*       ranges,
                        uint32_t                   rangeCount,
                        const uint8_t*             payload);

    ID3D12Resource* GetResource() const override { return m_buffer.Get(); }
    uint32_t        GetCapacity() const { return m_capacity; }
    uint32_t        GetStride()   const { return m_stride; }
    bool            AllowsUAV()   const { return m_allowUAV; }

private:
    ID3D12Device*          m_device   = nullptr;

    // Main GPU-resident buffer (DEFAULT heap, COMMON state between flushes).
    ComPtr<ID3D12Resource> m_buffer;

    uint32_t m_capacity = 0;
    uint32_t m_stride   = 0;
    bool     m_allowUAV = false;

    IUnityLog* m_log = nullptr;

    void Log(const char* msg) const;
    bool AllocBuffer(uint32_t capacity);
};
