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

// ---------------------------------------------------------------------------
// Readback request blob (shared contract with the C# NativeStructuredBuffer).
// Passed as the data pointer of IssuePluginEventAndData for a GPU->CPU readback;
// the render-thread callback records the copy into the buffer's READBACK staging
// resource and frees the blob. The CPU later polls TryReadback once the frame
// fence has passed the recorded target value.
// ---------------------------------------------------------------------------
struct NsbReadbackRequest
{
    uint64_t bufferHandle; // NativeStructuredBuffer*
    uint64_t srcByteOffset;
    uint64_t bytes;
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

    /// <summary>
    /// Render-thread: records a copy of [srcByteOffset, srcByteOffset+bytes) from the DEFAULT
    /// buffer into an internal READBACK-heap staging resource (lazily (re)allocated), then
    /// stamps |fenceTarget| as the frame-fence value at which the copy completes. The source
    /// transition to COPY_SOURCE goes through Unity's tracker so it composes with other passes.
    /// </summary>
    void RequestReadback(ID3D12GraphicsCommandList* cmdList,
                         const ResourceStateTracker& tracker,
                         uint64_t srcByteOffset,
                         uint64_t bytes,
                         uint64_t fenceTarget);

    /// <summary>
    /// Main-thread poll: if a readback is pending and the GPU has passed the recorded fence
    /// value (|completedFenceValue|), maps the staging resource, copies up to |dstBytes| into
    /// |dst|, clears the pending flag, and returns true. Returns false while still in flight or
    /// when no request is pending.
    /// </summary>
    bool TryReadback(uint64_t completedFenceValue, void* dst, uint64_t dstBytes);

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

    // GPU->CPU readback staging (READBACK heap, permanently in COPY_DEST). Lazily sized.
    ComPtr<ID3D12Resource> m_readbackBuffer;
    uint64_t               m_readbackCapacity   = 0;     // allocated bytes
    uint64_t               m_readbackBytes      = 0;     // bytes copied by the pending request
    uint64_t               m_readbackFenceTarget = 0;    // frame-fence value at which the copy is done
    bool                   m_readbackPending    = false;

    IUnityLog* m_log = nullptr;

    void Log(const char* msg) const;
    bool AllocBuffer(uint32_t capacity);
    bool EnsureReadbackCapacity(uint64_t bytes);
};
