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
    uint64_t bufferHandle; // NativeBuffer*
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
    uint64_t bufferHandle; // NativeBuffer*
    uint64_t srcByteOffset;
    uint64_t bytes;
};

// ---------------------------------------------------------------------------
// NativeBufferDesc
//   nvrhi BufferDesc analog. One description covers the three buffer roles the
//   plugin previously had as separate classes; the flags below select the heap
//   and the active upload/readback code path:
//
//     * Volatile constant/dynamic buffer  (the old NativeBuffer)
//         isVolatile = true, maxVersions = N (>1)
//         => N persistently-mapped UPLOAD-heap slots, one written per frame
//            (g_frameIndex). GetResource() resolves the current frame's slot.
//            CPU writes via Upload(); no copy command, no staging.
//
//     * GPU-resident DEFAULT buffer with staged CPU writes  (the old
//       NativeStructuredBuffer)
//         isVolatile = false, structStride > 0
//         => single DEFAULT-heap resource fed by UploadSnapshot() (staged through
//            the shared UPLOAD chunk pool) and optionally read back via
//            RequestReadback()/TryReadback().
//
//     * GPU-only UAV buffer  (the old NativeGpuBuffer)
//         isVolatile = false, canHaveUAVs = true, no CPU upload path used
//         => single DEFAULT-heap resource with ALLOW_UNORDERED_ACCESS; written
//            and read by compute shaders, optionally GPU-cleared.
//
//   canHaveUAVs may be combined with the staged-write path for mixed
//   CPU-upload + GPU-UAV buffers (e.g. RTXPT's lightsBuffer).
// ---------------------------------------------------------------------------
struct NativeBufferDesc
{
    uint64_t byteSize         = 0;     // total size in bytes
    uint32_t structStride     = 0;     // element stride; 0 = raw/typed buffer
    uint32_t maxVersions      = 1;     // >1 => per-frame UPLOAD ring (requires isVolatile)
    bool     canHaveUAVs      = false; // ALLOW_UNORDERED_ACCESS on the DEFAULT resource
    bool     isConstantBuffer = false; // round byteSize up to the 256-byte CBV alignment
    bool     isVolatile       = false; // UPLOAD heap, persistently mapped, per-frame versions
};

/// <summary>
/// Unified plugin buffer (nvrhi Buffer analog). The allocation is immutable in
/// size: to change capacity the owner disposes this instance and constructs a
/// new one (deferred-deleted via the plugin's fence-gated cleanup queue).
///
/// Behaviour is selected by <see cref="NativeBufferDesc"/> — see its comment for
/// the three roles (volatile upload ring / DEFAULT staged-copy / DEFAULT UAV).
/// </summary>
class NativeBuffer : public INativeResource
{
public:
    NativeBuffer() = default;
    ~NativeBuffer() override;

    bool Initialize(ID3D12Device* device, const NativeBufferDesc& desc, IUnityLog* log = nullptr);

    // --- Volatile / UPLOAD-ring path (the old NativeBuffer) ---------------------
    // Copy |bytes| into the current frame's mapped slot (g_frameIndex). No-op for
    // non-volatile buffers.
    void Upload(const void* data, uint32_t bytes);

    // --- DEFAULT staged-copy path (the old NativeStructuredBuffer) --------------

    /// <summary>
    /// Render-thread: stages the packed payload through one suballocation from the shared
    /// UPLOAD chunk pool and records a CopyBufferRegion per range. State transitions go
    /// through Unity's resource-state tracker (|tracker|) so the COPY_DEST transition
    /// composes with the SRV/UAV barriers of surrounding passes; when the tracker is
    /// unavailable it falls back to manual COMMON->COPY_DEST->COMMON barriers.
    /// Each range copies only its own slice; untouched regions keep their prior contents.
    /// </summary>
    void UploadSnapshot(ID3D12GraphicsCommandList* cmdList,
                        const ResourceStateTracker& tracker,
                        const NsbFlushRange*       ranges,
                        uint32_t                   rangeCount,
                        const uint8_t*             payload);

    /// <summary>
    /// Render-thread: records a copy of [srcByteOffset, srcByteOffset+bytes) into an internal
    /// READBACK-heap staging resource (lazily (re)allocated), then stamps |fenceTarget| as the
    /// frame-fence value at which the copy completes. The source transition to COPY_SOURCE goes
    /// through Unity's tracker so it composes with other passes.
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

    // The underlying D3D12 resource. For volatile buffers this resolves to the
    // current frame's slot (g_frameIndex); otherwise the single DEFAULT resource.
    ID3D12Resource*           GetResource() const override;
    D3D12_GPU_VIRTUAL_ADDRESS GetGPUVA() const;

    const NativeBufferDesc& GetDesc()  const { return m_desc; }
    uint32_t SizeInBytes() const { return static_cast<uint32_t>(m_desc.byteSize); }
    uint32_t GetStride()   const { return m_desc.structStride; }
    uint32_t GetCapacity() const { return m_desc.structStride ? static_cast<uint32_t>(m_desc.byteSize / m_desc.structStride) : 0; }
    bool     AllowsUAV()   const { return m_desc.canHaveUAVs; }

private:
    static constexpr uint32_t kMaxFrames = 3;

    ID3D12Device*    m_device = nullptr;
    NativeBufferDesc m_desc{};
    IUnityLog*       m_log = nullptr;

    // Volatile: m_versions[0..m_versionCount) are UPLOAD-heap, persistently mapped.
    // Non-volatile: only m_versions[0] is used (DEFAULT heap) and m_mapped is null.
    ComPtr<ID3D12Resource> m_versions[kMaxFrames];
    void*                  m_mapped[kMaxFrames] = {};
    uint32_t               m_versionCount = 1;

    // GPU->CPU readback staging (READBACK heap, permanently in COPY_DEST). Lazily sized.
    ComPtr<ID3D12Resource> m_readbackBuffer;
    uint64_t               m_readbackCapacity    = 0; // allocated bytes
    uint64_t               m_readbackBytes       = 0; // bytes copied by the pending request
    uint64_t               m_readbackFenceTarget = 0; // frame-fence value at which the copy is done
    bool                   m_readbackPending     = false;

    void Log(const char* msg) const;
    bool AllocUploadRing();
    bool AllocDefault();
    bool EnsureReadbackCapacity(uint64_t bytes);

    // Index of the slot the current frame reads/writes.
    uint32_t CurrentSlot() const;
};
