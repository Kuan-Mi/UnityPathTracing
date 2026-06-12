#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <cstdint>
#include <mutex>
#include <string>
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
// the render-thread callback records the copy into one slot of the buffer's
// READBACK staging ring and frees the blob. The CPU later polls TryReadback,
// which returns the newest request whose frame fence has passed.
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
//         isVolatile = true
//         => no persistent resource. Each Upload() suballocates a fresh slice
//            from the shared UPLOAD chunk pool (g_uploadPool), memcpys the data,
//            and records that slice's GPU VA. The buffer is bound as a root CBV
//            by that VA (GetGpuVirtualAddress); GetResource() returns nullptr.
//            This mirrors nvrhi's d3d12 volatile-constant-buffer model
//            (CommandList::writeBuffer + m_VolatileConstantBufferAddresses):
//            recycling is frame-fence-gated, not a fixed mod-N slot ring, so the
//            buffer must be written every frame before it is bound.
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
    uint32_t maxVersions      = 1;     // nvrhi-parity field; unused on D3D12 (pool handles versioning)
    bool     canHaveUAVs      = false; // ALLOW_UNORDERED_ACCESS on the DEFAULT resource
    bool     isConstantBuffer = false; // round byteSize up to the 256-byte CBV alignment
    bool     isVolatile       = false; // dynamic CB: per-write upload-pool suballocation, bound by VA
    std::string debugName;             // optional PIX/D3D12 debug name for the DEFAULT resource (empty = auto)
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

    // --- Volatile path (nvrhi-style dynamic constant buffer) --------------------
    // Suballocate a fresh slice from the shared UPLOAD pool, copy |bytes| into it,
    // and record its GPU VA for the next bind. Called on the render thread, before
    // the dispatch that reads it (event ordering guarantees within-frame currency).
    // No-op for non-volatile buffers.
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
    /// Render-thread: records a copy of [srcByteOffset, srcByteOffset+bytes) into the next slot
    /// of an internal READBACK-heap staging ring (lazily (re)allocated), then stamps
    /// |fenceTarget| as the frame-fence value at which that slot's copy completes. The ring
    /// (RTXPT's cReadbackLag idiom) lets a new request be issued every frame while earlier
    /// ones are still in flight — TryReadback harvests the newest completed one. The source
    /// transition to COPY_SOURCE goes through Unity's tracker so it composes with other passes.
    /// </summary>
    void RequestReadback(ID3D12GraphicsCommandList* cmdList,
                         const ResourceStateTracker& tracker,
                         uint64_t srcByteOffset,
                         uint64_t bytes,
                         uint64_t fenceTarget);

    /// <summary>
    /// Main-thread poll: if any unconsumed request's copy has completed on the GPU
    /// (|completedFenceValue| has passed its fence target), maps that slot, copies up to
    /// |dstBytes| into |dst|, retires it and everything older, and returns true. When several
    /// requests have completed, the newest wins. Returns false while all requests are still in
    /// flight or none is pending.
    /// </summary>
    bool TryReadback(uint64_t completedFenceValue, void* dst, uint64_t dstBytes);

    // The single DEFAULT-heap resource for non-volatile buffers; nullptr for
    // volatile buffers (they have no persistent resource — bound by VA).
    ID3D12Resource*           GetResource() const override;
    // Volatile: the current suballocation's offset-correct VA. Otherwise the
    // resource base VA.
    D3D12_GPU_VIRTUAL_ADDRESS GetGpuVirtualAddress() const override;

    const NativeBufferDesc& GetDesc()  const { return m_desc; }
    uint32_t SizeInBytes() const { return static_cast<uint32_t>(m_desc.byteSize); }
    uint32_t GetStride()   const { return m_desc.structStride; }
    uint32_t GetCapacity() const { return m_desc.structStride ? static_cast<uint32_t>(m_desc.byteSize / m_desc.structStride) : 0; }
    bool     AllowsUAV()   const { return m_desc.canHaveUAVs; }

private:
    ID3D12Device*    m_device = nullptr;
    NativeBufferDesc m_desc{};
    IUnityLog*       m_log = nullptr;

    // Non-volatile (DEFAULT heap): the single GPU-resident resource.
    ComPtr<ID3D12Resource> m_buffer;

    // Volatile: latest upload-pool suballocation recorded by Upload(). The chunk
    // resource is owned by g_uploadPool (not by this buffer); we only cache its VA.
    D3D12_GPU_VIRTUAL_ADDRESS m_volatileGpuVA = 0;

    // GPU->CPU readback staging: a ring of kReadbackLag slots within one READBACK-heap
    // resource (lazily sized, permanently in COPY_DEST). Request i lands in slot
    // (i % kReadbackLag), so up to kReadbackLag requests can be in flight before a slot is
    // reused — enough to issue one per frame without ever superseding an unharvested copy
    // (mirrors RTXPT ToneMappingPasses.cpp's cReadbackLag staging ring). m_readbackMutex
    // guards all bookkeeping: requests are recorded on the render thread, polls run on the
    // main thread.
    static constexpr uint32_t kReadbackLag = 3; // > Unity's max frames in flight (2)

    struct ReadbackSlot
    {
        uint64_t fenceTarget = 0; // frame-fence value at which this slot's copy is done
        uint64_t bytes       = 0; // bytes copied into this slot
    };

    ComPtr<ID3D12Resource> m_readbackBuffer;
    uint64_t               m_readbackSlotCapacity = 0; // per-slot bytes (resource = kReadbackLag x this)
    ReadbackSlot           m_readbackSlots[kReadbackLag];
    uint64_t               m_readbackIssued   = 0; // total requests recorded (render thread)
    uint64_t               m_readbackConsumed = 0; // requests retired by TryReadback (main thread)
    std::mutex             m_readbackMutex;

    void Log(const char* msg) const;
    bool AllocDefault();
    bool EnsureReadbackCapacity(uint64_t bytes);
};
