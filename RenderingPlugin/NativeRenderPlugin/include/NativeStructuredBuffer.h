#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <cstdint>
#include <array>
#include "IUnityLog.h"

using Microsoft::WRL::ComPtr;

/// <summary>
/// GPU-resident (DEFAULT heap) structured buffer with a triple-buffered UPLOAD
/// staging layer for CPU writes.
///
/// Write path  : UploadRange() → writes into staging[g_frameIndex]
/// Flush path  : FlushPendingCopies(cmdList) → COMMON→COPY_DEST, CopyBufferRegion,
///               COPY_DEST→COMMON; GPU buffer is then readable as SRV.
/// SRV access  : uses implicit state promotion (COMMON → NON_PIXEL_SHADER_RESOURCE).
/// </summary>
class NativeStructuredBuffer
{
public:
    static constexpr uint32_t kFrames = 3;

    /// <summary>Resources from a superseded allocation; caller must defer-delete them.</summary>
    struct GrowResult
    {
        ComPtr<ID3D12Resource> oldBuffer;
        ComPtr<ID3D12Resource> oldStaging[kFrames];
    };

    bool Initialize(ID3D12Device* device, uint32_t capacity, uint32_t elementStride, IUnityLog* log = nullptr);
    ~NativeStructuredBuffer();

    /// <summary>Writes elements into the current frame's staging buffer.</summary>
    void UploadRange(const void* data, uint32_t elementOffset, uint32_t elementCount);

    /// <summary>
    /// Issues COMMON→COPY_DEST barrier, CopyBufferRegion for the frame's dirty range,
    /// then COPY_DEST→COMMON barrier. Must be called with an active D3D12 command list
    /// before the buffer is used as an SRV.
    /// </summary>
    void FlushPendingCopies(ID3D12GraphicsCommandList* cmdList);

    /// <summary>
    /// Grows to at least <paramref name="newCapacity"/> elements.
    /// Old resources are returned in <paramref name="out"/> for deferred deletion by the caller.
    /// Returns false on allocation failure (old resources unchanged in that case).
    /// </summary>
    bool Grow(uint32_t newCapacity, GrowResult& out);

    ID3D12Resource* GetResource() const { return m_buffer.Get(); }
    uint32_t        GetCapacity() const { return m_capacity; }
    uint32_t        GetStride()   const { return m_stride; }

private:
    ID3D12Device*          m_device   = nullptr;

    // Main GPU-resident buffer (DEFAULT heap, COMMON state between flushes)
    ComPtr<ID3D12Resource> m_buffer;

    // Triple-buffered staging (UPLOAD heap, persistently mapped)
    ComPtr<ID3D12Resource> m_staging[kFrames];
    void*                  m_mappedStaging[kFrames] = {};

    // Per-frame dirty region (element indices, half-open [min, max))
    uint32_t m_dirtyMin[kFrames]  = {};
    uint32_t m_dirtyMax[kFrames]  = {};
    bool     m_hasPending[kFrames] = {};

    uint32_t m_capacity = 0;
    uint32_t m_stride   = 0;

    IUnityLog* m_log = nullptr;

    void Log(const char* msg) const;

    bool AllocBuffer (uint32_t capacity);
    bool AllocStaging(uint32_t capacity);
    void FreeStaging (GrowResult* outOld = nullptr); // moves old resources into outOld if non-null
};

