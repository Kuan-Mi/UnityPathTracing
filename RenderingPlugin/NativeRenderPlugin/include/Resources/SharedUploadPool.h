#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// SharedUploadPool
//   A GPU-completion-fenced pool of UPLOAD-heap chunks shared by all buffers
//   that need to stage CPU data into DEFAULT-heap resources. Modelled on nvrhi's
//   d3d12 UploadManager (a recyclable chunk pool), not a fixed-capacity ring:
//
//     - Allocate() bump-allocates from a "current" chunk; when a request doesn't
//       fit, the current chunk is retired and a fence-free pooled chunk (or a new
//       one of max(size, defaultChunkSize)) becomes current. Large requests get
//       their own right-sized chunk — there is no capacity to tune and no overflow
//       fallback path.
//     - A chunk is reusable only once the frame fence that last used it has
//       completed, exactly like TransientDescriptorRing and the deferred-delete
//       queue. We substitute the frame fence for nvrhi's per-submission instance.
//
//   Threading: Allocate() runs on the render thread during flush callbacks and
//   EndFrame() runs on the render thread at the frame-tick callback, so the two
//   are already serialised; the mutex is defensive only.
// ---------------------------------------------------------------------------
class SharedUploadPool
{
public:
    struct Allocation
    {
        ID3D12Resource* resource = nullptr; // CopyBufferRegion source
        uint64_t        offset   = 0;       // byte offset within resource
        uint8_t*        cpu      = nullptr; // mapped write pointer (resource base + offset)

        bool IsValid() const { return resource != nullptr; }
    };

    bool Initialize(ID3D12Device* device, uint64_t defaultChunkSize = (4ull << 20));
    void Shutdown();
    bool IsInitialized() const { return m_device != nullptr; }

    // Bump-allocate |size| bytes aligned to |alignment|. Returns an invalid
    // Allocation on failure (device missing / allocation failed).
    Allocation Allocate(uint64_t size, uint64_t alignment = 16);

    // Once per frame on the render thread: reclaim chunks whose stamped frame
    // fence has completed (<= completedFence), then stamp the chunks used this
    // frame with frameCompleteFence (the value the fence reaches once this frame's
    // GPU work finishes), so they cannot be reused until then.
    void EndFrame(uint64_t frameCompleteFence, uint64_t completedFence);

private:
    // Sentinel fence for a chunk that is in flight this frame but whose
    // frame-complete fence value is not yet known (assigned at EndFrame).
    static constexpr uint64_t kInFlightUnknown = UINT64_MAX;

    struct Chunk
    {
        ComPtr<ID3D12Resource> buffer;
        uint8_t* cpu          = nullptr;
        uint64_t size         = 0;
        uint64_t writePointer = 0;
        uint64_t fence        = 0; // 0 = free; kInFlightUnknown = used this frame; else frame-complete fence
    };

    Chunk* CreateChunk(uint64_t size);

    ID3D12Device* m_device           = nullptr;
    uint64_t      m_defaultChunkSize = 0;

    std::vector<std::unique_ptr<Chunk>> m_chunks;
    Chunk*                              m_current = nullptr;
    std::vector<Chunk*>                 m_usedThisFrame; // retired (full) chunks awaiting stamp
    std::mutex                          m_mutex;
};
