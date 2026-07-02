#pragma once
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <vector>
#include <list>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <cstdint>
#include "IUnityLog.h"
#include "IUnityGraphicsD3D12.h"
#include "rtxmu/D3D12AccelStructManager.h"

using Microsoft::WRL::ComPtr;

// Sentinel for "no RTXMU acceleration structure" (RTXMU ids start at 1).
static constexpr uint64_t kInvalidRtxmuId = ~0ull;

// ---------------------------------------------------------------------------
// NR_SubmeshDesc
//   Plain data descriptor for one sub-mesh within an AddInstance call.
//   Must match the C# NativeRenderPlugin.SubmeshDesc struct layout exactly.
// ---------------------------------------------------------------------------
// Flags for NR_SubmeshDesc::flags
// Bit 0: geometry is fully opaque (no alpha-clip / any-hit) — maps to D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE
#define NR_SUBMESH_FLAG_GEOMETRY_OPAQUE  0x1u

struct NR_SubmeshDesc
{
    uint32_t indexCount;      // number of indices in this sub-mesh
    uint32_t indexByteOffset; // byte offset of this sub-mesh's first index in the shared IB
    uint32_t baseVertex;      // value added to each index before reading a vertex (Unity SubMeshDescriptor.baseVertex)
    uint32_t flags;           // NR_SUBMESH_FLAG_* bitmask (bit 0 = GEOMETRY_OPAQUE)
};

// ---------------------------------------------------------------------------
// NR_SubmeshOMMDesc
//   Per-submesh pre-baked Opacity Micromap data passed inline to AddInstance.
//   Set arrayData = nullptr to indicate "no OMM for this submesh".
//   Layout: all pointers first (8-byte aligned), then all u32s packed together,
//   so no explicit padding is needed on 64-bit.
// ---------------------------------------------------------------------------
struct NR_SubmeshOMMDesc
{
    const void* arrayData;        // OMM array blob; nullptr = no OMM
    const void* descArray;        // OMM descriptor array blob
    const void* indexBuffer;      // OMM index buffer blob
    const void* histogramFlat;    // uint32[histogramCount * 3]: {count, subdivLevel, format}

    uint32_t    arrayDataSize;
    uint32_t    descArrayCount;
    uint32_t    indexCount;
    uint32_t    indexStride;      // 2 or 4
    uint32_t    histogramCount;
};

// ---------------------------------------------------------------------------
// NR_AddInstanceDesc
//   All per-instance parameters for NR_AS_AddInstance, except the AS handle.
//   Must match the C# NativeRenderPlugin.AddInstanceDesc struct layout exactly.
//   Layout: all pointers first (8-byte aligned), then all u32s packed together,
//   so no explicit padding is needed on 64-bit.
// ---------------------------------------------------------------------------
struct NR_AddInstanceDesc
{
    void*                    vbPtr;         // ID3D12Resource* vertex buffer
    void*                    ibPtr;         // ID3D12Resource* index buffer
    const NR_SubmeshDesc*    submeshDescs;  // pointer to NR_SubmeshDesc array
    const NR_SubmeshOMMDesc* ommDescs;      // nullable

    uint32_t                 instanceHandle;// unique handle (e.g. MeshRenderer.GetInstanceID())
    uint32_t                 vertexCount;
    uint32_t                 vertexStride;
    uint32_t                 indexStride;
    uint32_t                 submeshCount;
    uint32_t                 isDynamic;     // 1 = SkinnedMeshRenderer (BLAS rebuilt every frame)
    uint32_t                 hitGroupContribution; // InstanceContributionToHitGroupIndex, computed by C#
    uint32_t                 _pad;          // explicit padding to 64 bytes (8-byte struct alignment)
};

static_assert(sizeof(NR_AddInstanceDesc) == 64, "NR_AddInstanceDesc size mismatch with C# AddInstanceDesc");

// ---------------------------------------------------------------------------
// SubMeshData  –  per-submesh data (indices, material, optional OMM).
//   Fields shared across all submeshes of the same instance live in InstanceDef.
// ---------------------------------------------------------------------------
struct SubMeshData
{
    UINT        indexCount;
    UINT        indexByteOffset;  // byte offset of this sub-mesh's first index in the shared IB
    INT         baseVertex = 0;  // value added to each index before reading a vertex (BaseVertexLocation in D3D12)
    uint32_t    flags = 0;       // NR_SUBMESH_FLAG_* bitmask (bit 0 = NR_SUBMESH_FLAG_GEOMETRY_OPAQUE)

    bool hasBakedOMM = false;
    struct OMMBakedData
    {
        std::vector<uint8_t> arrayData;
        std::vector<uint8_t> descArray;
        uint32_t             descArrayCount = 0;
        std::vector<uint8_t> indexBuffer;
        uint32_t             indexCount  = 0;
        UINT                 indexStride = 2;
        DXGI_FORMAT          indexFormat = DXGI_FORMAT_R16_UINT;
        std::vector<D3D12_RAYTRACING_OPACITY_MICROMAP_HISTOGRAM_ENTRY> histogram;
    } ommBaked;
};

// Groups all sub-meshes belonging to one GameObject/Instance.
// The vertex/index buffer and layout fields are shared by all submeshes.
struct MeshInfo
{
    // Shared per-instance GPU buffers and vertex layout
    // NOTE: These are Unity-managed resources. We store raw pointers without AddRef
    // because Unity controls their lifetime. Using ComPtr would interfere with Unity's
    // resource management and cause premature or delayed deletion.
    //
    // LIFETIME INVARIANT: a BLAS build records GPU reads of these buffers that may
    // still be in flight for up to kGlobalNumFrames after the command list is
    // submitted. The caller MUST keep the underlying Unity mesh resources alive for
    // at least that many frames after RemoveInstance(); destroying the mesh in the
    // same frame it is removed is a use-after-free. (We cannot enforce this here
    // because we intentionally do not hold a reference.)
    ID3D12Resource* vertexBuffer = nullptr;
    UINT vertexCount;
    UINT vertexStride;
    ID3D12Resource* indexBuffer = nullptr;
    DXGI_FORMAT indexFormat; // DXGI_FORMAT_R16_UINT or DXGI_FORMAT_R32_UINT

    std::vector<SubMeshData> submeshes;
};

// ---------------------------------------------------------------------------
// MeshKey  –  identifies a unique mesh+submesh-subset combination.
//   Two AddInstance calls sharing the same VB+IB AND the same submesh subset
//   (identified by submeshHash) reuse the same BLAS.  Different subsets of the
//   same VB+IB (e.g. transparent vs. opaque groups), different geometry flags,
//   or different OMM presence get independent BLASes.
// ---------------------------------------------------------------------------
struct MeshKey
{
    uintptr_t vbPtr       = 0;
    uintptr_t ibPtr       = 0;
    uint64_t  submeshHash = 0; // hash of (indexCount, indexByteOffset, baseVertex, flags, hasOMM) for each submesh
    bool operator==(const MeshKey& o) const
    {
        return vbPtr == o.vbPtr && ibPtr == o.ibPtr && submeshHash == o.submeshHash;
    }
};

struct MeshKeyHash
{
    size_t operator()(const MeshKey& k) const noexcept
    {
        size_t h = k.vbPtr;
        h ^= k.ibPtr       + 0x9e3779b9ull + (h << 6) + (h >> 2);
        h ^= k.submeshHash + 0x9e3779b9ull + (h << 6) + (h >> 2);
        return h;
    }
};

struct BLASEntry
{
    // Legacy path: committed BLAS result buffer, optionally replaced by a
    // compacted committed buffer after the batched readback completes.
    ComPtr<ID3D12Resource> blas;

    // RTXMU path: result / scratch / update-scratch / compacted memory are
    // suballocated from RTXMU's pooled blocks and identified by this id.
    uint64_t rtxmuId = kInvalidRtxmuId;

    // Built OMM Array AS (consumed by the BLAS build and at trace time) — must
    // outlive the BLAS. Kept outside RTXMU (committed buffers), matching nvrhi,
    // which also manages OMM arrays separately from rtxmu. The OMM-array build
    // scratch and the raw array/desc input blobs are transient: they are
    // pool-owned and recycled by the frame fence, so they are not retained here.
    std::vector<ComPtr<ID3D12Resource>> ommArrays;
    std::vector<ComPtr<ID3D12Resource>> ommIndexBuffers;   // DEFAULT-heap, referenced by OMM linkage at trace time
    std::vector<DXGI_FORMAT>            ommIndexFormats;
    std::vector<UINT>                   ommIndexStrides;

    bool anyOMM   = false;
    int  refCount = 0;

    // Legacy dynamic BLAS path: cached update scratch size for PERFORM_UPDATE.
    UINT64 updateScratchSize = 0;
};
// ---------------------------------------------------------------------------
// AccelerationStructure
//   Unified class that manages the full instance lifecycle, BLAS cache, and TLAS.
//
//   Instance lifecycle:
//     AddInstance()         → indexed by opaque userHandle
//     RemoveInstance(h)     → removes instance; BLAS ref-decremented (deferred GPU delete)
//     SetInstanceTransform  → triggers TLAS refit next BuildOrUpdate
//     SetInstanceMask       → triggers TLAS full rebuild next BuildOrUpdate
//
//   BuildOrUpdate() (call each frame before ray dispatch):
//     A. Builds any pending new BLASes (cache hit = no GPU work)
//     B. If structural change (add/remove/mask): rebuilds TLAS
//     C. Else if only transforms changed: refits TLAS in-place (fast)
// ---------------------------------------------------------------------------
class AccelerationStructure
{
public:
    AccelerationStructure(ID3D12Device5* device, IUnityLog* log, bool useRtxmu = true, bool useCompaction = true);
    ~AccelerationStructure();

    // Optional: supply v8 interface so the AS can notify Unity of resource state changes
    // caused by implicit BLAS input buffer promotions.
    void SetUnityGraphics(IUnityGraphicsD3D12v8* iface) { m_d3d12v8 = iface; }

    // Remove all instances and reset the AS.
    void Clear();

    // Add one instance (one GameObject) with all its sub-meshes.
    //   desc.instanceHandle : caller-assigned opaque ID (e.g. Unity MeshRenderer.GetInstanceID()).
    //                         Must be unique among active instances; no-op if already registered.
    bool AddInstance(const NR_AddInstanceDesc& desc);

    // Remove instance identified by handle. No-op if handle is invalid.
    void RemoveInstance(uint32_t handle);

    // Update the vertex buffer pointer for a dynamic (SkinnedMeshRenderer) instance.
    // Discards the old BLAS (deferred 3-frame GPU delete) and schedules a rebuild.
    // vbPtr must be the current-frame ID3D12Resource* from GetVertexBuffer().
    void UpdateDynamicVertexBuffer(uint32_t handle, void* vbPtr);

    // Per-frame update: set world transform (row-major 3x4).
    void SetInstanceTransform(uint32_t handle, const float transform[12]);

    // Set the TLAS instance mask for visibility filtering (default 0xFF).
    void SetInstanceMask(uint32_t handle, uint8_t mask);

    // Set the custom InstanceID returned by InstanceID() in HLSL shaders.
    // Use this to align InstanceID() with an index into a structured buffer (e.g. t_InstanceData).
    void SetInstanceID(uint32_t handle, uint32_t id);

    // Set the TLAS emission order for this instance. BuildOrUpdate emits TLAS instances
    // sorted ascending by this value (stable sort; the default 0xFFFFFFFF keeps legacy
    // slot-order emission for callers that never assign one). Callers whose shaders index
    // per-instance buffers by InstanceIndex() must re-assign dense order indices after any
    // add/remove: freed slots are reused (LIFO), so raw slot order stops matching
    // registration order as soon as instances have been removed.
    void SetInstanceOrderIndex(uint32_t handle, uint32_t order);

    // Update InstanceContributionToHitGroupIndex for an existing instance — the base offset
    // of its geometries in the caller's flat shader table, which shifts for surviving
    // instances whenever the scene's geometry layout changes.
    void SetInstanceHitGroupContribution(uint32_t handle, uint32_t contribution);

    // Number of active (non-removed) instances.
    uint32_t GetInstanceCount() const { return m_activeCount; }

    // True if any BLAS in the cache uses OMM geometry.
    bool HasAnyOMM() const;

    // Called from the renderer before dispatch. Builds/updates BLAS+TLAS as needed.
    bool BuildOrUpdate(ID3D12GraphicsCommandList4* cmdList);

    // Debug: dump every active instance's state to the Unity log.  Also verifies
    // handle<->slot map consistency and flags duplicate vb+ib pairs across slots.
    // Safe to call every frame; intended for diagnostics only.
    void DumpInstances(const char* tag = nullptr) const;

    ID3D12Resource* GetTLAS() const;

private:
    // -----------------------------------------------------------------------
    // Internal BLAS types
    // -----------------------------------------------------------------------

    // RTXMU build lifecycle batch: the ids built (PopulateBuildCommandList) in
    // one frame travel together through the deferred stages —
    //   frame N   : build + inline compaction-size emit; size copy to readback
    //               (PopulateCompactionSizeCopiesCommandList)
    //   frame N+3 : GPU provably done → PopulateCompactionCommandList records
    //               the compaction copies (RTXMU reads the sizes from its
    //               mapped readback and switches GetAccelStructGPUVA over)
    //   frame N+6 : compaction copies provably done → GarbageCollection frees
    //               the transient result / scratch / size memory.
    // Mirrors nvrhi's rtxmuBuildIds → asBuildsCompleted → rtxmuCompactionIds
    // flow, with the frame-fence delay standing in for nvrhi's per-commandlist
    // fences. Non-compaction builds (dynamic BLASes) ride the same queue;
    // RTXMU skips them where compaction does not apply.
    struct RtxmuIdBatch
    {
        std::vector<uint64_t> ids;
        uint32_t              frame = 0;   // m_frameCounter when this stage was recorded
    };

    struct PendingCompactionBatch
    {
        std::vector<MeshKey>   keys;
        ComPtr<ID3D12Resource> sizeBuffer;
        ComPtr<ID3D12Resource> readbackBuffer;
        void*                  mappedReadback = nullptr;
        uint32_t               buildFrame     = 0;
    };

    struct TLASInstanceEntry
    {
        D3D12_GPU_VIRTUAL_ADDRESS blasVA;
        float    transform[12];
        uint32_t instanceID;
        uint8_t  mask;
        uint32_t hitGroupContribution;  // InstanceContributionToHitGroupIndex, computed by C#
        uint32_t submeshCount;          // number of geometries in this BLAS
    };

    // -----------------------------------------------------------------------
    // Slot system
    //   m_slots is sparse; holes from RemoveInstance are tracked in m_freeSlots.
    // -----------------------------------------------------------------------
    struct InstanceSlot
    {
        MeshInfo meshInfo;
        MeshKey     meshKey;
        float       transform[12] = {
            1,0,0,0,
            0,1,0,0,
            0,0,1,0
        };
        uint32_t customInstanceID = 0;
        uint8_t mask      = 0xFF;
        bool    active    = false;
        bool    needsBLAS = false;
        bool    isDynamic = false; // SkinnedMeshRenderer: BLAS updated each frame
        uint32_t hitGroupContribution = 0; // InstanceContributionToHitGroupIndex, computed by C#
        // TLAS emission order (SetInstanceOrderIndex). 0xFFFFFFFF = unordered: emitted after
        // all ordered instances, in slot order (stable sort), preserving legacy behavior.
        uint32_t tlasOrder = 0xFFFFFFFFu;
        // Legacy committed-buffer BLAS VA, used when RTXMU is disabled.
        D3D12_GPU_VIRTUAL_ADDRESS blasVA = 0;
        // RTXMU id of this instance's BLAS. The GPU VA is resolved per frame at
        // TLAS emission when RTXMU is enabled.
        uint64_t blasRtxmuId = kInvalidRtxmuId;
        // Persistent BLAS for dynamic (skinned) instances – reused every frame with PERFORM_UPDATE
        std::unique_ptr<BLASEntry> dynamicBlas;
    };

    // -----------------------------------------------------------------------
    // BLAS helpers
    // -----------------------------------------------------------------------
    bool EnsureBLAS(ID3D12GraphicsCommandList4* cmdList, InstanceSlot& slot);
    void ReleaseBLAS(const MeshKey& key);
    D3D12_GPU_VIRTUAL_ADDRESS GetBLASVA(const MeshKey& key) const;
    bool BuildOMMForSubmesh(ID3D12GraphicsCommandList4* cmdList,
                            BLASEntry& entry, size_t subIdx, const SubMeshData& mesh);

    // --- RTXMU lifecycle helpers ---
    // Advances the deferred stages (compaction copies, garbage collection) for
    // batches whose previous stage has provably completed on the GPU.
    void ProcessRtxmuCompaction(ID3D12GraphicsCommandList4* cmdList);
    // Records the compaction-size readback copy for this frame's builds and
    // queues them for the deferred stages. Must run after the frame's global
    // post-build UAV barrier.
    void FlushRtxmuBuilds(ID3D12GraphicsCommandList4* cmdList);
    void QueueCompactionSizeQueries(ID3D12GraphicsCommandList4* cmdList);
    void ProcessPendingCompactions(ID3D12GraphicsCommandList4* cmdList);
    // Defers RemoveAccelerationStructures(id) until the GPU has finished any
    // frame that may still reference it (shares the deferred-delete fence).
    void ScheduleRtxmuRemove(uint64_t id);
    // Removes |id| from every pending lifecycle batch so a released BLAS is
    // never compacted / garbage-collected after its removal was scheduled.
    void ScrubPendingRtxmuId(uint64_t id);
    // Releases everything a BLASEntry owns (RTXMU memory + OMM buffers), GPU-safely.
    void ReleaseBLASEntryResources(BLASEntry& e);
    // Current GPU VA for an RTXMU id (compacted VA once compaction is recorded), 0 if invalid.
    D3D12_GPU_VIRTUAL_ADDRESS ResolveBlasVA(uint64_t id) const;

    // TLAS helpers
    bool BuildTLAS(ID3D12GraphicsCommandList4* cmdList, const std::vector<TLASInstanceEntry>& entries);

    // -----------------------------------------------------------------------
    // Members
    // -----------------------------------------------------------------------
    IUnityLog*               m_log;
    IUnityGraphicsD3D12v8*   m_d3d12v8 = nullptr;
    ComPtr<ID3D12Device5> m_device;
    bool m_useRtxmu = true;
    // Controls whether static BLASes are actually copy-compacted after build.
    bool m_useCompaction = true;

    // BLAS cache
    std::unordered_map<MeshKey, BLASEntry, MeshKeyHash> m_blasCache;

    // RTXMU acceleration-structure memory manager (suballocated result/scratch/
    // compaction pools). shared_ptr so deferred-delete lambdas that still need
    // to call RemoveAccelerationStructures keep it alive past this object.
    std::shared_ptr<rtxmu::DxAccelStructManager> m_rtxmu;

    // RTXMU lifecycle queues (see RtxmuIdBatch).
    std::vector<uint64_t>     m_rtxmuBuildsThisFrame;
    std::vector<RtxmuIdBatch> m_rtxmuPendingCompaction;  // size copy recorded, awaiting GPU
    std::vector<RtxmuIdBatch> m_rtxmuPendingGC;          // compaction recorded, awaiting GPU
    // Frames to wait before trusting that a recorded stage has executed on the
    // GPU — same margin as the deferred-delete queue (kDeleteDelay).
    static constexpr uint32_t kRtxmuStageLatency = 3;
    // Suballocator block size; matches nvrhi's rtxMemUtil->Initialize(8388608).
    static constexpr uint32_t kRtxmuBlockSize = 8u * 1024u * 1024u;

    std::vector<MeshKey>                m_compactionQueue;
    std::vector<PendingCompactionBatch> m_pendingCompactionBatches;

    uint32_t m_frameCounter = 0;

    // Single persistent TLAS rebuilt in place each frame (nvrhi model). Serialized
    // against the previous frame's traversal reads by a UAV barrier in BuildTLAS.
    // Scratch and instance-desc upload come from g_scratchPool / g_uploadPool.
    ComPtr<ID3D12Resource> m_tlas;
    UINT64                 m_tlasResultCapacity = 0; // bytes allocated for m_tlas
    // Reused CPU-side staging array for instance descriptors (built then uploaded
    // through g_uploadPool), avoiding a slow per-instance write over PCIe.
    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> m_dxrInstances;

    // Slot system
    std::vector<InstanceSlot>              m_slots;
    std::vector<uint32_t>                  m_freeSlots;
    std::unordered_map<uint32_t, uint32_t> m_handleToSlot;
    uint32_t m_activeCount = 0;
    // Number of active instances that carry baked OMM data; backs HasAnyOMM() in O(1).
    uint32_t m_ommInstanceCount = 0;

    std::vector<TLASInstanceEntry> m_tlasEntries;
    // Reused scratch: active slot indices sorted by InstanceSlot::tlasOrder for TLAS emission.
    std::vector<uint32_t>          m_orderedSlotScratch;

    // Reused scratch for the per-frame dynamic-BLAS refit path in EnsureBLAS (geometry
    // descs reference the tri/linkage descs by pointer, so all three live together).
    // Guarded by m_stateMutex like everything else EnsureBLAS touches.
    std::vector<D3D12_RAYTRACING_GEOMETRY_DESC>             m_refitGeomDescs;
    std::vector<D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC>   m_refitOmmTriDescs;
    std::vector<D3D12_RAYTRACING_GEOMETRY_OMM_LINKAGE_DESC> m_refitOmmLinkages;

    // Mutex protecting shared state accessed from both Main Thread (Clear, AddInstance,
    // RemoveInstance, SetInstance*) and Render Thread (BuildOrUpdate / BuildTLAS).
    std::mutex m_stateMutex;
};
