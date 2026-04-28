#pragma once
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <vector>
#include <list>
#include <unordered_map>
#include <mutex>
#include <cstdint>
#include "IUnityLog.h"
#include "IUnityGraphicsD3D12.h"

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// NR_SubmeshDesc
//   Plain data descriptor for one sub-mesh within an AddInstance call.
//   Must match the C# NativeRenderPlugin.SubmeshDesc struct layout exactly.
// ---------------------------------------------------------------------------
struct NR_SubmeshDesc
{
    uint32_t indexCount;      // number of indices in this sub-mesh
    uint32_t indexByteOffset; // byte offset of this sub-mesh's first index in the shared IB
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
};

static_assert(sizeof(NR_AddInstanceDesc) == 56, "NR_AddInstanceDesc size mismatch with C# AddInstanceDesc");

// ---------------------------------------------------------------------------
// SubMeshData  –  per-submesh data (indices, material, optional OMM).
//   Fields shared across all submeshes of the same instance live in InstanceDef.
// ---------------------------------------------------------------------------
struct SubMeshData
{
    UINT        indexCount;
    UINT        indexByteOffset;  // byte offset of this sub-mesh's first index in the shared IB

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
    ID3D12Resource* vertexBuffer = nullptr;
    UINT vertexCount;
    UINT vertexStride;
    ID3D12Resource* indexBuffer = nullptr;
    DXGI_FORMAT indexFormat; // DXGI_FORMAT_R16_UINT or DXGI_FORMAT_R32_UINT

    std::vector<SubMeshData> submeshes;
};

// ---------------------------------------------------------------------------
// MeshKey  –  identifies a unique mesh by its GPU buffer pointers.
//   Two AddInstance calls sharing the same VB+IB reuse the same BLAS.
// ---------------------------------------------------------------------------
struct MeshKey
{
    uintptr_t vbPtr = 0;
    uintptr_t ibPtr = 0;
    bool operator==(const MeshKey& o) const { return vbPtr == o.vbPtr && ibPtr == o.ibPtr; }
};

struct MeshKeyHash
{
    size_t operator()(const MeshKey& k) const noexcept
    {
        size_t h = k.vbPtr;
        h ^= k.ibPtr + 0x9e3779b9ull + (h << 6) + (h >> 2);
        return h;
    }
};

struct BLASEntry
{
    ComPtr<ID3D12Resource> blas;
    ComPtr<ID3D12Resource> blasScratch;

    std::vector<ComPtr<ID3D12Resource>> ommArrays;
    std::vector<ComPtr<ID3D12Resource>> ommArrayScratch;
    std::vector<ComPtr<ID3D12Resource>> ommIndexBuffers;
    std::vector<ComPtr<ID3D12Resource>> ommDescArrayBuffers;
    std::vector<ComPtr<ID3D12Resource>> ommArrayDataBuffers;
    std::vector<DXGI_FORMAT>            ommIndexFormats;
    std::vector<UINT>                   ommIndexStrides;

    bool anyOMM   = false;
    int  refCount = 0;
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
    AccelerationStructure(ID3D12Device5* device, IUnityLog* log);
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
    void UpdateDynamicVertexBuffer(uint32_t handle, void* vbPtr, uint32_t vertexCount, uint32_t vertexStride);

    // Per-frame update: set world transform (row-major 3x4).
    void SetInstanceTransform(uint32_t handle, const float transform[12]);

    // Set the TLAS instance mask for visibility filtering (default 0xFF).
    void SetInstanceMask(uint32_t handle, uint8_t mask);

    // Set the custom InstanceID returned by InstanceID() in HLSL shaders.
    // Use this to align InstanceID() with an index into a structured buffer (e.g. t_InstanceData).
    void SetInstanceID(uint32_t handle, uint32_t id);

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

    ID3D12Resource* GetTLAS() const { return m_tlasResources[m_frameIndex].tlas.Get(); }

private:
    // -----------------------------------------------------------------------
    // Internal BLAS types
    // -----------------------------------------------------------------------


    struct TLASInstanceEntry
    {
        D3D12_GPU_VIRTUAL_ADDRESS blasVA;
        float    transform[12];
        uint32_t instanceID;
        uint8_t  mask;
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
        bool    isDynamic = false; // SkinnedMeshRenderer: BLAS rebuilt each frame
        D3D12_GPU_VIRTUAL_ADDRESS blasVA;
    };

    // -----------------------------------------------------------------------
    // BLAS helpers
    // -----------------------------------------------------------------------
    bool EnsureBLAS(ID3D12GraphicsCommandList4* cmdList, InstanceSlot& slot);
    void ReleaseBLAS(const MeshKey& key);
    D3D12_GPU_VIRTUAL_ADDRESS GetBLASVA(const MeshKey& key) const;
    bool BuildOMMForSubmesh(ID3D12GraphicsCommandList4* cmdList,
                            BLASEntry& entry, size_t subIdx, const SubMeshData& mesh);

    // TLAS helpers
    bool BuildTLAS(ID3D12GraphicsCommandList4* cmdList, const std::vector<TLASInstanceEntry>& entries);

    // TLAS per-frame double-buffer slot
    struct TLASFrameResources
    {
        ComPtr<ID3D12Resource> instanceDesc;
        void*                  mappedInstanceDesc   = nullptr;
        ComPtr<ID3D12Resource> tlas;
        ComPtr<ID3D12Resource> tlasScratch;
        uint32_t               instanceDescCapacity = 0;   // instances that fit in instanceDesc
        UINT64                 tlasResultCapacity   = 0;   // bytes allocated for tlas
        UINT64                 tlasScratchCapacity  = 0;   // bytes allocated for tlasScratch
    };

    // -----------------------------------------------------------------------
    // Members
    // -----------------------------------------------------------------------
    IUnityLog*               m_log;
    IUnityGraphicsD3D12v8*   m_d3d12v8 = nullptr;
    ComPtr<ID3D12Device5> m_device;

    // BLAS cache
    std::unordered_map<MeshKey, BLASEntry, MeshKeyHash> m_blasCache;

    // TLAS quadruple-buffered resources (indexed by m_frameIndex)
    TLASFrameResources     m_tlasResources[4];
    uint32_t               m_frameIndex           = 0;

    // Slot system
    std::vector<InstanceSlot>              m_slots;
    std::vector<uint32_t>                  m_freeSlots;
    std::unordered_map<uint32_t, uint32_t> m_handleToSlot;
    uint32_t m_activeCount = 0;

    std::vector<TLASInstanceEntry> m_tlasEntries;

    // Mutex protecting shared state accessed from both Main Thread (Clear, AddInstance,
    // RemoveInstance, SetInstance*) and Render Thread (BuildOrUpdate / BuildTLAS).
    std::mutex m_stateMutex;
};
