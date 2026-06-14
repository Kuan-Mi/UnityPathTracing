#include "AccelerationStructure.h"
#include "PluginInternal.h"
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <algorithm>

// Forward declaration of global deferred resource delete function from Plugin.cpp.
// RetireObject<T> (typed deferred delete) comes from the included PluginInternal.h.
extern void SafeReleaseResource(ComPtr<ID3D12Resource> resource);

// ---------------------------------------------------------------------------
// HashSubmeshDescs
//   Produces a stable 64-bit hash from the (indexCount, indexByteOffset,
//   baseVertex, flags) tuple of each submesh passed to AddInstance, plus a
//   per-submesh OMM-present bit derived from ommDescs.  Used as the third
//   component of MeshKey so that different submesh subsets of the same VB+IB
//   (e.g. transparent vs. opaque SubmeshGroups) are cached separately, and so
//   that BLAS builds with/without OMM or with different geometry flags never
//   share a cache entry.
//   Returns at least 1 (0 is reserved as an unset sentinel).
// ---------------------------------------------------------------------------
static uint64_t HashSubmeshDescs(const NR_SubmeshDesc* descs, uint32_t count,
                                  const NR_SubmeshOMMDesc* ommDescs = nullptr)
{
    uint64_t h = static_cast<uint64_t>(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        h ^= static_cast<uint64_t>(descs[i].indexCount)      * 2654435761ULL + (h << 7) + (h >> 5);
        h ^= static_cast<uint64_t>(descs[i].indexByteOffset) * 2246822519ULL + (h << 7) + (h >> 5);
        h ^= static_cast<uint64_t>(descs[i].baseVertex)      * 3266489917ULL + (h << 7) + (h >> 5);
        h ^= static_cast<uint64_t>(descs[i].flags)           * 2166136261ULL + (h << 7) + (h >> 5);
        // Encode OMM presence per-submesh (1 = has OMM, 0 = no OMM)
        const uint64_t hasOMM = (ommDescs && ommDescs[i].arrayData) ? 1ULL : 0ULL;
        h ^= hasOMM * 1000000007ULL + (h << 7) + (h >> 5);
    }
    return h ? h : 1u;
}

// ---------------------------------------------------------------------------
// Internal buffer helper
// ---------------------------------------------------------------------------
static ComPtr<ID3D12Resource> CreateBuffer(
    ID3D12Device *device,
    UINT64 size,
    D3D12_RESOURCE_FLAGS flags,
    D3D12_RESOURCE_STATES initialState,
    const D3D12_HEAP_PROPERTIES &heapProps,
    const wchar_t *name = nullptr)
{
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment = 0;
    desc.Width = size ? size : 1;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    ComPtr<ID3D12Resource> resource;
    HRESULT hr = device->CreateCommittedResource(
        &heapProps, D3D12_HEAP_FLAG_NONE,
        &desc, initialState,
        nullptr, IID_PPV_ARGS(&resource));
    if (SUCCEEDED(hr) && resource && name)
    {
        resource->SetName(name);
    }
    return SUCCEEDED(hr) ? resource : nullptr;
}

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------
static void AccelLogf(IUnityLog *log, UnityLogType type, const char *fmt, ...)
{
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    log->Log(type, buf, __FILE__, __LINE__);
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
AccelerationStructure::AccelerationStructure(ID3D12Device5 *device, IUnityLog *log)
    : m_device(device), m_log(log)
{
    // RTXMU owns all BLAS result/scratch/compaction memory from here on
    // (suballocated 8MB blocks, same Initialize size nvrhi uses).
    m_rtxmu = std::make_shared<rtxmu::DxAccelStructManager>(device);
    m_rtxmu->Initialize(kRtxmuBlockSize);
}

AccelerationStructure::~AccelerationStructure()
{
    AccelLogf(m_log, kUnityLogTypeLog, "[AccelerationStructure::~AccelerationStructure] Destructor complete");
}

// ---------------------------------------------------------------------------
// BuildOMMForSubmesh
//   Uploads OMM data to GPU and records a BuildRaytracingAccelerationStructure
//   command for the OMM Array AS into cmdList.
// ---------------------------------------------------------------------------
bool AccelerationStructure::BuildOMMForSubmesh(ID3D12GraphicsCommandList4 *cmdList, BLASEntry &entry, size_t subIdx, const SubMeshData &mesh)
{
    const SubMeshData::OMMBakedData &baked = mesh.ommBaked;
    AccelLogf(m_log, kUnityLogTypeLog,
              "[OMM] BuildOMMForSubmesh[%zu]: arrayData=%zu bytes, descs=%u, indices=%u",
              subIdx, baked.arrayData.size(), baked.descArrayCount, baked.indexCount);

    if (baked.histogram.empty())
    {
        AccelLogf(m_log, kUnityLogTypeError, "[OMM] BuildOMMForSubmesh[%zu]: histogram is empty, cannot build OMM array", subIdx);
        return false;
    }

    D3D12_HEAP_PROPERTIES defaultHeap = {};
    defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;

    // OMM-array build inputs (raw array blob + per-OMM desc array) are transient —
    // only read by the OMM-array build recorded below — so they are suballocated from
    // the shared upload pool (fence-recycled) rather than committed UPLOAD buffers.
    // D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_BYTE_ALIGNMENT (128). Inlined because the
    // constant is absent from some SDK headers even when the OMM types are available.
    constexpr UINT64 kOmmInputAlign = 128;

    // 1. Raw OMM array data
    const UINT64 arrayBytes = !baked.arrayData.empty() ? (UINT64)baked.arrayData.size() : 1;
    SharedUploadPool::Allocation arrayAlloc = g_uploadPool.Allocate(arrayBytes, kOmmInputAlign);
    if (!arrayAlloc.IsValid())
    {
        AccelLogf(m_log, kUnityLogTypeError, "[OMM] BuildOMMForSubmesh[%zu]: arrayData suballoc failed", subIdx);
        return false;
    }
    if (!baked.arrayData.empty())
        memcpy(arrayAlloc.cpu, baked.arrayData.data(), baked.arrayData.size());

    // 2. OMM desc array
    const UINT64 descArrayBytes = (UINT64)baked.descArrayCount * sizeof(D3D12_RAYTRACING_OPACITY_MICROMAP_DESC);
    SharedUploadPool::Allocation descAlloc = g_uploadPool.Allocate(descArrayBytes ? descArrayBytes : 1, kOmmInputAlign);
    if (!descAlloc.IsValid())
    {
        AccelLogf(m_log, kUnityLogTypeError, "[OMM] BuildOMMForSubmesh[%zu]: descArray suballoc failed", subIdx);
        return false;
    }
    if (!baked.descArray.empty())
    {
        size_t copyBytes = (std::min)((size_t)descArrayBytes, baked.descArray.size());
        memcpy(descAlloc.cpu, baked.descArray.data(), copyBytes);
    }

    // 3. OMM index buffer: lives in a DEFAULT heap because it is referenced by the
    //    BLAS OMM linkage both during the BLAS build and at ray-traversal time, so
    //    it must persist and be fast to read on the GPU. Stage through the upload pool,
    //    copy into a committed DEFAULT buffer, then transition to NON_PIXEL_SHADER_RESOURCE.
    UINT ommIdxBytes = baked.indexCount * baked.indexStride;
    SharedUploadPool::Allocation idxStaging = g_uploadPool.Allocate(ommIdxBytes ? ommIdxBytes : 1, 16);
    auto ommIdxBuf = CreateBuffer(m_device.Get(),
                                  ommIdxBytes ? ommIdxBytes : 1,
                                  D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, defaultHeap,
                                  L"OMM_IndexBuffer");
    if (!idxStaging.IsValid() || !ommIdxBuf)
    {
        AccelLogf(m_log, kUnityLogTypeError,
                  "[OMM] BuildOMMForSubmesh[%zu]: OMM index buf alloc failed", subIdx);
        return false;
    }
    if (ommIdxBytes)
        memcpy(idxStaging.cpu, baked.indexBuffer.data(), ommIdxBytes);
    cmdList->CopyBufferRegion(ommIdxBuf.Get(), 0, idxStaging.resource, idxStaging.offset, ommIdxBytes ? ommIdxBytes : 1);
    {
        D3D12_RESOURCE_BARRIER idxBarrier = {};
        idxBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        idxBarrier.Transition.pResource   = ommIdxBuf.Get();
        idxBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        idxBarrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        idxBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cmdList->ResourceBarrier(1, &idxBarrier);
    }

    // 4. Build OMM Array AS
    D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_DESC ommArrayDesc = {};
    ommArrayDesc.NumOmmHistogramEntries = (UINT)baked.histogram.size();
    ommArrayDesc.pOmmHistogram = baked.histogram.data();
    ommArrayDesc.InputBuffer = arrayAlloc.gpu;
    ommArrayDesc.PerOmmDescs.StartAddress = descAlloc.gpu;
    ommArrayDesc.PerOmmDescs.StrideInBytes = sizeof(D3D12_RAYTRACING_OPACITY_MICROMAP_DESC);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS ommInputs = {};
    ommInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_ARRAY;
    ommInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    ommInputs.NumDescs = 1;
    ommInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    ommInputs.pOpacityMicromapArrayDesc = &ommArrayDesc;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo = {};
    m_device->GetRaytracingAccelerationStructurePrebuildInfo(&ommInputs, &prebuildInfo);

    // OMM-array build scratch is transient — suballocated from the fence-recycled
    // scratch pool, consumed by the build command below, then reclaimed by the frame fence.
    SharedUploadPool::Allocation ommScratch = g_scratchPool.Allocate(
        prebuildInfo.ScratchDataSizeInBytes, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
    entry.ommArrays[subIdx] = CreateBuffer(m_device.Get(),
                                           prebuildInfo.ResultDataMaxSizeInBytes,
                                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                           D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, defaultHeap,
                                           L"OMM_Array");
    if (!ommScratch.IsValid() || !entry.ommArrays[subIdx])
    {
        AccelLogf(m_log, kUnityLogTypeError,
                  "[OMM] BuildOMMForSubmesh[%zu]: OMM array buf alloc failed", subIdx);
        return false;
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
    buildDesc.DestAccelerationStructureData = entry.ommArrays[subIdx]->GetGPUVirtualAddress();
    buildDesc.Inputs = ommInputs;
    buildDesc.ScratchAccelerationStructureData = ommScratch.gpu;
    cmdList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);

    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = entry.ommArrays[subIdx].Get();
    cmdList->ResourceBarrier(1, &barrier);

    // Persist only the built OMM Array AS and its index buffer; the index format/stride
    // are needed to wire up the OMM linkage in EnsureBLAS. (Scratch + array/desc inputs
    // are pool-owned and recycled by the frame fence — nothing to release here.)
    entry.ommIndexBuffers[subIdx] = std::move(ommIdxBuf);
    entry.ommIndexFormats[subIdx] = baked.indexFormat;
    entry.ommIndexStrides[subIdx] = baked.indexStride;

    AccelLogf(m_log, kUnityLogTypeLog,
              "[OMM] BuildOMMForSubmesh[%zu]: OMM Array AS recorded on cmdlist", subIdx);
    return true;
}

// ---------------------------------------------------------------------------
// EnsureBLAS
//   Cache hit:  increment refCount, return immediately (no GPU work).
//   Cache miss: build BLAS (+ OMM) and cache it.
//   isDynamic:  true for SkinnedMeshRenderer (rebuilt every frame with ALLOW_UPDATE flag)
// ---------------------------------------------------------------------------
bool AccelerationStructure::EnsureBLAS(ID3D12GraphicsCommandList4 *cmdList, InstanceSlot &slot)
{
    if(!slot.isDynamic){
        auto it = m_blasCache.find(slot.meshKey);
        if (it != m_blasCache.end())
        {
            it->second.refCount++;
            slot.blasRtxmuId = it->second.rtxmuId;
            // AccelLogf(m_log, kUnityLogTypeLog, "[BLAS] AddRef  vb=%p refCount=%d", (void *)slot.meshInfo.vertexBuffer, it->second.refCount);
            return true;
        }
    }

    auto &def = slot.meshInfo;
    auto isDynamic = slot.isDynamic;
    auto &key = slot.meshKey;

    BLASEntry blas;
    const size_t subCount = def.submeshes.size();
    if (subCount == 0)
    {
        AccelLogf(m_log, kUnityLogTypeError, "EnsureBLAS: instance has no submeshes");
        return false;
    }

    // CRITICAL: Request resource state transitions BEFORE accessing Unity's buffers.
    // Unity's skinning compute shader may leave vertex buffers in UNORDERED_ACCESS state,
    // but we need NON_PIXEL_SHADER_RESOURCE for BLAS builds. RequestResourceState ensures
    // Unity inserts the necessary barrier in the command list before our BLAS build command.
    //
    // NOTE: For dynamic meshes, we request state every time because the vertex buffer changes
    // each frame. Unity's state tracker should handle redundant requests efficiently.
    // AccelLogf(m_log, kUnityLogTypeLog, "[EnsureBLAS] Building %s BLAS for VB=%p IB=%p", isDynamic ? "DYNAMIC" : "STATIC", (void *)def.vertexBuffer, (void *)def.indexBuffer);
    m_d3d12v8->RequestResourceState(def.vertexBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    m_d3d12v8->RequestResourceState(def.indexBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // ---------------------------------------------------------------------------
    // Fast update path: reuse existing dynamic BLAS/Scratch with PERFORM_UPDATE.
    // Topology (indices, OMM) is unchanged; only vertex positions change each frame.
    // ---------------------------------------------------------------------------
    // Only take the in-place update path when the cached dynamic BLAS has the same
    // geometry count as the current submesh layout. A skinned mesh normally keeps a
    // fixed topology (only vertex positions change), but if the submesh count ever
    // changes the cached BLAS/scratch sizes no longer apply — fall through to a full
    // rebuild below (the stale BLAS is deferred-released before being replaced).
    if (isDynamic && slot.dynamicBlas && slot.dynamicBlas->rtxmuId != kInvalidRtxmuId &&
        slot.dynamicBlas->ommArrays.size() == subCount)
    {
        BLASEntry &existing = *slot.dynamicBlas;
        // Member scratch (capacity retained across frames): this path runs every frame
        // for every skinned instance, so avoid three heap allocations per refit.
        m_refitGeomDescs.resize(subCount);
        m_refitOmmTriDescs.resize(subCount);
        m_refitOmmLinkages.resize(subCount);
        auto &upGeomDescs   = m_refitGeomDescs;
        auto &upOmmTriDescs = m_refitOmmTriDescs;
        auto &upOmmLinkages = m_refitOmmLinkages;

        for (size_t j = 0; j < subCount; ++j)
        {
            const SubMeshData &sub = def.submeshes[j];
            D3D12_RAYTRACING_GEOMETRY_DESC &gd = upGeomDescs[j];
            gd = {};

            if (sub.hasBakedOMM && j < existing.ommArrays.size() && existing.ommArrays[j])
            {
                gd.Type  = D3D12_RAYTRACING_GEOMETRY_TYPE_OMM_TRIANGLES;
                gd.Flags = (sub.flags & NR_SUBMESH_FLAG_GEOMETRY_OPAQUE)
                    ? D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE
                    : D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;

                D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC &td = upOmmTriDescs[j];
                td = {};
                td.VertexBuffer.StartAddress  = def.vertexBuffer->GetGPUVirtualAddress()
                    + static_cast<UINT64>(sub.baseVertex) * def.vertexStride;
                td.VertexBuffer.StrideInBytes = def.vertexStride;
                td.VertexCount  = def.vertexCount - static_cast<UINT>(sub.baseVertex);
                td.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
                td.IndexBuffer  = def.indexBuffer->GetGPUVirtualAddress() + sub.indexByteOffset;
                td.IndexCount   = sub.indexCount;
                td.IndexFormat  = def.indexFormat;
                td.Transform3x4 = 0;

                D3D12_RAYTRACING_GEOMETRY_OMM_LINKAGE_DESC &ol = upOmmLinkages[j];
                ol = {};
                ol.OpacityMicromapArray                    = existing.ommArrays[j]->GetGPUVirtualAddress();
                ol.OpacityMicromapBaseLocation             = 0;
                ol.OpacityMicromapIndexBuffer.StartAddress = existing.ommIndexBuffers[j]->GetGPUVirtualAddress();
                ol.OpacityMicromapIndexBuffer.StrideInBytes = existing.ommIndexStrides[j];
                ol.OpacityMicromapIndexFormat              = existing.ommIndexFormats[j];

                gd.OmmTriangles.pTriangles  = &td;
                gd.OmmTriangles.pOmmLinkage = &ol;
            }
            else
            {
                gd.Type  = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
                gd.Flags = (sub.flags & NR_SUBMESH_FLAG_GEOMETRY_OPAQUE)
                    ? D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE
                    : D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
                gd.Triangles.VertexBuffer.StartAddress  = def.vertexBuffer->GetGPUVirtualAddress()
                    + static_cast<UINT64>(sub.baseVertex) * def.vertexStride;
                gd.Triangles.VertexBuffer.StrideInBytes = def.vertexStride;
                gd.Triangles.VertexCount  = def.vertexCount - static_cast<UINT>(sub.baseVertex);
                gd.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
                gd.Triangles.IndexBuffer  = def.indexBuffer->GetGPUVirtualAddress() + sub.indexByteOffset;
                gd.Triangles.IndexCount   = sub.indexCount;
                gd.Triangles.IndexFormat  = def.indexFormat;
                gd.Triangles.Transform3x4 = 0;
            }
        }

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS upFlags =
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE |
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD |
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
        if (existing.anyOMM)
            upFlags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_DISABLE_OMMS;

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS upInputs = {};
        upInputs.Type         = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        upInputs.Flags        = upFlags;
        upInputs.NumDescs     = static_cast<UINT>(subCount);
        upInputs.DescsLayout  = D3D12_ELEMENTS_LAYOUT_ARRAY;
        upInputs.pGeometryDescs = upGeomDescs.data();

        // RTXMU records the in-place refit: PERFORM_UPDATE|ALLOW_UPDATE routes to
        // its update path, which reuses the persistent update scratch suballocated
        // at the initial build — no per-frame prebuild query or scratch allocation.
        const std::vector<uint64_t> updateIds(1, existing.rtxmuId);
        m_rtxmu->PopulateUpdateCommandList(cmdList, &upInputs, 1, updateIds);

        slot.blasRtxmuId = existing.rtxmuId;
        return true;
    }

    blas.ommArrays.resize(subCount);
    blas.ommIndexBuffers.resize(subCount);
    blas.ommIndexFormats.resize(subCount, DXGI_FORMAT_R16_UINT);
    blas.ommIndexStrides.resize(subCount, 2);

    std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geomDescs(subCount);
    std::vector<D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC> ommTriDescs(subCount);
    std::vector<D3D12_RAYTRACING_GEOMETRY_OMM_LINKAGE_DESC> ommLinkages(subCount);
    bool instanceHasOMM = false;

    for (size_t j = 0; j < subCount; ++j)
    {
        const SubMeshData &sub = def.submeshes[j];
        D3D12_RAYTRACING_GEOMETRY_DESC &geomDesc = geomDescs[j];
        geomDesc = {};

        bool subUseOMM = false;
        if (sub.hasBakedOMM)
        {
            subUseOMM = BuildOMMForSubmesh(cmdList, blas, j, sub);
            if (subUseOMM)
            {
                blas.anyOMM = true;
                AccelLogf(m_log, kUnityLogTypeLog, "EnsureBLAS: submesh[%zu] OMM active", j);
            }
            else
            {
                AccelLogf(m_log, kUnityLogTypeWarning, "EnsureBLAS: submesh[%zu] OMM build failed, falling back to opaque", j);
            }
        }

        if (subUseOMM)
        {
            geomDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_OMM_TRIANGLES;
            geomDesc.Flags = (sub.flags & NR_SUBMESH_FLAG_GEOMETRY_OPAQUE)
                ? D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE
                : D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;

            D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC &td = ommTriDescs[j];
            td = {};
            td.VertexBuffer.StartAddress = def.vertexBuffer->GetGPUVirtualAddress()
                + static_cast<UINT64>(sub.baseVertex) * def.vertexStride;
            td.VertexBuffer.StrideInBytes = def.vertexStride;
            td.VertexCount = def.vertexCount - static_cast<UINT>(sub.baseVertex);
            td.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
            td.IndexBuffer = def.indexBuffer->GetGPUVirtualAddress() + sub.indexByteOffset;
            td.IndexCount = sub.indexCount;
            td.IndexFormat = def.indexFormat;
            td.Transform3x4 = 0;

            D3D12_RAYTRACING_GEOMETRY_OMM_LINKAGE_DESC &ol = ommLinkages[j];
            ol = {};
            ol.OpacityMicromapArray = blas.ommArrays[j]->GetGPUVirtualAddress();
            ol.OpacityMicromapBaseLocation = 0;
            ol.OpacityMicromapIndexBuffer.StartAddress = blas.ommIndexBuffers[j]->GetGPUVirtualAddress();
            ol.OpacityMicromapIndexBuffer.StrideInBytes = blas.ommIndexStrides[j];
            ol.OpacityMicromapIndexFormat = blas.ommIndexFormats[j];

            geomDesc.OmmTriangles.pTriangles = &td;
            geomDesc.OmmTriangles.pOmmLinkage = &ol;
            instanceHasOMM = true;
        }
        else
        {
            geomDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
            geomDesc.Flags = (sub.flags & NR_SUBMESH_FLAG_GEOMETRY_OPAQUE)
                ? D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE
                : D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
            geomDesc.Triangles.VertexBuffer.StartAddress = def.vertexBuffer->GetGPUVirtualAddress()
                + static_cast<UINT64>(sub.baseVertex) * def.vertexStride;
            geomDesc.Triangles.VertexBuffer.StrideInBytes = def.vertexStride;
            geomDesc.Triangles.VertexCount = def.vertexCount - static_cast<UINT>(sub.baseVertex);
            geomDesc.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
            geomDesc.Triangles.IndexBuffer = def.indexBuffer->GetGPUVirtualAddress() + sub.indexByteOffset;
            geomDesc.Triangles.IndexCount = sub.indexCount;
            geomDesc.Triangles.IndexFormat = def.indexFormat;
            geomDesc.Triangles.Transform3x4 = 0;
        }
    }

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS blasFlags;

    // Dynamic BLAS (SkinnedMesh): refit every frame — ALLOW_UPDATE for in-place refits
    // plus PREFER_FAST_BUILD since build/refit cost dominates over traversal cost.
    // Static BLAS: built once, use PREFER_FAST_TRACE for optimal ray tracing performance.
    // (The initial build flags must stay consistent with the PERFORM_UPDATE flags above.)
    if (isDynamic)
    {
        blasFlags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE |
                    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
    }
    else
    {
        blasFlags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    }

    if (instanceHasOMM)
        blasFlags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_DISABLE_OMMS;

    // Note: ALLOW_COMPACTION is incompatible with ALLOW_UPDATE, so only add it for static BLAS
    if (!isDynamic)
        blasFlags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
    inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    inputs.Flags = blasFlags;
    inputs.NumDescs = static_cast<UINT>(subCount);
    inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputs.pGeometryDescs = geomDescs.data();

    // RTXMU performs the prebuild query, suballocates result + scratch (+ the
    // persistent update scratch when ALLOW_UPDATE is set, + the compaction-size
    // slots when ALLOW_COMPACTION is set) and records the build with the inline
    // compacted-size emit.
    std::vector<uint64_t> newIds;
    m_rtxmu->PopulateBuildCommandList(cmdList, &inputs, 1, newIds);
    if (newIds.empty())
    {
        AccelLogf(m_log, kUnityLogTypeError, "EnsureBLAS: RTXMU build allocation failed");
        return false;
    }
    blas.rtxmuId = newIds[0];
    m_rtxmuBuildsThisFrame.push_back(newIds[0]);

    slot.blasRtxmuId = blas.rtxmuId;
    if(isDynamic){
        // If a stale dynamic BLAS is being replaced (e.g. submesh count changed, so
        // the update fast-path was skipped above), release its RTXMU memory and OMM
        // buffers — both go through GPU-safe deferred frees inside the helper.
        if (slot.dynamicBlas)
        {
            ReleaseBLASEntryResources(*slot.dynamicBlas);
            slot.dynamicBlas.reset();
        }
        // Store the BLASEntry in the slot so it can be reused (via PERFORM_UPDATE) next frame.
        // Ownership transfers here; ReleaseBLAS/RemoveInstance will release it when the
        // instance is destroyed.
        slot.dynamicBlas = std::make_unique<BLASEntry>(std::move(blas));
    }else{
        blas.refCount = 1;
        // AccelLogf(m_log, kUnityLogTypeLog, "[BLAS] Add     vb=%p refCount=1 (new, anyOMM=%d)",
        //           (void*)key.vbPtr, (int)blas.anyOMM);
        m_blasCache.emplace(key, std::move(blas));
    }

    return true;
}

// ---------------------------------------------------------------------------
// ReleaseBLAS
// ---------------------------------------------------------------------------
void AccelerationStructure::ReleaseBLAS(const MeshKey &key)
{
    auto it = m_blasCache.find(key);
    if (it == m_blasCache.end())
        return;

    if (--it->second.refCount > 0)
    {
        // AccelLogf(m_log, kUnityLogTypeLog, "[BLAS] Release vb=%p refCount=%d (still alive)",
        //           (void*)key.vbPtr, it->second.refCount);
        return;
    }

    // AccelLogf(m_log, kUnityLogTypeLog,
    //     "[BLAS] Release vb=%p refCount=0 \u2192 deferred GPU delete", (void*)key.vbPtr);

    ReleaseBLASEntryResources(it->second);
    m_blasCache.erase(it);
}

// ---------------------------------------------------------------------------
// ReleaseBLASEntryResources
//   Releases everything a BLASEntry owns: the RTXMU-managed acceleration
//   structure memory (deferred RemoveAccelerationStructures) and the OMM
//   buffers (deferred resource release). Safe to call on entries whose BLAS
//   was never built (rtxmuId invalid).
// ---------------------------------------------------------------------------
void AccelerationStructure::ReleaseBLASEntryResources(BLASEntry &e)
{
    if (e.rtxmuId != kInvalidRtxmuId)
    {
        ScrubPendingRtxmuId(e.rtxmuId);   // never compact/GC an id queued for removal
        ScheduleRtxmuRemove(e.rtxmuId);
        e.rtxmuId = kInvalidRtxmuId;
    }
    // OMM-array build scratch and array/desc input blobs are pool-owned;
    // only the built OMM Array AS and its index buffer remain to release here.
    for (auto &r : e.ommArrays)
        if (r)
            SafeReleaseResource(std::move(r));
    for (auto &r : e.ommIndexBuffers)
        if (r)
            SafeReleaseResource(std::move(r));
}

// ---------------------------------------------------------------------------
// ScheduleRtxmuRemove / ScrubPendingRtxmuId / ResolveBlasVA
// ---------------------------------------------------------------------------
void AccelerationStructure::ScheduleRtxmuRemove(uint64_t id)
{
    // The lambda holds a shared_ptr so the manager outlives this object if the
    // AccelerationStructure itself is retired before the cleanup drains.
    EnqueueCleanup([mgr = m_rtxmu, id]()
    {
        if (mgr->IsValid(id))
        {
            const std::vector<uint64_t> ids(1, id);
            mgr->RemoveAccelerationStructures(ids);
        }
    });
}

void AccelerationStructure::ScrubPendingRtxmuId(uint64_t id)
{
    auto scrub = [id](std::vector<uint64_t> &ids)
    {
        ids.erase(std::remove(ids.begin(), ids.end(), id), ids.end());
    };
    scrub(m_rtxmuBuildsThisFrame);
    for (auto &batch : m_rtxmuPendingCompaction) scrub(batch.ids);
    for (auto &batch : m_rtxmuPendingGC)         scrub(batch.ids);
}

D3D12_GPU_VIRTUAL_ADDRESS AccelerationStructure::ResolveBlasVA(uint64_t id) const
{
    return (id != kInvalidRtxmuId && m_rtxmu->IsValid(id))
        ? m_rtxmu->GetAccelStructGPUVA(id)
        : 0;
}

// ---------------------------------------------------------------------------
// FlushRtxmuBuilds
//   Records the compaction-size readback copy for every BLAS built this frame
//   and queues the ids for the deferred lifecycle stages. RTXMU copies its
//   compaction-size blocks wholesale, so this is one CopyResource per 64KB
//   size block regardless of how many BLASes were built.
//   Caller (BuildOrUpdate) must have issued the global post-build UAV barrier.
// ---------------------------------------------------------------------------
void AccelerationStructure::FlushRtxmuBuilds(ID3D12GraphicsCommandList4 *cmdList)
{
    if (m_rtxmuBuildsThisFrame.empty())
        return;

    m_rtxmu->PopulateCompactionSizeCopiesCommandList(cmdList, m_rtxmuBuildsThisFrame);

    RtxmuIdBatch batch;
    batch.ids   = std::move(m_rtxmuBuildsThisFrame);
    batch.frame = m_frameCounter;
    m_rtxmuPendingCompaction.push_back(std::move(batch));
    m_rtxmuBuildsThisFrame.clear();
}

// ---------------------------------------------------------------------------
// ProcessRtxmuCompaction
//   Called at the start of BuildOrUpdate each frame. Advances the deferred
//   RTXMU lifecycle stages for batches whose previous stage has provably
//   completed on the GPU (kRtxmuStageLatency frames, same margin as the
//   deferred-delete queue):
//     pendingCompaction → PopulateCompactionCommandList records the compaction
//       copies (RTXMU reads each size from its mapped readback, suballocates
//       the compacted memory, and GetAccelStructGPUVA switches over — the TLAS
//       emission later this frame picks up the new VA, after the copy).
//     pendingGC → GarbageCollection frees the transient result / scratch /
//       compaction-size memory of compacted BLASes (non-compaction builds keep
//       their memory; RTXMU skips what does not apply).
// ---------------------------------------------------------------------------
void AccelerationStructure::ProcessRtxmuCompaction(ID3D12GraphicsCommandList4 *cmdList)
{
    auto gcIt = m_rtxmuPendingGC.begin();
    while (gcIt != m_rtxmuPendingGC.end())
    {
        if (m_frameCounter - gcIt->frame < kRtxmuStageLatency)
        {
            ++gcIt;
            continue;
        }
        if (!gcIt->ids.empty())
            m_rtxmu->GarbageCollection(gcIt->ids);
        gcIt = m_rtxmuPendingGC.erase(gcIt);
    }

    auto it = m_rtxmuPendingCompaction.begin();
    while (it != m_rtxmuPendingCompaction.end())
    {
        if (m_frameCounter - it->frame < kRtxmuStageLatency)
        {
            ++it;
            continue;
        }
        if (!it->ids.empty())
        {
            m_rtxmu->PopulateCompactionCommandList(cmdList, it->ids);
            RtxmuIdBatch gc;
            gc.ids   = std::move(it->ids);
            gc.frame = m_frameCounter;
            m_rtxmuPendingGC.push_back(std::move(gc));
        }
        it = m_rtxmuPendingCompaction.erase(it);
    }
}

// ---------------------------------------------------------------------------
// GetBLASVA / HasAnyOMM
// ---------------------------------------------------------------------------
D3D12_GPU_VIRTUAL_ADDRESS AccelerationStructure::GetBLASVA(const MeshKey &key) const
{
    auto it = m_blasCache.find(key);
    return (it != m_blasCache.end()) ? ResolveBlasVA(it->second.rtxmuId) : 0;
}

ID3D12Resource* AccelerationStructure::GetTLAS() const
{
    return m_tlas.Get();
}

bool AccelerationStructure::HasAnyOMM() const
{
    // O(1): m_ommInstanceCount tracks active instances that were given baked OMM data
    // (covers both already-built and pending BLASes). Maintained in AddInstance /
    // RemoveInstance / Clear.
    return m_ommInstanceCount > 0;
}

// ---------------------------------------------------------------------------
// BuildTLAS  -  full rebuild of the single persistent TLAS each frame (nvrhi model).
//   * Instance descriptors are built CPU-side into m_dxrInstances, then uploaded
//     through the shared upload pool (g_uploadPool).
//   * Build scratch is suballocated from the shared scratch pool (g_scratchPool).
//   * m_tlas is a single persistent result buffer rebuilt in place; it is reused
//     while the prebuild size fits, otherwise reallocated. Because the rebuild
//     overwrites a buffer the previous frame's traversal may still be reading, a
//     UAV barrier is issued on m_tlas before the build (D3D12 keeps AS buffers in
//     RAYTRACING_ACCELERATION_STRUCTURE state, so Write-after-Read is a UAV barrier,
//     exactly as nvrhi's state tracker emits).
// ---------------------------------------------------------------------------
bool AccelerationStructure::BuildTLAS(ID3D12GraphicsCommandList4 *cmdList, const std::vector<TLASInstanceEntry> &entries)
{
    const uint32_t count = static_cast<uint32_t>(entries.size());

    // ------------------------------------------------------------------
    // 1. Build the instance-desc array CPU-side and upload it via the pool.
    //    (nvrhi fills as->dxrInstances then suballocates the upload — building in
    //     CPU memory and copying once is far faster than per-element PCIe writes.)
    // ------------------------------------------------------------------
    D3D12_GPU_VIRTUAL_ADDRESS instanceGpuVA = 0;
    if (count > 0)
    {
        m_dxrInstances.resize(count);
        for (uint32_t i = 0; i < count; ++i)
        {
            const TLASInstanceEntry &e = entries[i];
            D3D12_RAYTRACING_INSTANCE_DESC &inst = m_dxrInstances[i];
            memset(&inst, 0, sizeof(inst));
            memcpy(inst.Transform, e.transform, 12 * sizeof(float));
            inst.InstanceID = e.instanceID;
            inst.InstanceMask = e.mask;
            inst.InstanceContributionToHitGroupIndex = e.hitGroupContribution;
            inst.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
            inst.AccelerationStructure = e.blasVA;
        }

        const UINT64 uploadBytes = sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * count;
        SharedUploadPool::Allocation instAlloc =
            g_uploadPool.Allocate(uploadBytes, D3D12_RAYTRACING_INSTANCE_DESCS_BYTE_ALIGNMENT);
        if (!instAlloc.IsValid())
        {
            AccelLogf(m_log, kUnityLogTypeError, "BuildTLAS: instance-desc suballoc failed");
            return false;
        }
        memcpy(instAlloc.cpu, m_dxrInstances.data(), uploadBytes);
        instanceGpuVA = instAlloc.gpu;
    }

    // ------------------------------------------------------------------
    // 2. Query prebuild sizes for the instance count (full build each frame).
    // ------------------------------------------------------------------
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
    inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    inputs.NumDescs = count;
    inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputs.InstanceDescs = instanceGpuVA;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo = {};
    m_device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuildInfo);

    const UINT64 neededResult = prebuildInfo.ResultDataMaxSizeInBytes;

    // ------------------------------------------------------------------
    // 3. Single persistent TLAS result buffer: reuse while it fits, else realloc.
    //    Track whether we reused it so we can serialize the in-place rebuild
    //    against the previous frame's traversal reads with a UAV barrier.
    // ------------------------------------------------------------------
    bool reusedTlas = false;
    if (!m_tlas || neededResult > m_tlasResultCapacity)
    {
        if (m_tlas)
            SafeReleaseResource(std::move(m_tlas));

        D3D12_HEAP_PROPERTIES defaultHeap = {};
        defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;

        m_tlas = CreateBuffer(m_device.Get(), neededResult,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                              D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, defaultHeap,
                              L"TLAS_Result");
        m_tlasResultCapacity = neededResult;
        if (!m_tlas)
        {
            AccelLogf(m_log, kUnityLogTypeError, "BuildTLAS: TLAS result buffer allocation failed");
            return false;
        }
    }
    else
    {
        reusedTlas = true;
    }

    // ------------------------------------------------------------------
    // 4. Scratch from the shared scratch pool (full-build size).
    // ------------------------------------------------------------------
    SharedUploadPool::Allocation tlasScratch =
        g_scratchPool.Allocate(prebuildInfo.ScratchDataSizeInBytes, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
    if (!tlasScratch.IsValid())
    {
        AccelLogf(m_log, kUnityLogTypeError, "BuildTLAS: TLAS scratch suballoc failed");
        return false;
    }

    // ------------------------------------------------------------------
    // 5. Serialize the in-place rebuild against the previous frame's reads, then
    //    record the full-build command and a post-build UAV barrier for the dispatch.
    // ------------------------------------------------------------------
    if (reusedTlas)
    {
        D3D12_RESOURCE_BARRIER preBarrier = {};
        preBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        preBarrier.UAV.pResource = m_tlas.Get();
        cmdList->ResourceBarrier(1, &preBarrier);
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
    buildDesc.DestAccelerationStructureData = m_tlas->GetGPUVirtualAddress();
    buildDesc.Inputs = inputs;
    buildDesc.ScratchAccelerationStructureData = tlasScratch.gpu;
    cmdList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);

    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = m_tlas.Get();
    cmdList->ResourceBarrier(1, &barrier);
    return true;
}

// ===========================================================================
// High-level instance management
// ===========================================================================

// ---------------------------------------------------------------------------
// DumpInstances  -  per-frame diagnostic dump.
//
//   For every active slot prints:
//     slot index, userHandle (reverse-looked up from m_handleToSlot),
//     customInstanceID, mask, needsBLAS, submesh count, vb/ib pointers,
//     cached BLAS GPU VA, BLAS refCount, and translation part of the transform.
//
//   Also performs two self-checks:
//     1. handle<->slot map bidirectional consistency
//     2. duplicate (vbPtr, ibPtr) keys across slots (same mesh shared by
//        multiple renderers is legitimate, but logged so it can be correlated
//        with InstanceID() collisions in shader output).
// ---------------------------------------------------------------------------
void AccelerationStructure::DumpInstances(const char *tag) const
{
    const char *t = tag ? tag : "Dump";
    AccelLogf(m_log, kUnityLogTypeLog,
              "[AS][%s] ===== instances: active=%u slots=%zu free=%zu handles=%zu cache=%zu frame=%u =====",
              t, m_activeCount, m_slots.size(), m_freeSlots.size(),
              m_handleToSlot.size(), m_blasCache.size(), g_frameIndex);

    // Build reverse map: slotIndex -> userHandle (expect 1:1 for active slots).
    std::unordered_map<uint32_t, uint32_t> slotToHandle;
    slotToHandle.reserve(m_handleToSlot.size());
    for (const auto &kv : m_handleToSlot)
    {
        auto ins = slotToHandle.emplace(kv.second, kv.first);
        if (!ins.second)
        {
            AccelLogf(m_log, kUnityLogTypeError,
                      "[AS][%s] DUPLICATE slot %u mapped from handles %u and %u",
                      t, kv.second, ins.first->second, kv.first);
        }
    }

    // Track (vb,ib) duplicates across active slots.
    std::unordered_map<MeshKey, uint32_t, MeshKeyHash> seenKey;

    for (uint32_t i = 0; i < m_slots.size(); ++i)
    {
        const InstanceSlot &s = m_slots[i];
        if (!s.active)
            continue;

        uint32_t handle = 0xFFFFFFFFu;
        auto itH = slotToHandle.find(i);
        if (itH != slotToHandle.end())
            handle = itH->second;
        else
        {
            AccelLogf(m_log, kUnityLogTypeError,
                      "[AS][%s] slot %u active but has no handle in m_handleToSlot", t, i);
        }

        // Cross-check: handleToSlot should round-trip.
        if (handle != 0xFFFFFFFFu)
        {
            auto itBack = m_handleToSlot.find(handle);
            if (itBack == m_handleToSlot.end() || itBack->second != i)
            {
                AccelLogf(m_log, kUnityLogTypeError,
                          "[AS][%s] handleToSlot round-trip failed: handle=%u expects slot=%u got=%d",
                          t, handle, i,
                          itBack == m_handleToSlot.end() ? -1 : (int)itBack->second);
            }
        }

        D3D12_GPU_VIRTUAL_ADDRESS blasVA = GetBLASVA(s.meshKey);
        int refCount = 0;
        auto itC = m_blasCache.find(s.meshKey);
        if (itC != m_blasCache.end())
            refCount = itC->second.refCount;

        // Flag duplicated (vb,ib) across slots.
        auto dupIt = seenKey.find(s.meshKey);
        bool dup = (dupIt != seenKey.end());
        if (!dup)
            seenKey.emplace(s.meshKey, i);

        AccelLogf(m_log, kUnityLogTypeLog,
                  "[AS][%s] slot=%-4u handle=%-10u cid=%-6u mask=0x%02X needsBLAS=%d sub=%-3zu "
                  "vb=%p ib=%p blasVA=0x%llx blasRef=%d T=(%.2f,%.2f,%.2f)%s",
                  t, i, handle, s.customInstanceID, s.mask,
                  (int)s.needsBLAS, s.meshInfo.submeshes.size(),
                  (void *)s.meshKey.vbPtr, (void *)s.meshKey.ibPtr,
                  (unsigned long long)blasVA, refCount,
                  s.transform[3], s.transform[7], s.transform[11],
                  dup ? "  [DUP vb+ib shared with earlier slot]" : "");
    }
}

// ---------------------------------------------------------------------------
// Clear
// ---------------------------------------------------------------------------
void AccelerationStructure::Clear()
{
    std::lock_guard<std::mutex> lock(m_stateMutex);
    AccelLogf(m_log, kUnityLogTypeLog, "[AS::Clear] BEGIN - activeSlots=%u, blasCache=%zu", m_activeCount, m_blasCache.size());

    // Release all BLAS ref-counts (deferred GPU delete when they reach 0)
    int releasedBLAS = 0;
    for (const auto &slot : m_slots)
    {
        if (slot.active && !slot.needsBLAS)
        {
            ReleaseBLAS(slot.meshKey);
            releasedBLAS++;
        }
    }
    AccelLogf(m_log, kUnityLogTypeLog, "[AS::Clear] Released %d BLAS entries", releasedBLAS);

    // Move the single TLAS result buffer to pending delete. Instance-desc uploads and
    // TLAS build scratch are pool-owned (g_uploadPool / g_scratchPool) and recycled by
    // the frame fence — nothing to release here.
    if (m_tlas)
        SafeReleaseResource(std::move(m_tlas));
    m_tlasResultCapacity = 0;
    m_dxrInstances.clear();

    // Release remaining BLAS resources (RTXMU memory via deferred remove, OMM
    // buffers via deferred resource release). ReleaseBLAS above only handled
    // entries whose refcount dropped to 0; m_blasCache may still hold entries
    // from shared meshes or ref > 0, and dynamic slots own their BLAS directly.
    for (auto &kv : m_blasCache)
        ReleaseBLASEntryResources(kv.second);
    m_blasCache.clear();

    for (auto &slot : m_slots)
    {
        if (slot.dynamicBlas)
        {
            ReleaseBLASEntryResources(*slot.dynamicBlas);
            slot.dynamicBlas.reset();
        }
    }

    // Abandon the RTXMU lifecycle queues — the scheduled removes above free the
    // memory; compacting / GC'ing removed ids would be invalid.
    m_rtxmuBuildsThisFrame.clear();
    m_rtxmuPendingCompaction.clear();
    m_rtxmuPendingGC.clear();

    // NOTE: We no longer defer deletion of vertex/index buffers from slots because we don't own them.
    // Unity manages these resources, and we only store raw pointers without AddRef.
    // Simply clear the slots - the pointers will be nulled out automatically.
    m_slots.clear();
    m_freeSlots.clear();
    m_handleToSlot.clear();
    m_activeCount = 0;
    m_ommInstanceCount = 0;
    m_tlasEntries.clear();

    AccelLogf(m_log, kUnityLogTypeLog,
              "[AS::Clear] END");
}

// ---------------------------------------------------------------------------
// AddInstance
// ---------------------------------------------------------------------------
bool AccelerationStructure::AddInstance(const NR_AddInstanceDesc &desc)
{
    std::lock_guard<std::mutex> lock(m_stateMutex);
    auto *vb = static_cast<ID3D12Resource *>(desc.vbPtr);
    auto *ib = static_cast<ID3D12Resource *>(desc.ibPtr);
    const auto *submeshes = desc.submeshDescs;
    const uint32_t submeshCount = desc.submeshCount;
    const uint32_t userHandle = desc.instanceHandle;

    if (!vb || !ib || !submeshes || submeshCount == 0)
    {
        AccelLogf(m_log, kUnityLogTypeError, "AddInstance: null buffer or empty submesh list");
        return false;
    }
    if (m_handleToSlot.count(userHandle))
    {
        AccelLogf(m_log, kUnityLogTypeWarning, "AddInstance: handle already registered, ignoring");
        return false;
    }

    const DXGI_FORMAT idxFmt = (desc.indexStride == 4) ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_R16_UINT;

    InstanceSlot slot;
    slot.active = true;
    slot.needsBLAS = true;
    slot.isDynamic = (desc.isDynamic != 0);
    slot.hitGroupContribution = desc.hitGroupContribution;
    slot.mask = 0xFF;
    float identity[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    memcpy(slot.transform, identity, 48);

    // Dynamic (skinned) instances use a per-instance key derived from the handle
    // with the high bit set, so multiple skinned instances sharing the same index
    // buffer do not alias the same BLAS cache entry.
    if (slot.isDynamic)
        slot.meshKey = {static_cast<uintptr_t>(userHandle) | (static_cast<uintptr_t>(1) << 63),
                        reinterpret_cast<uintptr_t>(ib)};
    else
        slot.meshKey = {
            reinterpret_cast<uintptr_t>(vb),
            reinterpret_cast<uintptr_t>(ib),
            HashSubmeshDescs(submeshes, submeshCount, desc.ommDescs)
        };

    // Set descriptive names for Unity-provided buffers to aid debugging
    wchar_t vbName[64], ibName[64];
    if (slot.isDynamic)
    {
        swprintf(vbName, 64, L"Unity_VB_Dynamic_Handle%u", userHandle);
        swprintf(ibName, 64, L"Unity_IB_Dynamic_Handle%u", userHandle);
    }
    else
    {
        swprintf(vbName, 64, L"Unity_VB_Static_%p", (void *)vb);
        swprintf(ibName, 64, L"Unity_IB_Static_%p", (void *)ib);
    }
    vb->SetName(vbName);
    ib->SetName(ibName);
    // AccelLogf(m_log, kUnityLogTypeLog, "[AddInstance] Set names: VB=%p '%ls', IB=%p '%ls', isDynamic=%d", (void *)vb, vbName, (void *)ib, ibName, (int)slot.isDynamic);

    slot.meshInfo.vertexBuffer = vb;
    slot.meshInfo.vertexCount = desc.vertexCount;
    slot.meshInfo.vertexStride = desc.vertexStride;
    slot.meshInfo.indexBuffer = ib;
    slot.meshInfo.indexFormat = idxFmt;

    slot.meshInfo.submeshes.resize(submeshCount);
    for (uint32_t j = 0; j < submeshCount; ++j)
    {
        SubMeshData &md = slot.meshInfo.submeshes[j];
        md.indexCount = submeshes[j].indexCount;
        md.indexByteOffset = submeshes[j].indexByteOffset;
        md.baseVertex = static_cast<INT>(submeshes[j].baseVertex);
        md.flags = submeshes[j].flags;
        md.hasBakedOMM = false;

        if (desc.ommDescs && desc.ommDescs[j].arrayData && desc.ommDescs[j].arrayDataSize > 0)
        {
            const NR_SubmeshOMMDesc &o = desc.ommDescs[j];
            md.hasBakedOMM = true;
            auto &baked = md.ommBaked;
            const uint8_t *pArray = static_cast<const uint8_t *>(o.arrayData);
            baked.arrayData.assign(pArray, pArray + o.arrayDataSize);
            const uint8_t *pDesc = static_cast<const uint8_t *>(o.descArray);
            baked.descArray.assign(pDesc, pDesc + o.descArrayCount * 8u);
            baked.descArrayCount = o.descArrayCount;
            const uint8_t *pIdx = static_cast<const uint8_t *>(o.indexBuffer);
            baked.indexBuffer.assign(pIdx, pIdx + o.indexCount * o.indexStride);
            baked.indexCount = o.indexCount;
            baked.indexStride = o.indexStride;
            baked.indexFormat = (o.indexStride == 4) ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_R16_UINT;
            const uint32_t *src = static_cast<const uint32_t *>(o.histogramFlat);
            baked.histogram.resize(o.histogramCount);
            for (uint32_t h = 0; h < o.histogramCount; ++h, src += 3)
            {
                auto &he = baked.histogram[h];
                he.Count = src[0];
                he.SubdivisionLevel = static_cast<UINT16>(src[1]);
                he.Format = static_cast<D3D12_RAYTRACING_OPACITY_MICROMAP_FORMAT>(src[2]);
            }
        }
    }

    // Track OMM presence for O(1) HasAnyOMM().
    for (const auto &md : slot.meshInfo.submeshes)
    {
        if (md.hasBakedOMM)
        {
            ++m_ommInstanceCount;
            break;
        }
    }

    uint32_t slotIndex;
    if (!m_freeSlots.empty())
    {
        slotIndex = m_freeSlots.back();
        m_freeSlots.pop_back();
        m_slots[slotIndex] = std::move(slot);
    }
    else
    {
        slotIndex = static_cast<uint32_t>(m_slots.size());
        m_slots.push_back(std::move(slot));
    }
    m_handleToSlot[userHandle] = slotIndex;
    ++m_activeCount;
    return true;
}

// ---------------------------------------------------------------------------
// RemoveInstance
// ---------------------------------------------------------------------------
void AccelerationStructure::RemoveInstance(uint32_t handle)
{
    std::lock_guard<std::mutex> lock(m_stateMutex);
    auto it = m_handleToSlot.find(handle);
    if (it == m_handleToSlot.end())
        return;

    const uint32_t slotIndex = it->second;
    InstanceSlot &slot = m_slots[slotIndex];
    if (!slot.active)
        return;

    if (!slot.needsBLAS)
        ReleaseBLAS(slot.meshKey);

    // Release the persistent dynamic BLAS (RTXMU remove + OMM release are both
    // deferred internally, safe after 3 frames).
    if (slot.dynamicBlas)
    {
        ReleaseBLASEntryResources(*slot.dynamicBlas);
        slot.dynamicBlas.reset();
    }

    // Keep the OMM-presence counter in sync (mirror of AddInstance).
    for (const auto &md : slot.meshInfo.submeshes)
    {
        if (md.hasBakedOMM)
        {
            --m_ommInstanceCount;
            break;
        }
    }

    // NOTE: We no longer defer deletion of vertex/index buffers because we don't own them.
    // Unity manages these resources, and we only store raw pointers without AddRef.
    slot.active = false;
    slot.needsBLAS = false;
    slot.meshInfo.submeshes.clear();
    slot.meshInfo.vertexBuffer = nullptr;
    slot.meshInfo.indexBuffer = nullptr;
    m_freeSlots.push_back(slotIndex);
    m_handleToSlot.erase(it);
    --m_activeCount;
}

// ---------------------------------------------------------------------------
// UpdateDynamicVertexBuffer
//   For SkinnedMeshRenderer instances: swap in the new GPU vertex buffer
//   produced by Unity's skinning pass, discard the stale BLAS (deferred GPU
//   delete after 3 frames), and schedule a rebuild for next BuildOrUpdate.
// ---------------------------------------------------------------------------
void AccelerationStructure::UpdateDynamicVertexBuffer(uint32_t handle, void *vbPtr)
{
    std::lock_guard<std::mutex> lock(m_stateMutex);
    auto it = m_handleToSlot.find(handle);
    if (it == m_handleToSlot.end())
        return;

    InstanceSlot &slot = m_slots[it->second];
    if (!slot.active || !slot.isDynamic)
        return;

    auto *newVb = static_cast<ID3D12Resource *>(vbPtr);
    if (!newVb)
        return;

    //wchar_t vbName[64];
    //swprintf(vbName, 64, L"Unity_VB_Dynamic_Handle%u_Updated", handle);
    //newVb->SetName(vbName);
    //AccelLogf(m_log, kUnityLogTypeLog, "[UpdateDynamicVB] Handle=%u, oldVB=%p, newVB=%p '%ls'", handle, (void *)slot.meshInfo.vertexBuffer, (void *)newVb, vbName);

    slot.meshInfo.vertexBuffer = newVb;
    slot.needsBLAS = true;
}

// ---------------------------------------------------------------------------
// SetInstanceTransform / SetInstanceMask
// ---------------------------------------------------------------------------
void AccelerationStructure::SetInstanceTransform(uint32_t handle, const float transform[12])
{
    std::lock_guard<std::mutex> lock(m_stateMutex);
    auto it = m_handleToSlot.find(handle);
    if (it == m_handleToSlot.end())
        return;
    InstanceSlot &slot = m_slots[it->second];
    if (!slot.active)
        return;
    memcpy(slot.transform, transform, 48);
}

void AccelerationStructure::SetInstanceMask(uint32_t handle, uint8_t mask)
{
    std::lock_guard<std::mutex> lock(m_stateMutex);
    auto it = m_handleToSlot.find(handle);
    if (it == m_handleToSlot.end())
        return;
    InstanceSlot &slot = m_slots[it->second];
    if (!slot.active || slot.mask == mask)
        return;
    slot.mask = mask;
}

void AccelerationStructure::SetInstanceID(uint32_t handle, uint32_t id)
{
    std::lock_guard<std::mutex> lock(m_stateMutex);
    auto it = m_handleToSlot.find(handle);
    if (it == m_handleToSlot.end())
        return;
    InstanceSlot &slot = m_slots[it->second];
    if (!slot.active || slot.customInstanceID == id)
        return;
    slot.customInstanceID = id;
}

void AccelerationStructure::SetInstanceOrderIndex(uint32_t handle, uint32_t order)
{
    std::lock_guard<std::mutex> lock(m_stateMutex);
    auto it = m_handleToSlot.find(handle);
    if (it == m_handleToSlot.end())
        return;
    InstanceSlot &slot = m_slots[it->second];
    if (!slot.active)
        return;
    slot.tlasOrder = order;
}

void AccelerationStructure::SetInstanceHitGroupContribution(uint32_t handle, uint32_t contribution)
{
    std::lock_guard<std::mutex> lock(m_stateMutex);
    auto it = m_handleToSlot.find(handle);
    if (it == m_handleToSlot.end())
        return;
    InstanceSlot &slot = m_slots[it->second];
    if (!slot.active)
        return;
    slot.hitGroupContribution = contribution;
}

// ---------------------------------------------------------------------------
// BuildOrUpdate  -  called every frame before ray dispatch.
// ---------------------------------------------------------------------------
bool AccelerationStructure::BuildOrUpdate(ID3D12GraphicsCommandList4 *cmdList)
{
    std::lock_guard<std::mutex> lock(m_stateMutex);

    // Per-frame diagnostic dump (one line per active instance).
    // DumpInstances("BuildOrUpdate");

    // -------------------------------------------------------------------
    // Step 0: Advance frame counter and compact any ready pending BLASes
    // -------------------------------------------------------------------
    ++m_frameCounter;
    ProcessRtxmuCompaction(cmdList);

    // -------------------------------------------------------------------
    // Step A: Build any pending new BLASes.
    //   Only expensive first-time/static full builds are throttled, to spread a
    //   mass scene load across frames and avoid a GPU TDR. Dynamic (skinned) BLAS
    //   refits are cheap PERFORM_UPDATE passes that must run every frame to track
    //   animation, so they are never throttled. Statics skipped this frame keep
    //   needsBLAS=true and are retried next frame.
    // -------------------------------------------------------------------
    static constexpr int kMaxStaticBLASBuildsPerFrame = 256;
    bool anyNewBLAS = false;
    int staticBuildsThisFrame = 0;
    for (auto &slot : m_slots)
    {
        if (!slot.active || !slot.needsBLAS)
            continue;

        if (!slot.isDynamic && staticBuildsThisFrame >= kMaxStaticBLASBuildsPerFrame)
            continue; // budget exhausted; retry this static build next frame

        if (!EnsureBLAS(cmdList, slot))
        {
            AccelLogf(m_log, kUnityLogTypeError, "BuildOrUpdate: EnsureBLAS failed");
            return false;
        }
        if (!slot.isDynamic)
        {
            slot.needsBLAS = false;
            ++staticBuildsThisFrame;
        }
        anyNewBLAS = true;
    }
    // Emit a single global UAV barrier covering all newly-built BLASes.
    if (anyNewBLAS)
    {
        D3D12_RESOURCE_BARRIER blasBarrier = {};
        blasBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        blasBarrier.UAV.pResource = nullptr; // nullptr = all UAV resources
        cmdList->ResourceBarrier(1, &blasBarrier);
    }

    // Record the compaction-size readback copies for this frame's RTXMU builds
    // (must follow the global UAV barrier above so the builds are complete).
    FlushRtxmuBuilds(cmdList);

    // Emit TLAS instances in caller-specified order (SetInstanceOrderIndex), not raw slot
    // order: freed slots are reused, so slot order diverges from registration order after
    // any remove. Shaders that index per-instance buffers by InstanceIndex() (RTXPT
    // t_InstanceData) rely on this order matching their CPU-side array. Slots without an
    // assigned order keep the 0xFFFFFFFF default and the stable sort emits them last, in
    // slot order — identical to the legacy behavior for callers that never set one.
    m_orderedSlotScratch.clear();
    for (uint32_t s = 0; s < static_cast<uint32_t>(m_slots.size()); ++s)
        if (m_slots[s].active)
            m_orderedSlotScratch.push_back(s);
    std::stable_sort(m_orderedSlotScratch.begin(), m_orderedSlotScratch.end(),
                     [this](uint32_t a, uint32_t b) { return m_slots[a].tlasOrder < m_slots[b].tlasOrder; });

    m_tlasEntries.clear();
    for (uint32_t s : m_orderedSlotScratch)
    {
        const InstanceSlot &slot = m_slots[s];
        TLASInstanceEntry e;
        // Resolve the BLAS VA per frame — RTXMU moves the BLAS to its compacted
        // location once compaction is recorded, and the id tracks that.
        e.blasVA = ResolveBlasVA(slot.blasRtxmuId);
        e.instanceID = slot.customInstanceID;
        e.mask = slot.mask;
        memcpy(e.transform, slot.transform, 48);
        e.hitGroupContribution = slot.hitGroupContribution;
        e.submeshCount = static_cast<uint32_t>(slot.meshInfo.submeshes.size());
        m_tlasEntries.push_back(e);
    }

    if (!BuildTLAS(cmdList, m_tlasEntries))
    {
        AccelLogf(m_log, kUnityLogTypeError, "BuildOrUpdate: BuildTLAS failed");
        return false;
    }
    return true;
}