#include "AccelerationStructure.h"
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <algorithm>

// ---------------------------------------------------------------------------
// Internal buffer helper
// ---------------------------------------------------------------------------
static ComPtr<ID3D12Resource> CreateBuffer(
    ID3D12Device* device,
    UINT64 size,
    D3D12_RESOURCE_FLAGS flags,
    D3D12_RESOURCE_STATES initialState,
    const D3D12_HEAP_PROPERTIES& heapProps)
{
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment          = 0;
    desc.Width              = size ? size : 1;
    desc.Height             = 1;
    desc.DepthOrArraySize   = 1;
    desc.MipLevels          = 1;
    desc.Format             = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count   = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags              = flags;
    ComPtr<ID3D12Resource> resource;
    HRESULT hr = device->CreateCommittedResource(
        &heapProps, D3D12_HEAP_FLAG_NONE,
        &desc, initialState,
        nullptr, IID_PPV_ARGS(&resource));
    return SUCCEEDED(hr) ? resource : nullptr;
}

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------
static void AccelLogf(IUnityLog* log, UnityLogType type, const char* fmt, ...)
{
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    if (log)
        log->Log(type, buf, __FILE__, __LINE__);
    else
        printf("[AccelStructure] %s\n", buf);
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
AccelerationStructure::AccelerationStructure(ID3D12Device5* device, IUnityLog* log)
    : m_device(device)
    , m_log(log)
{
}

// ---------------------------------------------------------------------------
// TickDeferredDeletes  -  age GPU resource delete queue each frame.
// ---------------------------------------------------------------------------
void AccelerationStructure::TickDeferredDeletes()
{
    for (auto it = m_pendingDeletes.begin(); it != m_pendingDeletes.end(); )
    {
        if (--it->framesRemaining <= 0)
            it = m_pendingDeletes.erase(it);
        else
            ++it;
    }
}

// ---------------------------------------------------------------------------
// BuildOMMForSubmesh
//   Uploads OMM data to GPU and records a BuildRaytracingAccelerationStructure
//   command for the OMM Array AS into cmdList.
// ---------------------------------------------------------------------------
bool AccelerationStructure::BuildOMMForSubmesh(
    ID3D12GraphicsCommandList4* cmdList,
    BLASEntry& entry, size_t subIdx, const MeshData& mesh)
{
    const MeshData::OMMBakedData& baked = mesh.ommBaked;
    AccelLogf(m_log, kUnityLogTypeLog,
        "[OMM] BuildOMMForSubmesh[%zu]: arrayData=%zu bytes, descs=%u, indices=%u",
        subIdx, baked.arrayData.size(), baked.descArrayCount, baked.indexCount);

    if (baked.histogram.empty())
    {
        AccelLogf(m_log, kUnityLogTypeError,
            "[OMM] BuildOMMForSubmesh[%zu]: histogram is empty, cannot build OMM array", subIdx);
        return false;
    }

    D3D12_HEAP_PROPERTIES uploadHeap = {};
    uploadHeap.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_HEAP_PROPERTIES defaultHeap = {};
    defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;
    void* mapped = nullptr;

    // 1. Upload raw OMM array data
    auto arrayDataBuf = CreateBuffer(m_device.Get(),
        !baked.arrayData.empty() ? (UINT64)baked.arrayData.size() : 1,
        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, uploadHeap);
    if (!arrayDataBuf)
    {
        AccelLogf(m_log, kUnityLogTypeError,
            "[OMM] BuildOMMForSubmesh[%zu]: arrayData buf alloc failed", subIdx);
        return false;
    }
    arrayDataBuf->Map(0, nullptr, &mapped);
    if (!baked.arrayData.empty()) memcpy(mapped, baked.arrayData.data(), baked.arrayData.size());
    arrayDataBuf->Unmap(0, nullptr);

    // 2. Upload OMM desc array
    UINT64 descArrayBytes = (UINT64)baked.descArrayCount * sizeof(D3D12_RAYTRACING_OPACITY_MICROMAP_DESC);
    auto descArrayBuf = CreateBuffer(m_device.Get(),
        descArrayBytes ? descArrayBytes : 1,
        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, uploadHeap);
    if (!descArrayBuf)
    {
        AccelLogf(m_log, kUnityLogTypeError,
            "[OMM] BuildOMMForSubmesh[%zu]: descArray buf alloc failed", subIdx);
        return false;
    }
    descArrayBuf->Map(0, nullptr, &mapped);
    if (!baked.descArray.empty())
    {
        size_t copyBytes = (std::min)((size_t)descArrayBytes, baked.descArray.size());
        memcpy(mapped, baked.descArray.data(), copyBytes);
    }
    descArrayBuf->Unmap(0, nullptr);

    // 3. Upload OMM index buffer
    UINT ommIdxBytes = baked.indexCount * baked.indexStride;
    auto ommIdxBuf = CreateBuffer(m_device.Get(),
        ommIdxBytes ? ommIdxBytes : 1,
        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, uploadHeap);
    if (!ommIdxBuf)
    {
        AccelLogf(m_log, kUnityLogTypeError,
            "[OMM] BuildOMMForSubmesh[%zu]: OMM index buf alloc failed", subIdx);
        return false;
    }
    ommIdxBuf->Map(0, nullptr, &mapped);
    if (ommIdxBytes) memcpy(mapped, baked.indexBuffer.data(), ommIdxBytes);
    ommIdxBuf->Unmap(0, nullptr);

    // 4. Build OMM Array AS
    D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_DESC ommArrayDesc = {};
    ommArrayDesc.NumOmmHistogramEntries        = (UINT)baked.histogram.size();
    ommArrayDesc.pOmmHistogram                 = baked.histogram.data();
    ommArrayDesc.InputBuffer                   = arrayDataBuf->GetGPUVirtualAddress();
    ommArrayDesc.PerOmmDescs.StartAddress      = descArrayBuf->GetGPUVirtualAddress();
    ommArrayDesc.PerOmmDescs.StrideInBytes     = sizeof(D3D12_RAYTRACING_OPACITY_MICROMAP_DESC);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS ommInputs = {};
    ommInputs.Type                      = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_ARRAY;
    ommInputs.Flags                     = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    ommInputs.NumDescs                  = 1;
    ommInputs.DescsLayout               = D3D12_ELEMENTS_LAYOUT_ARRAY;
    ommInputs.pOpacityMicromapArrayDesc = &ommArrayDesc;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo = {};
    m_device->GetRaytracingAccelerationStructurePrebuildInfo(&ommInputs, &prebuildInfo);

    entry.ommArrayScratch[subIdx] = CreateBuffer(m_device.Get(),
        prebuildInfo.ScratchDataSizeInBytes,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, defaultHeap);
    entry.ommArrays[subIdx] = CreateBuffer(m_device.Get(),
        prebuildInfo.ResultDataMaxSizeInBytes,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, defaultHeap);
    if (!entry.ommArrayScratch[subIdx] || !entry.ommArrays[subIdx])
    {
        AccelLogf(m_log, kUnityLogTypeError,
            "[OMM] BuildOMMForSubmesh[%zu]: OMM array buf alloc failed", subIdx);
        return false;
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
    buildDesc.DestAccelerationStructureData    = entry.ommArrays[subIdx]->GetGPUVirtualAddress();
    buildDesc.Inputs                           = ommInputs;
    buildDesc.ScratchAccelerationStructureData = entry.ommArrayScratch[subIdx]->GetGPUVirtualAddress();
    cmdList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);

    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = entry.ommArrays[subIdx].Get();
    cmdList->ResourceBarrier(1, &barrier);

    entry.ommIndexBuffers[subIdx]     = std::move(ommIdxBuf);
    entry.ommDescArrayBuffers[subIdx] = std::move(descArrayBuf);
    entry.ommArrayDataBuffers[subIdx] = std::move(arrayDataBuf);
    entry.ommIndexFormats[subIdx]     = baked.indexFormat;
    entry.ommIndexStrides[subIdx]     = baked.indexStride;

    AccelLogf(m_log, kUnityLogTypeLog,
        "[OMM] BuildOMMForSubmesh[%zu]: OMM Array AS recorded on cmdlist", subIdx);
    return true;
}

// ---------------------------------------------------------------------------
// EnsureBLAS
//   Cache hit:  increment refCount, return immediately (no GPU work).
//   Cache miss: build BLAS (+ OMM) and cache it.
// ---------------------------------------------------------------------------
bool AccelerationStructure::EnsureBLAS(
    ID3D12GraphicsCommandList4* cmdList,
    const MeshKey& key, const InstanceDef& def)
{
    auto it = m_blasCache.find(key);
    if (it != m_blasCache.end())
    {
        it->second.refCount++;
        AccelLogf(m_log, kUnityLogTypeLog,
            "[BLAS] AddRef  vb=%p refCount=%d", (void*)key.vbPtr, it->second.refCount);
        return true;
    }

    BLASEntry entry;
    const size_t subCount = def.submeshes.size();
    if (subCount == 0)
    {
        AccelLogf(m_log, kUnityLogTypeError, "EnsureBLAS: instance has no submeshes");
        return false;
    }

    entry.ommArrays.resize(subCount);
    entry.ommArrayScratch.resize(subCount);
    entry.ommIndexBuffers.resize(subCount);
    entry.ommDescArrayBuffers.resize(subCount);
    entry.ommArrayDataBuffers.resize(subCount);
    entry.ommIndexFormats.resize(subCount, DXGI_FORMAT_R16_UINT);
    entry.ommIndexStrides.resize(subCount, 2);

    const MeshData& firstSub = def.submeshes[0];
    std::vector<D3D12_RAYTRACING_GEOMETRY_DESC>             geomDescs(subCount);
    std::vector<D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC>   ommTriDescs(subCount);
    std::vector<D3D12_RAYTRACING_GEOMETRY_OMM_LINKAGE_DESC> ommLinkages(subCount);
    bool instanceHasOMM = false;

    for (size_t j = 0; j < subCount; ++j)
    {
        const MeshData& sub = def.submeshes[j];
        D3D12_RAYTRACING_GEOMETRY_DESC& geomDesc = geomDescs[j];
        geomDesc = {};

        bool subUseOMM = false;
        if (sub.hasBakedOMM)
        {
            subUseOMM = BuildOMMForSubmesh(cmdList, entry, j, sub);
            if (subUseOMM)
            {
                entry.anyOMM = true;
                AccelLogf(m_log, kUnityLogTypeLog, "EnsureBLAS: submesh[%zu] OMM active", j);
            }
            else
            {
                AccelLogf(m_log, kUnityLogTypeWarning,
                    "EnsureBLAS: submesh[%zu] OMM build failed, falling back to opaque", j);
            }
        }

        if (subUseOMM)
        {
            geomDesc.Type  = D3D12_RAYTRACING_GEOMETRY_TYPE_OMM_TRIANGLES;
            geomDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;

            D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC& td = ommTriDescs[j];
            td = {};
            td.VertexBuffer.StartAddress  = firstSub.vertexBuffer->GetGPUVirtualAddress();
            td.VertexBuffer.StrideInBytes = firstSub.vertexStride;
            td.VertexCount                = firstSub.vertexCount;
            td.VertexFormat               = DXGI_FORMAT_R32G32B32_FLOAT;
            td.IndexBuffer                = firstSub.indexBuffer
                ? firstSub.indexBuffer->GetGPUVirtualAddress() + sub.indexByteOffset : 0;
            td.IndexCount                 = firstSub.indexBuffer ? sub.indexCount : 0;
            td.IndexFormat                = firstSub.indexBuffer ? sub.indexFormat : DXGI_FORMAT_UNKNOWN;
            td.Transform3x4               = 0;

            D3D12_RAYTRACING_GEOMETRY_OMM_LINKAGE_DESC& ol = ommLinkages[j];
            ol = {};
            ol.OpacityMicromapArray                     = entry.ommArrays[j]->GetGPUVirtualAddress();
            ol.OpacityMicromapBaseLocation              = 0;
            ol.OpacityMicromapIndexBuffer.StartAddress  = entry.ommIndexBuffers[j]->GetGPUVirtualAddress();
            ol.OpacityMicromapIndexBuffer.StrideInBytes = entry.ommIndexStrides[j];
            ol.OpacityMicromapIndexFormat               = entry.ommIndexFormats[j];

            geomDesc.OmmTriangles.pTriangles  = &td;
            geomDesc.OmmTriangles.pOmmLinkage = &ol;
            instanceHasOMM = true;
        }
        else
        {
            geomDesc.Type  = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
            geomDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
            geomDesc.Triangles.VertexBuffer.StartAddress  = firstSub.vertexBuffer->GetGPUVirtualAddress();
            geomDesc.Triangles.VertexBuffer.StrideInBytes = firstSub.vertexStride;
            geomDesc.Triangles.VertexCount                = firstSub.vertexCount;
            geomDesc.Triangles.VertexFormat               = DXGI_FORMAT_R32G32B32_FLOAT;
            geomDesc.Triangles.IndexBuffer                = firstSub.indexBuffer
                ? firstSub.indexBuffer->GetGPUVirtualAddress() + sub.indexByteOffset : 0;
            geomDesc.Triangles.IndexCount                 = firstSub.indexBuffer ? sub.indexCount : 0;
            geomDesc.Triangles.IndexFormat                = firstSub.indexBuffer ? sub.indexFormat : DXGI_FORMAT_UNKNOWN;
            geomDesc.Triangles.Transform3x4               = 0;
        }
    }

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS blasFlags =
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    if (instanceHasOMM)
        blasFlags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_DISABLE_OMMS;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
    inputs.Type           = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    inputs.Flags          = blasFlags;
    inputs.NumDescs       = static_cast<UINT>(subCount);
    inputs.DescsLayout    = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputs.pGeometryDescs = geomDescs.data();

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo = {};
    m_device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuildInfo);

    D3D12_HEAP_PROPERTIES defaultHeap = {};
    defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;

    entry.blasScratch = CreateBuffer(m_device.Get(),
        prebuildInfo.ScratchDataSizeInBytes,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON, defaultHeap);
    entry.blas = CreateBuffer(m_device.Get(),
        prebuildInfo.ResultDataMaxSizeInBytes,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, defaultHeap);
    if (!entry.blasScratch || !entry.blas)
    {
        AccelLogf(m_log, kUnityLogTypeError, "EnsureBLAS: buffer allocation failed");
        return false;
    }

    // Use RequestResourceState to insert an explicit COMMON→NON_PIXEL_SHADER_RESOURCE barrier
    // into Unity's current command list BEFORE the BLAS build. This prevents Unity's parallel
    // (async) command lists from recording a barrier that assumes COMMON state for these
    // buffers — which would fail validation when submitted after the implicit promotion.
    if (m_d3d12v8)
    {
        m_d3d12v8->RequestResourceState(
            firstSub.vertexBuffer.Get(),
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (firstSub.indexBuffer)
            m_d3d12v8->RequestResourceState(
                firstSub.indexBuffer.Get(),
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
    buildDesc.DestAccelerationStructureData    = entry.blas->GetGPUVirtualAddress();
    buildDesc.Inputs                           = inputs;
    buildDesc.ScratchAccelerationStructureData = entry.blasScratch->GetGPUVirtualAddress();
    cmdList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);

    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = entry.blas.Get();
    cmdList->ResourceBarrier(1, &barrier);

    // Scratch is only needed during the build command; release it after 3 frames.
    {
        PendingDelete pd;
        pd.framesRemaining = 3;
        pd.resources.push_back(std::move(entry.blasScratch));
        m_pendingDeletes.push_back(std::move(pd));
    }

    entry.refCount = 1;
    // AccelLogf(m_log, kUnityLogTypeLog, "[BLAS] Add     vb=%p refCount=1 (new, anyOMM=%d)",
    //           (void*)key.vbPtr, (int)entry.anyOMM);
    m_blasCache.emplace(key, std::move(entry));
    return true;
}

// ---------------------------------------------------------------------------
// ReleaseBLAS
// ---------------------------------------------------------------------------
void AccelerationStructure::ReleaseBLAS(const MeshKey& key)
{
    auto it = m_blasCache.find(key);
    if (it == m_blasCache.end()) return;

    if (--it->second.refCount > 0)
    {
        // AccelLogf(m_log, kUnityLogTypeLog, "[BLAS] Release vb=%p refCount=%d (still alive)",
        //           (void*)key.vbPtr, it->second.refCount);
        return;
    }

    // AccelLogf(m_log, kUnityLogTypeLog,
    //     "[BLAS] Release vb=%p refCount=0 \u2192 deferred GPU delete", (void*)key.vbPtr);

    PendingDelete pd;
    pd.framesRemaining = 3;
    BLASEntry& e = it->second;
    pd.resources.push_back(std::move(e.blas));
    // e.blasScratch was already moved to pending delete at build time
    for (auto& r : e.ommArrays)           if (r) pd.resources.push_back(std::move(r));
    for (auto& r : e.ommArrayScratch)     if (r) pd.resources.push_back(std::move(r));
    for (auto& r : e.ommIndexBuffers)     if (r) pd.resources.push_back(std::move(r));
    for (auto& r : e.ommDescArrayBuffers) if (r) pd.resources.push_back(std::move(r));
    for (auto& r : e.ommArrayDataBuffers) if (r) pd.resources.push_back(std::move(r));
    m_pendingDeletes.push_back(std::move(pd));
    m_blasCache.erase(it);
}

// ---------------------------------------------------------------------------
// GetBLASVA / HasAnyOMM
// ---------------------------------------------------------------------------
D3D12_GPU_VIRTUAL_ADDRESS AccelerationStructure::GetBLASVA(const MeshKey& key) const
{
    auto it = m_blasCache.find(key);
    return (it != m_blasCache.end() && it->second.blas)
        ? it->second.blas->GetGPUVirtualAddress() : 0;
}

bool AccelerationStructure::HasAnyOMM() const
{
    for (const auto& kv : m_blasCache)
        if (kv.second.anyOMM) return true;
    // Also check pending slots not yet built
    for (const auto& slot : m_slots)
        if (slot.active)
            for (const auto& sub : slot.mesh.submeshes)
                if (sub.hasBakedOMM) return true;
    return false;
}

// ---------------------------------------------------------------------------
// BuildTLAS  -  (re)build of the TLAS with capacity-aware buffer reuse.
//   * m_instanceDesc is reused when count fits within the existing allocation.
//   * m_tlas / tlasScratch (per-frame) are reused when the prebuild sizes fit within the
//     existing capacities.  Only when capacity is exceeded are old buffers
//     moved to the deferred-delete queue and new (larger) ones allocated.
// ---------------------------------------------------------------------------
bool AccelerationStructure::BuildTLAS(
    ID3D12GraphicsCommandList4* cmdList,
    const std::vector<TLASInstanceEntry>& entries)
{
    const uint32_t count = static_cast<uint32_t>(entries.size());
    TLASFrameResources& res = m_tlasResources[m_frameIndex];

    // ------------------------------------------------------------------
    // 1. Instance-desc buffer: reuse current frame's slot if capacity is
    //    sufficient; otherwise defer-delete it and allocate a larger one.
    //    The previous frame's slot is untouched — GPU may still read it.
    // ------------------------------------------------------------------
    if (count > 0)
    {
        if (count > res.instanceDescCapacity || !res.instanceDesc)
        {
            if (res.instanceDesc)
            {
                PendingDelete pd;
                pd.framesRemaining = 3;
                pd.resources.push_back(std::move(res.instanceDesc));
                res.mappedInstanceDesc   = nullptr;
                res.instanceDescCapacity = 0;
                m_pendingDeletes.push_back(std::move(pd));
            }

            D3D12_HEAP_PROPERTIES uploadHeap = {};
            uploadHeap.Type = D3D12_HEAP_TYPE_UPLOAD;

            // Over-allocate by 1.5x to reduce future reallocations.
            const uint32_t newCapacity = static_cast<uint32_t>(count * 3 / 2) + 1;
            UINT64 instanceDescSize = sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * newCapacity;
            res.instanceDesc = CreateBuffer(m_device.Get(), instanceDescSize,
                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, uploadHeap);
            if (!res.instanceDesc)
            {
                AccelLogf(m_log, kUnityLogTypeError, "BuildTLAS: instance desc buffer allocation failed");
                return false;
            }

            // Persistently map so subsequent frames can write without re-mapping.
            res.instanceDesc->Map(0, nullptr, &res.mappedInstanceDesc);
            res.instanceDescCapacity = newCapacity;
        }

        // Write instance descriptors into this frame's mapped buffer.
        // The previous frame's buffer is still safe because GPU reads that slot.
        auto* descs = static_cast<D3D12_RAYTRACING_INSTANCE_DESC*>(res.mappedInstanceDesc);
        for (uint32_t i = 0; i < count; ++i)
        {
            const TLASInstanceEntry& e = entries[i];
            D3D12_RAYTRACING_INSTANCE_DESC& inst = descs[i];
            memset(&inst, 0, sizeof(inst));
            memcpy(inst.Transform, e.transform, 12 * sizeof(float));
            inst.InstanceID                          = e.instanceID;
            inst.InstanceMask                        = e.mask;
            inst.InstanceContributionToHitGroupIndex = 0;
            inst.Flags                               = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
            inst.AccelerationStructure               = e.blasVA;
        }

        m_lastTransforms.resize(count * 12);
        for (uint32_t i = 0; i < count; ++i)
            memcpy(m_lastTransforms.data() + i * 12, entries[i].transform, 12 * sizeof(float));
    }
    else
    {
        // count == 0: no instance-desc buffer needed; keep any existing allocation.
        m_lastTransforms.clear();
    }

    // ------------------------------------------------------------------
    // 2. Query prebuild sizes for the new instance count.
    // ------------------------------------------------------------------
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
    inputs.Type          = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    inputs.Flags         = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
                         | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
    inputs.NumDescs      = count;
    inputs.DescsLayout   = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputs.InstanceDescs = (count > 0 && res.instanceDesc) ? res.instanceDesc->GetGPUVirtualAddress() : 0;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo = {};
    m_device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuildInfo);

    // Scratch must cover both full-build and update-refit sizes.
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputsForUpdate = inputs;
    inputsForUpdate.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE
                          | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfoUpdate = {};
    m_device->GetRaytracingAccelerationStructurePrebuildInfo(&inputsForUpdate, &prebuildInfoUpdate);

    const UINT64 neededResult  = prebuildInfo.ResultDataMaxSizeInBytes;
    const UINT64 neededScratch = max(prebuildInfo.ScratchDataSizeInBytes,
                                     prebuildInfoUpdate.UpdateScratchDataSizeInBytes);

    // ------------------------------------------------------------------
    // 3a. TLAS result buffer: reuse current frame's slot or reallocate.
    //     Only this frame's slot is touched; GPU is busy with the other.
    // ------------------------------------------------------------------
    if (!res.tlas || neededResult > res.tlasResultCapacity)
    {
        if (res.tlas)
        {
            PendingDelete pd;
            pd.framesRemaining = 3;
            pd.resources.push_back(std::move(res.tlas));
            m_pendingDeletes.push_back(std::move(pd));
        }

        D3D12_HEAP_PROPERTIES defaultHeap = {};
        defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;

        res.tlas = CreateBuffer(m_device.Get(), neededResult,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, defaultHeap);
        res.tlasResultCapacity = neededResult;

        if (!res.tlas)
        {
            AccelLogf(m_log, kUnityLogTypeError, "BuildTLAS: TLAS result buffer allocation failed");
            return false;
        }
    }

    // ------------------------------------------------------------------
    // 3b. Scratch buffer: per-frame slot — each frame owns its scratch so
    //     concurrent GPU work on different frames never aliases the buffer.
    // ------------------------------------------------------------------
    if (!res.tlasScratch || neededScratch > res.tlasScratchCapacity)
    {
        if (res.tlasScratch)
        {
            PendingDelete pd;
            pd.framesRemaining = 3;
            pd.resources.push_back(std::move(res.tlasScratch));
            m_pendingDeletes.push_back(std::move(pd));
        }

        D3D12_HEAP_PROPERTIES defaultHeap = {};
        defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;

        res.tlasScratch = CreateBuffer(m_device.Get(), neededScratch,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON, defaultHeap);
        res.tlasScratchCapacity = neededScratch;

        if (!res.tlasScratch)
        {
            AccelLogf(m_log, kUnityLogTypeError, "BuildTLAS: TLAS scratch buffer allocation failed");
            return false;
        }
    }

    // ------------------------------------------------------------------
    // 4. Record the full-build command.
    // ------------------------------------------------------------------
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
    buildDesc.DestAccelerationStructureData    = res.tlas->GetGPUVirtualAddress();
    buildDesc.Inputs                           = inputs;
    buildDesc.ScratchAccelerationStructureData = res.tlasScratch->GetGPUVirtualAddress();
    cmdList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);

    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = res.tlas.Get();
    cmdList->ResourceBarrier(1, &barrier);
    return true;
}

// ===========================================================================
// High-level instance management
// ===========================================================================

// ---------------------------------------------------------------------------
// Clear
// ---------------------------------------------------------------------------
void AccelerationStructure::Clear()
{
    // Release all BLAS ref-counts (deferred GPU delete when they reach 0)
    for (const auto& slot : m_slots)
    {
        if (slot.active && !slot.needsBLAS)
            ReleaseBLAS(slot.meshKey);
    }

    // Move TLAS resources to pending delete (both frame slots + shared scratch).
    {
        PendingDelete pd;
        pd.framesRemaining = 3;
        for (auto& r : m_tlasResources)
        {
            if (r.instanceDesc) pd.resources.push_back(std::move(r.instanceDesc));
            if (r.tlas)         pd.resources.push_back(std::move(r.tlas));
            if (r.tlasScratch)  pd.resources.push_back(std::move(r.tlasScratch));
            r.mappedInstanceDesc   = nullptr;
            r.instanceDescCapacity = 0;
            r.tlasResultCapacity   = 0;
            r.tlasScratchCapacity  = 0;
        }
        if (!pd.resources.empty())
            m_pendingDeletes.push_back(std::move(pd));
    }
    m_frameIndex = 0;

    // Move remaining BLAS resources to deferred-delete queue
    // (ReleaseBLAS only handles slots whose BLAS was already built;
    //  m_blasCache may still hold entries from shared meshes or ref > 0)
    {
        PendingDelete pd;
        pd.framesRemaining = 3;
        for (auto& kv : m_blasCache)
        {
            BLASEntry& e = kv.second;
            if (e.blas) pd.resources.push_back(std::move(e.blas));
            if (e.blasScratch) pd.resources.push_back(std::move(e.blasScratch));
            for (auto& r : e.ommArrays)           if (r) pd.resources.push_back(std::move(r));
            for (auto& r : e.ommArrayScratch)     if (r) pd.resources.push_back(std::move(r));
            for (auto& r : e.ommIndexBuffers)     if (r) pd.resources.push_back(std::move(r));
            for (auto& r : e.ommDescArrayBuffers) if (r) pd.resources.push_back(std::move(r));
            for (auto& r : e.ommArrayDataBuffers) if (r) pd.resources.push_back(std::move(r));
        }
        if (!pd.resources.empty())
            m_pendingDeletes.push_back(std::move(pd));
    }
    m_blasCache.clear();
    m_slots.clear();
    m_freeSlots.clear();
    m_handleToSlot.clear();
    m_activeCount      = 0;
    m_activeDefs.clear();
    m_tlasEntries.clear();
    m_lastTransforms.clear();
    m_tlasRebuildPendingSlots = 3;
    m_transformsDirty         = false;
}

// ---------------------------------------------------------------------------
// AddInstance
// ---------------------------------------------------------------------------
bool AccelerationStructure::AddInstance(
    uint32_t userHandle,
    ID3D12Resource* vb, uint32_t vertexCount, uint32_t vertexStride,
    uint32_t posOff, uint32_t normOff, uint32_t uvOff, uint32_t tanOff,
    ID3D12Resource* ib, uint32_t indexStride,
    const NR_SubmeshDesc* submeshes, uint32_t submeshCount,
    const NR_SubmeshOMMDesc* ommDescs)
{
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

    const DXGI_FORMAT idxFmt = (indexStride == 4) ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_R16_UINT;

    InstanceSlot slot;
    slot.active    = true;
    slot.needsBLAS = true;
    slot.meshKey   = { reinterpret_cast<uintptr_t>(vb), reinterpret_cast<uintptr_t>(ib) };
    slot.mask      = 0xFF;
    float identity[12] = { 1,0,0,0, 0,1,0,0, 0,0,1,0 };
    memcpy(slot.transform, identity, 48);

    slot.mesh.submeshes.resize(submeshCount);
    for (uint32_t j = 0; j < submeshCount; ++j)
    {
        MeshData& md       = slot.mesh.submeshes[j];
        md.vertexBuffer    = vb;
        md.vertexCount     = vertexCount;
        md.vertexStride    = vertexStride;
        md.positionOffset  = posOff;
        md.normalOffset    = normOff;
        md.texCoord1Offset = uvOff;
        md.tangentOffset   = tanOff;
        md.indexBuffer     = ib;
        md.indexCount      = submeshes[j].indexCount;
        md.indexByteOffset = submeshes[j].indexByteOffset;
        md.indexFormat     = idxFmt;
        md.materialIndex   = submeshes[j].materialIndex;
        md.hasBakedOMM     = false;

        if (ommDescs && ommDescs[j].arrayData && ommDescs[j].arrayDataSize > 0)
        {
            const NR_SubmeshOMMDesc& o = ommDescs[j];
            md.hasBakedOMM = true;
            auto& baked = md.ommBaked;
            const uint8_t* pArray = static_cast<const uint8_t*>(o.arrayData);
            baked.arrayData.assign(pArray, pArray + o.arrayDataSize);
            const uint8_t* pDesc = static_cast<const uint8_t*>(o.descArray);
            baked.descArray.assign(pDesc, pDesc + o.descArrayCount * 8u);
            baked.descArrayCount = o.descArrayCount;
            const uint8_t* pIdx = static_cast<const uint8_t*>(o.indexBuffer);
            baked.indexBuffer.assign(pIdx, pIdx + o.indexCount * o.indexStride);
            baked.indexCount  = o.indexCount;
            baked.indexStride = o.indexStride;
            baked.indexFormat = (o.indexStride == 4) ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_R16_UINT;
            const uint32_t* src = static_cast<const uint32_t*>(o.histogramFlat);
            baked.histogram.resize(o.histogramCount);
            for (uint32_t h = 0; h < o.histogramCount; ++h, src += 3)
            {
                auto& he          = baked.histogram[h];
                he.Count            = src[0];
                he.SubdivisionLevel = static_cast<UINT16>(src[1]);
                he.Format           = static_cast<D3D12_RAYTRACING_OPACITY_MICROMAP_FORMAT>(src[2]);
            }
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
    m_tlasRebuildPendingSlots = 3;
    return true;
}

// ---------------------------------------------------------------------------
// RemoveInstance
// ---------------------------------------------------------------------------
void AccelerationStructure::RemoveInstance(uint32_t handle)
{
    auto it = m_handleToSlot.find(handle);
    if (it == m_handleToSlot.end()) return;

    const uint32_t slotIndex = it->second;
    InstanceSlot& slot = m_slots[slotIndex];
    if (!slot.active) return;

    if (!slot.needsBLAS)
        ReleaseBLAS(slot.meshKey);

    slot.active    = false;
    slot.needsBLAS = false;
    slot.mesh.submeshes.clear();
    m_freeSlots.push_back(slotIndex);
    m_handleToSlot.erase(it);
    --m_activeCount;
    m_tlasRebuildPendingSlots = 3;
}

// ---------------------------------------------------------------------------
// SetInstanceTransform / SetInstanceMask
// ---------------------------------------------------------------------------
void AccelerationStructure::SetInstanceTransform(uint32_t handle, const float transform[12])
{
    auto it = m_handleToSlot.find(handle);
    if (it == m_handleToSlot.end()) return;
    InstanceSlot& slot = m_slots[it->second];
    if (!slot.active) return;
    if (memcmp(slot.transform, transform, 48) != 0)
    {
        memcpy(slot.transform, transform, 48);
        m_transformsDirty = true;
    }
}

void AccelerationStructure::SetInstanceMask(uint32_t handle, uint8_t mask)
{
    auto it = m_handleToSlot.find(handle);
    if (it == m_handleToSlot.end()) return;
    InstanceSlot& slot = m_slots[it->second];
    if (!slot.active || slot.mask == mask) return;
    slot.mask                 = mask;
    m_tlasRebuildPendingSlots = 3;
}

void AccelerationStructure::SetInstanceID(uint32_t handle, uint32_t id)
{
    auto it = m_handleToSlot.find(handle);
    if (it == m_handleToSlot.end()) return;
    InstanceSlot& slot = m_slots[it->second];
    if (!slot.active || slot.customInstanceID == id) return;
    slot.customInstanceID     = id;
    m_tlasRebuildPendingSlots = 3;
}

// ---------------------------------------------------------------------------
// BuildOrUpdate  -  called every frame before ray dispatch.
// ---------------------------------------------------------------------------
bool AccelerationStructure::BuildOrUpdate(ID3D12GraphicsCommandList4* cmdList)
{
    TickDeferredDeletes();

    // Advance to the next double-buffer slot each frame.
    // The GPU is currently consuming the previous slot; we now own this slot.
    m_frameIndex = (m_frameIndex + 1) % 3;

    // If a structural rebuild is pending for this slot, mark it as needed now.
    // m_tlasRebuildPendingSlots is set to 2 on any structural change so that
    // BOTH slots are brought up to date before we fall back to refit-only.
    const bool tlasNeedsRebuild = (m_tlasRebuildPendingSlots > 0);

    // -------------------------------------------------------------------
    // Step A: Build any pending new BLASes (throttled to avoid GPU TDR)
    // -------------------------------------------------------------------
    static constexpr int kMaxBLASBuildsPerFrame = 50;
    bool anyNewBLAS = false;
    int  blasBuildsThisFrame = 0;
    for (auto& slot : m_slots)
    {
        if (!slot.active || !slot.needsBLAS) continue;
        if (blasBuildsThisFrame >= kMaxBLASBuildsPerFrame)
        {
            // More BLASes remain; ensure we keep rebuilding next frame.
            m_tlasRebuildPendingSlots = (std::max)(m_tlasRebuildPendingSlots, 1);
            break;
        }
        if (!EnsureBLAS(cmdList, slot.meshKey, slot.mesh))
        {
            AccelLogf(m_log, kUnityLogTypeError, "BuildOrUpdate: EnsureBLAS failed");
            return false;
        }
        slot.needsBLAS = false;
        anyNewBLAS     = true;
        ++blasBuildsThisFrame;
    }
    // New BLASes require all slots to rebuild; reset the counter to 3.
    if (anyNewBLAS)
        m_tlasRebuildPendingSlots = 3;

    // -------------------------------------------------------------------
    // Step B: Structural change - full TLAS rebuild
    // -------------------------------------------------------------------
    if (m_tlasRebuildPendingSlots > 0)
    {
        m_activeDefs.clear();
        m_tlasEntries.clear();
        for (const auto& slot : m_slots)
        {
            if (!slot.active) continue;
            m_activeDefs.push_back(slot.mesh);
            TLASInstanceEntry e;
            e.blasVA     = GetBLASVA(slot.meshKey);
            e.instanceID = slot.customInstanceID;
            e.mask       = slot.mask;
            memcpy(e.transform, slot.transform, 48);
            m_tlasEntries.push_back(e);
        }
        AccelLogf(m_log, kUnityLogTypeLog, "BuildOrUpdate: rebuilding TLAS (pending=%d)...", m_tlasRebuildPendingSlots);
        if (!BuildTLAS(cmdList, m_tlasEntries))
        {
            AccelLogf(m_log, kUnityLogTypeError, "BuildOrUpdate: BuildTLAS failed");
            return false;
        }
        --m_tlasRebuildPendingSlots;
        m_transformsDirty = false;
        return true;
    }

    // -------------------------------------------------------------------
    // Step C: No structural change — update transforms if dirty, then
    // always call BuildTLAS to write the current state into this frame's
    // slot.  With triple buffering each slot has its own instanceDesc
    // buffer, so skipping BuildTLAS would leave stale data in the slot.
    // -------------------------------------------------------------------
    if (m_transformsDirty)
    {
        uint32_t denseIdx = 0;
        for (const auto& slot : m_slots)
        {
            if (!slot.active) continue;
            memcpy(m_tlasEntries[denseIdx].transform, slot.transform, 48);
            ++denseIdx;
        }
        m_transformsDirty = false;
    }
    BuildTLAS(cmdList, m_tlasEntries);
    return true;
}
