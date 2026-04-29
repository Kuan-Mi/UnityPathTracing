#include "AccelerationStructure.h"
#include "PluginInternal.h"
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <algorithm>

// Forward declaration of global deferred resource delete function from Plugin.cpp
extern void SafeReleaseResource(ComPtr<ID3D12Resource> resource);
extern void EnqueueDeferredDelete(void *ptr, DeferredType type);

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
}

AccelerationStructure::~AccelerationStructure()
{
    AccelLogf(m_log, kUnityLogTypeLog, "[AccelerationStructure::~AccelerationStructure] Destructor complete");
}

bool AccelerationStructure::EnsureBLAS(ID3D12GraphicsCommandList4 *cmdList, InstanceSlot &slot)
{
    auto &def = slot.meshInfo;

    BLASEntry blas;
    const size_t subCount = def.submeshes.size();
    if (subCount == 0)
    {
        AccelLogf(m_log, kUnityLogTypeError, "EnsureBLAS: instance has no submeshes");
        return false;
    }

    AccelLogf(m_log, kUnityLogTypeLog, "[EnsureBLAS] Building BLAS for VB=%p IB=%p", (void *)def.vertexBuffer, (void *)def.indexBuffer);
    m_d3d12v8->RequestResourceState(def.vertexBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    m_d3d12v8->RequestResourceState(def.indexBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geomDescs(subCount);

    for (size_t j = 0; j < subCount; ++j)
    {
        const SubMeshData &sub = def.submeshes[j];
        D3D12_RAYTRACING_GEOMETRY_DESC &geomDesc = geomDescs[j];
        geomDesc = {};

        geomDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        geomDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        geomDesc.Triangles.VertexBuffer.StartAddress = def.vertexBuffer->GetGPUVirtualAddress();
        geomDesc.Triangles.VertexBuffer.StrideInBytes = def.vertexStride;
        geomDesc.Triangles.VertexCount = def.vertexCount;
        geomDesc.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        geomDesc.Triangles.IndexBuffer = def.indexBuffer->GetGPUVirtualAddress() + sub.indexByteOffset;
        geomDesc.Triangles.IndexCount = sub.indexCount;
        geomDesc.Triangles.IndexFormat = def.indexFormat;
        geomDesc.Triangles.Transform3x4 = 0;
    }

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS blasFlags;
    blasFlags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
    inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    inputs.Flags = blasFlags;
    inputs.NumDescs = static_cast<UINT>(subCount);
    inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputs.pGeometryDescs = geomDescs.data();

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo = {};
    m_device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuildInfo);

    D3D12_HEAP_PROPERTIES defaultHeap = {};
    defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;

    wchar_t blasName[64], scratchName[64];

    swprintf(blasName, 64, L"BLAS_VB_%u", slot.customInstanceID);
    swprintf(scratchName, 64, L"BLASScratch_VB_%u", slot.customInstanceID);

    blas.blasScratch = CreateBuffer(m_device.Get(),
                                    prebuildInfo.ScratchDataSizeInBytes,
                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                    D3D12_RESOURCE_STATE_COMMON, defaultHeap, scratchName);
    blas.blas = CreateBuffer(m_device.Get(),
                             prebuildInfo.ResultDataMaxSizeInBytes,
                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                             D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, defaultHeap, blasName);
    if (!blas.blasScratch || !blas.blas)
    {
        AccelLogf(m_log, kUnityLogTypeError, "EnsureBLAS: buffer allocation failed");
        return false;
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
    buildDesc.DestAccelerationStructureData = blas.blas->GetGPUVirtualAddress();
    buildDesc.Inputs = inputs;
    buildDesc.ScratchAccelerationStructureData = blas.blasScratch->GetGPUVirtualAddress();
    cmdList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);

    slot.blasVA = blas.blas->GetGPUVirtualAddress();
    EnqueueDeferredDelete(new BLASEntry(std::move(blas)), DeferredType::AccelStructBlas);

    return true;
}

// ---------------------------------------------------------------------------
// BuildTLAS  -  (re)build of the TLAS with capacity-aware buffer reuse.
//   * m_instanceDesc is reused when count fits within the existing allocation.
//   * m_tlas / tlasScratch (per-frame) are reused when the prebuild sizes fit within the
//     existing capacities.  Only when capacity is exceeded are old buffers
//     moved to the deferred-delete queue and new (larger) ones allocated.
// ---------------------------------------------------------------------------
bool AccelerationStructure::BuildTLAS(ID3D12GraphicsCommandList4 *cmdList, const std::vector<TLASInstanceEntry> &entries)
{
    const uint32_t count = static_cast<uint32_t>(entries.size());
    TLASFrameResources &res = m_tlasResources[m_frameIndex];

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
                SafeReleaseResource(std::move(res.instanceDesc));
                res.mappedInstanceDesc = nullptr;
                res.instanceDescCapacity = 0;
            }

            D3D12_HEAP_PROPERTIES uploadHeap = {};
            uploadHeap.Type = D3D12_HEAP_TYPE_UPLOAD;

            // Over-allocate by 1.5x to reduce future reallocations.
            const uint32_t newCapacity = static_cast<uint32_t>(count * 3 / 2) + 1;
            UINT64 instanceDescSize = sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * newCapacity;

            wchar_t name[64];
            swprintf(name, 64, L"TLAS_InstanceDesc_Frame%u", m_frameIndex);
            res.instanceDesc = CreateBuffer(m_device.Get(), instanceDescSize, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, uploadHeap, name);
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
        auto *descs = static_cast<D3D12_RAYTRACING_INSTANCE_DESC *>(res.mappedInstanceDesc);
        for (uint32_t i = 0; i < count; ++i)
        {
            const TLASInstanceEntry &e = entries[i];
            D3D12_RAYTRACING_INSTANCE_DESC &inst = descs[i];
            memset(&inst, 0, sizeof(inst));
            memcpy(inst.Transform, e.transform, 12 * sizeof(float));
            inst.InstanceID = e.instanceID;
            inst.InstanceMask = e.mask;
            inst.InstanceContributionToHitGroupIndex = 0;
            inst.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
            inst.AccelerationStructure = e.blasVA;
        }
    }

    // ------------------------------------------------------------------
    // 2. Query prebuild sizes for the new instance count.
    // ------------------------------------------------------------------
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
    inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
    inputs.NumDescs = count;
    inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputs.InstanceDescs = (count > 0 && res.instanceDesc) ? res.instanceDesc->GetGPUVirtualAddress() : 0;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo = {};
    m_device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &prebuildInfo);

    // Scratch must cover both full-build and update-refit sizes.
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputsForUpdate = inputs;
    inputsForUpdate.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfoUpdate = {};
    m_device->GetRaytracingAccelerationStructurePrebuildInfo(&inputsForUpdate, &prebuildInfoUpdate);

    const UINT64 neededResult = prebuildInfo.ResultDataMaxSizeInBytes;
    const UINT64 neededScratch = max(prebuildInfo.ScratchDataSizeInBytes, prebuildInfoUpdate.UpdateScratchDataSizeInBytes);

    // ------------------------------------------------------------------
    // 3a. TLAS result buffer: reuse current frame's slot or reallocate.
    //     Only this frame's slot is touched; GPU is busy with the other.
    // ------------------------------------------------------------------
    if (!res.tlas || neededResult > res.tlasResultCapacity)
    {
        if (res.tlas)
        {
            SafeReleaseResource(std::move(res.tlas));
        }

        D3D12_HEAP_PROPERTIES defaultHeap = {};
        defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;

        wchar_t name[64];
        swprintf(name, 64, L"TLAS_Result_Frame%u", m_frameIndex);
        res.tlas = CreateBuffer(m_device.Get(), neededResult,
                                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, defaultHeap, name);
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
            SafeReleaseResource(std::move(res.tlasScratch));
        }

        D3D12_HEAP_PROPERTIES defaultHeap = {};
        defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;

        wchar_t name[64];
        swprintf(name, 64, L"TLAS_Scratch_Frame%u", m_frameIndex);
        res.tlasScratch = CreateBuffer(m_device.Get(), neededScratch,
                                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                       D3D12_RESOURCE_STATE_COMMON, defaultHeap, name);
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
    buildDesc.DestAccelerationStructureData = res.tlas->GetGPUVirtualAddress();
    buildDesc.Inputs = inputs;
    buildDesc.ScratchAccelerationStructureData = res.tlasScratch->GetGPUVirtualAddress();
    cmdList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);

    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
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
    std::lock_guard<std::mutex> lock(m_stateMutex);
    AccelLogf(m_log, kUnityLogTypeLog, "[AS::Clear] BEGIN - activeSlots=%u", m_activeCount);

    // Move TLAS resources to pending delete (both frame slots + shared scratch).
    {
        for (auto &r : m_tlasResources)
        {
            if (r.instanceDesc)
                SafeReleaseResource(std::move(r.instanceDesc));
            if (r.tlas)
                SafeReleaseResource(std::move(r.tlas));
            if (r.tlasScratch)
                SafeReleaseResource(std::move(r.tlasScratch));
            r.mappedInstanceDesc = nullptr;
            r.instanceDescCapacity = 0;
            r.tlasResultCapacity = 0;
            r.tlasScratchCapacity = 0;
        }
    }
    m_frameIndex = 0;

    // NOTE: We no longer defer deletion of vertex/index buffers from slots because we don't own them.
    // Unity manages these resources, and we only store raw pointers without AddRef.
    // Simply clear the slots - the pointers will be nulled out automatically.
    m_slots.clear();
    m_freeSlots.clear();
    m_handleToSlot.clear();
    m_activeCount = 0;
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
    slot.mask = 0xFF;
    float identity[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    memcpy(slot.transform, identity, 48);

    // Set descriptive names for Unity-provided buffers to aid debugging
    wchar_t vbName[64], ibName[64];

    swprintf(vbName, 64, L"Unity_VB_%p", (void *)vb);
    swprintf(ibName, 64, L"Unity_IB_%p", (void *)ib);

    vb->SetName(vbName);
    ib->SetName(ibName);
    AccelLogf(m_log, kUnityLogTypeLog, "[AddInstance] Set names: VB=%p '%ls', IB=%p '%ls'", (void *)vb, vbName, (void *)ib, ibName);

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

    // NOTE: We no longer defer deletion of vertex/index buffers because we don't own them.
    // Unity manages these resources, and we only store raw pointers without AddRef.
    slot.active = false;
    slot.meshInfo.submeshes.clear();
    slot.meshInfo.vertexBuffer = nullptr;
    slot.meshInfo.indexBuffer = nullptr;
    m_freeSlots.push_back(slotIndex);
    m_handleToSlot.erase(it);
    --m_activeCount;
}

void AccelerationStructure::UpdateDynamicVertexBuffer(uint32_t handle, void *vbPtr, uint32_t vertexCount, uint32_t vertexStride)
{
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

// ---------------------------------------------------------------------------
// BuildOrUpdate  -  called every frame before ray dispatch.
// ---------------------------------------------------------------------------
bool AccelerationStructure::BuildOrUpdate(ID3D12GraphicsCommandList4 *cmdList)
{
    std::lock_guard<std::mutex> lock(m_stateMutex);

    // Per-frame diagnostic dump (one line per active instance).
    // DumpInstances("BuildOrUpdate");

    // Advance to the next double-buffer slot each frame.
    // The GPU is currently consuming the previous slot; we now own this slot.
    m_frameIndex = (m_frameIndex + 1) % 10;

    ID3D12Fence *frameFence = m_d3d12v8->GetFrameFence();

    auto completedValue = frameFence->GetCompletedValue();

    AccelLogf(m_log, kUnityLogTypeLog, "BuildOrUpdate: frame=%u, wait fence value=%llu, completed value=%llu", m_frameIndex, m_tlasFenceValues[m_frameIndex], completedValue);

    if (completedValue < m_tlasFenceValues[m_frameIndex])
    {

        AccelLogf(m_log, kUnityLogTypeError, "BuildOrUpdate: waiting for GPU to finish with frame %u (fence value %llu, completed value %llu)", m_frameIndex, m_tlasFenceValues[m_frameIndex], completedValue);
        // 如果 GPU 还没跑完，必须等待。
        // 使用 Event 阻塞 CPU，防止篡改正在被 GPU 读取的 TLAS
        HANDLE event = CreateEventEx(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
        frameFence->SetEventOnCompletion(m_tlasFenceValues[m_frameIndex], event);
        WaitForSingleObject(event, INFINITE);
        CloseHandle(event);
    }

    m_tlasFenceValues[m_frameIndex] = m_d3d12v8->GetNextFrameFenceValue();

    AccelLogf(m_log, kUnityLogTypeLog, "BuildOrUpdate: frame=%u, NextFrameFenceValue %llu", m_frameIndex, m_tlasFenceValues[m_frameIndex]);

    // -------------------------------------------------------------------
    // Step A: Build any pending new BLASes (throttled to avoid GPU TDR)
    // -------------------------------------------------------------------
    static constexpr int kMaxBLASBuildsPerFrame = 10000;
    bool anyNewBLAS = false;
    int blasBuildsThisFrame = 0;
    for (auto &slot : m_slots)
    {
        if (!slot.active)
            continue;

        if (blasBuildsThisFrame >= kMaxBLASBuildsPerFrame)
            break;

        if (!EnsureBLAS(cmdList, slot))
        {
            AccelLogf(m_log, kUnityLogTypeError, "BuildOrUpdate: EnsureBLAS failed");
            return false;
        }
        anyNewBLAS = true;
        ++blasBuildsThisFrame;
    }
    // Emit a single global UAV barrier covering all newly-built BLASes.
    if (anyNewBLAS)
    {
        D3D12_RESOURCE_BARRIER blasBarrier = {};
        blasBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        blasBarrier.UAV.pResource = nullptr; // nullptr = all UAV resources
        cmdList->ResourceBarrier(1, &blasBarrier);
    }

    m_tlasEntries.clear();
    for (const auto &slot : m_slots)
    {
        if (!slot.active)
            continue;
        TLASInstanceEntry e;
        e.blasVA = slot.blasVA;
        e.instanceID = slot.customInstanceID;
        e.mask = slot.mask;
        memcpy(e.transform, slot.transform, 48);
        m_tlasEntries.push_back(e);
    }

    if (!BuildTLAS(cmdList, m_tlasEntries))
    {
        AccelLogf(m_log, kUnityLogTypeError, "BuildOrUpdate: BuildTLAS failed");
        return false;
    }
    return true;
}