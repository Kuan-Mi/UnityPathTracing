#include "DescriptorSetBase.h"
#include "ComputeShader.h"
#include "RayTraceShader.h"
#include "AccelerationStructure.h"
#include "BindlessTexture.h"
#include "BindlessBuffer.h"
#include "BindlessUAVTexture.h"
#include "INativeResource.h"
#include "TransientDescriptorRing.h"
#include "PluginInternal.h"
#include <cstdio>
#include <cstdarg>

// ---------------------------------------------------------------------------
// ResolveBoundResource
//   Resolves the ID3D12Resource* a singular-resource binding slot points at.
//   With the unified objectPtr there are exactly two cases for the slot kinds
//   that reach here (SRV / UAV / CBV / non-AS ROOT_SRV):
//     - NativeBuffer: objectPtr is an INativeResource*; GetResource() resolves the
//       per-frame slot of volatile buffers dynamically.
//     - None:         objectPtr is already a raw ID3D12Resource* (Unity textures,
//       AS-backed buffers, the stable DEFAULT-buffer pointer, ...).
//   AccelStruct / Bindless* slots are handled by their callers and never reach
//   this function, so no separate raw-pointer field is needed.
// ---------------------------------------------------------------------------
static ID3D12Resource* ResolveBoundResource(const BindingSlot& slot)
{
    if (slot.objectKind == BindingObjectKind::NativeBuffer)
        return slot.objectPtr ? reinterpret_cast<INativeResource*>(slot.objectPtr)->GetResource() : nullptr;
    return reinterpret_cast<ID3D12Resource*>(slot.objectPtr);
}

// ===========================================================================
// Constructor
// ===========================================================================

template<typename ShaderT>
DescriptorSetBase<ShaderT>::DescriptorSetBase(
    ShaderT*                  shader,
    ID3D12Device*             device,
    IUnityLog*                log,
    DescriptorHeapAllocator*  allocator,
    IUnityGraphicsD3D12v8*    d3d12v8)
    : m_shader(shader), m_device(device), m_log(log), m_allocator(allocator),
      m_tracker(d3d12v8)
{
}

// ===========================================================================
// Logging
// ===========================================================================

template<typename ShaderT>
void DescriptorSetBase<ShaderT>::Log(UnityLogType type, const char* msg) const
{
    if (m_log) m_log->Log(type, msg, __FILE__, __LINE__);
    else        printf("[DescriptorSetBase] %s\n", msg);
}

template<typename ShaderT>
void DescriptorSetBase<ShaderT>::Logf(UnityLogType type, const char* fmt, ...) const
{
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    Log(type, buf);
}

// ===========================================================================
// EnsureCategoryIndices
//   Bucket every binding into its per-type index list exactly once, and size the
//   view-descriptor cache to match.  Cheap one-time work amortised over every
//   subsequent dispatch.
// ===========================================================================

template<typename ShaderT>
void DescriptorSetBase<ShaderT>::EnsureCategoryIndices()
{
    if (m_categoriesBuilt) return;

    const auto& bindings = m_shader->GetBindings();
    m_viewCache.assign(bindings.size(), CachedView{});

    for (uint32_t i = 0; i < bindings.size(); ++i)
    {
        switch (bindings[i].type)
        {
        case BindingType::SRV:            m_idxSRV.push_back(i);       break;
        case BindingType::TLAS:           m_idxTLAS.push_back(i);      break;
        case BindingType::UAV:            m_idxUAV.push_back(i);       break;
        case BindingType::CBV:            m_idxCBV.push_back(i);       break;
        case BindingType::ROOT_SRV:       m_idxRootSRV.push_back(i);   break;
        case BindingType::ROOT_CONSTANTS: m_idxRootConst.push_back(i); break;
        case BindingType::SRV_ARRAY:      m_idxSRVArray.push_back(i);  break;
        case BindingType::UAV_ARRAY:      m_idxUAVArray.push_back(i);  break;
        }
    }

    m_categoriesBuilt = true;
}

// ===========================================================================
// AllocateTransientTables
//   Allocates one contiguous block of (numSRV + numUAV) slots from the
//   transient ring.  Doing it as a single allocation eliminates the
//   partial-failure case (succeed for SRV but fail for UAV) and halves the
//   per-dispatch ring bookkeeping.  The SRV table occupies [base, base+numSRV)
//   and the UAV table [base+numSRV, base+numSRV+numUAV); the two are bound as
//   separate root descriptor tables in BindRootParams.
// ===========================================================================

template<typename ShaderT>
bool DescriptorSetBase<ShaderT>::AllocateTransientTables(
    uint32_t& outSrvBase, uint32_t& outUavBase) const
{
    const uint32_t numSRV = m_shader->GetNumSRV();
    const uint32_t numUAV = m_shader->GetNumUAV();
    const uint32_t total  = numSRV + numUAV;

    if (total == 0)
    {
        outSrvBase = kInvalidAlloc;
        outUavBase = kInvalidAlloc;
        return true;
    }

    uint32_t base = g_transientRing.Allocate(total);
    if (base == UINT32_MAX)
    {
        Logf(kUnityLogTypeError,
             "DescriptorSet::Dispatch [%s]: transient descriptor ring exhausted "
             "(needed %u slots) - skipping dispatch",
             m_shader->GetName(), total);
        return false;
    }

    outSrvBase = (numSRV > 0) ? base                : kInvalidAlloc;
    outUavBase = (numUAV > 0) ? (base + numSRV)     : kInvalidAlloc;
    return true;
}

// ===========================================================================
// WriteDescriptors
//   Writes all SRV/UAV descriptors into the transient table bases.
//   CBVs and ROOT_SRVs are bound as inline root descriptors in BindRootParams.
//   SRV_ARRAY / UAV_ARRAY bindings use their own heap (Bindless* objects).
// ===========================================================================

template<typename ShaderT>
void DescriptorSetBase<ShaderT>::WriteDescriptors(
    const BindingSlot* slots, uint32_t slotCount,
    uint32_t srvBase, uint32_t uavBase)
{
    EnsureCategoryIndices();
    const auto& bindings = m_shader->GetBindings();

    // --- SRV / TLAS ---
    if (srvBase != kInvalidAlloc)
    {
        // TLAS — never cached: the acceleration structure is rebuilt every frame,
        // so its GPU virtual address changes and the view must be re-derived.
        for (uint32_t i : m_idxTLAS)
        {
            const auto& b           = bindings[i];
            const BindingSlot& slot = (i < slotCount) ? slots[i] : BindingSlot{};

            D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
            s.ViewDimension           = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
            s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

            ID3D12Resource* tlas = nullptr;
            if (slot.objectKind == BindingObjectKind::AccelStruct && slot.objectPtr)
                tlas = reinterpret_cast<AccelerationStructure*>(slot.objectPtr)->GetTLAS();
            else
                tlas = reinterpret_cast<ID3D12Resource*>(slot.objectPtr); // raw TLAS resource (kind None)

            s.RaytracingAccelerationStructure.Location = tlas ? tlas->GetGPUVirtualAddress() : 0;
            m_device->CreateShaderResourceView(nullptr, &s,
                m_allocator->GetCPUHandle(srvBase + b.heapOffset));
        }

        for (uint32_t i : m_idxSRV)
        {
            const auto& b           = bindings[i];
            const BindingSlot& slot = (i < slotCount) ? slots[i] : BindingSlot{};
            D3D12_CPU_DESCRIPTOR_HANDLE h = m_allocator->GetCPUHandle(srvBase + b.heapOffset);
            ID3D12Resource* res = ResolveBoundResource(slot);
            if (res)
            {
                CachedView& cv = m_viewCache[i];
                if (!cv.valid || cv.res != reinterpret_cast<uint64_t>(res) ||
                    cv.count != slot.count || cv.stride != slot.stride || cv.format != slot.format)
                {
                    D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
                    s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                    auto rd = res->GetDesc();
                    if (rd.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
                    {
                        if (slot.stride > 0 && slot.count > 0)
                        {
                            s.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                            s.Format                     = DXGI_FORMAT_UNKNOWN;
                            s.Buffer.NumElements         = slot.count;
                            s.Buffer.StructureByteStride = slot.stride;
                        }
                        else if (slot.count > 0 && slot.format != 0)
                        {
                            s.ViewDimension      = D3D12_SRV_DIMENSION_BUFFER;
                            s.Format             = static_cast<DXGI_FORMAT>(slot.format);
                            s.Buffer.NumElements = slot.count;
                        }
                        else
                        {
                            s.ViewDimension      = D3D12_SRV_DIMENSION_BUFFER;
                            s.Format             = DXGI_FORMAT_R32_TYPELESS;
                            s.Buffer.Flags       = D3D12_BUFFER_SRV_FLAG_RAW;
                            s.Buffer.NumElements = static_cast<UINT>(rd.Width / 4);
                        }
                    }
                    else if (rd.DepthOrArraySize == 6)
                    {
                        // Cube map: DepthOrArraySize==6 → TEXTURECUBE SRV.
                        // A TEXTURE2DARRAY view would be incompatible with a
                        // TextureCube binding in the shader.
                        s.ViewDimension                   = D3D12_SRV_DIMENSION_TEXTURECUBE;
                        s.Format                          = rd.Format;
                        s.TextureCube.MostDetailedMip     = 0;
                        s.TextureCube.MipLevels           = rd.MipLevels;
                        s.TextureCube.ResourceMinLODClamp = 0.0f;
                    }
                    else if (rd.DepthOrArraySize > 1)
                    {
                        s.ViewDimension                        = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                        s.Format                               = rd.Format;
                        s.Texture2DArray.MostDetailedMip       = 0;
                        s.Texture2DArray.MipLevels             = rd.MipLevels;
                        s.Texture2DArray.FirstArraySlice       = 0;
                        s.Texture2DArray.ArraySize             = rd.DepthOrArraySize;
                        s.Texture2DArray.PlaneSlice            = 0;
                        s.Texture2DArray.ResourceMinLODClamp   = 0.0f;
                    }
                    else
                    {
                        s.ViewDimension       = D3D12_SRV_DIMENSION_TEXTURE2D;
                        s.Format              = rd.Format;
                        s.Texture2D.MipLevels = rd.MipLevels;
                    }
                    cv.srv    = s;
                    cv.res    = reinterpret_cast<uint64_t>(res);
                    cv.count  = slot.count;
                    cv.stride = slot.stride;
                    cv.format = slot.format;
                    cv.valid  = true;
                }
                m_device->CreateShaderResourceView(res, &cv.srv, h);
            }
            else
            {
                // Null descriptor — write a safe raw buffer placeholder.
                // Drop the cache so a later non-null bind re-derives the real view.
                m_viewCache[i].valid = false;
                D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
                s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                s.ViewDimension      = D3D12_SRV_DIMENSION_BUFFER;
                s.Format             = DXGI_FORMAT_R32_TYPELESS;
                s.Buffer.Flags       = D3D12_BUFFER_SRV_FLAG_RAW;
                s.Buffer.NumElements = 1;
                m_device->CreateShaderResourceView(nullptr, &s, h);
            }
        }
    }

    // --- UAV ---
    if (uavBase != kInvalidAlloc)
    {
        for (uint32_t i : m_idxUAV)
        {
            const auto& b           = bindings[i];
            const BindingSlot& slot = (i < slotCount) ? slots[i] : BindingSlot{};
            ID3D12Resource* res = ResolveBoundResource(slot);
            D3D12_CPU_DESCRIPTOR_HANDLE h = m_allocator->GetCPUHandle(uavBase + b.heapOffset);
            if (res)
            {
                CachedView& cv = m_viewCache[i];
                if (!cv.valid || cv.res != reinterpret_cast<uint64_t>(res) ||
                    cv.count != slot.count || cv.stride != slot.stride || cv.format != slot.format)
                {
                    D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
                    auto rd = res->GetDesc();
                    if (rd.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
                    {
                        u.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                        if (slot.stride > 0 && slot.count > 0)
                        {
                            u.Format                     = DXGI_FORMAT_UNKNOWN;
                            u.Buffer.NumElements         = slot.count;
                            u.Buffer.StructureByteStride = slot.stride;
                        }
                        else if (slot.count > 0 && slot.format != 0)
                        {
                            u.Format             = static_cast<DXGI_FORMAT>(slot.format);
                            u.Buffer.NumElements = slot.count;
                        }
                        else
                        {
                            u.Format             = DXGI_FORMAT_R32_TYPELESS;
                            u.Buffer.Flags       = D3D12_BUFFER_UAV_FLAG_RAW;
                            u.Buffer.NumElements = static_cast<UINT>(rd.Width / 4);
                        }
                    }
                    else if (rd.DepthOrArraySize > 1)
                    {
                        u.ViewDimension                  = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                        u.Format                         = rd.Format;
                        u.Texture2DArray.MipSlice        = 0;
                        u.Texture2DArray.FirstArraySlice = 0;
                        u.Texture2DArray.ArraySize       = rd.DepthOrArraySize;
                        u.Texture2DArray.PlaneSlice      = 0;
                    }
                    else
                    {
                        u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                        u.Format        = rd.Format;
                    }
                    cv.uav    = u;
                    cv.res    = reinterpret_cast<uint64_t>(res);
                    cv.count  = slot.count;
                    cv.stride = slot.stride;
                    cv.format = slot.format;
                    cv.valid  = true;
                }
                m_device->CreateUnorderedAccessView(res, nullptr, &cv.uav, h);
            }
            else
            {
                m_viewCache[i].valid = false;
                D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
                u.ViewDimension      = D3D12_UAV_DIMENSION_BUFFER;
                u.Format             = DXGI_FORMAT_R32_TYPELESS;
                u.Buffer.Flags       = D3D12_BUFFER_UAV_FLAG_RAW;
                u.Buffer.NumElements = 1;
                m_device->CreateUnorderedAccessView(nullptr, nullptr, &u, h);
            }
        }
    }
}

// ===========================================================================
// RequestResourceStates
// ===========================================================================

template<typename ShaderT>
void DescriptorSetBase<ShaderT>::RequestResourceStates(
    const BindingSlot* slots, uint32_t slotCount)
{
    if (!m_tracker.Valid()) return;
    EnsureCategoryIndices();

    auto slotOf = [&](uint32_t i) -> BindingSlot { return (i < slotCount) ? slots[i] : BindingSlot{}; };

    for (uint32_t i : m_idxSRV)
        m_tracker.Require(ResolveBoundResource(slotOf(i)), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    for (uint32_t i : m_idxRootSRV)
    {
        // A ROOT_SRV resolves the same way SRVs do (ResolveBoundResource handles the
        // NativeBuffer handle path and the raw-resource objectPtr fallback), with
        // the AccelStruct/TLAS special case layered on top.
        const BindingSlot slot = slotOf(i);
        ID3D12Resource* srvRes = (slot.objectKind == BindingObjectKind::AccelStruct && slot.objectPtr)
            ? reinterpret_cast<AccelerationStructure*>(slot.objectPtr)->GetTLAS()
            : ResolveBoundResource(slot);
        m_tracker.Require(srvRes, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }

    for (uint32_t i : m_idxUAV)
        m_tracker.Require(ResolveBoundResource(slotOf(i)), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    for (uint32_t i : m_idxCBV)
        m_tracker.Require(ResolveBoundResource(slotOf(i)), D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);

    for (uint32_t i : m_idxSRVArray)
    {
        const BindingSlot slot = slotOf(i);
        if (slot.objectKind == BindingObjectKind::BindlessTexture && slot.objectPtr)
        {
            auto* bt = reinterpret_cast<BindlessTexture*>(slot.objectPtr);
            for (uint32_t k = 0; k < bt->Capacity(); ++k)
                m_tracker.Require(bt->GetTexture(k), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        else if (slot.objectKind == BindingObjectKind::BindlessBuffer && slot.objectPtr)
        {
            auto* bb = reinterpret_cast<BindlessBuffer*>(slot.objectPtr);
            for (uint32_t k = 0; k < bb->Capacity(); ++k)
                m_tracker.Require(bb->GetBuffer(k), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
    }

    for (uint32_t i : m_idxUAVArray)
    {
        const BindingSlot slot = slotOf(i);
        if (slot.objectKind == BindingObjectKind::BindlessUAVTexture && slot.objectPtr)
        {
            auto* bt = reinterpret_cast<BindlessUAVTexture*>(slot.objectPtr);
            for (uint32_t k = 0; k < bt->Capacity(); ++k)
                m_tracker.Require(bt->GetTexture(k), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
    }
}

// ===========================================================================
// NotifyResourceStates
//   Notifies Unity about the post-dispatch state of each resource.
//   The UAVAccess=true flag on UAV resources is critical: it tells Unity
//   to insert a UAV barrier before the next use of that resource, even
//   when the state does not change (UAV -> UAV write-after-write hazard).
// ===========================================================================

template<typename ShaderT>
void DescriptorSetBase<ShaderT>::NotifyResourceStates(
    const BindingSlot* slots, uint32_t slotCount)
{
    if (!m_tracker.Valid()) return;
    EnsureCategoryIndices();

    auto slotOf = [&](uint32_t i) -> BindingSlot { return (i < slotCount) ? slots[i] : BindingSlot{}; };

    for (uint32_t i : m_idxUAV)
        m_tracker.Notify(ResolveBoundResource(slotOf(i)), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, /*uavAccess=*/true);

    for (uint32_t i : m_idxUAVArray)
    {
        const BindingSlot slot = slotOf(i);
        if (slot.objectKind == BindingObjectKind::BindlessUAVTexture && slot.objectPtr)
        {
            // Notify every element written through the bindless array — mirrors the
            // per-element Require() in RequestResourceStates so the post-dispatch UAV
            // barrier covers all textures, not just the base resource.
            auto* bt = reinterpret_cast<BindlessUAVTexture*>(slot.objectPtr);
            for (uint32_t k = 0; k < bt->Capacity(); ++k)
                m_tracker.Notify(bt->GetTexture(k), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, /*uavAccess=*/true);
        }
    }
}

// ===========================================================================
// ValidateBindings
// ===========================================================================

template<typename ShaderT>
bool DescriptorSetBase<ShaderT>::ValidateBindings(
    const BindingSlot* slots, uint32_t slotCount) const
{
    const auto& bindings = m_shader->GetBindings();
    bool anyMissing = false;

    for (size_t i = 0; i < bindings.size(); ++i)
    {
        const auto& b = bindings[i];
        const BindingSlot& slot = (i < slotCount) ? slots[i] : BindingSlot{};
        bool ok = false;
        const char* kind = "?";

        // A non-null objectPtr is required; the kind set narrows which wrappers are
        // valid for the binding type (None == a raw ID3D12Resource* in objectPtr).
        const bool isNone        = slot.objectKind == BindingObjectKind::None;
        const bool isNativeBuf   = slot.objectKind == BindingObjectKind::NativeBuffer;
        const bool isAccelStruct = slot.objectKind == BindingObjectKind::AccelStruct;

        switch (b.type)
        {
        case BindingType::TLAS:
            kind = "TLAS";
            ok = slot.objectPtr != 0 && (isNone || isAccelStruct);
            break;
        case BindingType::SRV:
            kind = "SRV";
            ok = slot.objectPtr != 0 && (isNone || isNativeBuf);
            break;
        case BindingType::UAV:
            kind = "UAV";
            ok = slot.objectPtr != 0 && (isNone || isNativeBuf);
            break;
        case BindingType::ROOT_SRV:
            kind = "ROOT_SRV";
            ok = slot.objectPtr != 0 && (isNone || isAccelStruct || isNativeBuf);
            break;
        case BindingType::CBV:
            kind = "CBV";
            ok = slot.objectPtr != 0 && (isNone || isNativeBuf);
            break;
        case BindingType::SRV_ARRAY:
            kind = "SRV_ARRAY";
            ok = (slot.objectKind == BindingObjectKind::BindlessTexture && slot.objectPtr != 0) ||
                 (slot.objectKind == BindingObjectKind::BindlessBuffer  && slot.objectPtr != 0);
            break;
        case BindingType::UAV_ARRAY:
            kind = "UAV_ARRAY";
            ok = (slot.objectKind == BindingObjectKind::BindlessUAVTexture && slot.objectPtr != 0);
            break;
        case BindingType::ROOT_CONSTANTS:
            kind = "ROOT_CONSTANTS";
            ok = slot.objectPtr != 0;
            break;
        }

        if (!ok)
        {
            Logf(kUnityLogTypeError,
                 "DescriptorSet::Dispatch: '%s' binding '%s' (%s, space%u, reg%u) is not set",
                 m_shader->GetName(), b.name.c_str(), kind, b.space, b.registerIndex);
            anyMissing = true;
        }
    }
    return !anyMissing;
}

// ===========================================================================
// BindRootParams
//   Binds the global heap, root signature, and all root parameters.
//   The base ID3D12GraphicsCommandList* interface exposes all SetComputeRoot*
//   methods, so this works for both Dispatch and DispatchRays command lists.
// ===========================================================================

template<typename ShaderT>
void DescriptorSetBase<ShaderT>::BindRootParams(
    ID3D12GraphicsCommandList* cmdList,
    const BindingSlot*      slots,
    uint32_t                   slotCount,
    uint32_t                   srvBase,
    uint32_t                   uavBase)
{
    EnsureCategoryIndices();

    // Bind the global shared heap
    ID3D12DescriptorHeap* heapsToBind[1] = { m_allocator->GetHeap() };
    cmdList->SetDescriptorHeaps(1, heapsToBind);

    cmdList->SetComputeRootSignature(m_shader->GetRootSignature());

    const auto& bindings        = m_shader->GetBindings();
    const uint32_t rootParamSRV     = m_shader->GetRootParamSRV();
    const uint32_t rootParamUAV     = m_shader->GetRootParamUAV();
    const uint32_t rootParamCBVBase = m_shader->GetRootParamCBVBase();

    auto slotOf = [&](uint32_t i) -> BindingSlot { return (i < slotCount) ? slots[i] : BindingSlot{}; };

    // SRV descriptor table
    if (rootParamSRV != kInvalidAlloc && srvBase != kInvalidAlloc)
        cmdList->SetComputeRootDescriptorTable(rootParamSRV,
            m_allocator->GetGPUHandle(srvBase));

    // UAV descriptor table
    if (rootParamUAV != kInvalidAlloc && uavBase != kInvalidAlloc)
        cmdList->SetComputeRootDescriptorTable(rootParamUAV,
            m_allocator->GetGPUHandle(uavBase));

    // SRV_ARRAY bindings — each has its own root parameter
    for (uint32_t i : m_idxSRVArray)
    {
        const auto& b = bindings[i];
        if (b.rootParam == kInvalidAlloc) continue;
        const BindingSlot slot = slotOf(i);
        if (slot.objectKind == BindingObjectKind::BindlessTexture && slot.objectPtr)
            cmdList->SetComputeRootDescriptorTable(b.rootParam,
                reinterpret_cast<BindlessTexture*>(slot.objectPtr)->GetGPUHandle());
        else if (slot.objectKind == BindingObjectKind::BindlessBuffer && slot.objectPtr)
            cmdList->SetComputeRootDescriptorTable(b.rootParam,
                reinterpret_cast<BindlessBuffer*>(slot.objectPtr)->GetGPUHandle());
    }

    // UAV_ARRAY bindings — each has its own root parameter
    for (uint32_t i : m_idxUAVArray)
    {
        const auto& b = bindings[i];
        if (b.rootParam == kInvalidAlloc) continue;
        const BindingSlot slot = slotOf(i);
        if (slot.objectKind == BindingObjectKind::BindlessUAVTexture && slot.objectPtr)
            cmdList->SetComputeRootDescriptorTable(b.rootParam,
                reinterpret_cast<BindlessUAVTexture*>(slot.objectPtr)->GetGPUHandle());
    }

    // Root CBV per CBV binding
    if (rootParamCBVBase != kInvalidAlloc)
    {
        for (uint32_t i : m_idxCBV)
        {
            const auto& b = bindings[i];
            const BindingSlot slot = slotOf(i);
            // A NativeBuffer resolves its own VA: volatile buffers have no resource
            // and return the current upload-pool suballocation's address; everything
            // else binds the raw resource's base VA.
            D3D12_GPU_VIRTUAL_ADDRESS addr = 0;
            if (slot.objectKind == BindingObjectKind::NativeBuffer && slot.objectPtr)
            {
                addr = reinterpret_cast<INativeResource*>(slot.objectPtr)->GetGpuVirtualAddress();
            }
            else
            {
                ID3D12Resource* res = ResolveBoundResource(slot);
                addr = res ? res->GetGPUVirtualAddress() : 0;
            }
            cmdList->SetComputeRootConstantBufferView(rootParamCBVBase + b.heapOffset, addr);
        }
    }

    // Inline root SRV per ROOT_SRV binding
    if (m_shader->GetRootParamRootSRVBase() != kInvalidAlloc)
    {
        for (uint32_t i : m_idxRootSRV)
        {
            const auto& b = bindings[i];
            if (b.rootParam == kInvalidAlloc) continue;
            const BindingSlot slot = slotOf(i);
            D3D12_GPU_VIRTUAL_ADDRESS va = 0;
            if (slot.objectKind == BindingObjectKind::AccelStruct && slot.objectPtr)
            {
                ID3D12Resource* tlas = reinterpret_cast<AccelerationStructure*>(slot.objectPtr)->GetTLAS();
                if (tlas) va = tlas->GetGPUVirtualAddress();
            }
            else
            {
                // Resolves a NativeBuffer handle to its resource, or falls back to the
                // raw-resource objectPtr — the root descriptor binds the buffer's base GPU VA.
                ID3D12Resource* res = ResolveBoundResource(slot);
                if (res) va = res->GetGPUVirtualAddress();
            }
            if (va) cmdList->SetComputeRootShaderResourceView(b.rootParam, va);
        }
    }

    // Root 32-bit constants per ROOT_CONSTANTS binding
    for (uint32_t i : m_idxRootConst)
    {
        const auto& b = bindings[i];
        if (b.rootParam == kInvalidAlloc) continue;
        const BindingSlot slot = slotOf(i);
        if (!slot.objectPtr) continue;
        UINT num32      = slot.count > 0 ? slot.count : b.num32BitValues;
        UINT destOffset = slot.stride; // destOffsetIn32BitValues
        cmdList->SetComputeRoot32BitConstants(
            b.rootParam,
            num32,
            reinterpret_cast<const void*>(static_cast<uintptr_t>(slot.objectPtr)),
            destOffset);
    }
}

// ===========================================================================
// Explicit template instantiations
// ===========================================================================

template class DescriptorSetBase<ComputeShader>;
template class DescriptorSetBase<RayTraceShader>;
