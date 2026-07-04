#include "DescriptorSetBase.h"
#include "ComputePipeline.h"
#include "RayTraceShader.h"
#include "RasterShader.h"
#include "AccelerationStructure.h"
#include "BindlessTexture.h"
#include "BindlessBuffer.h"
#include "BindlessUAVTexture.h"
#include "INativeResource.h"
#include "TransientDescriptorRing.h"
#include "PluginInternal.h"
#include <algorithm>
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
        case BindingType::SAMPLER:        m_idxSampler.push_back(i);   break;
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
    uint32_t& outSrvBase, uint32_t& outUavBase, uint32_t& outSamplerBase) const
{
    // Slot totals (not binding counts): a bounded array occupies arrayCount contiguous slots.
    const uint32_t numSRV = m_shader->GetNumSRVSlots();
    const uint32_t numUAV = m_shader->GetNumUAVSlots();
    const uint32_t total  = numSRV + numUAV;

    if (total == 0)
    {
        outSrvBase = kInvalidAlloc;
        outUavBase = kInvalidAlloc;
    }
    else
    {
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
    }

    const uint32_t numSampler = m_shader->GetNumSamplerSlots();
    if (numSampler == 0)
    {
        outSamplerBase = kInvalidAlloc;
        return true;
    }

    if (!g_samplerHeapAllocator || !g_samplerTransientRing.IsInitialized())
    {
        Logf(kUnityLogTypeError,
             "DescriptorSet::Dispatch [%s]: sampler descriptor table requested but sampler heap is not initialized",
             m_shader->GetName());
        return false;
    }

    outSamplerBase = g_samplerTransientRing.Allocate(numSampler);
    if (outSamplerBase == UINT32_MAX)
    {
        Logf(kUnityLogTypeError,
             "DescriptorSet::Dispatch [%s]: transient sampler descriptor ring exhausted "
             "(needed %u slots) - skipping dispatch",
             m_shader->GetName(), numSampler);
        return false;
    }
    return true;
}

// ===========================================================================
// EnsureStagingHeap
//   Creates the persistent CPU-only staging heap backing WriteDescriptors
//   (one descriptor per transient table slot). On failure staging is disabled
//   permanently for this set and WriteDescriptors creates descriptors directly
//   into the transient range every dispatch, exactly as before.
// ===========================================================================

template<typename ShaderT>
bool DescriptorSetBase<ShaderT>::EnsureStagingHeap(uint32_t totalSlots, uint32_t srvSlots)
{
    if (m_stagingHeap) return true;
    if (m_stagingFailed || !m_device || totalSlots == 0) return false;

    D3D12_DESCRIPTOR_HEAP_DESC hd = {};
    hd.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = totalSlots;
    hd.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_NONE; // CPU-only: CopyDescriptors source
    if (FAILED(m_device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&m_stagingHeap))))
    {
        m_stagingFailed = true;
        Logf(kUnityLogTypeWarning,
             "DescriptorSet [%s]: staging descriptor heap creation failed - "
             "falling back to per-dispatch descriptor creation",
             m_shader->GetName());
        return false;
    }
    m_stagingBase = m_stagingHeap->GetCPUDescriptorHandleForHeapStart();
    m_descSize    = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    auto slotAt = [&](uint32_t i) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE h = m_stagingBase;
        h.ptr += static_cast<SIZE_T>(i) * m_descSize;
        return h;
    };
    auto writeNullSRV = [&](D3D12_CPU_DESCRIPTOR_HANDLE h) {
        D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
        s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        s.ViewDimension      = D3D12_SRV_DIMENSION_BUFFER;
        s.Format             = DXGI_FORMAT_R32_TYPELESS;
        s.Buffer.Flags       = D3D12_BUFFER_SRV_FLAG_RAW;
        s.Buffer.NumElements = 1;
        m_device->CreateShaderResourceView(nullptr, &s, h);
    };
    auto writeNullUAV = [&](D3D12_CPU_DESCRIPTOR_HANDLE h) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
        u.ViewDimension      = D3D12_UAV_DIMENSION_BUFFER;
        u.Format             = DXGI_FORMAT_R32_TYPELESS;
        u.Buffer.Flags       = D3D12_BUFFER_UAV_FLAG_RAW;
        u.Buffer.NumElements = 1;
        m_device->CreateUnorderedAccessView(nullptr, nullptr, &u, h);
    };
    auto writeNullCBV = [&](D3D12_CPU_DESCRIPTOR_HANDLE h) {
        D3D12_CONSTANT_BUFFER_VIEW_DESC c = {};   // BufferLocation 0, SizeInBytes 0 = null CBV
        m_device->CreateConstantBufferView(&c, h);
    };

    if (m_shader->UsesSharedLayout())
    {
        // Shared-layout combined table: slot kinds interleave per declaration
        // order, so prefill each item's slots with the matching null view.
        for (const auto& item : m_shader->GetSharedLayout())
        {
            using K = ShaderBase::SharedLayoutKind;
            if (item.kind != K::SRV && item.kind != K::TLAS &&
                item.kind != K::UAV && item.kind != K::CBV)
                continue;
            for (uint32_t k = 0; k < item.count && item.tableOffset + k < totalSlots; ++k)
            {
                D3D12_CPU_DESCRIPTOR_HANDLE h = slotAt(item.tableOffset + k);
                if (item.kind == K::UAV)      writeNullUAV(h);
                else if (item.kind == K::CBV) writeNullCBV(h);
                else                          writeNullSRV(h);
            }
        }
        return true;
    }

    for (uint32_t i = 0; i < totalSlots; ++i)
    {
        if (i < srvSlots) writeNullSRV(slotAt(i));
        else              writeNullUAV(slotAt(i));
    }
    return true;
}

// ===========================================================================
// WriteDescriptors
//   Maintains all SRV/TLAS/UAV descriptors in the persistent staging heap and
//   publishes them into the transient table bases with one CopyDescriptorsSimple.
//   A binding's descriptor is only (re)created when its resolved resource or
//   view params changed since the last dispatch — the common case for the path
//   tracer (same resources rebound every frame) writes zero descriptors.
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

    const uint32_t numSRV = m_shader->GetNumSRVSlots();
    const uint32_t numUAV = m_shader->GetNumUAVSlots();
    const uint32_t total  = numSRV + numUAV;
    if (total == 0) return;

    // Shared-layout mode uses ONE combined SRV/UAV/CBV table (nvrhi SRVetc):
    // GetNumSRVSlots() covers the whole table, every table binding's heapOffset
    // is absolute within it, and UAV/CBV slots share the SRV block/base.
    const bool combined = m_shader->UsesSharedLayout();
    if (combined) uavBase = srvBase;

    const bool staged = EnsureStagingHeap(total, numSRV);

    // Destination for an SRV-table / UAV-table slot. The staging layout mirrors
    // the transient block AllocateTransientTables hands out: SRV slots occupy
    // [0, numSRV) and UAV slots [numSRV, total) — except in combined mode where
    // offsets are already absolute within the single table.
    const uint32_t uavStagingBias = combined ? 0 : numSRV;
    auto srvDest = [&](uint32_t off) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        if (!staged) return m_allocator->GetCPUHandle(srvBase + off);
        D3D12_CPU_DESCRIPTOR_HANDLE h = m_stagingBase;
        h.ptr += static_cast<SIZE_T>(off) * m_descSize;
        return h;
    };
    auto uavDest = [&](uint32_t off) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        if (!staged) return m_allocator->GetCPUHandle(uavBase + off);
        D3D12_CPU_DESCRIPTOR_HANDLE h = m_stagingBase;
        h.ptr += static_cast<SIZE_T>(uavStagingBias + off) * m_descSize;
        return h;
    };

    // --- SRV / TLAS ---
    if (srvBase != kInvalidAlloc)
    {
        // TLAS — the acceleration structure is rebuilt every frame but in place:
        // the result buffer (and so the GPU VA the view points at) only changes
        // when it is reallocated to fit more instances. Cache on the VA itself
        // (stored in cv.res) so the view is re-derived exactly when that happens.
        for (uint32_t i : m_idxTLAS)
        {
            const auto& b           = bindings[i];
            const BindingSlot& slot = (i < slotCount) ? slots[i] : BindingSlot{};

            ID3D12Resource* tlas = nullptr;
            if (slot.objectKind == BindingObjectKind::AccelStruct && slot.objectPtr)
                tlas = reinterpret_cast<AccelerationStructure*>(slot.objectPtr)->GetTLAS();
            else
                tlas = reinterpret_cast<ID3D12Resource*>(slot.objectPtr); // raw TLAS resource (kind None)

            const uint64_t va = tlas ? tlas->GetGPUVirtualAddress() : 0;
            CachedView& cv = m_viewCache[i];
            if (staged && cv.valid && cv.res == va)
                continue;

            D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
            s.ViewDimension           = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
            s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            s.RaytracingAccelerationStructure.Location = va;
            m_device->CreateShaderResourceView(nullptr, &s, srvDest(b.heapOffset));
            cv.res    = va;
            cv.count  = 0;
            cv.stride = 0;
            cv.format = 0;
            cv.valid  = staged;
        }

        for (uint32_t i : m_idxSRV)
        {
            const auto& b           = bindings[i];
            const BindingSlot& slot = (i < slotCount) ? slots[i] : BindingSlot{};
            ID3D12Resource* res = ResolveBoundResource(slot);

            // Bounded SRV array (e.g. Texture2D t[N]): arrayCount single-mip Texture2D views
            // starting at mip slot.stride (consecutive). Texture-only. The cache entry keys
            // the whole run on (res, stride); the view descs themselves are not stored.
            if (b.arrayCount > 1)
            {
                CachedView& cv = m_viewCache[i];
                if (staged && cv.valid && cv.res == reinterpret_cast<uint64_t>(res) &&
                    cv.stride == slot.stride)
                    continue;

                D3D12_RESOURCE_DESC rd = res ? res->GetDesc() : D3D12_RESOURCE_DESC{};
                uint32_t maxMip = (res && rd.MipLevels > 0) ? rd.MipLevels - 1 : 0;
                for (uint32_t k = 0; k < b.arrayCount; ++k)
                {
                    D3D12_CPU_DESCRIPTOR_HANDLE hk = srvDest(b.heapOffset + k);
                    D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
                    s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                    if (res)
                    {
                        s.ViewDimension             = D3D12_SRV_DIMENSION_TEXTURE2D;
                        s.Format                    = rd.Format;
                        s.Texture2D.MostDetailedMip = (slot.stride + k < maxMip) ? (slot.stride + k) : maxMip;
                        s.Texture2D.MipLevels       = 1;
                        m_device->CreateShaderResourceView(res, &s, hk);
                    }
                    else
                    {
                        s.ViewDimension      = D3D12_SRV_DIMENSION_BUFFER;
                        s.Format             = DXGI_FORMAT_R32_TYPELESS;
                        s.Buffer.Flags       = D3D12_BUFFER_SRV_FLAG_RAW;
                        s.Buffer.NumElements = 1;
                        m_device->CreateShaderResourceView(nullptr, &s, hk);
                    }
                }
                cv.res    = reinterpret_cast<uint64_t>(res);
                cv.count  = 0;
                cv.stride = slot.stride;
                cv.format = 0;
                cv.valid  = staged;
                continue;
            }

            if (res)
            {
                CachedView& cv = m_viewCache[i];
                const bool match = cv.valid && cv.res == reinterpret_cast<uint64_t>(res) &&
                    cv.count == slot.count && cv.stride == slot.stride && cv.format == slot.format;
                if (staged && match)
                    continue;   // staging already holds this exact view
                if (!match)
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
                    else if (slot.format != 0)
                    {
                        // Explicit typed Texture2DArray SRV (slot.format = DXGI_FORMAT, non-zero).
                        // Lets a cubemap's faces be read as a Texture2DArray<...> instead of the
                        // default TextureCube — e.g. BC6UCompress gathers the env cube mip i as a
                        // 6-slice array. Mirrors nvrhi's Texture_SRV(...).setDimension(Texture2DArray).
                        // slot.stride = MostDetailedMip, slot.count = MipLevels (0 ⇒ 1). Only ever set
                        // by the SetTextureArraySlice overload; all other texture SRVs leave format=0,
                        // so this branch is dormant for them.
                        s.ViewDimension                      = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                        s.Format                             = static_cast<DXGI_FORMAT>(slot.format);
                        s.Texture2DArray.MostDetailedMip     = slot.stride;
                        s.Texture2DArray.MipLevels           = (slot.count > 0) ? slot.count : 1;
                        s.Texture2DArray.FirstArraySlice     = 0;
                        s.Texture2DArray.ArraySize           = rd.DepthOrArraySize;
                        s.Texture2DArray.PlaneSlice          = 0;
                        s.Texture2DArray.ResourceMinLODClamp = 0.0f;
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
                        // Single Texture2D. slot.stride = most-detailed mip, slot.count = mip count
                        // (0 = all remaining) — both 0 for the default whole-texture SetTexture, so
                        // this is identical to before unless a per-mip SetTexture overload is used.
                        s.ViewDimension             = D3D12_SRV_DIMENSION_TEXTURE2D;
                        s.Format                    = rd.Format;
                        s.Texture2D.MostDetailedMip = slot.stride;
                        s.Texture2D.MipLevels       = (slot.count > 0)
                            ? slot.count
                            : (rd.MipLevels > slot.stride ? rd.MipLevels - slot.stride : 1);
                    }
                    cv.srv    = s;
                    cv.res    = reinterpret_cast<uint64_t>(res);
                    cv.count  = slot.count;
                    cv.stride = slot.stride;
                    cv.format = slot.format;
                    cv.valid  = true;
                }
                m_device->CreateShaderResourceView(res, &cv.srv, srvDest(b.heapOffset));
            }
            else
            {
                // Null descriptor — a safe raw buffer placeholder, cached with res == 0
                // so staging is not rewritten every dispatch; a later non-null bind
                // mismatches on res and re-derives the real view.
                CachedView& cv = m_viewCache[i];
                if (staged && cv.valid && cv.res == 0)
                    continue;
                D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
                s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                s.ViewDimension      = D3D12_SRV_DIMENSION_BUFFER;
                s.Format             = DXGI_FORMAT_R32_TYPELESS;
                s.Buffer.Flags       = D3D12_BUFFER_SRV_FLAG_RAW;
                s.Buffer.NumElements = 1;
                m_device->CreateShaderResourceView(nullptr, &s, srvDest(b.heapOffset));
                cv.res    = 0;
                cv.count  = 0;
                cv.stride = 0;
                cv.format = 0;
                cv.valid  = staged;
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

            // Bounded UAV array (e.g. RWTexture2D u[NUM_LODS]): arrayCount per-mip Texture2D UAVs
            // starting at mip slot.stride (consecutive). Texture-only. The cache entry keys
            // the whole run on (res, stride); the view descs themselves are not stored.
            if (b.arrayCount > 1)
            {
                CachedView& cv = m_viewCache[i];
                if (staged && cv.valid && cv.res == reinterpret_cast<uint64_t>(res) &&
                    cv.stride == slot.stride)
                    continue;

                D3D12_RESOURCE_DESC rd = res ? res->GetDesc() : D3D12_RESOURCE_DESC{};
                uint32_t maxMip = (res && rd.MipLevels > 0) ? rd.MipLevels - 1 : 0;
                for (uint32_t k = 0; k < b.arrayCount; ++k)
                {
                    D3D12_CPU_DESCRIPTOR_HANDLE hk = uavDest(b.heapOffset + k);
                    D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
                    if (res)
                    {
                        u.ViewDimension      = D3D12_UAV_DIMENSION_TEXTURE2D;
                        u.Format             = rd.Format;
                        u.Texture2D.MipSlice = (slot.stride + k < maxMip) ? (slot.stride + k) : maxMip;
                        m_device->CreateUnorderedAccessView(res, nullptr, &u, hk);
                    }
                    else
                    {
                        u.ViewDimension      = D3D12_UAV_DIMENSION_BUFFER;
                        u.Format             = DXGI_FORMAT_R32_TYPELESS;
                        u.Buffer.Flags       = D3D12_BUFFER_UAV_FLAG_RAW;
                        u.Buffer.NumElements = 1;
                        m_device->CreateUnorderedAccessView(nullptr, nullptr, &u, hk);
                    }
                }
                cv.res    = reinterpret_cast<uint64_t>(res);
                cv.count  = 0;
                cv.stride = slot.stride;
                cv.format = 0;
                cv.valid  = staged;
                continue;
            }

            if (res)
            {
                CachedView& cv = m_viewCache[i];
                const bool match = cv.valid && cv.res == reinterpret_cast<uint64_t>(res) &&
                    cv.count == slot.count && cv.stride == slot.stride && cv.format == slot.format;
                if (staged && match)
                    continue;   // staging already holds this exact view
                if (!match)
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
                        // Texture2DArray / cubemap UAV. slot.stride = mip slice (0 = mip 0, the
                        // default). Honoring it lets a single mip-chained cube be written one mip
                        // at a time (e.g. EnvMapBaker base layer writes mip 0+1, MIPReduceCS fills
                        // the rest) — mirroring nvrhi's .setDimension(Texture2DArray) per-mip views.
                        u.ViewDimension                  = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                        u.Format                         = rd.Format;
                        u.Texture2DArray.MipSlice        = slot.stride;
                        u.Texture2DArray.FirstArraySlice = 0;
                        u.Texture2DArray.ArraySize       = rd.DepthOrArraySize;
                        u.Texture2DArray.PlaneSlice      = 0;
                    }
                    else
                    {
                        // Single Texture2D UAV. slot.stride = mip slice (0 = mip 0, the default).
                        u.ViewDimension      = D3D12_UAV_DIMENSION_TEXTURE2D;
                        u.Format             = rd.Format;
                        u.Texture2D.MipSlice = slot.stride;
                    }
                    cv.uav    = u;
                    cv.res    = reinterpret_cast<uint64_t>(res);
                    cv.count  = slot.count;
                    cv.stride = slot.stride;
                    cv.format = slot.format;
                    cv.valid  = true;
                }
                m_device->CreateUnorderedAccessView(res, nullptr, &cv.uav, uavDest(b.heapOffset));
            }
            else
            {
                // Null UAV placeholder, cached with res == 0 (see SRV null branch).
                CachedView& cv = m_viewCache[i];
                if (staged && cv.valid && cv.res == 0)
                    continue;
                D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
                u.ViewDimension      = D3D12_UAV_DIMENSION_BUFFER;
                u.Format             = DXGI_FORMAT_R32_TYPELESS;
                u.Buffer.Flags       = D3D12_BUFFER_UAV_FLAG_RAW;
                u.Buffer.NumElements = 1;
                m_device->CreateUnorderedAccessView(nullptr, nullptr, &u, uavDest(b.heapOffset));
                cv.res    = 0;
                cv.count  = 0;
                cv.stride = 0;
                cv.format = 0;
                cv.valid  = staged;
            }
        }
    }

    // --- Static (table) CBVs — shared-layout mode only ---
    // Volatile CBVs stay root-bound in BindRootParams; a static CBV occupies a
    // slot in the combined table (nvrhi ConstantBuffer), so its view is written
    // here. CBV size comes from the resource width (256-aligned, D3D12 caps a
    // CBV at 65536 bytes); slots carry no explicit size.
    if (combined && srvBase != kInvalidAlloc)
    {
        for (uint32_t i : m_idxCBV)
        {
            const auto& b = bindings[i];
            if (b.rootParam != kInvalidAlloc)
                continue;   // volatile CBV → root descriptor
            const BindingSlot& slot = (i < slotCount) ? slots[i] : BindingSlot{};

            D3D12_GPU_VIRTUAL_ADDRESS va = 0;
            uint64_t width = 0;
            ID3D12Resource* res = nullptr;
            if (slot.objectKind == BindingObjectKind::NativeBuffer && slot.objectPtr)
                res = reinterpret_cast<INativeResource*>(slot.objectPtr)->GetResource();
            else
                res = ResolveBoundResource(slot);
            if (res)
            {
                va    = res->GetGPUVirtualAddress();
                width = res->GetDesc().Width;
            }
            const uint32_t sizeBytes = static_cast<uint32_t>(
                (std::min<uint64_t>)(((width + 255ull) & ~255ull), 65536ull));

            CachedView& cv = m_viewCache[i];
            if (staged && cv.valid && cv.res == va && cv.count == sizeBytes)
                continue;

            D3D12_CONSTANT_BUFFER_VIEW_DESC c = {};
            c.BufferLocation = va;
            c.SizeInBytes    = va ? sizeBytes : 0;
            m_device->CreateConstantBufferView(&c, srvDest(b.heapOffset));
            cv.res    = va;
            cv.count  = sizeBytes;
            cv.stride = 0;
            cv.format = 0;
            cv.valid  = staged;
        }
    }

    // --- Publish: one copy from staging into this dispatch's transient block ---
    // AllocateTransientTables hands out a single contiguous block (SRV slots then
    // UAV slots), so the whole region is covered by one CopyDescriptorsSimple.
    if (staged)
    {
        const uint32_t destBase = (numSRV > 0) ? srvBase : uavBase;
        if (destBase != kInvalidAlloc)
            m_device->CopyDescriptorsSimple(total,
                m_allocator->GetCPUHandle(destBase),
                m_stagingBase,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }
}

// ===========================================================================
// WriteSamplerDescriptors
//   Writes the shared-layout sampler descriptor table. Sampler slots encode
//   NativeSampler fields directly: count=filter, stride=packed address/mips,
//   format=maxAnisotropy. This keeps sampler binding descriptor-set local
//   without adding a separate native sampler object lifetime.
// ===========================================================================

template<typename ShaderT>
void DescriptorSetBase<ShaderT>::WriteSamplerDescriptors(
    const BindingSlot* slots, uint32_t slotCount,
    uint32_t samplerBase)
{
    if (samplerBase == kInvalidAlloc || !g_samplerHeapAllocator) return;
    EnsureCategoryIndices();

    const auto& bindings = m_shader->GetBindings();
    auto AddressFrom = [](uint32_t a) -> D3D12_TEXTURE_ADDRESS_MODE {
        switch (a)
        {
        case 1:  return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        case 2:  return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
        case 3:  return D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
        case 4:  return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
        default: return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        }
    };
    auto FilterFrom = [](uint32_t f) -> D3D12_FILTER {
        switch (f)
        {
        case 0:  return D3D12_FILTER_MIN_MAG_MIP_POINT;
        case 2:  return D3D12_FILTER_ANISOTROPIC;
        default: return D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        }
    };

    for (uint32_t i : m_idxSampler)
    {
        const auto& b = bindings[i];
        const BindingSlot slot = (i < slotCount) ? slots[i] : BindingSlot{};

        const uint32_t packed = slot.stride;
        const uint32_t addrU  =  packed        & 0xffu;
        const uint32_t addrV  = (packed >> 8)  & 0xffu;
        const uint32_t addrW  = (packed >> 16) & 0xffu;
        const bool     mips   = ((packed >> 24) & 0x1u) != 0;

        D3D12_SAMPLER_DESC sd = {};
        sd.Filter         = FilterFrom(slot.count);
        sd.AddressU       = AddressFrom(addrU);
        sd.AddressV       = AddressFrom(addrV);
        sd.AddressW       = AddressFrom(addrW);
        sd.MipLODBias     = 0.0f;
        sd.MaxAnisotropy  = (sd.Filter == D3D12_FILTER_ANISOTROPIC)
                           ? (slot.format ? slot.format : 16u)
                           : 0u;
        sd.ComparisonFunc = D3D12_COMPARISON_FUNC_NONE;
        sd.BorderColor[0] = 0.0f;
        sd.BorderColor[1] = 0.0f;
        sd.BorderColor[2] = 0.0f;
        sd.BorderColor[3] = 0.0f;
        sd.MinLOD         = 0.0f;
        sd.MaxLOD         = mips ? 16.0f : 0.0f;

        m_device->CreateSampler(&sd, g_samplerHeapAllocator->GetCPUHandle(samplerBase + b.heapOffset));
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

    m_subresReadBarriers.clear();

    // Resources bound as a UAV in this dispatch, collected once. An SRV that targets
    // one of these is the same-resource SRV+UAV case (donut mipmapgen): it must be
    // handled with manual per-subresource barriers rather than a conflicting
    // whole-resource Require.
    m_uavResScratch.clear();
    for (uint32_t j : m_idxUAV)
        if (ID3D12Resource* r = ResolveBoundResource(slotOf(j)))
            m_uavResScratch.push_back(r);
    for (uint32_t j : m_idxUAVArray)
    {
        const BindingSlot s = slotOf(j);
        if (s.objectKind == BindingObjectKind::BindlessUAVTexture && s.objectPtr)
        {
            auto* bt = reinterpret_cast<BindlessUAVTexture*>(s.objectPtr);
            for (uint32_t k = 0, n = bt->UsedCount(); k < n; ++k)
                if (ID3D12Resource* r = bt->GetTexture(k))
                    m_uavResScratch.push_back(r);
        }
    }
    auto boundAsUAV = [&](ID3D12Resource* res) -> bool {
        if (!res) return false;
        for (ID3D12Resource* r : m_uavResScratch)
            if (r == res) return true;
        return false;
    };

    for (uint32_t i : m_idxSRV)
    {
        const BindingSlot slot = slotOf(i);
        ID3D12Resource*   res  = ResolveBoundResource(slot);
        if (boundAsUAV(res))
        {
            // Keep |res| in UNORDERED_ACCESS (the UAV Require below does that) and
            // transition only the SRV-read mip(s) per-subresource in the Dispatch.
            // slot.stride = MostDetailedMip, slot.count = MipLevels (0 => 1 read mip).
            D3D12_RESOURCE_DESC rd = res->GetDesc();
            const UINT mipLevels   = rd.MipLevels ? rd.MipLevels : 1;
            const UINT arraySlices = (rd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D)
                                   ? 1 : (rd.DepthOrArraySize ? rd.DepthOrArraySize : 1);
            const UINT firstMip    = slot.stride;
            const UINT mipCount    = slot.count ? slot.count : 1;
            for (UINT a = 0; a < arraySlices; ++a)
            {
                // D3D12 subresource index = mip + arraySlice*mipLevels + plane*mipLevels*arraySize
                // (plane 0 for these color formats) — computed inline to avoid the d3dx12 helper.
                const UINT firstSub = firstMip + a * mipLevels;
                m_subresReadBarriers.push_back({ res, firstSub, mipCount });
            }
            continue;  // do NOT Require(SHADER_RESOURCE) on the whole resource
        }
        m_tracker.Require(res, m_srvReadState);
    }

    for (uint32_t i : m_idxRootSRV)
    {
        // A ROOT_SRV resolves the same way SRVs do (ResolveBoundResource handles the
        // NativeBuffer handle path and the raw-resource objectPtr fallback), with
        // the AccelStruct/TLAS special case layered on top.
        const BindingSlot slot = slotOf(i);
        ID3D12Resource* srvRes = (slot.objectKind == BindingObjectKind::AccelStruct && slot.objectPtr)
            ? reinterpret_cast<AccelerationStructure*>(slot.objectPtr)->GetTLAS()
            : ResolveBoundResource(slot);
        m_tracker.Require(srvRes, m_srvReadState);
    }

    for (uint32_t i : m_idxUAV)
        m_tracker.Require(ResolveBoundResource(slotOf(i)), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    for (uint32_t i : m_idxCBV)
        m_tracker.Require(ResolveBoundResource(slotOf(i)), D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);

    // SRV arrays: read-only within a frame, so the per-element Require sweep only
    // needs to run on the first dispatch that binds the array each frame (or after
    // its contents change) — the donut model is setPermanentTextureState on loaded
    // textures; once-per-frame is the safe adaptation with Unity owning the tracker.
    for (uint32_t i : m_idxSRVArray)
    {
        const BindingSlot slot = slotOf(i);
        if (slot.objectKind == BindingObjectKind::BindlessTexture && slot.objectPtr)
        {
            auto* bt = reinterpret_cast<BindlessTexture*>(slot.objectPtr);
            if (!bt->NeedsStateSweep(g_frameSerial, m_srvReadState)) continue;
            for (uint32_t k = 0, n = bt->UsedCount(); k < n; ++k)
                m_tracker.Require(bt->GetTexture(k), m_srvReadState);
            bt->MarkStateSweep(g_frameSerial, m_srvReadState);
        }
        else if (slot.objectKind == BindingObjectKind::BindlessBuffer && slot.objectPtr)
        {
            auto* bb = reinterpret_cast<BindlessBuffer*>(slot.objectPtr);
            if (!bb->NeedsStateSweep(g_frameSerial, m_srvReadState)) continue;
            for (uint32_t k = 0, n = bb->UsedCount(); k < n; ++k)
                m_tracker.Require(bb->GetBuffer(k), m_srvReadState);
            bb->MarkStateSweep(g_frameSerial, m_srvReadState);
        }
    }

    for (uint32_t i : m_idxUAVArray)
    {
        const BindingSlot slot = slotOf(i);
        if (slot.objectKind == BindingObjectKind::BindlessUAVTexture && slot.objectPtr)
        {
            auto* bt = reinterpret_cast<BindlessUAVTexture*>(slot.objectPtr);
            for (uint32_t k = 0, n = bt->UsedCount(); k < n; ++k)
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
            for (uint32_t k = 0, n = bt->UsedCount(); k < n; ++k)
                m_tracker.Notify(bt->GetTexture(k), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, /*uavAccess=*/true);
        }
    }
}

// ===========================================================================
// EmitSubresourceReadBarriers / RestoreSubresourceUAVStates
//   Balanced manual per-subresource barriers for the same-resource SRV+UAV case
//   (donut mipmapgen). The resource is in UNORDERED_ACCESS (RequestResourceStates
//   kept it there); we flip only the SRV-read subresource(s) to the read state for
//   the dispatch, then flip them back so Unity's per-resource tracker — which still
//   believes the resource is UNORDERED_ACCESS — stays correct. Both directions emit
//   one transition barrier per subresource (no ALL_SUBRESOURCES, so the UAV mips are
//   untouched and the writer/reader split is honoured within the resource).
// ===========================================================================

template<typename ShaderT>
void DescriptorSetBase<ShaderT>::EmitSubresourceReadBarriers(ID3D12GraphicsCommandList* cmd)
{
    if (!cmd || m_subresReadBarriers.empty()) return;
    std::vector<D3D12_RESOURCE_BARRIER> bars;
    for (const auto& b : m_subresReadBarriers)
        for (UINT k = 0; k < b.subCount; ++k)
        {
            D3D12_RESOURCE_BARRIER rb = {};
            rb.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            rb.Transition.pResource   = b.res;
            rb.Transition.Subresource = b.firstSub + k;
            rb.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            rb.Transition.StateAfter  = m_srvReadState;
            bars.push_back(rb);
        }
    if (!bars.empty()) cmd->ResourceBarrier((UINT)bars.size(), bars.data());
}

template<typename ShaderT>
void DescriptorSetBase<ShaderT>::RestoreSubresourceUAVStates(ID3D12GraphicsCommandList* cmd)
{
    if (!cmd || m_subresReadBarriers.empty()) return;
    std::vector<D3D12_RESOURCE_BARRIER> bars;
    for (const auto& b : m_subresReadBarriers)
        for (UINT k = 0; k < b.subCount; ++k)
        {
            D3D12_RESOURCE_BARRIER rb = {};
            rb.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            rb.Transition.pResource   = b.res;
            rb.Transition.Subresource = b.firstSub + k;
            rb.Transition.StateBefore = m_srvReadState;
            rb.Transition.StateAfter  = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            bars.push_back(rb);
        }
    if (!bars.empty()) cmd->ResourceBarrier((UINT)bars.size(), bars.data());
}

// ===========================================================================
// ValidateBindings
//   The binding table is derived from the declared layout, which is usually a
//   superset of what any one shader touches (e.g. the RTXPT global layout).
//   An UNBOUND slot is therefore legal — it gets a null descriptor, exactly
//   like nvrhi's unmatched binding-set slots — and only a slot that IS bound
//   but with an object kind incompatible with the layout item is an error.
// ===========================================================================

template<typename ShaderT>
bool DescriptorSetBase<ShaderT>::ValidateBindings(
    const BindingSlot* slots, uint32_t slotCount) const
{
    const auto& bindings = m_shader->GetBindings();
    bool anyBad = false;

    for (size_t i = 0; i < bindings.size(); ++i)
    {
        const auto& b = bindings[i];
        const BindingSlot& slot = (i < slotCount) ? slots[i] : BindingSlot{};

        // Unbound → null descriptor / skipped root param.
        if (slot.objectPtr == 0 && slot.objectKind == BindingObjectKind::None)
            continue;

        bool ok = false;
        const char* kind = "?";

        const bool isNone        = slot.objectKind == BindingObjectKind::None;
        const bool isNativeBuf   = slot.objectKind == BindingObjectKind::NativeBuffer;
        const bool isAccelStruct = slot.objectKind == BindingObjectKind::AccelStruct;

        switch (b.type)
        {
        case BindingType::TLAS:
            kind = "TLAS";
            ok = isNone || isAccelStruct;
            break;
        case BindingType::SRV:
            kind = "SRV";
            ok = isNone || isNativeBuf;
            break;
        case BindingType::UAV:
            kind = "UAV";
            ok = isNone || isNativeBuf;
            break;
        case BindingType::ROOT_SRV:
            kind = "ROOT_SRV";
            ok = isNone || isAccelStruct || isNativeBuf;
            break;
        case BindingType::CBV:
            kind = "CBV";
            ok = isNone || isNativeBuf;
            break;
        case BindingType::SRV_ARRAY:
            kind = "SRV_ARRAY";
            ok = slot.objectKind == BindingObjectKind::BindlessTexture ||
                 slot.objectKind == BindingObjectKind::BindlessBuffer;
            break;
        case BindingType::UAV_ARRAY:
            kind = "UAV_ARRAY";
            ok = slot.objectKind == BindingObjectKind::BindlessUAVTexture;
            break;
        case BindingType::ROOT_CONSTANTS:
            kind = "ROOT_CONSTANTS";
            ok = slot.objectKind == BindingObjectKind::RootConstants || isNone;
            break;
        case BindingType::SAMPLER:
            kind = "SAMPLER";
            ok = slot.objectKind == BindingObjectKind::Sampler;
            break;
        }

        if (!ok)
        {
            Logf(kUnityLogTypeError,
                 "DescriptorSet::Dispatch: '%s' slot %zu (%s, space%u, reg%u) is bound with an "
                 "incompatible object kind %u",
                 m_shader->GetName(), i, kind, b.space, b.registerIndex,
                 static_cast<uint32_t>(slot.objectKind));
            anyBad = true;
        }
    }
    return !anyBad;
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
    uint32_t                   uavBase,
    uint32_t                   samplerBase)
{
    EnsureCategoryIndices();

    // Compute vs graphics share the same root layout; only the setter family
    // differs.  These thin shims pick SetGraphicsRoot* / SetComputeRoot* once
    // per call site based on the pipeline kind set by the subclass.
    const bool gfx = m_graphicsRootBinding;
    auto SetRootSig = [&](ID3D12RootSignature* rs) {
        if (gfx) cmdList->SetGraphicsRootSignature(rs);
        else     cmdList->SetComputeRootSignature(rs);
    };
    auto SetTable = [&](UINT p, D3D12_GPU_DESCRIPTOR_HANDLE h) {
        if (gfx) cmdList->SetGraphicsRootDescriptorTable(p, h);
        else     cmdList->SetComputeRootDescriptorTable(p, h);
    };
    auto SetCBV = [&](UINT p, D3D12_GPU_VIRTUAL_ADDRESS a) {
        if (gfx) cmdList->SetGraphicsRootConstantBufferView(p, a);
        else     cmdList->SetComputeRootConstantBufferView(p, a);
    };
    auto SetSRV = [&](UINT p, D3D12_GPU_VIRTUAL_ADDRESS a) {
        if (gfx) cmdList->SetGraphicsRootShaderResourceView(p, a);
        else     cmdList->SetComputeRootShaderResourceView(p, a);
    };
    auto SetConstants = [&](UINT p, UINT n, const void* d, UINT off) {
        if (gfx) cmdList->SetGraphicsRoot32BitConstants(p, n, d, off);
        else     cmdList->SetComputeRoot32BitConstants(p, n, d, off);
    };

    // Bind the global shared heaps. D3D12 has separate heap classes for
    // CBV/SRV/UAV and samplers; shared-layout sampler tables need both.
    ID3D12DescriptorHeap* heapsToBind[2] = { m_allocator->GetHeap(), nullptr };
    UINT heapCount = 1;
    if (g_samplerHeapAllocator && m_shader->GetRootParamSampler() != kInvalidAlloc)
        heapsToBind[heapCount++] = g_samplerHeapAllocator->GetHeap();
    cmdList->SetDescriptorHeaps(heapCount, heapsToBind);

    SetRootSig(m_shader->GetRootSignature());

    const auto& bindings        = m_shader->GetBindings();
    const uint32_t rootParamSRV     = m_shader->GetRootParamSRV();
    const uint32_t rootParamUAV     = m_shader->GetRootParamUAV();
    const uint32_t rootParamSampler = m_shader->GetRootParamSampler();
    const uint32_t rootParamCBVBase = m_shader->GetRootParamCBVBase();

    auto slotOf = [&](uint32_t i) -> BindingSlot { return (i < slotCount) ? slots[i] : BindingSlot{}; };

    // SRV descriptor table
    if (rootParamSRV != kInvalidAlloc && srvBase != kInvalidAlloc)
        SetTable(rootParamSRV, m_allocator->GetGPUHandle(srvBase));

    // UAV descriptor table
    if (rootParamUAV != kInvalidAlloc && uavBase != kInvalidAlloc)
        SetTable(rootParamUAV, m_allocator->GetGPUHandle(uavBase));

    if (rootParamSampler != kInvalidAlloc && samplerBase != kInvalidAlloc && g_samplerHeapAllocator)
        SetTable(rootParamSampler, g_samplerHeapAllocator->GetGPUHandle(samplerBase));

    // Bindless (*_ARRAY) bindings. In shared-layout mode a consecutive run of
    // bindless items shares ONE root parameter (one table, several unbounded
    // ranges all at offset 0 — nvrhi bindless layout), so every binding in the
    // group must resolve to the same descriptor-table base. Set each param
    // once and flag disagreements instead of silently letting the last win.
    UINT     setParams[8];
    uint64_t setHandles[8];
    uint32_t numSetParams = 0;
    auto SetTableOnce = [&](UINT p, D3D12_GPU_DESCRIPTOR_HANDLE h, const char* bindingName) {
        for (uint32_t k = 0; k < numSetParams; ++k)
        {
            if (setParams[k] != p) continue;
            if (setHandles[k] != h.ptr)
                Logf(kUnityLogTypeError,
                     "DescriptorSet [%s]: grouped bindless binding '%s' resolves to a "
                     "different descriptor table than its group — all bindings sharing "
                     "one bindless root param must use the same table",
                     m_shader->GetName(), bindingName);
            return;
        }
        SetTable(p, h);
        if (numSetParams < 8)
        {
            setParams[numSetParams]  = p;
            setHandles[numSetParams] = h.ptr;
            ++numSetParams;
        }
    };

    for (uint32_t i : m_idxSRVArray)
    {
        const auto& b = bindings[i];
        if (b.rootParam == kInvalidAlloc) continue;
        const BindingSlot slot = slotOf(i);
        if (slot.objectKind == BindingObjectKind::BindlessTexture && slot.objectPtr)
            SetTableOnce(b.rootParam,
                reinterpret_cast<BindlessTexture*>(slot.objectPtr)->GetGPUHandle(), b.name.c_str());
        else if (slot.objectKind == BindingObjectKind::BindlessBuffer && slot.objectPtr)
            SetTableOnce(b.rootParam,
                reinterpret_cast<BindlessBuffer*>(slot.objectPtr)->GetGPUHandle(), b.name.c_str());
    }

    for (uint32_t i : m_idxUAVArray)
    {
        const auto& b = bindings[i];
        if (b.rootParam == kInvalidAlloc) continue;
        const BindingSlot slot = slotOf(i);
        if (slot.objectKind == BindingObjectKind::BindlessUAVTexture && slot.objectPtr)
            SetTableOnce(b.rootParam,
                reinterpret_cast<BindlessUAVTexture*>(slot.objectPtr)->GetGPUHandle(), b.name.c_str());
    }

    // Root CBVs. Shared-layout mode: each volatile CBV carries its own root
    // param on the binding (static CBVs live in the combined table and were
    // written by WriteDescriptors). Legacy mode: params are contiguous from
    // rootParamCBVBase, indexed by heapOffset.
    auto resolveCbvVA = [&](const BindingSlot& slot) -> D3D12_GPU_VIRTUAL_ADDRESS {
        // A NativeBuffer resolves its own VA: volatile buffers have no resource
        // and return the current upload-pool suballocation's address; everything
        // else binds the raw resource's base VA.
        if (slot.objectKind == BindingObjectKind::NativeBuffer && slot.objectPtr)
            return reinterpret_cast<INativeResource*>(slot.objectPtr)->GetGpuVirtualAddress();
        ID3D12Resource* res = ResolveBoundResource(slot);
        return res ? res->GetGPUVirtualAddress() : 0;
    };
    if (m_shader->UsesSharedLayout())
    {
        for (uint32_t i : m_idxCBV)
        {
            const auto& b = bindings[i];
            if (b.rootParam == kInvalidAlloc) continue;   // table CBV
            // Unbound volatile CBV (superset layout, shader doesn't read it):
            // leave the root param unset rather than binding VA 0.
            const D3D12_GPU_VIRTUAL_ADDRESS va = resolveCbvVA(slotOf(i));
            if (va) SetCBV(b.rootParam, va);
        }
    }
    else if (rootParamCBVBase != kInvalidAlloc)
    {
        for (uint32_t i : m_idxCBV)
        {
            const auto& b = bindings[i];
            SetCBV(rootParamCBVBase + b.heapOffset, resolveCbvVA(slotOf(i)));
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
            if (va) SetSRV(b.rootParam, va);
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
        SetConstants(
            b.rootParam,
            num32,
            reinterpret_cast<const void*>(static_cast<uintptr_t>(slot.objectPtr)),
            destOffset);
    }
}

// ===========================================================================
// Explicit template instantiations
// ===========================================================================

template class DescriptorSetBase<ComputePipeline>;
template class DescriptorSetBase<RayTraceShader>;
template class DescriptorSetBase<RasterShader>;
