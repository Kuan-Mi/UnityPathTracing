#include "ComputeDescriptorSet.h"
#include "AccelerationStructure.h"
#include "BindlessTexture.h"
#include "BindlessBuffer.h"
#include "BindlessUAVTexture.h"
#include "NativeBuffer.h"
#include "PluginInternal.h"
#include <cstdio>
#include <cstdarg>

// ---------------------------------------------------------------------------

ComputeDescriptorSet::ComputeDescriptorSet(ComputeShader*            cs,
                                           ID3D12Device*             device,
                                           IUnityLog*                log,
                                           DescriptorHeapAllocator*  allocator,
                                           IUnityGraphicsD3D12v8*    d3d12v8)
    : m_cs(cs), m_device(device), m_log(log), m_allocator(allocator), m_d3d12v8(d3d12v8)
    , m_cachedNumSRV(cs ? cs->GetNumSRV() : 0)
    , m_cachedNumUAV(cs ? cs->GetNumUAV() : 0)
{
}

ComputeDescriptorSet::~ComputeDescriptorSet()
{
    FreeAllocations();
}

// ---------------------------------------------------------------------------

void ComputeDescriptorSet::Log(UnityLogType type, const char* msg) const
{
    if (m_log) m_log->Log(type, msg, __FILE__, __LINE__);
    else        printf("[ComputeDescriptorSet] %s\n", msg);
}

void ComputeDescriptorSet::Logf(UnityLogType type, const char* fmt, ...) const
{
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    Log(type, buf);
}

// ---------------------------------------------------------------------------
// FreeAllocations
// ---------------------------------------------------------------------------
void ComputeDescriptorSet::FreeAllocations()
{
    if (!m_allocator) return;
    // Use cached counts — m_cs may already be deleted (ComputeShader is enqueued for
    // deferred-delete before ComputeDescriptorSet, so dereferencing m_cs here is UAF).
    uint32_t numSRV = m_cachedNumSRV;
    uint32_t numUAV = m_cachedNumUAV;
    for (uint32_t f = 0; f < kNumSlots; ++f)
    {
        if (m_srvAllocBase[f] != kInvalidAlloc && numSRV > 0) { m_allocator->Free(m_srvAllocBase[f], numSRV); m_srvAllocBase[f] = kInvalidAlloc; }
        if (m_uavAllocBase[f] != kInvalidAlloc && numUAV > 0) { m_allocator->Free(m_uavAllocBase[f], numUAV); m_uavAllocBase[f] = kInvalidAlloc; }
    }
}

// ---------------------------------------------------------------------------
// AllocateAndWriteDescriptors
// ---------------------------------------------------------------------------
bool ComputeDescriptorSet::AllocateAndWriteDescriptors(const CS_BindingSlot* slots, uint32_t slotCount, uint32_t slotIdx)
{
    if (!m_allocator) return false;
    uint32_t numSRV = m_cs->GetNumSRV();
    uint32_t numUAV = m_cs->GetNumUAV();
    if (m_srvAllocBase[slotIdx] == kInvalidAlloc && numSRV > 0)
        m_srvAllocBase[slotIdx] = m_allocator->Allocate(numSRV);
    if (m_uavAllocBase[slotIdx] == kInvalidAlloc && numUAV > 0)
        m_uavAllocBase[slotIdx] = m_allocator->Allocate(numUAV);
    UpdateDescriptors(slots, slotCount, slotIdx);
    return true;
}

// ---------------------------------------------------------------------------
// UpdateDescriptors
//   Writes all SRV/UAV descriptors using the per-dispatch slot array.
//   CBVs are bound as inline root descriptors in Dispatch.
//   SRV_ARRAY bindings use their own heap in BindlessTexture/BindlessBuffer.
// ---------------------------------------------------------------------------
void ComputeDescriptorSet::UpdateDescriptors(const CS_BindingSlot* slots, uint32_t slotCount, uint32_t slotIdx)
{
    const auto& bindings = m_cs->GetBindings();

    // --- SRV / TLAS ---
    const uint32_t f = slotIdx;
    if (m_srvAllocBase[f] != kInvalidAlloc)
    {
        for (size_t i = 0; i < bindings.size(); ++i)
        {
            const auto& b = bindings[i];
            const CS_BindingSlot& slot = (i < slotCount) ? slots[i] : CS_BindingSlot{};

            if (b.type == ComputeBindingType::ROOT_SRV)
            {
                continue; // handled as inline root descriptor in Dispatch
            }
            else if (b.type == ComputeBindingType::TLAS)
            {
                D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
                s.ViewDimension           = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
                s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

                ID3D12Resource* tlas = nullptr;
                if (slot.objectKind == CS_BindingObjectKind::AccelStruct && slot.objectPtr)
                    tlas = reinterpret_cast<AccelerationStructure*>(slot.objectPtr)->GetTLAS();
                else
                    tlas = reinterpret_cast<ID3D12Resource*>(slot.resourcePtr);

                auto tlasVA = tlas ? tlas->GetGPUVirtualAddress() : 0;
                s.RaytracingAccelerationStructure.Location = tlasVA;
                m_device->CreateShaderResourceView(nullptr, &s,
                    m_allocator->GetCPUHandle(m_srvAllocBase[f] + b.heapOffset));
            }
            else if (b.type == ComputeBindingType::SRV)
            {
                D3D12_CPU_DESCRIPTOR_HANDLE h = m_allocator->GetCPUHandle(m_srvAllocBase[f] + b.heapOffset);
                D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
                s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                ID3D12Resource* res = reinterpret_cast<ID3D12Resource*>(slot.resourcePtr);
                if (res)
                {
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
                            // Typed buffer (e.g. Buffer<float2> -> DXGI_FORMAT_R32G32_FLOAT)
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
                    else
                    {
                        s.ViewDimension       = D3D12_SRV_DIMENSION_TEXTURE2D;
                        s.Format              = rd.Format;
                        s.Texture2D.MipLevels = rd.MipLevels;
                    }
                    m_device->CreateShaderResourceView(res, &s, h);
                }
                else
                {
                    s.ViewDimension      = D3D12_SRV_DIMENSION_BUFFER;
                    s.Format             = DXGI_FORMAT_R32_TYPELESS;
                    s.Buffer.Flags       = D3D12_BUFFER_SRV_FLAG_RAW;
                    s.Buffer.NumElements = 1;
                    m_device->CreateShaderResourceView(nullptr, &s, h);
                }
            }
        }
    }

    // --- UAV ---
    if (m_uavAllocBase[f] != kInvalidAlloc)
    {
        for (size_t i = 0; i < bindings.size(); ++i)
        {
            const auto& b = bindings[i];
            if (b.type != ComputeBindingType::UAV) continue;
            const CS_BindingSlot& slot = (i < slotCount) ? slots[i] : CS_BindingSlot{};
            ID3D12Resource* res = reinterpret_cast<ID3D12Resource*>(slot.resourcePtr);
            D3D12_CPU_DESCRIPTOR_HANDLE h = m_allocator->GetCPUHandle(m_uavAllocBase[f] + b.heapOffset);
            D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
            if (res)
            {
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
                        // Typed buffer (e.g. RWBuffer<uint2> -> DXGI_FORMAT_R32G32_UINT)
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
                else
                {
                    u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                    u.Format        = rd.Format;
                }
                m_device->CreateUnorderedAccessView(res, nullptr, &u, h);
            }
            else
            {
                u.ViewDimension      = D3D12_UAV_DIMENSION_BUFFER;
                u.Format             = DXGI_FORMAT_R32_TYPELESS;
                u.Buffer.Flags       = D3D12_BUFFER_UAV_FLAG_RAW;
                u.Buffer.NumElements = 1;
                m_device->CreateUnorderedAccessView(nullptr, nullptr, &u, h);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RequestResourceStates
// ---------------------------------------------------------------------------
void ComputeDescriptorSet::RequestResourceStates(const CS_BindingSlot* slots, uint32_t slotCount)
{
    if (!m_d3d12v8) return;
    const auto& bindings = m_cs->GetBindings();
    const char* shaderName = m_cs->GetName();

    for (size_t i = 0; i < bindings.size(); ++i)
    {
        const auto& b = bindings[i];
        const CS_BindingSlot& slot = (i < slotCount) ? slots[i] : CS_BindingSlot{};
        ID3D12Resource* res = reinterpret_cast<ID3D12Resource*>(slot.resourcePtr);
        if (b.type == ComputeBindingType::SRV && res)
        {
            m_d3d12v8->RequestResourceState(res, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        else if (b.type == ComputeBindingType::ROOT_SRV)
        {
            ID3D12Resource* srvRes = nullptr;
            if (slot.objectKind == CS_BindingObjectKind::AccelStruct && slot.objectPtr)
                srvRes = reinterpret_cast<AccelerationStructure*>(slot.objectPtr)->GetTLAS();
            else
                srvRes = res;
            if (srvRes)
                m_d3d12v8->RequestResourceState(srvRes, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        else if (b.type == ComputeBindingType::UAV && res)
        {
            // Logf(kUnityLogTypeLog, "  [%s] RequestResourceState: '%s' (UAV) res=%p -> UNORDERED_ACCESS (0x%X)",
            //      shaderName, b.name.c_str(), (void*)res, (UINT)D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            m_d3d12v8->RequestResourceState(res, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        else if (b.type == ComputeBindingType::CBV)
        {
            ID3D12Resource* cbvRes = (slot.objectKind == CS_BindingObjectKind::NativeBuffer && slot.objectPtr)
                ? reinterpret_cast<::NativeBuffer*>(slot.objectPtr)->GetResource()
                : res;
            if (cbvRes)
            {
                // Logf(kUnityLogTypeLog, "  [%s] RequestResourceState: '%s' (CBV) res=%p -> VERTEX_AND_CONSTANT_BUFFER (0x%X)",
                //      shaderName, b.name.c_str(), (void*)cbvRes, (UINT)D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
                m_d3d12v8->RequestResourceState(cbvRes, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
            }
        }
        else if (b.type == ComputeBindingType::SRV_ARRAY &&
                 slot.objectKind == CS_BindingObjectKind::BindlessTexture && slot.objectPtr)
        {
            auto* bt = reinterpret_cast<BindlessTexture*>(slot.objectPtr);
            for (uint32_t k = 0; k < bt->Capacity(); ++k)
            {
                ID3D12Resource* tex = bt->GetTexture(k);
                if (tex)
                    m_d3d12v8->RequestResourceState(tex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
        }
        else if (b.type == ComputeBindingType::SRV_ARRAY &&
                 slot.objectKind == CS_BindingObjectKind::BindlessBuffer && slot.objectPtr)
        {
            auto* bb = reinterpret_cast<BindlessBuffer*>(slot.objectPtr);
            for (uint32_t k = 0; k < bb->Capacity(); ++k)
            {
                ID3D12Resource* buf = bb->GetBuffer(k);
                if (buf)
                    m_d3d12v8->RequestResourceState(buf, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
        }
        else if (b.type == ComputeBindingType::UAV_ARRAY &&
                 slot.objectKind == CS_BindingObjectKind::BindlessUAVTexture && slot.objectPtr)
        {
            // Individual mip-level UAVs inside BindlessUAVTexture share the same resource.
            // Request UNORDERED_ACCESS on each unique resource.
            auto* uavArr = reinterpret_cast<BindlessUAVTexture*>(slot.objectPtr);
            // We only have per-slot resource access via the SlotInfo inside BindlessUAVTexture.
            // Since BindlessUAVTexture doesn't expose per-slot resources directly, request state
            // on the resource passed via resourcePtr (the base texture).
            ID3D12Resource* res = reinterpret_cast<ID3D12Resource*>(slot.resourcePtr);
            if (res)
                m_d3d12v8->RequestResourceState(res, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------
void ComputeDescriptorSet::Dispatch(
    ID3D12GraphicsCommandList* cmdList,
    UINT threadGroupX, UINT threadGroupY, UINT threadGroupZ,
    const CS_BindingSlot* slots, uint32_t slotCount)
{
    if (!m_cs || !m_cs->GetPSO() || !m_cs->GetRootSignature() || !m_allocator) return;
    if (!slots && slotCount > 0) return;

    const auto& bindings = m_cs->GetBindings();

    // --- Validate all bindings ---
    {
        bool anyMissing = false;
        for (size_t i = 0; i < bindings.size(); ++i)
        {
            const auto& b = bindings[i];
            const CS_BindingSlot& slot = (i < slotCount) ? slots[i] : CS_BindingSlot{};
            bool ok = false;
            const char* kind = "?";
            switch (b.type)
            {
            case ComputeBindingType::TLAS:
                kind = "TLAS";
                ok = slot.resourcePtr != 0 ||
                     (slot.objectKind == CS_BindingObjectKind::AccelStruct && slot.objectPtr != 0);
                break;
            case ComputeBindingType::SRV:
                kind = "SRV";
                ok = slot.resourcePtr != 0;
                break;
            case ComputeBindingType::UAV:
                kind = "UAV";
                ok = slot.resourcePtr != 0;
                break;
            case ComputeBindingType::ROOT_SRV:
                kind = "ROOT_SRV";
                ok = slot.resourcePtr != 0 ||
                     (slot.objectKind == CS_BindingObjectKind::AccelStruct && slot.objectPtr != 0);
                break;
            case ComputeBindingType::CBV:
                kind = "CBV";
                ok = slot.resourcePtr != 0 ||
                     (slot.objectKind == CS_BindingObjectKind::NativeBuffer && slot.objectPtr != 0);
                break;
            case ComputeBindingType::SRV_ARRAY:
                kind = "SRV_ARRAY";
                ok = (slot.objectKind == CS_BindingObjectKind::BindlessTexture && slot.objectPtr != 0) ||
                     (slot.objectKind == CS_BindingObjectKind::BindlessBuffer  && slot.objectPtr != 0);
                break;
            case ComputeBindingType::UAV_ARRAY:
                kind = "UAV_ARRAY";
                ok = (slot.objectKind == CS_BindingObjectKind::BindlessUAVTexture && slot.objectPtr != 0);
                break;
            case ComputeBindingType::ROOT_CONSTANTS:
                kind = "ROOT_CONSTANTS";
                ok = slot.objectPtr != 0;
                break;
            }
            if (!ok)
            {
                Logf(kUnityLogTypeError,
                     "ComputeDescriptorSet::Dispatch: '%s' binding '%s' (%s, space%u, reg%u) is not set",
                        m_cs->GetName(),
                      b.name.c_str(), kind, b.space, b.registerIndex);
                anyMissing = true;
            }
        }
        if (anyMissing) return;
    }

    // Compute per-eye slot index: resets each new frame, increments each Dispatch call.
    // This gives eye0 and eye1 independent GPU heap descriptor slices within the same frame.
    if (g_frameIndex != m_lastFrameIndex)
    {
        m_subFrameIdx    = 0;
        m_lastFrameIndex = g_frameIndex;
    }
    if (m_subFrameIdx >= kMaxEyesPerFrame)
    {
        Logf(kUnityLogTypeError,
             "ComputeDescriptorSet::Dispatch [%s]: more than %u dispatches in one frame "
             "(g_frameIndex=%u). Increase kMaxEyesPerFrame or check for unexpected re-use.",
             m_cs ? m_cs->GetName() : "?", kMaxEyesPerFrame, g_frameIndex);
        // Clamp to last valid slot to avoid out-of-bounds access.
        m_subFrameIdx = kMaxEyesPerFrame - 1;
    }
    const uint32_t slotIdx = g_frameIndex * kMaxEyesPerFrame + m_subFrameIdx;
    ++m_subFrameIdx;

    // Allocate heap slots for this slot on first use, then write descriptors
    uint32_t numSRV = m_cs->GetNumSRV();
    uint32_t numUAV = m_cs->GetNumUAV();
    if ((numSRV > 0 && m_srvAllocBase[slotIdx] == kInvalidAlloc) ||
        (numUAV > 0 && m_uavAllocBase[slotIdx] == kInvalidAlloc))
    {
        if (!AllocateAndWriteDescriptors(slots, slotCount, slotIdx)) return;
    }
    else
    {
        UpdateDescriptors(slots, slotCount, slotIdx);
    }

    // Bind the global shared heap
    ID3D12DescriptorHeap* heapsToBind[1] = { m_allocator->GetHeap() };
    cmdList->SetDescriptorHeaps(1, heapsToBind);

    cmdList->SetPipelineState(m_cs->GetPSO());
    cmdList->SetComputeRootSignature(m_cs->GetRootSignature());

    uint32_t rootParamSRV     = m_cs->GetRootParamSRV();
    uint32_t rootParamUAV     = m_cs->GetRootParamUAV();
    uint32_t rootParamCBVBase = m_cs->GetRootParamCBVBase();

    // SRV table
    if (rootParamSRV != kInvalidAlloc && m_srvAllocBase[slotIdx] != kInvalidAlloc)
        cmdList->SetComputeRootDescriptorTable(rootParamSRV,
            m_allocator->GetGPUHandle(m_srvAllocBase[slotIdx]));

    // UAV table
    if (rootParamUAV != kInvalidAlloc && m_uavAllocBase[slotIdx] != kInvalidAlloc)
        cmdList->SetComputeRootDescriptorTable(rootParamUAV,
            m_allocator->GetGPUHandle(m_uavAllocBase[slotIdx]));

    // SRV_ARRAY bindings – each has its own root parameter
    for (size_t i = 0; i < bindings.size(); ++i)
    {
        const auto& b = bindings[i];
        if (b.type != ComputeBindingType::SRV_ARRAY) continue;
        if (b.rootParam == kInvalidAlloc) continue;
        const CS_BindingSlot& slot = (i < slotCount) ? slots[i] : CS_BindingSlot{};
        if (slot.objectKind == CS_BindingObjectKind::BindlessTexture && slot.objectPtr)
            cmdList->SetComputeRootDescriptorTable(b.rootParam,
                reinterpret_cast<BindlessTexture*>(slot.objectPtr)->GetGPUHandle());
        else if (slot.objectKind == CS_BindingObjectKind::BindlessBuffer && slot.objectPtr)
            cmdList->SetComputeRootDescriptorTable(b.rootParam,
                reinterpret_cast<BindlessBuffer*>(slot.objectPtr)->GetGPUHandle());
    }

    // UAV_ARRAY bindings – each has its own root parameter
    for (size_t i = 0; i < bindings.size(); ++i)
    {
        const auto& b = bindings[i];
        if (b.type != ComputeBindingType::UAV_ARRAY) continue;
        if (b.rootParam == kInvalidAlloc) continue;
        const CS_BindingSlot& slot = (i < slotCount) ? slots[i] : CS_BindingSlot{};
        if (slot.objectKind == CS_BindingObjectKind::BindlessUAVTexture && slot.objectPtr)
            cmdList->SetComputeRootDescriptorTable(b.rootParam,
                reinterpret_cast<BindlessUAVTexture*>(slot.objectPtr)->GetGPUHandle());
    }

    // Root CBV per CBV binding
    if (rootParamCBVBase != kInvalidAlloc)
    {
        for (size_t i = 0; i < bindings.size(); ++i)
        {
            const auto& b = bindings[i];
            if (b.type != ComputeBindingType::CBV) continue;
            const CS_BindingSlot& slot = (i < slotCount) ? slots[i] : CS_BindingSlot{};
            ID3D12Resource* res = nullptr;
            if (slot.objectKind == CS_BindingObjectKind::NativeBuffer && slot.objectPtr)
                res = reinterpret_cast<::NativeBuffer*>(slot.objectPtr)->GetResource();
            else
                res = reinterpret_cast<ID3D12Resource*>(slot.resourcePtr);
            D3D12_GPU_VIRTUAL_ADDRESS addr = res ? res->GetGPUVirtualAddress() : 0;
            cmdList->SetComputeRootConstantBufferView(rootParamCBVBase + b.heapOffset, addr);
        }
    }

    // Root inline SRV per ROOT_SRV binding
    if (m_cs->GetRootParamRootSRVBase() != kInvalidAlloc)
    {
        for (size_t i = 0; i < bindings.size(); ++i)
        {
            const auto& b = bindings[i];
            if (b.type != ComputeBindingType::ROOT_SRV) continue;
            if (b.rootParam == kInvalidAlloc) continue;
            const CS_BindingSlot& slot = (i < slotCount) ? slots[i] : CS_BindingSlot{};
            D3D12_GPU_VIRTUAL_ADDRESS va = 0;
            if (slot.objectKind == CS_BindingObjectKind::AccelStruct && slot.objectPtr)
            {
                ID3D12Resource* tlas = reinterpret_cast<AccelerationStructure*>(slot.objectPtr)->GetTLAS();
                if (tlas) va = tlas->GetGPUVirtualAddress();
            }
            else
            {
                auto* res = reinterpret_cast<ID3D12Resource*>(slot.resourcePtr);
                if (res) va = res->GetGPUVirtualAddress();
            }
            if (va) cmdList->SetComputeRootShaderResourceView(b.rootParam, va);
        }
    }

    // Root 32-bit constants per ROOT_CONSTANTS binding
    for (size_t i = 0; i < bindings.size(); ++i)
    {
        const auto& b = bindings[i];
        if (b.type != ComputeBindingType::ROOT_CONSTANTS) continue;
        if (b.rootParam == kInvalidAlloc) continue;
        const CS_BindingSlot& slot = (i < slotCount) ? slots[i] : CS_BindingSlot{};
        if (!slot.objectPtr) continue;
        UINT num32      = slot.count > 0 ? slot.count : b.num32BitValues;
        UINT destOffset = slot.stride; // destOffsetIn32BitValues
        cmdList->SetComputeRoot32BitConstants(
            b.rootParam,
            num32,
            reinterpret_cast<const void*>(static_cast<uintptr_t>(slot.objectPtr)),
            destOffset);
    }

    // Resource state requests
    RequestResourceStates(slots, slotCount);

    cmdList->Dispatch(threadGroupX, threadGroupY, threadGroupZ);
}
