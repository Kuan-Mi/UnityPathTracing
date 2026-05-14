#include "DescriptorSetBase.h"
#include "ComputeShader.h"
#include "RayTraceShader.h"
#include "AccelerationStructure.h"
#include "BindlessTexture.h"
#include "BindlessBuffer.h"
#include "BindlessUAVTexture.h"
#include "NativeBuffer.h"
#include "PluginInternal.h"
#include <cstdio>
#include <cstdarg>

// ===========================================================================
// Constructor / Destructor
// ===========================================================================

template<typename ShaderT>
DescriptorSetBase<ShaderT>::DescriptorSetBase(
    ShaderT*                  shader,
    ID3D12Device*             device,
    IUnityLog*                log,
    DescriptorHeapAllocator*  allocator,
    IUnityGraphicsD3D12v8*    d3d12v8)
    : m_shader(shader), m_device(device), m_log(log), m_allocator(allocator), m_d3d12v8(d3d12v8)
    , m_cachedNumSRV(shader ? shader->GetNumSRV() : 0)
    , m_cachedNumUAV(shader ? shader->GetNumUAV() : 0)
{
}

template<typename ShaderT>
DescriptorSetBase<ShaderT>::~DescriptorSetBase()
{
    FreeAllocations();
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
// FreeAllocations
// ===========================================================================

template<typename ShaderT>
void DescriptorSetBase<ShaderT>::FreeAllocations()
{
    if (!m_allocator) return;
    // Use cached counts — m_shader may already be deleted (shader is enqueued for
    // deferred-delete before descriptor set, so m_shader can be a dangling pointer).
    const uint32_t numSRV = m_cachedNumSRV;
    const uint32_t numUAV = m_cachedNumUAV;
    for (uint32_t f = 0; f < kNumSlots; ++f)
    {
        if (m_srvAllocBase[f] != kInvalidAlloc && numSRV > 0)
            { m_allocator->Free(m_srvAllocBase[f], numSRV); m_srvAllocBase[f] = kInvalidAlloc; }
        if (m_uavAllocBase[f] != kInvalidAlloc && numUAV > 0)
            { m_allocator->Free(m_uavAllocBase[f], numUAV); m_uavAllocBase[f] = kInvalidAlloc; }
    }
}

// ===========================================================================
// AllocateAndWriteDescriptors
// ===========================================================================

template<typename ShaderT>
bool DescriptorSetBase<ShaderT>::AllocateAndWriteDescriptors(
    const CS_BindingSlot* slots, uint32_t slotCount, uint32_t slotIdx)
{
    if (!m_allocator) return false;
    const uint32_t numSRV = m_shader->GetNumSRV();
    const uint32_t numUAV = m_shader->GetNumUAV();
    if (m_srvAllocBase[slotIdx] == kInvalidAlloc && numSRV > 0)
        m_srvAllocBase[slotIdx] = m_allocator->Allocate(numSRV);
    if (m_uavAllocBase[slotIdx] == kInvalidAlloc && numUAV > 0)
        m_uavAllocBase[slotIdx] = m_allocator->Allocate(numUAV);
    UpdateDescriptors(slots, slotCount, slotIdx);
    return true;
}

// ===========================================================================
// UpdateDescriptors
//   Writes all SRV/UAV descriptors from the per-dispatch slot array.
//   CBVs and ROOT_SRVs are bound as inline root descriptors in BindRootParams.
//   SRV_ARRAY / UAV_ARRAY bindings use their own heap (Bindless* objects).
// ===========================================================================

template<typename ShaderT>
void DescriptorSetBase<ShaderT>::UpdateDescriptors(
    const CS_BindingSlot* slots, uint32_t slotCount, uint32_t slotIdx)
{
    const auto& bindings = m_shader->GetBindings();
    const uint32_t f = slotIdx;

    // --- SRV / TLAS ---
    if (m_srvAllocBase[f] != kInvalidAlloc)
    {
        for (size_t i = 0; i < bindings.size(); ++i)
        {
            const auto& b    = bindings[i];
            const CS_BindingSlot& slot = (i < slotCount) ? slots[i] : CS_BindingSlot{};

            if (b.type == ComputeBindingType::ROOT_SRV)
            {
                continue; // bound as inline root descriptor in BindRootParams
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

                s.RaytracingAccelerationStructure.Location = tlas ? tlas->GetGPUVirtualAddress() : 0;
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
                    // Null descriptor — write a safe raw buffer placeholder
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

// ===========================================================================
// RequestResourceStates
// ===========================================================================

template<typename ShaderT>
void DescriptorSetBase<ShaderT>::RequestResourceStates(
    const CS_BindingSlot* slots, uint32_t slotCount)
{
    if (!m_d3d12v8) return;
    const auto& bindings = m_shader->GetBindings();

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
            m_d3d12v8->RequestResourceState(res, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        else if (b.type == ComputeBindingType::CBV)
        {
            ID3D12Resource* cbvRes = (slot.objectKind == CS_BindingObjectKind::NativeBuffer && slot.objectPtr)
                ? reinterpret_cast<::NativeBuffer*>(slot.objectPtr)->GetResource()
                : res;
            if (cbvRes)
                m_d3d12v8->RequestResourceState(cbvRes, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
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
            if (res)
                m_d3d12v8->RequestResourceState(res, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
    }
}

// ===========================================================================
// ValidateBindings
// ===========================================================================

template<typename ShaderT>
bool DescriptorSetBase<ShaderT>::ValidateBindings(
    const CS_BindingSlot* slots, uint32_t slotCount) const
{
    const auto& bindings = m_shader->GetBindings();
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
                 "DescriptorSet::Dispatch: '%s' binding '%s' (%s, space%u, reg%u) is not set",
                 m_shader->GetName(), b.name.c_str(), kind, b.space, b.registerIndex);
            anyMissing = true;
        }
    }
    return !anyMissing;
}

// ===========================================================================
// AcquireSlot
// ===========================================================================

template<typename ShaderT>
void DescriptorSetBase<ShaderT>::AcquireSlot(uint32_t& outSlotIdx)
{
    if (g_frameIndex != m_lastFrameIndex)
    {
        m_subFrameIdx    = 0;
        m_lastFrameIndex = g_frameIndex;
    }
    if (m_subFrameIdx >= kMaxEyesPerFrame)
    {
        Logf(kUnityLogTypeError,
             "DescriptorSet::Dispatch [%s]: more than %u dispatches in one frame. Clamping.",
             m_shader->GetName(), kMaxEyesPerFrame);
        m_subFrameIdx = kMaxEyesPerFrame - 1;
    }
    outSlotIdx = g_frameIndex * kMaxEyesPerFrame + m_subFrameIdx;
    ++m_subFrameIdx;
}

// ===========================================================================
// EnsureDescriptors
// ===========================================================================

template<typename ShaderT>
void DescriptorSetBase<ShaderT>::EnsureDescriptors(
    const CS_BindingSlot* slots, uint32_t slotCount, uint32_t slotIdx)
{
    const uint32_t numSRV = m_shader->GetNumSRV();
    const uint32_t numUAV = m_shader->GetNumUAV();
    if ((numSRV > 0 && m_srvAllocBase[slotIdx] == kInvalidAlloc) ||
        (numUAV > 0 && m_uavAllocBase[slotIdx] == kInvalidAlloc))
    {
        AllocateAndWriteDescriptors(slots, slotCount, slotIdx);
    }
    else
    {
        UpdateDescriptors(slots, slotCount, slotIdx);
    }
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
    const CS_BindingSlot*      slots,
    uint32_t                   slotCount,
    uint32_t                   slotIdx)
{
    // Bind the global shared heap
    ID3D12DescriptorHeap* heapsToBind[1] = { m_allocator->GetHeap() };
    cmdList->SetDescriptorHeaps(1, heapsToBind);

    cmdList->SetComputeRootSignature(m_shader->GetRootSignature());

    const auto& bindings        = m_shader->GetBindings();
    const uint32_t rootParamSRV     = m_shader->GetRootParamSRV();
    const uint32_t rootParamUAV     = m_shader->GetRootParamUAV();
    const uint32_t rootParamCBVBase = m_shader->GetRootParamCBVBase();

    // SRV descriptor table
    if (rootParamSRV != kInvalidAlloc && m_srvAllocBase[slotIdx] != kInvalidAlloc)
        cmdList->SetComputeRootDescriptorTable(rootParamSRV,
            m_allocator->GetGPUHandle(m_srvAllocBase[slotIdx]));

    // UAV descriptor table
    if (rootParamUAV != kInvalidAlloc && m_uavAllocBase[slotIdx] != kInvalidAlloc)
        cmdList->SetComputeRootDescriptorTable(rootParamUAV,
            m_allocator->GetGPUHandle(m_uavAllocBase[slotIdx]));

    // SRV_ARRAY bindings — each has its own root parameter
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

    // UAV_ARRAY bindings — each has its own root parameter
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

    // Inline root SRV per ROOT_SRV binding
    if (m_shader->GetRootParamRootSRVBase() != kInvalidAlloc)
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
}

// ===========================================================================
// Explicit template instantiations
// ===========================================================================

template class DescriptorSetBase<ComputeShader>;
template class DescriptorSetBase<RayTraceShader>;
