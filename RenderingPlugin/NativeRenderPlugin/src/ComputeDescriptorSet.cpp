#include "ComputeDescriptorSet.h"
#include "AccelerationStructure.h"
#include "BindlessTexture.h"
#include "BindlessBuffer.h"
#include <cstdio>
#include <cstdarg>

// ---------------------------------------------------------------------------

ComputeDescriptorSet::ComputeDescriptorSet(ComputeShader*            cs,
                                           ID3D12Device*             device,
                                           IUnityLog*                log,
                                           DescriptorHeapAllocator*  allocator,
                                           IUnityGraphicsD3D12v8*    d3d12v8)
    : m_cs(cs), m_device(device), m_log(log), m_allocator(allocator), m_d3d12v8(d3d12v8)
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
    uint32_t numSRV = m_cs ? m_cs->GetNumSRV() : 0;
    uint32_t numUAV = m_cs ? m_cs->GetNumUAV() : 0;
    if (m_srvAllocBase != kInvalidAlloc && numSRV > 0) { m_allocator->Free(m_srvAllocBase, numSRV); m_srvAllocBase = kInvalidAlloc; }
    if (m_uavAllocBase != kInvalidAlloc && numUAV > 0) { m_allocator->Free(m_uavAllocBase, numUAV); m_uavAllocBase = kInvalidAlloc; }
}

// ---------------------------------------------------------------------------
// AllocateAndWriteDescriptors
// ---------------------------------------------------------------------------
bool ComputeDescriptorSet::AllocateAndWriteDescriptors(const CS_BindingSlot* slots, uint32_t slotCount)
{
    if (!m_allocator) return false;
    uint32_t numSRV = m_cs->GetNumSRV();
    uint32_t numUAV = m_cs->GetNumUAV();
    if (m_srvAllocBase == kInvalidAlloc && numSRV > 0)
        m_srvAllocBase = m_allocator->Allocate(numSRV);
    if (m_uavAllocBase == kInvalidAlloc && numUAV > 0)
        m_uavAllocBase = m_allocator->Allocate(numUAV);
    UpdateDescriptors(slots, slotCount);
    return true;
}

// ---------------------------------------------------------------------------
// UpdateDescriptors
//   Writes all SRV/UAV descriptors using the per-dispatch slot array.
//   CBVs are bound as inline root descriptors in Dispatch.
//   SRV_ARRAY bindings use their own heap in BindlessTexture/BindlessBuffer.
// ---------------------------------------------------------------------------
void ComputeDescriptorSet::UpdateDescriptors(const CS_BindingSlot* slots, uint32_t slotCount)
{
    const auto& bindings = m_cs->GetBindings();

    // --- SRV / TLAS ---
    if (m_srvAllocBase != kInvalidAlloc)
    {
        for (size_t i = 0; i < bindings.size(); ++i)
        {
            const auto& b = bindings[i];
            const CS_BindingSlot& slot = (i < slotCount) ? slots[i] : CS_BindingSlot{};

            if (b.type == ComputeBindingType::TLAS)
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
                    m_allocator->GetCPUHandle(m_srvAllocBase + b.heapOffset));
            }
            else if (b.type == ComputeBindingType::SRV)
            {
                D3D12_CPU_DESCRIPTOR_HANDLE h = m_allocator->GetCPUHandle(m_srvAllocBase + b.heapOffset);
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
    if (m_uavAllocBase != kInvalidAlloc)
    {
        for (size_t i = 0; i < bindings.size(); ++i)
        {
            const auto& b = bindings[i];
            if (b.type != ComputeBindingType::UAV) continue;
            const CS_BindingSlot& slot = (i < slotCount) ? slots[i] : CS_BindingSlot{};
            ID3D12Resource* res = reinterpret_cast<ID3D12Resource*>(slot.resourcePtr);
            D3D12_CPU_DESCRIPTOR_HANDLE h = m_allocator->GetCPUHandle(m_uavAllocBase + b.heapOffset);
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
            Logf(kUnityLogTypeLog, "  [%s] RequestResourceState: '%s' (SRV) res=%p -> NON_PIXEL_SHADER_RESOURCE (0x%X)",
                 shaderName, b.name.c_str(), (void*)res, (UINT)D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            m_d3d12v8->RequestResourceState(res, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        else if (b.type == ComputeBindingType::UAV && res)
        {
            Logf(kUnityLogTypeLog, "  [%s] RequestResourceState: '%s' (UAV) res=%p -> UNORDERED_ACCESS (0x%X)",
                 shaderName, b.name.c_str(), (void*)res, (UINT)D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            m_d3d12v8->RequestResourceState(res, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        else if (b.type == ComputeBindingType::CBV && res)
        {
            Logf(kUnityLogTypeLog, "  [%s] RequestResourceState: '%s' (CBV) res=%p -> VERTEX_AND_CONSTANT_BUFFER (0x%X)",
                 shaderName, b.name.c_str(), (void*)res, (UINT)D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
            m_d3d12v8->RequestResourceState(res, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
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
            case ComputeBindingType::CBV:
                kind = "CBV";
                ok = slot.resourcePtr != 0;
                break;
            case ComputeBindingType::SRV_ARRAY:
                kind = "SRV_ARRAY";
                ok = (slot.objectKind == CS_BindingObjectKind::BindlessTexture && slot.objectPtr != 0) ||
                     (slot.objectKind == CS_BindingObjectKind::BindlessBuffer  && slot.objectPtr != 0);
                break;
            }
            if (!ok)
            {
                Logf(kUnityLogTypeError,
                     "ComputeDescriptorSet::Dispatch: binding '%s' (%s, space%u, reg%u) is not set",
                     b.name.c_str(), kind, b.space, b.registerIndex);
                anyMissing = true;
            }
        }
        if (anyMissing) return;
    }

    // Allocate heap slots on first call, then write all descriptors every dispatch
    uint32_t numSRV = m_cs->GetNumSRV();
    uint32_t numUAV = m_cs->GetNumUAV();
    if ((numSRV > 0 && m_srvAllocBase == kInvalidAlloc) ||
        (numUAV > 0 && m_uavAllocBase == kInvalidAlloc))
    {
        if (!AllocateAndWriteDescriptors(slots, slotCount)) return;
    }
    else
    {
        UpdateDescriptors(slots, slotCount);
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
    if (rootParamSRV != kInvalidAlloc && m_srvAllocBase != kInvalidAlloc)
        cmdList->SetComputeRootDescriptorTable(rootParamSRV,
            m_allocator->GetGPUHandle(m_srvAllocBase));

    // UAV table
    if (rootParamUAV != kInvalidAlloc && m_uavAllocBase != kInvalidAlloc)
        cmdList->SetComputeRootDescriptorTable(rootParamUAV,
            m_allocator->GetGPUHandle(m_uavAllocBase));

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

    // Root CBV per CBV binding
    if (rootParamCBVBase != kInvalidAlloc)
    {
        for (size_t i = 0; i < bindings.size(); ++i)
        {
            const auto& b = bindings[i];
            if (b.type != ComputeBindingType::CBV) continue;
            const CS_BindingSlot& slot = (i < slotCount) ? slots[i] : CS_BindingSlot{};
            ID3D12Resource* res = reinterpret_cast<ID3D12Resource*>(slot.resourcePtr);
            D3D12_GPU_VIRTUAL_ADDRESS addr = res ? res->GetGPUVirtualAddress() : 0;
            cmdList->SetComputeRootConstantBufferView(rootParamCBVBase + b.heapOffset, addr);
        }
    }

    // Resource state requests
    RequestResourceStates(slots, slotCount);

    cmdList->Dispatch(threadGroupX, threadGroupY, threadGroupZ);
}
