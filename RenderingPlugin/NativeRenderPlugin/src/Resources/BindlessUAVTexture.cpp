#include "BindlessUAVTexture.h"
#include "PluginInternal.h"
#include <cstdio>
#include <cassert>

// ---------------------------------------------------------------------------
// BindlessUAVTexture
// ---------------------------------------------------------------------------

static void WriteNullUAV(ID3D12Device* device, D3D12_CPU_DESCRIPTOR_HANDLE h)
{
    // Null RWTexture2D<float> UAV — one dummy R32_FLOAT texel.
    D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
    u.ViewDimension        = D3D12_UAV_DIMENSION_TEXTURE2D;
    u.Format               = DXGI_FORMAT_R32_FLOAT;
    u.Texture2D.MipSlice   = 0;
    u.Texture2D.PlaneSlice = 0;
    device->CreateUnorderedAccessView(nullptr, nullptr, &u, h);
}

// ---------------------------------------------------------------------------

void BindlessUAVTexture::Log(UnityLogType type, const char* msg) const
{
    if (m_log) m_log->Log(type, msg, __FILE__, __LINE__);
    else        printf("[BindlessUAVTexture] %s\n", msg);
}

bool BindlessUAVTexture::Initialize(ID3D12Device*            device,
                                    DescriptorHeapAllocator* allocator,
                                    uint32_t                 capacity,
                                    IUnityLog*               log)
{
    assert(!m_initialized && "BindlessUAVTexture::Initialize called twice");
    m_device    = device;
    m_allocator = allocator;
    m_log       = log;

    if (capacity == 0)
    {
        Log(kUnityLogTypeWarning, "BindlessUAVTexture: capacity 0 – at least 1 required");
        capacity = 1;
    }

    m_capacity  = capacity;
    m_allocBase = m_allocator->Allocate(capacity);
    m_slots.resize(capacity);

    // Write null descriptors for all initial slots
    for (uint32_t i = 0; i < capacity; ++i)
        WriteNullDescriptor(i);

    m_initialized = true;
    return true;
}

BindlessUAVTexture::~BindlessUAVTexture()
{
    if (m_initialized && m_allocator && m_capacity > 0)
        m_allocator->Free(m_allocBase, m_capacity);
}

void BindlessUAVTexture::SetTexture(uint32_t index, ID3D12Resource* resource,
                                    uint32_t mipSlice, DXGI_FORMAT format)
{
    if (index >= m_capacity)
    {
        Log(kUnityLogTypeWarning, "BindlessUAVTexture::SetTexture: index out of range");
        return;
    }
    m_slots[index].resource = resource;
    m_slots[index].mipSlice = mipSlice;
    m_slots[index].format   = format;
    WriteDescriptor(index);
}

void BindlessUAVTexture::Resize(uint32_t newCapacity)
{
    if (newCapacity == 0) newCapacity = 1;
    if (newCapacity == m_capacity) return;

    // Defer freeing the old descriptor range until the GPU is done with it.
    NR_EnqueueDescriptorRangeFree(m_allocator, m_allocBase, m_capacity);
    m_allocBase = m_allocator->Allocate(newCapacity);

    uint32_t oldCapacity = m_capacity;
    m_slots.resize(newCapacity);
    m_capacity = newCapacity;

    // Re-write all descriptors in the new range
    for (uint32_t i = 0; i < newCapacity; ++i)
    {
        if (i < oldCapacity)
            WriteDescriptor(i);
        else
            WriteNullDescriptor(i);
    }
}

D3D12_GPU_DESCRIPTOR_HANDLE BindlessUAVTexture::GetGPUHandle() const
{
    return m_allocator->GetGPUHandle(m_allocBase);
}

void BindlessUAVTexture::WriteDescriptor(uint32_t index)
{
    D3D12_CPU_DESCRIPTOR_HANDLE h = m_allocator->GetCPUHandle(m_allocBase + index);
    const SlotInfo& s = m_slots[index];

    if (!s.resource)
    {
        WriteNullDescriptor(index);
        return;
    }

    D3D12_RESOURCE_DESC rd = s.resource->GetDesc();
    if (rd.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D)
    {
        Log(kUnityLogTypeWarning, "BindlessUAVTexture::WriteDescriptor: resource is not Texture2D");
        WriteNullDescriptor(index);
        return;
    }

    // Derive format from resource if caller passed DXGI_FORMAT_UNKNOWN
    DXGI_FORMAT fmt = (s.format != DXGI_FORMAT_UNKNOWN) ? s.format : rd.Format;

    D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
    u.Format               = fmt;
    u.ViewDimension        = D3D12_UAV_DIMENSION_TEXTURE2D;
    u.Texture2D.MipSlice   = s.mipSlice;
    u.Texture2D.PlaneSlice = 0;

    m_device->CreateUnorderedAccessView(s.resource, nullptr, &u, h);
}

void BindlessUAVTexture::WriteNullDescriptor(uint32_t index)
{
    D3D12_CPU_DESCRIPTOR_HANDLE h = m_allocator->GetCPUHandle(m_allocBase + index);
    WriteNullUAV(m_device, h);
}
