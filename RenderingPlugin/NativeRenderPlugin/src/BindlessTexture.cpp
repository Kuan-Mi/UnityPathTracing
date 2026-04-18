#include "BindlessTexture.h"
#include <cstdio>
#include <cassert>

// ---------------------------------------------------------------------------
// BindlessTexture
// ---------------------------------------------------------------------------

static void WriteNullSRV(ID3D12Device* device, D3D12_CPU_DESCRIPTOR_HANDLE h)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
    s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    s.ViewDimension           = D3D12_SRV_DIMENSION_TEXTURE2D;
    s.Format                  = DXGI_FORMAT_R8G8B8A8_UNORM;
    s.Texture2D.MipLevels     = 1;
    device->CreateShaderResourceView(nullptr, &s, h);
}

// ---------------------------------------------------------------------------

void BindlessTexture::Log(UnityLogType type, const char* msg) const
{
    if (m_log) m_log->Log(type, msg, __FILE__, __LINE__);
    else        printf("[BindlessTexture] %s\n", msg);
}

bool BindlessTexture::Initialize(ID3D12Device*            device,
                                 DescriptorHeapAllocator* allocator,
                                 uint32_t                 capacity,
                                 IUnityLog*               log)
{
    assert(!m_initialized && "BindlessTexture::Initialize called twice");
    m_device    = device;
    m_allocator = allocator;
    m_log       = log;

    if (capacity == 0)
    {
        Log(kUnityLogTypeWarning, "BindlessTexture: capacity 0 – at least 1 required");
        capacity = 1;
    }

    m_capacity  = capacity;
    m_allocBase = m_allocator->Allocate(capacity);
    m_textures.assign(capacity, nullptr);

    // Write null descriptors for all initial slots
    for (uint32_t i = 0; i < capacity; ++i)
        WriteNullDescriptor(i);

    m_initialized = true;
    return true;
}

BindlessTexture::~BindlessTexture()
{
    if (m_initialized && m_allocator && m_capacity > 0)
        m_allocator->Free(m_allocBase, m_capacity);
}

void BindlessTexture::SetTexture(uint32_t index, ID3D12Resource* resource)
{
    if (index >= m_capacity)
    {
        Log(kUnityLogTypeWarning, "BindlessTexture::SetTexture: index out of range");
        return;
    }
    m_textures[index] = resource;
    WriteDescriptor(index, resource);
}

void BindlessTexture::Resize(uint32_t newCapacity)
{
    if (newCapacity == 0) newCapacity = 1;
    if (newCapacity == m_capacity) return;

    // Free old range and allocate new one
    m_allocator->Free(m_allocBase, m_capacity);
    m_allocBase = m_allocator->Allocate(newCapacity);

    // Resize texture pointer array (preserving existing entries up to min)
    uint32_t oldCapacity = m_capacity;
    m_textures.resize(newCapacity, nullptr);
    m_capacity = newCapacity;

    // Re-write all descriptors in the new range
    for (uint32_t i = 0; i < newCapacity; ++i)
    {
        ID3D12Resource* res = (i < m_textures.size()) ? m_textures[i] : nullptr;
        WriteDescriptor(i, res);
    }
    (void)oldCapacity;
}

D3D12_GPU_DESCRIPTOR_HANDLE BindlessTexture::GetGPUHandle() const
{
    return m_allocator->GetGPUHandle(m_allocBase);
}

void BindlessTexture::WriteDescriptor(uint32_t index, ID3D12Resource* resource)
{
    D3D12_CPU_DESCRIPTOR_HANDLE h = m_allocator->GetCPUHandle(m_allocBase + index);

    if (!resource)
    {
        WriteNullSRV(m_device, h);
        return;
    }

    D3D12_RESOURCE_DESC rd = resource->GetDesc();
    D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
    s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

    if (rd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D)
    {
        if (rd.DepthOrArraySize == 6)
        {
            s.ViewDimension          = D3D12_SRV_DIMENSION_TEXTURECUBE;
            s.Format                 = rd.Format;
            s.TextureCube.MipLevels  = rd.MipLevels;
        }
        else
        {
            s.ViewDimension       = D3D12_SRV_DIMENSION_TEXTURE2D;
            s.Format              = rd.Format;
            s.Texture2D.MipLevels = rd.MipLevels;
        }
    }
    else if (rd.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
    {
        // Treat raw buffers as ByteAddressBuffer SRV
        s.ViewDimension          = D3D12_SRV_DIMENSION_BUFFER;
        s.Format                 = DXGI_FORMAT_R32_TYPELESS;
        s.Buffer.Flags           = D3D12_BUFFER_SRV_FLAG_RAW;
        s.Buffer.NumElements     = static_cast<UINT>(rd.Width / 4);
    }
    else
    {
        WriteNullSRV(m_device, h);
        return;
    }

    m_device->CreateShaderResourceView(resource, &s, h);
}

void BindlessTexture::WriteNullDescriptor(uint32_t index)
{
    D3D12_CPU_DESCRIPTOR_HANDLE h = m_allocator->GetCPUHandle(m_allocBase + index);
    WriteNullSRV(m_device, h);
}
