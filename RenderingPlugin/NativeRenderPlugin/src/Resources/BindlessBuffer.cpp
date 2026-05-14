#include "BindlessBuffer.h"
#include <cstdio>
#include <cassert>

// ---------------------------------------------------------------------------
// BindlessBuffer
// ---------------------------------------------------------------------------

static void WriteNullBufferSRV(ID3D12Device* device, D3D12_CPU_DESCRIPTOR_HANDLE h)
{
    // A null raw buffer SRV – 1 dummy R32_TYPELESS element, raw flag
    D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
    s.Shader4ComponentMapping  = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    s.ViewDimension            = D3D12_SRV_DIMENSION_BUFFER;
    s.Format                   = DXGI_FORMAT_R32_TYPELESS;
    s.Buffer.Flags             = D3D12_BUFFER_SRV_FLAG_RAW;
    s.Buffer.NumElements       = 1;
    device->CreateShaderResourceView(nullptr, &s, h);
}

// ---------------------------------------------------------------------------

void BindlessBuffer::Log(UnityLogType type, const char* msg) const
{
    if (m_log) m_log->Log(type, msg, __FILE__, __LINE__);
    else        printf("[BindlessBuffer] %s\n", msg);
}

bool BindlessBuffer::Initialize(ID3D12Device*            device,
                                DescriptorHeapAllocator* allocator,
                                uint32_t                 capacity,
                                IUnityLog*               log)
{
    assert(!m_initialized && "BindlessBuffer::Initialize called twice");
    m_device    = device;
    m_allocator = allocator;
    m_log       = log;

    if (capacity == 0)
    {
        Log(kUnityLogTypeWarning, "BindlessBuffer: capacity 0 – at least 1 required");
        capacity = 1;
    }

    m_capacity  = capacity;
    m_allocBase = m_allocator->Allocate(capacity);
    m_buffers.assign(capacity, nullptr);

    // Write null descriptors for all initial slots
    for (uint32_t i = 0; i < capacity; ++i)
        WriteNullDescriptor(i);

    m_initialized = true;
    return true;
}

BindlessBuffer::~BindlessBuffer()
{
    if (m_initialized && m_allocator && m_capacity > 0)
        m_allocator->Free(m_allocBase, m_capacity);
}

void BindlessBuffer::SetBuffer(uint32_t index, ID3D12Resource* resource)
{
    if (index >= m_capacity)
    {
        Log(kUnityLogTypeWarning, "BindlessBuffer::SetBuffer: index out of range");
        return;
    }
    m_buffers[index] = resource;
    WriteDescriptor(index, resource);
}

void BindlessBuffer::Resize(uint32_t newCapacity)
{
    if (newCapacity == 0) newCapacity = 1;
    if (newCapacity == m_capacity) return;

    // Free old range and allocate new one
    m_allocator->Free(m_allocBase, m_capacity);
    m_allocBase = m_allocator->Allocate(newCapacity);

    // Resize buffer pointer array (preserving existing entries up to min)
    uint32_t oldCapacity = m_capacity;
    m_buffers.resize(newCapacity, nullptr);
    m_capacity = newCapacity;

    // Re-write all descriptors in the new range
    for (uint32_t i = 0; i < newCapacity; ++i)
    {
        ID3D12Resource* res = (i < m_buffers.size()) ? m_buffers[i] : nullptr;
        WriteDescriptor(i, res);
    }
    (void)oldCapacity;
}

D3D12_GPU_DESCRIPTOR_HANDLE BindlessBuffer::GetGPUHandle() const
{
    return m_allocator->GetGPUHandle(m_allocBase);
}

void BindlessBuffer::WriteDescriptor(uint32_t index, ID3D12Resource* resource)
{
    D3D12_CPU_DESCRIPTOR_HANDLE h = m_allocator->GetCPUHandle(m_allocBase + index);

    if (!resource)
    {
        WriteNullBufferSRV(m_device, h);
        return;
    }

    D3D12_RESOURCE_DESC rd = resource->GetDesc();
    if (rd.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER)
    {
        Log(kUnityLogTypeWarning, "BindlessBuffer::SetBuffer: resource is not a buffer");
        WriteNullBufferSRV(m_device, h);
        return;
    }

    D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
    s.Shader4ComponentMapping  = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    s.ViewDimension            = D3D12_SRV_DIMENSION_BUFFER;
    s.Format                   = DXGI_FORMAT_R32_TYPELESS;
    s.Buffer.Flags             = D3D12_BUFFER_SRV_FLAG_RAW;
    s.Buffer.NumElements       = static_cast<UINT>(rd.Width / 4);

    m_device->CreateShaderResourceView(resource, &s, h);
}

void BindlessBuffer::WriteNullDescriptor(uint32_t index)
{
    D3D12_CPU_DESCRIPTOR_HANDLE h = m_allocator->GetCPUHandle(m_allocBase + index);
    WriteNullBufferSRV(m_device, h);
}
