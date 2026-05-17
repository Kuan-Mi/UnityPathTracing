#include "DescriptorHeapAllocator.h"
#include <cassert>
#include <algorithm>
#include <cstdio>

bool DescriptorHeapAllocator::Initialize(ID3D12Device* device)
{
    D3D12_DESCRIPTOR_HEAP_DESC d = {};
    d.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    d.NumDescriptors = kCapacity;
    d.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device->CreateDescriptorHeap(&d, IID_PPV_ARGS(&m_heap))))
    {
        printf("[DescriptorHeapAllocator] CreateDescriptorHeap failed\n");
        return false;
    }
    //Set Name for debugging
    m_heap->SetName(L"DescriptorHeapAllocator::m_heap");

    m_incSize  = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_bumpNext = 0;
    m_freeList.clear();
    return true;
}

void DescriptorHeapAllocator::Shutdown()
{
    m_heap.Reset();
    m_bumpNext = 0;
    m_freeList.clear();
}

uint32_t DescriptorHeapAllocator::Allocate(uint32_t count)
{
    assert(count > 0 && "DescriptorHeapAllocator::Allocate: count must be > 0");
    std::lock_guard<std::mutex> lk(m_mutex);

    // First-fit search in the free list
    for (size_t i = 0; i < m_freeList.size(); ++i)
    {
        if (m_freeList[i].count >= count)
        {
            uint32_t base = m_freeList[i].base;
            if (m_freeList[i].count == count)
                m_freeList.erase(m_freeList.begin() + static_cast<ptrdiff_t>(i));
            else
            {
                m_freeList[i].base  += count;
                m_freeList[i].count -= count;
            }
            return base;
        }
    }

    // Bump allocate
    uint32_t base = m_bumpNext;
    if (base + count > kCapacity)
    {
        printf("[DescriptorHeapAllocator] ERROR: out of descriptor slots "
               "(requested %u, bumpNext=%u, capacity=%u) – increase kCapacity\n",
               count, m_bumpNext, kCapacity);
        return UINT32_MAX;
    }
    m_bumpNext += count;
    return base;
}

void DescriptorHeapAllocator::Free(uint32_t base, uint32_t count)
{
    if (count == 0) return;
    std::lock_guard<std::mutex> lk(m_mutex);
    m_freeList.push_back({ base, count });

    // Sort by base address and merge adjacent ranges to prevent fragmentation.
    std::sort(m_freeList.begin(), m_freeList.end(),
              [](const FreeRange& a, const FreeRange& b){ return a.base < b.base; });
    for (size_t i = 0; i + 1 < m_freeList.size(); )
    {
        FreeRange& cur  = m_freeList[i];
        FreeRange& next = m_freeList[i + 1];
        if (cur.base + cur.count == next.base)
        {
            cur.count += next.count;
            m_freeList.erase(m_freeList.begin() + static_cast<ptrdiff_t>(i + 1));
        }
        else ++i;
    }
}

D3D12_CPU_DESCRIPTOR_HANDLE DescriptorHeapAllocator::GetCPUHandle(uint32_t slot) const
{
    D3D12_CPU_DESCRIPTOR_HANDLE h = m_heap->GetCPUDescriptorHandleForHeapStart();
    h.ptr += static_cast<SIZE_T>(slot) * m_incSize;
    return h;
}

D3D12_GPU_DESCRIPTOR_HANDLE DescriptorHeapAllocator::GetGPUHandle(uint32_t slot) const
{
    D3D12_GPU_DESCRIPTOR_HANDLE h = m_heap->GetGPUDescriptorHandleForHeapStart();
    h.ptr += static_cast<UINT64>(slot) * m_incSize;
    return h;
}
