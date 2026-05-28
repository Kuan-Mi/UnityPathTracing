#include "TransientDescriptorRing.h"
#include <cstdio>

bool TransientDescriptorRing::Initialize(DescriptorHeapAllocator* heap, uint32_t capacity)
{
    if (!heap || capacity == 0) return false;

    uint32_t base = heap->Allocate(capacity);
    if (base == UINT32_MAX)
    {
        printf("[TransientDescriptorRing] failed to reserve %u heap slots\n", capacity);
        return false;
    }

    m_heap     = heap;
    m_base     = base;
    m_capacity = capacity;
    m_head     = 0;
    m_tail     = 0;
    m_markers.clear();
    return true;
}

void TransientDescriptorRing::Shutdown()
{
    m_heap     = nullptr;
    m_base     = 0;
    m_capacity = 0;
    m_head     = 0;
    m_tail     = 0;
    m_markers.clear();
}

uint32_t TransientDescriptorRing::Allocate(uint32_t count)
{
    if (!m_heap || count == 0 || count > m_capacity) return UINT32_MAX;

    std::lock_guard<std::mutex> lk(m_mutex);

    uint64_t allocHead = m_head;
    uint32_t phys      = static_cast<uint32_t>(allocHead % m_capacity);

    // A descriptor table must be contiguous, so it cannot straddle the wrap
    // boundary.  Pad to the next boundary (the skipped tail slots are simply
    // wasted until reclaimed) when the allocation would cross it.
    if (phys + count > m_capacity)
        allocHead += (m_capacity - phys);

    // Refuse if honouring this allocation would overwrite slots still in flight.
    if ((allocHead + count) - m_tail > m_capacity)
    {
        printf("[TransientDescriptorRing] out of slots (count=%u, inflight=%llu, capacity=%u) "
               "- increase capacity\n",
               count, static_cast<unsigned long long>(m_head - m_tail), m_capacity);
        return UINT32_MAX;
    }

    uint32_t offset = static_cast<uint32_t>(allocHead % m_capacity);
    m_head          = allocHead + count;
    return m_base + offset;
}

void TransientDescriptorRing::EndFrame(uint64_t frameCompleteFenceValue, uint64_t completedFenceValue)
{
    if (!m_heap) return;

    std::lock_guard<std::mutex> lk(m_mutex);

    // Reclaim every frame the GPU has finished.  Markers are pushed in frame
    // order so their fenceValues are ascending; pop from the front.
    while (!m_markers.empty() && m_markers.front().fenceValue <= completedFenceValue)
    {
        m_tail = m_markers.front().head;
        m_markers.pop_front();
    }

    // Record this frame's high-water mark, unless it produced no allocations.
    if (m_markers.empty() || m_markers.back().head != m_head)
        m_markers.push_back({ frameCompleteFenceValue, m_head });
}
