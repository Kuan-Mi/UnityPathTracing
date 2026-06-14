#include "DeferredDeleteQueue.h"

void DeferredDeleteQueue::Initialize(FenceGetter getCompleted, FenceGetter getNext, uint32_t frameDelay)
{
    std::lock_guard<std::mutex> lk(m_mutex);
    m_getCompleted = std::move(getCompleted);
    m_getNext      = std::move(getNext);
    m_frameDelay   = frameDelay;
}

uint64_t DeferredDeleteQueue::ComputeStamp() const
{
    // In a packaged player GetNextFrameFenceValue() can return a stale, too-low value
    // (observed: 1) while the fence has already completed a higher value (observed: 8).
    // Tagging against the stale value would schedule deletion at a fence the GPU has
    // *already* passed, freeing the resource while still in use -> GPU hang
    // (DXGI_ERROR_DEVICE_HUNG 0x887a0006). Clamp the base to the completed value so the
    // tag is always ahead of GPU progress by m_frameDelay frames.
    const uint64_t completed = m_getCompleted ? m_getCompleted() : 0;
    const uint64_t next      = m_getNext      ? m_getNext()      : 0;
    const uint64_t base      = next > completed ? next : completed;
    return base + m_frameDelay;
}

void DeferredDeleteQueue::Enqueue(Task&& task)
{
    if (!task) return;

    std::lock_guard<std::mutex> lk(m_mutex);

    // Stamp the open bucket on first use this frame. All of a frame's deletions share
    // one stamp (computed once instead of per task); the bucket is closed into the
    // in-flight deque at the next Drain().
    if (m_current.fenceValue == 0)
        m_current.fenceValue = ComputeStamp();

    m_current.tasks.emplace_back(std::move(task));
}

void DeferredDeleteQueue::Drain(bool force)
{
    // Query completed progress outside the lock. force => run everything.
    const uint64_t completed = force ? UINT64_MAX : (m_getCompleted ? m_getCompleted() : 0);

    std::vector<RetireBucket> toRun;
    {
        std::lock_guard<std::mutex> lk(m_mutex);

        // Close this frame's open bucket into the in-flight deque. Stamps are monotonic
        // across frames, so the deque stays sorted with the oldest (lowest) at the front.
        if (!m_current.tasks.empty())
        {
            m_inFlight.push_back(std::move(m_current));
            m_current = RetireBucket{};
        }

        // Run buckets whose fence has completed. Sorted front-to-back, so stop at the
        // first one still in flight.
        while (!m_inFlight.empty())
        {
            if (!force && m_inFlight.front().fenceValue > completed)
                break;
            toRun.push_back(std::move(m_inFlight.front()));
            m_inFlight.pop_front();
        }
    }

    // Run destructors outside the lock to avoid deadlock and reduce lock hold time.
    for (auto& bucket : toRun)
        for (auto& task : bucket.tasks)
            if (task) task();
}

void DeferredDeleteQueue::Shutdown()
{
    Drain(/*force*/ true);
}
