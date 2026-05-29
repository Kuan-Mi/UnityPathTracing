#pragma once
#include <cstdint>
#include <deque>
#include <vector>
#include <mutex>
#include <functional>

// ---------------------------------------------------------------------------
// DeferredDeleteQueue
//   Delays destruction of GPU-facing objects until the GPU has finished executing
//   all commands that may reference them.
//
//   Modelled on nvrhi's per-submission keep-alive lists (Queue::commandListsInFlight):
//   deletions are grouped into one "retire bucket" per frame. The open bucket
//   accumulates the current frame's tasks and is stamped — once, on first use — with
//   the fence value the frame's GPU work will reach plus a safety margin, never below
//   current GPU progress (see ComputeStamp for why the clamp is required). Each frame
//   Drain() closes the open bucket into the in-flight deque (ascending fence order,
//   oldest at front) and runs every bucket whose stamped fence has completed.
//
//   Decoupled from D3D12/Unity: the caller supplies fence-value getters at
//   Initialize(). Thread-safe — the internal mutex guards enqueue vs. drain, which
//   may run on different threads.
// ---------------------------------------------------------------------------
class DeferredDeleteQueue
{
public:
    using Task        = std::function<void()>;
    using FenceGetter = std::function<uint64_t()>;

    // getCompleted: current GPU-completed fence value (0 if no fence exists yet).
    // getNext:      fence value the current frame's GPU work will reach.
    // frameDelay:   extra frames of safety margin before a bucket is freed.
    void Initialize(FenceGetter getCompleted, FenceGetter getNext, uint32_t frameDelay);

    // Force-run every remaining task, ignoring fences (teardown). Safe to call repeatedly.
    void Shutdown();

    // Defer a task until this frame's GPU work (+ margin) has completed.
    void Enqueue(Task&& task);

    // Once per frame: close the open bucket and run tasks whose fence has completed.
    // force=true runs everything regardless of fence (teardown).
    void Drain(bool force = false);

private:
    struct RetireBucket
    {
        uint64_t          fenceValue = 0; // 0 = open/unstamped
        std::vector<Task> tasks;
    };

    uint64_t ComputeStamp() const; // max(next, completed) + frameDelay

    FenceGetter              m_getCompleted;
    FenceGetter              m_getNext;
    uint32_t                 m_frameDelay = 3;
    std::deque<RetireBucket> m_inFlight;  // ascending fenceValue, oldest at front
    RetireBucket             m_current;   // open bucket for the current frame
    std::mutex               m_mutex;
};
