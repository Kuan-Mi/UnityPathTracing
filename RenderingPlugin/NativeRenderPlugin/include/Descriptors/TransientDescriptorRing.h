#pragma once
#include <cstdint>
#include <deque>
#include <mutex>
#include "DescriptorHeapAllocator.h"

// ---------------------------------------------------------------------------
// TransientDescriptorRing
//   A GPU-completion-fenced ring allocator carved out of the shared
//   DescriptorHeapAllocator.  Every Dispatch / DispatchRays bump-allocates the
//   SRV+UAV descriptor table it needs for that single call; the slots are
//   reclaimed automatically once the GPU has finished the frame that used them.
//
//   This replaces the old fixed per-descriptor-set reservation
//   (kNumFrames * kMaxEyesPerFrame slots), which capped each descriptor set to
//   a hard-coded number of dispatches per frame and silently corrupted live
//   descriptors past that cap.  The only limit now is the total number of
//   in-flight transient descriptors across all not-yet-completed frames, bounded
//   by the ring capacity.
//
//   Threading: Allocate() runs on the render thread during dispatch callbacks
//   and EndFrame() runs on the render thread at the frame-tick callback, so the
//   two are already serialised; the mutex is defensive only.
// ---------------------------------------------------------------------------
class TransientDescriptorRing
{
public:
    // Reserve a contiguous `capacity`-slot region from the shared heap.
    // Must be called once after DescriptorHeapAllocator::Initialize and before
    // any descriptor set dispatches.
    bool Initialize(DescriptorHeapAllocator* heap, uint32_t capacity);

    // Reset all state.  The reserved heap range is released implicitly when the
    // owning DescriptorHeapAllocator is shut down.
    void Shutdown();

    // Allocate `count` contiguous slots.  Returns the absolute heap slot index
    // (usable with DescriptorHeapAllocator::GetCPUHandle / GetGPUHandle), or
    // UINT32_MAX if the ring is full of in-flight (not-yet-completed) frames.
    uint32_t Allocate(uint32_t count);

    // Demarcate the end of a frame's allocations:
    //   1. reclaim every frame whose GPU work has completed (the frame fence has
    //      reached its recorded value <= completedFenceValue), then
    //   2. record the current high-water mark against frameCompleteFenceValue —
    //      the fence value the GPU will reach once this frame's work is done.
    // Called once per frame from the frame-tick render-thread callback.
    void EndFrame(uint64_t frameCompleteFenceValue, uint64_t completedFenceValue);

    bool IsInitialized() const { return m_heap != nullptr; }

private:
    DescriptorHeapAllocator* m_heap     = nullptr;
    uint32_t                 m_base     = 0; // first heap slot owned by the ring
    uint32_t                 m_capacity = 0; // ring size in slots

    // Absolute, monotonically-increasing cursors.  They never wrap; a physical
    // ring offset is (cursor % m_capacity).  (m_head - m_tail) is the number of
    // in-flight slots and is kept <= m_capacity by Allocate().
    uint64_t m_head = 0; // next free position
    uint64_t m_tail = 0; // oldest position still potentially read by the GPU

    struct FrameMarker { uint64_t fenceValue; uint64_t head; };
    std::deque<FrameMarker> m_markers; // one per in-flight frame, fenceValue ascending
    std::mutex              m_mutex;
};
