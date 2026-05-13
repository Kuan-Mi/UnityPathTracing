#pragma once
#include <cstdint>
#include <d3d12.h>
#include <dxgi1_6.h>
#include "IUnityLog.h"
#include "IUnityGraphicsD3D12.h"
#include "DescriptorHeapAllocator.h"
#include "RayTraceShader.h"
#include "ComputeShader.h"   // CS_BindingSlot, CS_BindingObjectKind

// ---------------------------------------------------------------------------
// RayTraceDescriptorSet
//   Owns the GPU-heap slice (SRV / UAV allocations) for one logical descriptor
//   set tied to a RayTraceShader.
//
//   Motivation: a RayTraceShader may be dispatched multiple times per frame
//   with different resource bindings (e.g. two XR eyes).  Each
//   NativeRayTraceDescriptorSet on the C# side has a corresponding
//   RayTraceDescriptorSet here so allocations are independent and dispatches
//   do not clobber each other.
//
//   Resource bindings are passed per-dispatch as a CS_BindingSlot[] array
//   (same layout as ComputeDescriptorSet) from C# via IssuePluginEventAndData.
//
//   Lifetime: created via NR_RTS_CreateDescriptorSet /
//             destroyed via NR_RTS_DestroyDescriptorSet (both called from C#).
// ---------------------------------------------------------------------------
class RayTraceDescriptorSet
{
public:
    RayTraceDescriptorSet(RayTraceShader*          shader,
                          ID3D12Device*             device,
                          IUnityLog*                log,
                          DescriptorHeapAllocator*  allocator,
                          IUnityGraphicsD3D12v8*    d3d12v8);
    ~RayTraceDescriptorSet();

    // Execute a DispatchRays call.  All descriptor writing, resource-state
    // requests, and root-parameter setup happen here.
    void Dispatch(ID3D12GraphicsCommandList4* cmdList,
                  UINT width, UINT height,
                  const CS_BindingSlot* slots, uint32_t slotCount);

private:
    bool AllocateAndWriteDescriptors(const CS_BindingSlot* slots, uint32_t slotCount, uint32_t slotIdx);
    void UpdateDescriptors          (const CS_BindingSlot* slots, uint32_t slotCount, uint32_t slotIdx);
    void FreeAllocations();
    void RequestResourceStates(const CS_BindingSlot* slots, uint32_t slotCount);

    void Log (UnityLogType type, const char* msg)      const;
    void Logf(UnityLogType type, const char* fmt, ...) const;

    RayTraceShader*          m_shader    = nullptr;
    ID3D12Device*            m_device    = nullptr;
    IUnityLog*               m_log       = nullptr;
    DescriptorHeapAllocator* m_allocator = nullptr;
    IUnityGraphicsD3D12v8*   m_d3d12v8   = nullptr;

    // Cached at construction — used by FreeAllocations so we never touch
    // m_shader after it may have been deleted (shader is enqueued for
    // deferred-delete before the descriptor set).
    uint32_t m_cachedNumSRV = 0;
    uint32_t m_cachedNumUAV = 0;

    static constexpr uint32_t kInvalidAlloc   = UINT32_MAX;
    static constexpr uint32_t kNumFrames       = 3;  // triple-buffering
    static constexpr uint32_t kMaxEyesPerFrame = 2;  // XR stereo
    static constexpr uint32_t kNumSlots        = kNumFrames * kMaxEyesPerFrame; // 6

    // Per-dispatch slot index: slotIdx = g_frameIndex * kMaxEyesPerFrame + m_subFrameIdx
    // m_subFrameIdx resets to 0 on each new g_frameIndex, then increments per Dispatch.
    uint32_t m_lastFrameIndex = UINT32_MAX;
    uint32_t m_subFrameIdx    = 0;

    uint32_t m_srvAllocBase[kNumSlots] = {
        kInvalidAlloc, kInvalidAlloc, kInvalidAlloc,
        kInvalidAlloc, kInvalidAlloc, kInvalidAlloc };
    uint32_t m_uavAllocBase[kNumSlots] = {
        kInvalidAlloc, kInvalidAlloc, kInvalidAlloc,
        kInvalidAlloc, kInvalidAlloc, kInvalidAlloc };
};
