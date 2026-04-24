#pragma once
#include <cstdint>
#include <d3d12.h>
#include <dxgi1_6.h>
#include "IUnityLog.h"
#include "IUnityGraphicsD3D12.h"
#include "DescriptorHeapAllocator.h"
#include "ComputeShader.h"

// ---------------------------------------------------------------------------
// ComputeDescriptorSet
//   Owns the GPU-heap slice (SRV / UAV allocations) for one logical
//   descriptor-set tied to a ComputeShader.
//
//   Motivation: ComputeShader may be dispatched multiple times per frame with
//   different resource bindings.  Each NativeComputeDescriptorSet on the C#
//   side has a corresponding ComputeDescriptorSet here so allocations are
//   independent and dispatches do not clobber each other.
//
//   Lifetime: created via NR_CS_CreateDescriptorSet / destroyed via
//   NR_CS_DestroyDescriptorSet (both called from C#).
// ---------------------------------------------------------------------------
class ComputeDescriptorSet
{
public:
    ComputeDescriptorSet(ComputeShader* cs,
                         ID3D12Device*             device,
                         IUnityLog*                log,
                         DescriptorHeapAllocator*  allocator,
                         IUnityGraphicsD3D12v8*    d3d12v8);
    ~ComputeDescriptorSet();

    // Execute the compute dispatch.  All resource binding, descriptor writing,
    // resource-state requests, and root-parameter setup happen here.
    void Dispatch(ID3D12GraphicsCommandList* cmdList,
                  UINT threadGroupX, UINT threadGroupY, UINT threadGroupZ,
                  const CS_BindingSlot* slots, uint32_t slotCount);

private:
    bool AllocateAndWriteDescriptors(const CS_BindingSlot* slots, uint32_t slotCount);
    void UpdateDescriptors          (const CS_BindingSlot* slots, uint32_t slotCount);
    void FreeAllocations();
    void RequestResourceStates(const CS_BindingSlot* slots, uint32_t slotCount);

    void Log (UnityLogType type, const char* msg)        const;
    void Logf(UnityLogType type, const char* fmt, ...)   const;

    ComputeShader*           m_cs        = nullptr;
    ID3D12Device*            m_device    = nullptr;
    IUnityLog*               m_log       = nullptr;
    DescriptorHeapAllocator* m_allocator = nullptr;
    IUnityGraphicsD3D12v8*   m_d3d12v8   = nullptr;

    static constexpr uint32_t kInvalidAlloc = UINT32_MAX;
    uint32_t m_srvAllocBase = kInvalidAlloc;
    uint32_t m_uavAllocBase = kInvalidAlloc;
};
