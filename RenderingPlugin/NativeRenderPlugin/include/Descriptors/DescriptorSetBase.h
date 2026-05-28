#pragma once
#include <cstdint>
#include <d3d12.h>
#include <dxgi1_6.h>
#include "IUnityLog.h"
#include "IUnityGraphicsD3D12.h"
#include "DescriptorHeapAllocator.h"
#include "ResourceStateTracker.h"
#include "ComputeShader.h"   // CS_BindingSlot, CS_BindingObjectKind, ComputeBinding, ComputeBindingType

// ---------------------------------------------------------------------------
// DescriptorSetBase<ShaderT>
//   Common state and descriptor-management logic shared between
//   ComputeDescriptorSet (Dispatch) and RayTraceDescriptorSet (DispatchRays).
//
//   The SRV+UAV descriptor table backing each Dispatch is bump-allocated from
//   the global TransientDescriptorRing (see TransientDescriptorRing.h) and
//   reclaimed automatically once the GPU finishes the frame that used it.
//   The descriptor set itself owns no per-frame heap state.
//
//   ShaderT must expose:
//     const char*                        GetName()                const
//     uint32_t                           GetNumSRV()              const
//     uint32_t                           GetNumUAV()              const
//     uint32_t                           GetRootParamSRV()        const
//     uint32_t                           GetRootParamUAV()        const
//     uint32_t                           GetRootParamCBVBase()    const
//     uint32_t                           GetRootParamRootSRVBase()const
//     const std::vector<Binding>&        GetBindings()            const
//     ID3D12RootSignature*               GetRootSignature()       const
//
//   Explicit template instantiations for ComputeShader and RayTraceShader
//   are provided in DescriptorSetBase.cpp.
// ---------------------------------------------------------------------------
template<typename ShaderT>
class DescriptorSetBase
{
public:
    DescriptorSetBase(ShaderT*                  shader,
                      ID3D12Device*             device,
                      IUnityLog*                log,
                      DescriptorHeapAllocator*  allocator,
                      IUnityGraphicsD3D12v8*    d3d12v8);

protected:
    // --- Logging ---
    void Log (UnityLogType type, const char* msg)      const;
    void Logf(UnityLogType type, const char* fmt, ...) const;

    // --- Dispatch helpers ---

    // Validates all binding slots; logs errors and returns false on any missing binding.
    bool ValidateBindings(const BindingSlot* slots, uint32_t slotCount) const;

    // Bump-allocate the SRV and UAV descriptor tables for one dispatch from the
    // global transient ring.  On success returns true and fills outSrvBase /
    // outUavBase with absolute heap slot indices (or kInvalidAlloc if the
    // shader uses zero SRVs / zero UAVs).  On ring exhaustion logs an error
    // and returns false; callers must skip the dispatch in that case.
    bool AllocateTransientTables(uint32_t& outSrvBase, uint32_t& outUavBase) const;

    // Write SRV/TLAS and UAV descriptors into the supplied transient table
    // bases.  ROOT_SRV / CBV / ROOT_CONSTANTS / SRV_ARRAY / UAV_ARRAY are
    // bound elsewhere (BindRootParams) and are skipped here.
    void WriteDescriptors(const BindingSlot* slots, uint32_t slotCount,
                          uint32_t srvBase, uint32_t uavBase);

    void RequestResourceStates(const BindingSlot* slots, uint32_t slotCount);
    void NotifyResourceStates (const BindingSlot* slots, uint32_t slotCount);

    // Binds the global heap, the root signature, and all root parameters
    // (descriptor tables, inline CBVs, inline SRVs, root constants).
    // Takes the base ID3D12GraphicsCommandList* so it works for both
    // Dispatch (cmdList) and DispatchRays (cmdList4 implicitly upcast).
    void BindRootParams(ID3D12GraphicsCommandList* cmdList,
                        const BindingSlot*      slots,
                        uint32_t                   slotCount,
                        uint32_t                   srvBase,
                        uint32_t                   uavBase);

    // --- State ---
    ShaderT*                 m_shader    = nullptr;
    ID3D12Device*            m_device    = nullptr;
    IUnityLog*               m_log       = nullptr;
    DescriptorHeapAllocator* m_allocator = nullptr;
    ResourceStateTracker     m_tracker;   // facade over the IUnityGraphicsD3D12v8 state tracker

    static constexpr uint32_t kInvalidAlloc = UINT32_MAX;
};
