#pragma once
#include <cstdint>
#include <d3d12.h>
#include <dxgi1_6.h>
#include "IUnityLog.h"
#include "IUnityGraphicsD3D12.h"
#include "DescriptorHeapAllocator.h"
#include "ComputeShader.h"   // CS_BindingSlot, CS_BindingObjectKind, ComputeBinding, ComputeBindingType

// ---------------------------------------------------------------------------
// DescriptorSetBase<ShaderT>
//   Common state and descriptor-management logic shared between
//   ComputeDescriptorSet (Dispatch) and RayTraceDescriptorSet (DispatchRays).
//
//   ShaderT must expose:
//     const char*                        GetName()                const
//     uint32_t                           GetNumSRV()              const
//     uint32_t                           GetNumUAV()              const
//     uint32_t                           GetRootParamSRV()        const
//     uint32_t                           GetRootParamUAV()        const
//     uint32_t                           GetRootParamCBVBase()    const
//     uint32_t                           GetRootParamRootSRVBase()const
//     const std::vector<ComputeBinding>& GetBindings()            const
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
    ~DescriptorSetBase();

protected:
    // --- Logging ---
    void Log (UnityLogType type, const char* msg)      const;
    void Logf(UnityLogType type, const char* fmt, ...) const;

    // --- Descriptor heap management ---
    void FreeAllocations();
    bool AllocateAndWriteDescriptors(const BindingSlot* slots, uint32_t slotCount, uint32_t slotIdx);
    void UpdateDescriptors          (const BindingSlot* slots, uint32_t slotCount, uint32_t slotIdx);
    void RequestResourceStates      (const BindingSlot* slots, uint32_t slotCount);

    // --- Dispatch helpers ---

    // Validates all binding slots; logs errors and returns false on any missing binding.
    bool ValidateBindings(const BindingSlot* slots, uint32_t slotCount) const;

    // Computes per-dispatch slotIdx (ring-buffer across frames × eyes) and
    // increments m_subFrameIdx.  Always succeeds; clamps on overflow.
    void AcquireSlot(uint32_t& outSlotIdx);

    // Allocates heap slots on first use then writes descriptors; or just
    // re-writes descriptors if already allocated.
    void EnsureDescriptors(const BindingSlot* slots, uint32_t slotCount, uint32_t slotIdx);

    // Binds the global heap, the root signature, and all root parameters
    // (descriptor tables, inline CBVs, inline SRVs, root constants).
    // Takes the base ID3D12GraphicsCommandList* so it works for both
    // Dispatch (cmdList) and DispatchRays (cmdList4 implicitly upcast).
    void BindRootParams(ID3D12GraphicsCommandList* cmdList,
                        const BindingSlot*      slots,
                        uint32_t                   slotCount,
                        uint32_t                   slotIdx);

    // --- State ---
    ShaderT*                 m_shader    = nullptr;
    ID3D12Device*            m_device    = nullptr;
    IUnityLog*               m_log       = nullptr;
    DescriptorHeapAllocator* m_allocator = nullptr;
    IUnityGraphicsD3D12v8*   m_d3d12v8   = nullptr;

    // Cached at construction — FreeAllocations must not dereference m_shader
    // because the shader object may already be destroyed by the time the
    // descriptor set's destructor runs (deferred-delete ordering).
    uint32_t m_cachedNumSRV = 0;
    uint32_t m_cachedNumUAV = 0;

    static constexpr uint32_t kInvalidAlloc    = UINT32_MAX;
    static constexpr uint32_t kNumFrames        = 3;  // triple-buffering
    static constexpr uint32_t kMaxEyesPerFrame  = 2;  // XR stereo
    static constexpr uint32_t kNumSlots         = kNumFrames * kMaxEyesPerFrame; // 6

    // Per-dispatch slot index: slotIdx = g_frameIndex * kMaxEyesPerFrame + m_subFrameIdx.
    // m_subFrameIdx resets to 0 on each new frame, then increments per Dispatch call,
    // giving eye0 and eye1 independent GPU heap slices within the same frame.
    uint32_t m_lastFrameIndex = UINT32_MAX;
    uint32_t m_subFrameIdx    = 0;

    uint32_t m_srvAllocBase[kNumSlots] = {
        kInvalidAlloc, kInvalidAlloc, kInvalidAlloc,
        kInvalidAlloc, kInvalidAlloc, kInvalidAlloc };
    uint32_t m_uavAllocBase[kNumSlots] = {
        kInvalidAlloc, kInvalidAlloc, kInvalidAlloc,
        kInvalidAlloc, kInvalidAlloc, kInvalidAlloc };
};
