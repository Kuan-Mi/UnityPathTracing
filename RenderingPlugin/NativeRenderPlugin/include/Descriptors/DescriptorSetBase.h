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

    // --- Per-subresource SRV+UAV on the same resource (donut mipmapgen pattern) ---
    // Unity's IUnityGraphicsD3D12 tracks state PER RESOURCE, so a resource bound as
    // an SRV on one mip and a UAV on other mips within the same dispatch cannot be
    // expressed through the tracker (one whole-resource state). RequestResourceStates
    // detects that case: it keeps the resource in UNORDERED_ACCESS (via the UAV
    // binding) and records the SRV-read subresources here instead of requesting
    // SHADER_RESOURCE on the whole resource. The compute Dispatch then brackets the
    // GPU dispatch with manual, balanced per-subresource barriers — transitioning
    // just the SRV mip(s) UAV->read before, and read->UAV after — so the resource is
    // handed back to Unity in exactly the UNORDERED_ACCESS state the tracker believes
    // it is in. This is the ONLY place the plugin issues its own ResourceBarrier; it
    // is balanced within one Dispatch so it never desyncs Unity's per-resource map.
    struct SubresReadBarrier { ID3D12Resource* res; UINT firstSub; UINT subCount; };
    std::vector<SubresReadBarrier> m_subresReadBarriers;  // filled by RequestResourceStates

    // Reused per-dispatch scratch: every resource bound as a UAV (singular or via a
    // bindless UAV array) in the current RequestResourceStates call. Built once per
    // dispatch so the same-resource SRV+UAV detection is a lookup in this small list
    // instead of a re-walk of all UAV bindings per SRV.
    std::vector<ID3D12Resource*> m_uavResScratch;

    void EmitSubresourceReadBarriers (ID3D12GraphicsCommandList* cmd);  // UAV  -> m_srvReadState
    void RestoreSubresourceUAVStates (ID3D12GraphicsCommandList* cmd);  // read -> UAV

    // Build the per-type binding index lists (and size the view cache) once.
    // The reflection in m_shader is immutable for the lifetime of this object
    // (a hot-reload destroys and recreates the descriptor set), so the lists are
    // computed lazily on the first dispatch and reused thereafter.  This lets
    // every per-dispatch pass touch only the bindings it cares about instead of
    // re-scanning the whole bindings vector once per pass.
    void EnsureCategoryIndices();

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

    // Pipeline-kind knobs, set by the subclass constructor.  Compute and ray
    // tracing leave the defaults; the raster set flips them so BindRootParams
    // emits SetGraphicsRoot* and SRV inputs are requested in the
    // pixel-shader-readable state.
    bool                 m_graphicsRootBinding = false;
    D3D12_RESOURCE_STATES m_srvReadState       = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

    // --- Per-type binding index lists (built by EnsureCategoryIndices) ---
    // Each holds the indices into m_shader->GetBindings() of one BindingType, so
    // a pass over (say) the CBVs iterates only m_idxCBV rather than filtering the
    // whole bindings vector with a per-element `if (b.type != X) continue;`.
    std::vector<uint32_t> m_idxSRV;        // BindingType::SRV
    std::vector<uint32_t> m_idxTLAS;       // BindingType::TLAS
    std::vector<uint32_t> m_idxUAV;        // BindingType::UAV
    std::vector<uint32_t> m_idxCBV;        // BindingType::CBV
    std::vector<uint32_t> m_idxRootSRV;    // BindingType::ROOT_SRV
    std::vector<uint32_t> m_idxRootConst;  // BindingType::ROOT_CONSTANTS
    std::vector<uint32_t> m_idxSRVArray;   // BindingType::SRV_ARRAY
    std::vector<uint32_t> m_idxUAVArray;   // BindingType::UAV_ARRAY
    bool                  m_categoriesBuilt = false;

    // Cached SRV/UAV view descriptors, one entry per binding index.  The transient
    // ring hands out a fresh heap slot every dispatch, so CreateView still runs each
    // time, but the GetDesc() round-trip and view-desc derivation are skipped while
    // the resolved resource and buffer view params (count/stride/format) are unchanged
    // — the common case for the path tracer, which rebinds the same resources every frame.
    struct CachedView
    {
        uint64_t res    = 0;     // resolved ID3D12Resource* the cached desc was built for
        uint32_t count  = 0;
        uint32_t stride = 0;
        uint32_t format = 0;
        bool     valid  = false;
        union
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC  srv;
            D3D12_UNORDERED_ACCESS_VIEW_DESC uav;
        };
        CachedView() : srv{} {}
    };
    std::vector<CachedView> m_viewCache;

    static constexpr uint32_t kInvalidAlloc = UINT32_MAX;
};
