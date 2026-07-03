#pragma once
#include <cstdint>
#include <d3d12.h>
#include <d3d12shader.h>
#include <dxgi1_6.h>
#include <dxcapi.h>
#include <wrl/client.h>
#include <string>
#include <vector>
#include "IUnityLog.h"
#include "IUnityGraphicsD3D12.h"
#include "DescriptorHeapAllocator.h"
#include "ShaderBindings.h"  // BindingType, Binding, BindingSlot

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// ShaderBase
//   Common state and logic shared by ComputeShader, RayTraceShader and
//   RasterShader.
//
//   Binding model (nvrhi-style, declaration-driven):
//     The application (C#) declares an explicit binding layout before load —
//     an ordered item list mirroring nvrhi::BindingLayoutDesc — plus optional
//     static-sampler definitions. The layout is the single contract between
//     HLSL register assignments and the root signature; this class never
//     reflects shader resource bindings. BuildSharedRootSignature turns the
//     layout into the root signature (nvrhi d3d12::BindingLayout semantics)
//     and BuildBindingsFromLayout derives the per-slot binding table the
//     descriptor sets consume (one BindingSlot per layout item, in item
//     order). Mismatches between the layout and the actual shader registers
//     are the caller's responsibility (C# validates against import-time
//     reflection; the D3D12 debug layer is the runtime backstop — exactly
//     nvrhi's model).
//
//   Subclasses provide:
//     – Initialize (may specialize device usage)
//     – BuildPipeline / GetPSO (different PSO types)
// ---------------------------------------------------------------------------
class ShaderBase
{
public:
    ShaderBase()          = default;
    virtual ~ShaderBase() = default;

    enum class SharedLayoutKind : uint32_t
    {
        SRV = 0,
        UAV = 1,
        CBV = 2,
        VolatileCBV = 3,
        PushConstants = 4,
        TLAS = 5,
        BindlessSRV = 6,
        BindlessUAV = 7,
        RootSRV = 8,
        Sampler = 9,
    };

    void ClearSharedLayout();
    void AddSharedLayoutItem(SharedLayoutKind kind, uint32_t shaderRegister,
                             uint32_t space, uint32_t count, uint32_t num32BitValues,
                             uint32_t visibility = D3D12_SHADER_VISIBILITY_ALL,
                             uint32_t bindlessLayoutIndex = UINT32_MAX);

    struct SharedLayoutItem
    {
        SharedLayoutKind kind = SharedLayoutKind::SRV;
        uint32_t shaderRegister = 0;
        uint32_t space = 0;
        uint32_t count = 1;
        uint32_t num32BitValues = 0;
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL;
        // Bindless items only: share the previous bindless item's root parameter
        // (one table, several unbounded ranges — the nvrhi bindless layout with
        // multiple registerSpaces, e.g. donut's shared descriptor table). When
        // false each bindless item gets its own root parameter/table.
        uint32_t bindlessLayoutIndex = kInvalidAlloc;
        uint32_t tableOffset = 0;
        uint32_t rootParam = kInvalidAlloc;
    };
    const std::vector<SharedLayoutItem>& GetSharedLayout() const { return m_sharedLayout; }

    // Explicit static-sampler definition (C# resolves sampler registers and
    // filter/address config from import-time reflection + editor hints; the
    // plugin just serializes them into the root signature).
    struct StaticSamplerDef
    {
        uint32_t reg           = 0;
        uint32_t space         = 0;
        uint32_t filter        = 1;   // 0 point, 1 linear, 2 aniso
        uint32_t addressU      = 0;   // 0 wrap, 1 clamp, 2 mirror, 3 mirror-once, 4 border
        uint32_t addressV      = 0;
        uint32_t addressW      = 0;
        bool     mips          = false;
        uint32_t maxAnisotropy = 16;
    };
    void AddStaticSampler(const StaticSamplerDef& def);

    // --- Common accessors ---
    ID3D12RootSignature*        GetRootSignature() const { return m_rootSig.Get(); }
    const std::vector<Binding>& GetBindings()      const { return m_bindings; }
    uint32_t GetBindingCount()         const { return static_cast<uint32_t>(m_bindings.size()); }
    uint32_t GetRootParamSRV()         const { return m_rootParamSRV; }
    uint32_t GetRootParamUAV()         const { return m_rootParamUAV; }
    uint32_t GetRootParamSampler()     const { return m_rootParamSampler; }
    uint32_t GetRootParamCBVBase()     const { return m_rootParamCBVBase; }
    uint32_t GetRootParamRootSRVBase() const { return m_rootParamRootSRVBase; }
    uint32_t GetNumSRV()               const { return m_numSRV; }
    uint32_t GetNumUAV()               const { return m_numUAV; }
    // Total descriptor slots in the combined SRV/UAV/CBV table (bounded arrays
    // count as arrayCount slots, singles as 1).
    uint32_t GetNumSRVSlots()          const { return m_numSRVSlots; }
    uint32_t GetNumUAVSlots()          const { return m_numUAVSlots; }
    uint32_t GetNumSamplerSlots()      const { return m_numSamplerSlots; }
    const char* GetName()              const { return m_name.c_str(); }
    bool UsesSharedLayout()            const { return !m_sharedLayout.empty(); }

    static constexpr uint32_t kInvalidAlloc = UINT32_MAX;

protected:
    // --- Logging ---
    void Log (UnityLogType type, const char* msg)      const;
    void Logf(UnityLogType type, const char* fmt, ...) const;

    // --- Shared build steps ---
    //   BuildSharedRootSignature: layout → root signature (nvrhi order), fills
    //   each item's tableOffset / rootParam. Fails when the layout is empty —
    //   an explicit layout is mandatory.
    //   BuildBindingsFromLayout: layout → m_bindings (one entry per item, in
    //   item order — the dispatch BindingSlot payload contract).
    bool BuildSharedRootSignature();
    void BuildBindingsFromLayout();

    // --- Shared state ---
    IUnityLog*               m_log       = nullptr;
    ComPtr<ID3D12Device5>    m_device;           // unified to Device5; CS uses as Device
    IUnityGraphicsD3D12v8*   m_d3d12v8   = nullptr;
    std::string              m_name;

    ComPtr<ID3D12RootSignature> m_rootSig;

    std::vector<Binding>          m_bindings;      // derived from m_sharedLayout
    std::vector<StaticSamplerDef> m_staticSamplers;

    // Root parameter indices (populated by BuildSharedRootSignature)
    uint32_t m_rootParamSRV         = kInvalidAlloc;
    uint32_t m_rootParamUAV         = kInvalidAlloc;
    uint32_t m_rootParamSampler     = kInvalidAlloc;
    uint32_t m_rootParamCBVBase     = kInvalidAlloc;
    uint32_t m_rootParamRootSRVBase = kInvalidAlloc;

    // Binding counts (populated by BuildBindingsFromLayout)
    uint32_t m_numSRV           = 0;    // # of SRV/TLAS bindings
    uint32_t m_numUAV           = 0;    // # of UAV bindings
    uint32_t m_numSRVSlots      = 0;    // combined-table descriptor count (SRV+UAV+CBV)
    uint32_t m_numUAVSlots      = 0;    // always 0 (combined table); kept for descriptor sets
    uint32_t m_numSamplerSlots  = 0;    // total sampler descriptors in the sampler table
    uint32_t m_numCBV           = 0;
    uint32_t m_numSRVArray      = 0;
    uint32_t m_numUAVArray      = 0;
    uint32_t m_numRootConstants = 0;
    uint32_t m_numRootSRV       = 0;
    uint32_t m_numSampler       = 0;

    std::vector<SharedLayoutItem> m_sharedLayout;
};
