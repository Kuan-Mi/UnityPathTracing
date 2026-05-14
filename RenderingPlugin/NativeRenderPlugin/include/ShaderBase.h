#pragma once
#include <cstdint>
#include <d3d12.h>
#include <d3d12shader.h>
#include <dxgi1_6.h>
#include <dxcapi.h>
#include <wrl/client.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "IUnityLog.h"
#include "IUnityGraphicsD3D12.h"
#include "DescriptorHeapAllocator.h"
#include "ShaderBindings.h"  // ComputeBindingType, ComputeBinding, CS_BindingSlot

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// ShaderBase
//   Common state and logic shared by ComputeShader and RayTraceShader:
//     – device / log / allocator storage (all using ID3D12Device5)
//     – binding metadata (m_bindings, m_bindingIndex, heap-offset assignment)
//     – pre-load hints (SetRootConstantsHint, SetRootSRVHint)
//     – binding queries (GetBindingCount / GetSlotIndex / GetBindingName)
//     – BuildRootSignature (identical for both pipeline types)
//     – Log / Logf helpers
//
//   Subclasses provide:
//     – Initialize (may specialize device usage)
//     – ReflectBindings (CS uses ID3D12ShaderReflection; RT uses ID3D12LibraryReflection)
//     – BuildPipeline / GetPSO (different PSO types)
// ---------------------------------------------------------------------------
class ShaderBase
{
public:
    ShaderBase()          = default;
    virtual ~ShaderBase() = default;

    // --- Pre-load hints (must be called BEFORE LoadShaderFromBytes) ---
    void SetRootConstantsHint(const char* name, uint32_t num32BitValues);
    void SetRootSRVHint(const char* name);

    // --- Binding metadata queries ---
    uint32_t    GetBindingCount() const;
    uint32_t    GetSlotIndex   (const char* name) const;
    const char* GetBindingName (uint32_t index)   const;

    // --- Common accessors ---
    ID3D12RootSignature*               GetRootSignature()       const { return m_rootSig.Get(); }
    const std::vector<Binding>& GetBindings()            const { return m_bindings; }
    uint32_t GetRootParamSRV()         const { return m_rootParamSRV; }
    uint32_t GetRootParamUAV()         const { return m_rootParamUAV; }
    uint32_t GetRootParamCBVBase()     const { return m_rootParamCBVBase; }
    uint32_t GetRootParamRootSRVBase() const { return m_rootParamRootSRVBase; }
    uint32_t GetNumSRV()               const { return m_numSRV; }
    uint32_t GetNumUAV()               const { return m_numUAV; }
    const char* GetName()              const { return m_name.c_str(); }

    static constexpr uint32_t kInvalidAlloc = UINT32_MAX;

protected:
    // --- Logging ---
    void Log (UnityLogType type, const char* msg)      const;
    void Logf(UnityLogType type, const char* fmt, ...) const;

    // --- Shared build step: root signature ---
    //   Called by both ComputeShader::LoadShaderFromBytes and
    //   RayTraceShader::LoadShaderFromBytes after ReflectBindings().
    bool BuildRootSignature();

    // --- Shared reflection helpers ---
    //   ClassifyBinding: fills type/num fields on a ComputeBinding from a
    //   D3D12_SHADER_INPUT_BIND_DESC, using current hints.  Returns false if
    //   the binding should be skipped (samplers → caller handles separately).
    bool ClassifyBinding(const D3D12_SHADER_INPUT_BIND_DESC& bind,
                         const std::string& name,
                         Binding& out);

    //   AssignHeapOffsets: assigns consecutive heapOffset values per type group
    //   (SRV/TLAS → SRV table, UAV → UAV table, CBV → CBV inline).
    void AssignHeapOffsets();

    // --- Shared state ---
    IUnityLog*               m_log       = nullptr;
    ComPtr<ID3D12Device5>    m_device;           // unified to Device5; CS uses as Device
    DescriptorHeapAllocator* m_allocator = nullptr;
    IUnityGraphicsD3D12v8*   m_d3d12v8   = nullptr;
    std::string              m_name;

    ComPtr<ID3D12RootSignature> m_rootSig;

    struct SamplerReflection { std::string name; uint32_t reg; uint32_t space; };

    std::vector<Binding>             m_bindings;
    std::unordered_map<std::string, size_t> m_bindingIndex;
    std::vector<SamplerReflection>          m_samplerBindings;

    // Root parameter indices (populated by BuildRootSignature)
    uint32_t m_rootParamSRV         = kInvalidAlloc;
    uint32_t m_rootParamUAV         = kInvalidAlloc;
    uint32_t m_rootParamCBVBase     = kInvalidAlloc;
    uint32_t m_rootParamRootSRVBase = kInvalidAlloc;

    // Binding counts (populated by ReflectBindings in subclass)
    uint32_t m_numSRV           = 0;
    uint32_t m_numUAV           = 0;
    uint32_t m_numCBV           = 0;
    uint32_t m_numSRVArray      = 0;
    uint32_t m_numUAVArray      = 0;
    uint32_t m_numRootConstants = 0;
    uint32_t m_numRootSRV       = 0;

    // Pre-load hints
    std::unordered_map<std::string, uint32_t> m_rootConstantsHints;
    std::unordered_set<std::string>           m_rootSRVHints;
};
