#pragma once
#include "ShaderBase.h"  // pulls in all D3D12/DXC/Unity headers + ComputeBinding types

// RayTraceBindingType and RayTraceBinding are type aliases for the Compute
// counterparts — shared descriptor-set logic uses ComputeBinding directly.
using RayTraceBindingType = BindingType;
using RayTraceBinding     = Binding;

// ---------------------------------------------------------------------------
// RayTraceShader
//   One self-contained DXR pipeline.  Common binding metadata, root-signature
//   build, logging, and hints are provided by ShaderBase.
//
// Root parameter layout (built dynamically from reflection):
//   0   – SRV descriptor table (one range per SRV/TLAS binding)     optional
//   1   – UAV descriptor table (one range per UAV binding)           optional
//   2+  – one descriptor table per SRV_ARRAY binding
//   N+  – one descriptor table per UAV_ARRAY binding
//   M+  – one root CBV descriptor per CBV binding
//   P+  – one inline root SRV per ROOT_SRV binding
//   Q+  – one root 32-bit constants slot per ROOT_CONSTANTS binding
// ---------------------------------------------------------------------------
class RayTraceShader : public ShaderBase
{
public:
    RayTraceShader()  = default;
    ~RayTraceShader() = default;

    bool Initialize(ID3D12Device5* device, IUnityLog* log,
                    DescriptorHeapAllocator* allocator, IUnityGraphicsD3D12v8* d3d12v8);

    // Build DXR pipeline from pre-compiled DXIL lib bytes.
    // flags: bit 0 = allow D3D12_RAYTRACING_PIPELINE_FLAG_ALLOW_OPACITY_MICROMAPS (lib_6_9+).
    // maxPayloadSizeInBytes: MaxPayloadSizeInBytes for D3D12_RAYTRACING_SHADER_CONFIG.
    // rayGenName: RayGeneration entry point to use for DispatchRays. Null/empty = first discovered.
    bool LoadShaderFromBytes(const uint8_t* dxilBytes, uint32_t size,
                             const char* name = nullptr,
                             uint32_t flags = 0,
                             uint32_t maxPayloadSizeInBytes = 4,
                             const char* rayGenName = nullptr);

    // Allow Opacity Micromaps in the pipeline (requires lib_6_9+ DXIL and GPU support).
    void SetAllowOpacityMicromaps(bool allow) { m_allowOpacityMicromaps = allow; }

    // --- Accessors for RayTraceDescriptorSet ---
    ID3D12StateObject* GetPSO()         const { return m_pso.Get(); }

    // Shader table accessors for RayTraceDescriptorSet::Dispatch
    ID3D12Resource* GetRayGenTable()    const { return m_rayGenTable.Get(); }
    ID3D12Resource* GetMissTable()      const { return m_missTable.Get(); }
    ID3D12Resource* GetHitGroupTable()  const { return m_hitGroupTable.Get(); }
    uint32_t        GetMissCount()      const { return static_cast<uint32_t>(m_missShaders.size()); }
    uint32_t        GetHitGroupCount()  const { return static_cast<uint32_t>(m_hitGroups.size()); }

private:
    bool ReflectBindings(IDxcBlob* shaderLib);
    bool BuildPipeline  (IDxcBlob* shaderLib);
    bool BuildShaderTable();

    // Hit group info — one per discovered (ClosestHit*/AnyHit*) group
    struct HitGroupInfo
    {
        std::wstring groupExport;       // e.g. L"HitGroup_Primary"
        std::wstring closestHitExport;
        std::wstring anyHitExport;
    };

    // DXR-specific state
    ComPtr<ID3D12StateObject>   m_pso;

    // Shader tables (upload heap; GPU reads at DispatchRays time)
    ComPtr<ID3D12Resource> m_rayGenTable;
    ComPtr<ID3D12Resource> m_missTable;
    ComPtr<ID3D12Resource> m_hitGroupTable;

    // Shader entry points
    std::vector<std::wstring>                m_rayGenShaders;  // all discovered; m_rayGenName selects which one
    std::wstring                             m_rayGenName;     // requested entry; empty = use [0]
    std::vector<std::wstring>                m_missShaders;
    std::vector<HitGroupInfo>                m_hitGroups;
    std::unordered_map<std::wstring, size_t> m_hitGroupIndex;  // groupKey → index

    bool     m_allowOpacityMicromaps = false;
    uint32_t m_maxPayloadSizeInBytes = 4;
};
