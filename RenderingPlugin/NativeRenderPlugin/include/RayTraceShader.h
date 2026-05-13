#pragma once
#include <cstdint>
#include <d3d12.h>
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

using Microsoft::WRL::ComPtr;

class BindlessTexture;
class BindlessBuffer;
class BindlessUAVTexture;
class AccelerationStructure;
class NativeBuffer;

// ---------------------------------------------------------------------------
// RayTraceBindingType
//   Mirrors ComputeBindingType — fully aligned with the CS slot-based system.
// ---------------------------------------------------------------------------
enum class RayTraceBindingType
{
    SRV,            // single StructuredBuffer<T> / ByteAddressBuffer / Texture2D (SRV)
    UAV,            // single RWTexture2D / RWStructuredBuffer / RWBuffer (UAV)
    CBV,            // ConstantBuffer<T> bound as inline root CBV descriptor
    SRV_ARRAY,      // unbounded Texture2D[] / ByteAddressBuffer[] via BindlessTexture/BindlessBuffer
    UAV_ARRAY,      // unbounded RWTexture2D[] via BindlessUAVTexture
    TLAS,           // RaytracingAccelerationStructure
    ROOT_CONSTANTS, // ConstantBuffer<T> pushed via SetComputeRoot32BitConstants
    ROOT_SRV,       // buffer SRV / TLAS promoted to inline root SRV descriptor
};

// ---------------------------------------------------------------------------
// RayTraceBinding
//   One reflected binding.  Stateless: no bound* fields.
//   Resource data is passed per-dispatch via CS_BindingSlot[].
// ---------------------------------------------------------------------------
struct RayTraceBinding
{
    std::string         name;           // HLSL variable name
    RayTraceBindingType type;           // SRV / UAV / CBV / SRV_ARRAY / UAV_ARRAY / TLAS / ...
    uint32_t            space;          // register space
    uint32_t            registerIndex;  // t/u/b register number
    uint32_t            heapOffset;     // offset within shared SRV or UAV alloc range
    uint32_t            rootParam;      // root parameter index (SRV_ARRAY/UAV_ARRAY/ROOT_CONSTANTS/ROOT_SRV: own; others: shared table)
    uint32_t            num32BitValues; // ROOT_CONSTANTS only
};

// ---------------------------------------------------------------------------
// RayTraceShader
//   One self-contained DXR pipeline.  Manages its own PSO, root signature,
//   and shader tables.  Heap management is delegated to RayTraceDescriptorSet.
//
// Resource data is passed per-dispatch via CS_BindingSlot[] (same layout as
// ComputeShader) so the same C# DescriptorSet ring-buffer infrastructure works.
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
class RayTraceShader
{
public:
    RayTraceShader();
    ~RayTraceShader();

    bool Initialize(ID3D12Device5* device, IUnityLog* log,
                    DescriptorHeapAllocator* allocator, IUnityGraphicsD3D12v8* d3d12v8);

    // Build DXR pipeline from pre-compiled DXIL lib bytes.
    // name is used as the D3D12 debug name (optional).
    bool LoadShaderFromBytes(const uint8_t* dxilBytes, uint32_t size, const char* name = nullptr);

    // Pre-load hints — must be called BEFORE LoadShaderFromBytes.
    // Promote a CBV to inline root 32-bit constants.
    void SetRootConstantsHint(const char* name, uint32_t num32BitValues);
    // Promote a buffer SRV or TLAS to an inline root SRV descriptor.
    void SetRootSRVHint(const char* name);

    // --- Binding metadata queries (main thread, called from C# to build slot arrays) ---
    uint32_t    GetBindingCount() const;
    uint32_t    GetSlotIndex   (const char* name) const;
    const char* GetBindingName (uint32_t index)   const;

    // --- Accessors for RayTraceDescriptorSet ---
    ID3D12StateObject*                    GetPSO()            const { return m_pso.Get(); }
    ID3D12RootSignature*                  GetRootSignature()  const { return m_rootSig.Get(); }
    const std::vector<RayTraceBinding>&   GetBindings()       const { return m_bindings; }
    uint32_t GetRootParamSRV()            const { return m_rootParamSRV; }
    uint32_t GetRootParamUAV()            const { return m_rootParamUAV; }
    uint32_t GetRootParamCBVBase()        const { return m_rootParamCBVBase; }
    uint32_t GetRootParamRootSRVBase()    const { return m_rootParamRootSRVBase; }
    uint32_t GetNumSRV()                  const { return m_numSRV; }
    uint32_t GetNumUAV()                  const { return m_numUAV; }

    // Shader table accessors for RayTraceDescriptorSet::Dispatch
    ID3D12Resource* GetRayGenTable()      const { return m_rayGenTable.Get(); }
    ID3D12Resource* GetMissTable()        const { return m_missTable.Get(); }
    ID3D12Resource* GetHitGroupTable()    const { return m_hitGroupTable.Get(); }
    uint32_t        GetMissCount()        const { return static_cast<uint32_t>(m_missShaders.size()); }
    uint32_t        GetHitGroupCount()    const { return static_cast<uint32_t>(m_hitGroups.size()); }

    const char* GetName() const { return m_name.c_str(); }

    static constexpr uint32_t kInvalidAlloc = UINT32_MAX;

private:
    bool ReflectBindings(IDxcBlob* shaderLib);
    bool BuildRootSignature();
    bool BuildPipeline(IDxcBlob* shaderLib);
    bool BuildShaderTable();

    void Log (UnityLogType type, const char* msg)      const;
    void Logf(UnityLogType type, const char* fmt, ...) const;

    IUnityLog*               m_log       = nullptr;
    ComPtr<ID3D12Device5>    m_device;
    DescriptorHeapAllocator* m_allocator = nullptr;
    IUnityGraphicsD3D12v8*   m_d3d12v8   = nullptr;
    std::string              m_name;

    // Pipeline
    ComPtr<ID3D12StateObject>   m_pso;
    ComPtr<ID3D12RootSignature> m_rootSig;

    // Shader tables (upload heap; GPU reads at DispatchRays time)
    ComPtr<ID3D12Resource> m_rayGenTable;
    ComPtr<ID3D12Resource> m_missTable;
    ComPtr<ID3D12Resource> m_hitGroupTable;

    // Sampler reflection
    struct SamplerReflection { std::string name; uint32_t reg; uint32_t space; };

    // Hit group info — one per discovered (ClosestHit*/AnyHit*) group
    struct HitGroupInfo
    {
        std::wstring groupExport;       // e.g. L"HitGroup_Primary"
        std::wstring closestHitExport;
        std::wstring anyHitExport;
    };

    // All reflected bindings (all types except samplers)
    std::vector<RayTraceBinding>            m_bindings;
    std::unordered_map<std::string, size_t> m_bindingIndex; // name → index

    std::vector<SamplerReflection>          m_samplerBindings;

    // Root parameter indices (set during BuildRootSignature)
    uint32_t m_rootParamSRV         = kInvalidAlloc;
    uint32_t m_rootParamUAV         = kInvalidAlloc;
    uint32_t m_rootParamCBVBase     = kInvalidAlloc;
    uint32_t m_rootParamRootSRVBase = kInvalidAlloc;

    // Binding counts
    uint32_t m_numSRV           = 0;
    uint32_t m_numUAV           = 0;
    uint32_t m_numCBV           = 0;
    uint32_t m_numSRVArray      = 0;
    uint32_t m_numUAVArray      = 0;
    uint32_t m_numRootConstants = 0;
    uint32_t m_numRootSRV       = 0;

    // Pre-load hints
    std::unordered_map<std::string, uint32_t> m_rootConstantsHints; // name → num32BitValues
    std::unordered_set<std::string>           m_rootSRVHints;       // names promoted to root SRV

    // Shader entry points
    std::vector<std::wstring>                 m_rayGenShaders;  // [0] used for Dispatch
    std::vector<std::wstring>                 m_missShaders;
    std::vector<HitGroupInfo>                 m_hitGroups;
    std::unordered_map<std::wstring, size_t>  m_hitGroupIndex;  // groupKey → index
};
