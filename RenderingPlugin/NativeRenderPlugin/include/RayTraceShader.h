#pragma once
#include <cstdint>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <dxcapi.h>
#include <wrl/client.h>
#include <string>
#include <vector>
#include <unordered_map>
#include "IUnityLog.h"
#include "IUnityGraphicsD3D12.h"
#include "DescriptorHeapAllocator.h"

using Microsoft::WRL::ComPtr;

class BindlessTexture;
class BindlessBuffer;
class AccelerationStructure;

// ---------------------------------------------------------------------------
// Describes one resource binding discovered via DXC reflection.
// ---------------------------------------------------------------------------
enum class UserBindingType
{
    SRV,        // single StructuredBuffer<T> or ByteAddressBuffer (SRV)
    UAV,        // single RWTexture2D / RWStructuredBuffer (UAV)
    CBV,        // ConstantBuffer<T>
    SRV_ARRAY,  // unbounded array[] bound via BindlessTexture or BindlessBuffer
    TLAS,       // RaytracingAccelerationStructure
};

struct UserBinding
{
    std::string     name;               // HLSL variable name
    UserBindingType type;               // SRV / UAV / CBV / SRV_ARRAY / TLAS
    uint32_t        space;              // register space
    uint32_t        registerIndex;      // tn / un / bn number
    uint32_t        heapOffset;         // offset within the shared SRV/UAV alloc range
    uint32_t        rootParam;          // root parameter index (SRV_ARRAY: own root param; others: shared table)
    ID3D12Resource*    boundResource;      // currently bound resource (non-owning)
    AccelerationStructure* boundAS;    // bound AccelerationStructure for dynamic TLAS lookup (non-owning)
    BindlessTexture* boundBT;           // bound BindlessTexture (SRV_ARRAY, texture)
    BindlessBuffer*  boundBB;           // bound BindlessBuffer  (SRV_ARRAY, buffer)
    UINT             boundCount;        // element count for StructuredBuffer SRVs
    UINT             boundStride;       // element stride for StructuredBuffer SRVs (0 = raw ByteAddressBuffer)
};

// ---------------------------------------------------------------------------
// RayTraceShader
//   One self-contained ray tracing shader object.  Manages its own pipeline
//   state, root signature, and a slice of the global DescriptorHeapAllocator.
//
// Fully generic: no register/slot conventions are assumed in C++.
// All resources are reflected from HLSL and bound by name from C#.
//
// Root parameter layout (built dynamically from reflection):
//   0   – SRV descriptor table (one range per SRV/TLAS binding)    optional
//   1   – UAV descriptor table (one range per UAV binding)          optional
//   2+  – one descriptor table per SRV_ARRAY (unbounded) binding
//   N+  – one root CBV descriptor per CBV binding
// ---------------------------------------------------------------------------
class RayTraceShader
{
public:
    RayTraceShader();
    ~RayTraceShader();

    bool Initialize(ID3D12Device5* device, IUnityLog* log, DescriptorHeapAllocator* allocator, IUnityGraphicsD3D12v8* d3d12v8);

    // Build pipeline from pre-compiled DXIL bytes (e.g. deserialized from Unity asset cache).
    // Skips file I/O and shader compilation entirely.
    bool LoadShaderFromBytes(const uint8_t* dxilBytes, uint32_t size);

    // --- Resource binding (all spaces) ---
    // Bind by HLSL variable name. Returns false if name not found or type mismatch.

    // ByteAddressBuffer / unstructured SRV
    bool SetBuffer       (const char* name, ID3D12Resource* resource);
    // StructuredBuffer<T> SRV — caller must supply element count and byte stride
    bool SetStructuredBuffer(const char* name, ID3D12Resource* resource, UINT elementCount, UINT elementStride);
    // RaytracingAccelerationStructure (maps to TLAS binding, e.g. "SceneBVH")
    bool SetAccelerationStructure(const char* name, ID3D12Resource* tlas);
    // Preferred: binds by AccelerationStructure object — TLAS ptr is read dynamically at Dispatch time.
    bool SetAccelerationStructureObject(const char* name, AccelerationStructure* as);
    // UAV
    bool SetRWBuffer           (const char* name, ID3D12Resource* resource);
    bool SetRWStructuredBuffer (const char* name, ID3D12Resource* resource, UINT elementCount, UINT elementStride);
    bool SetRWTexture          (const char* name, ID3D12Resource* resource);
    // Texture SRV
    bool SetTexture      (const char* name, ID3D12Resource* resource);
    // CBV
    bool SetConstantBuffer(const char* name, ID3D12Resource* resource);

    // Unbounded array bindings (via BindlessTexture / BindlessBuffer)
    bool SetBindlessTexture(const char* name, BindlessTexture* bt);
    bool SetBindlessBuffer (const char* name, BindlessBuffer*  bb);

    // --- Dispatch ---
    // All resources (OutputTexture, SceneConstants, TLAS, etc.) must be bound
    // by name via SetXxx before calling Dispatch each frame.
    void Dispatch(
        ID3D12GraphicsCommandList4*  cmdList,
        UINT width, UINT height);

private:
    bool ReflectUserBindings(IDxcBlob* shaderLib);
    bool BuildRootSignature();
    bool BuildPipeline(IDxcBlob* shaderLib);
    bool BuildShaderTable();

    // Allocate global-heap slots and write all descriptors.
    // Called once (first Dispatch) or after shader reload.
    bool AllocateAndWriteDescriptors();

    // Re-write all mutable descriptors every frame.
    void UpdateUserDescriptors();

    void FreeAllAllocations();

    void Log(UnityLogType type, const char* msg) const;
    void Logf(UnityLogType type, const char* fmt, ...) const;

    IUnityLog*               m_log       = nullptr;
    ComPtr<ID3D12Device5>    m_device;
    DescriptorHeapAllocator* m_allocator = nullptr;
    IUnityGraphicsD3D12v8*   m_d3d12v8   = nullptr;

    // Pipeline
    ComPtr<ID3D12StateObject>   m_pso;
    ComPtr<ID3D12RootSignature> m_rootSig;

    // Shader tables
    ComPtr<ID3D12Resource> m_rayGenTable;
    ComPtr<ID3D12Resource> m_missTable;
    ComPtr<ID3D12Resource> m_hitGroupTable;

    // Global-heap allocations
    //   m_srvAllocBase : N slots – all single SRV + TLAS bindings (in reflection order)
    //   m_uavAllocBase : M slots – all single UAV bindings
    //   SRV_ARRAY bindings own their heap slots inside BindlessTexture/BindlessBuffer.
    static constexpr uint32_t kInvalidAlloc = UINT32_MAX;
    uint32_t m_srvAllocBase = kInvalidAlloc;
    uint32_t m_uavAllocBase = kInvalidAlloc;

    // Sampler binding discovered via reflection — properties parsed from name (Unity inline sampler convention).
    struct SamplerReflection
    {
        std::string name;
        uint32_t    reg;
        uint32_t    space;
    };

    // One triangle hit group derived from shader reflection + naming convention.
    // Naming rule: strip type prefix ("ClosestHit"/"AnyHit") then optional leading "_" to get groupKey.
    //   groupKey == ""        → groupExport = L"HitGroup"
    //   groupKey != ""        → groupExport = L"HitGroup_" + groupKey
    // Example: ClosestHitShader + AnyHitShader → key="Shader" → L"HitGroup_Shader"
    //          ClosestHit_Primary              → key="Primary" → L"HitGroup_Primary"
    struct HitGroupInfo
    {
        std::wstring groupExport;        // exported hit group name, e.g. L"HitGroup_Shader"
        std::wstring closestHitExport;   // HLSL function name, may be empty
        std::wstring anyHitExport;       // HLSL function name, may be empty
    };

    // All reflected bindings (all spaces, all types except samplers)
    std::vector<UserBinding>                m_userBindings;
    std::unordered_map<std::string, size_t> m_bindingIndex; // name → index
    std::vector<SamplerReflection>          m_samplerBindings;

    // Root parameter indices (set during BuildRootSignature)
    uint32_t m_rootParamSRV    = kInvalidAlloc; // SRV+TLAS table
    uint32_t m_rootParamUAV    = kInvalidAlloc; // UAV table
    uint32_t m_rootParamCBVBase = kInvalidAlloc; // first root CBV (sequential)
    // SRV_ARRAY bindings store their root param inside UserBinding::rootParam.

    // Binding counts
    uint32_t m_numSRV      = 0; // single SRV + TLAS bindings (go into SRV table)
    uint32_t m_numUAV      = 0; // single UAV bindings
    uint32_t m_numCBV      = 0; // CBV bindings
    uint32_t m_numSRVArray = 0; // unbounded SRV array bindings

    // Shader entry points discovered via reflection (populated in ReflectUserBindings)
    std::vector<std::wstring>                        m_rayGenShaders; // [0] is always used for Dispatch
    std::vector<std::wstring>                        m_missShaders;
    std::vector<HitGroupInfo>                        m_hitGroups;
    std::unordered_map<std::wstring, size_t>         m_hitGroupIndex; // groupKey → index in m_hitGroups

};
