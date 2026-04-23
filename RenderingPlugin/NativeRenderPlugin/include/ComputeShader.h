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
enum class ComputeBindingType
{
    SRV,        // single StructuredBuffer<T> or ByteAddressBuffer or Texture (SRV)
    UAV,        // single RWTexture2D / RWStructuredBuffer / RWBuffer (UAV)
    CBV,        // ConstantBuffer<T>
    SRV_ARRAY,  // unbounded array[] bound via BindlessTexture or BindlessBuffer
    TLAS,       // RaytracingAccelerationStructure
};

struct ComputeBinding
{
    std::string       name;             // HLSL variable name
    ComputeBindingType type;            // SRV / UAV / CBV / SRV_ARRAY
    uint32_t          space;            // register space
    uint32_t          registerIndex;    // tn / un / bn number
    uint32_t          heapOffset;       // offset within the shared SRV/UAV alloc range
    uint32_t          rootParam;        // root parameter index (SRV_ARRAY: own; others: shared table)
    ID3D12Resource*   boundResource;    // currently bound resource (non-owning)
    AccelerationStructure* boundAS;     // bound AccelerationStructure for dynamic TLAS lookup (non-owning)
    BindlessTexture*  boundBT;          // bound BindlessTexture (SRV_ARRAY, texture)
    BindlessBuffer*   boundBB;          // bound BindlessBuffer  (SRV_ARRAY, buffer)
    UINT              boundCount;       // element count for StructuredBuffer SRVs
    UINT              boundStride;      // element stride (0 = raw ByteAddressBuffer)
};

// ---------------------------------------------------------------------------
// ComputeShader
//   One self-contained compute shader object.  Manages its own pipeline
//   state, root signature, and a slice of the global DescriptorHeapAllocator.
//
// Fully generic: no register/slot conventions are assumed in C++.
// All resources are reflected from HLSL and bound by name from C#.
//
// Root parameter layout (built dynamically from reflection):
//   0   – SRV descriptor table (one range per SRV binding)           optional
//   1   – UAV descriptor table (one range per UAV binding)            optional
//   2+  – one descriptor table per SRV_ARRAY (unbounded) binding
//   N+  – one root CBV descriptor per CBV binding
// ---------------------------------------------------------------------------
class ComputeShader
{
public:
    ComputeShader();
    ~ComputeShader();

    bool Initialize(ID3D12Device* device, IUnityLog* log, DescriptorHeapAllocator* allocator, IUnityGraphicsD3D12v8* d3d12v8);

    // Build pipeline from pre-compiled DXIL bytes (compiled as cs_6_x).
    // name is used as the D3D12 debug name for the PSO and root signature (optional).
    bool LoadShaderFromBytes(const uint8_t* dxilBytes, uint32_t size, const char* name = nullptr);

    // --- Resource binding (all spaces) ---
    // Bind by HLSL variable name. Returns false if name not found or type mismatch.

    // ByteAddressBuffer / unstructured SRV
    bool SetBuffer          (const char* name, ID3D12Resource* resource);
    // StructuredBuffer<T> SRV — caller must supply element count and byte stride
    bool SetStructuredBuffer(const char* name, ID3D12Resource* resource, UINT elementCount, UINT elementStride);
    // RaytracingAccelerationStructure (maps to TLAS binding)
    bool SetAccelerationStructure(const char* name, ID3D12Resource* tlas);
    // Preferred: binds by AccelerationStructure object — TLAS ptr is read dynamically at Dispatch time.
    bool SetAccelerationStructureObject(const char* name, AccelerationStructure* as);
    // Texture SRV
    bool SetTexture         (const char* name, ID3D12Resource* resource);
    // UAV
    bool SetRWBuffer           (const char* name, ID3D12Resource* resource);
    bool SetRWStructuredBuffer (const char* name, ID3D12Resource* resource, UINT elementCount, UINT elementStride);
    bool SetRWTexture          (const char* name, ID3D12Resource* resource);
    // CBV
    bool SetConstantBuffer  (const char* name, ID3D12Resource* resource);

    // Unbounded array bindings (via BindlessTexture / BindlessBuffer)
    bool SetBindlessTexture (const char* name, BindlessTexture* bt);
    bool SetBindlessBuffer  (const char* name, BindlessBuffer*  bb);

    // --- Dispatch ---
    // All resources must be bound by name via SetXxx before calling Dispatch each frame.
    // threadGroupX/Y/Z are the number of thread groups (not threads) to dispatch.
    void Dispatch(ID3D12GraphicsCommandList* cmdList,
                  UINT threadGroupX, UINT threadGroupY, UINT threadGroupZ);

private:
    bool ReflectBindings(IDxcBlob* shaderBlob);
    bool BuildRootSignature();
    bool BuildPipeline(IDxcBlob* shaderBlob);

    // Allocate global-heap slots and write all descriptors.
    // Called once (first Dispatch) or after shader reload.
    bool AllocateAndWriteDescriptors();

    // Re-write all mutable descriptors every frame.
    void UpdateDescriptors();

    void FreeAllAllocations();

    void Log (UnityLogType type, const char* msg) const;
    void Logf(UnityLogType type, const char* fmt, ...) const;

    IUnityLog*               m_log       = nullptr;
    ComPtr<ID3D12Device>     m_device;
    DescriptorHeapAllocator* m_allocator = nullptr;
    IUnityGraphicsD3D12v8*   m_d3d12v8   = nullptr;
    std::string              m_name;       // D3D12 debug name (set from C# via LoadShaderFromBytes)

    // Pipeline
    ComPtr<ID3D12PipelineState> m_pso;
    ComPtr<ID3D12RootSignature> m_rootSig;

    // Sampler binding discovered via reflection — properties parsed from name (Unity inline sampler convention).
    struct SamplerReflection
    {
        std::string name;
        uint32_t    reg;
        uint32_t    space;
    };

    // Global-heap allocations
    static constexpr uint32_t kInvalidAlloc = UINT32_MAX;
    uint32_t m_srvAllocBase = kInvalidAlloc;
    uint32_t m_uavAllocBase = kInvalidAlloc;

    // All reflected bindings (all spaces, all types except samplers)
    std::vector<ComputeBinding>             m_bindings;
    std::unordered_map<std::string, size_t> m_bindingIndex; // name → index
    std::vector<SamplerReflection>          m_samplerBindings;

    // Root parameter indices (set during BuildRootSignature)
    uint32_t m_rootParamSRV     = kInvalidAlloc; // SRV table
    uint32_t m_rootParamUAV     = kInvalidAlloc; // UAV table
    uint32_t m_rootParamCBVBase = kInvalidAlloc; // first root CBV (sequential)
    // SRV_ARRAY bindings store their root param inside ComputeBinding::rootParam.

    // Binding counts
    uint32_t m_numSRV      = 0;
    uint32_t m_numUAV      = 0;
    uint32_t m_numCBV      = 0;
    uint32_t m_numSRVArray = 0;
};
