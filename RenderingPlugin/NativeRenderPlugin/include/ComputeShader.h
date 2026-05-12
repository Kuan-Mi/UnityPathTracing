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
// Describes one resource binding discovered via DXC reflection.
// ---------------------------------------------------------------------------
enum class ComputeBindingType
{
    SRV,            // single StructuredBuffer<T> or ByteAddressBuffer or Texture (SRV)
    UAV,            // single RWTexture2D / RWStructuredBuffer / RWBuffer (UAV)
    CBV,            // ConstantBuffer<T> bound as inline root CBV descriptor
    SRV_ARRAY,      // unbounded array[] bound via BindlessTexture or BindlessBuffer
    UAV_ARRAY,      // unbounded RWTexture2D[] bound via BindlessUAVTexture
    TLAS,           // RaytracingAccelerationStructure
    ROOT_CONSTANTS, // ConstantBuffer<T> pushed via SetComputeRoot32BitConstants
    ROOT_SRV,       // buffer SRV / TLAS promoted to inline root descriptor (SetComputeRootShaderResourceView)
};

struct ComputeBinding
{
    std::string       name;             // HLSL variable name
    ComputeBindingType type;            // SRV / UAV / CBV / SRV_ARRAY / TLAS / ROOT_CONSTANTS
    uint32_t          space;            // register space
    uint32_t          registerIndex;    // tn / un / bn number
    uint32_t          heapOffset;       // offset within the shared SRV/UAV alloc range
    uint32_t          rootParam;        // root parameter index (SRV_ARRAY: own; others: shared table)
    uint32_t          num32BitValues;   // ROOT_CONSTANTS only: total DWORD count from hint
    // NOTE: No bound* fields here — bindings are passed per-dispatch via CS_BindingSlot[].
};

// ---------------------------------------------------------------------------
// CS_BindingSlot
//   One slot per reflected binding, passed from C# via IssuePluginEventAndData.
//   C# fills these on the main thread and the render-thread callback reads them.
//   Must match NativeRenderPlugin.CS_BindingSlot exactly (Pack=4, 32 bytes).
// ---------------------------------------------------------------------------
#pragma pack(push, 4)
enum class CS_BindingObjectKind : uint32_t
{
    None               = 0,
    AccelStruct        = 1,
    BindlessTexture    = 2,
    BindlessBuffer     = 3,
    BindlessUAVTexture = 6,
    NativeBuffer       = 5,
};

struct CS_BindingSlot
{
    uint64_t             resourcePtr;   // ID3D12Resource* (may be 0)
    uint64_t             objectPtr;     // AccelerationStructure* | BindlessTexture* | BindlessBuffer*
    uint32_t             count;         // element count  (StructuredBuffer or typed buffer)
    uint32_t             stride;        // element stride (StructuredBuffer; 0 = raw/typed)
    CS_BindingObjectKind objectKind;    // what objectPtr points to
    uint32_t             format;        // DXGI_FORMAT for typed buffer UAV/SRV (0 = raw/structured)
}; // 32 bytes
#pragma pack(pop)

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
//   2+  – one descriptor table per SRV_ARRAY (unbounded SRV) binding
//   N+  – one descriptor table per UAV_ARRAY (unbounded UAV) binding
//   M+  – one root CBV descriptor per CBV binding
//   P+  – one root 32-bit constants slot per ROOT_CONSTANTS binding
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

    // Must be called BEFORE LoadShaderFromBytes to promote a CBV to root 32-bit constants.
    // num32BitValues: total number of 32-bit values in the constant buffer.
    void SetRootConstantsHint(const char* name, uint32_t num32BitValues);

    // Must be called BEFORE LoadShaderFromBytes to promote a buffer SRV or TLAS binding
    // to an inline root SRV descriptor (SetComputeRootShaderResourceView).
    // Only valid for buffer resources (StructuredBuffer, ByteAddressBuffer, TLAS).
    void SetRootSRVHint(const char* name);

    // --- Binding metadata queries (called from C# on main thread to build slot arrays) ---

    // Total number of reflected bindings (excluding samplers).
    uint32_t    GetBindingCount() const;
    // Returns the slot index for a given HLSL variable name, or UINT32_MAX if not found.
    uint32_t    GetSlotIndex(const char* name) const;
    // Returns the HLSL variable name for a given slot index, or nullptr if out of range.
    const char* GetBindingName(uint32_t index) const;

    // --- Accessors for ComputeDescriptorSet ---
    ID3D12PipelineState*              GetPSO()            const { return m_pso.Get(); }
    ID3D12RootSignature*              GetRootSignature()  const { return m_rootSig.Get(); }
    const std::vector<ComputeBinding>& GetBindings()      const { return m_bindings; }
    uint32_t GetRootParamSRV()           const { return m_rootParamSRV; }
    uint32_t GetRootParamUAV()           const { return m_rootParamUAV; }
    uint32_t GetRootParamCBVBase()       const { return m_rootParamCBVBase; }
    uint32_t GetRootParamRootSRVBase()   const { return m_rootParamRootSRVBase; }
    uint32_t GetNumSRV()                 const { return m_numSRV; }
    uint32_t GetNumUAV()                 const { return m_numUAV; }
    uint32_t GetNumUAVArray()            const { return m_numUAVArray; }
    uint32_t GetNumRootConstants()       const { return m_numRootConstants; }
    uint32_t GetNumRootSRV()             const { return m_numRootSRV; }
    const char* GetName()          const { return m_name.c_str(); }

    static constexpr uint32_t kInvalidAlloc = UINT32_MAX;

private:
    bool ReflectBindings(IDxcBlob* shaderBlob);
    bool BuildRootSignature();
    bool BuildPipeline(IDxcBlob* shaderBlob);

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

    // All reflected bindings (all spaces, all types except samplers)
    std::vector<ComputeBinding>             m_bindings;
    std::unordered_map<std::string, size_t> m_bindingIndex; // name → index
    std::vector<SamplerReflection>          m_samplerBindings;

    // Root parameter indices (set during BuildRootSignature)
    uint32_t m_rootParamSRV         = kInvalidAlloc; // SRV table
    uint32_t m_rootParamUAV         = kInvalidAlloc; // UAV table
    uint32_t m_rootParamCBVBase     = kInvalidAlloc; // first root CBV (sequential)
    uint32_t m_rootParamRootSRVBase = kInvalidAlloc; // first inline root SRV (sequential)
    // SRV_ARRAY, ROOT_CONSTANTS, and ROOT_SRV bindings store their root param inside ComputeBinding::rootParam.

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
    std::unordered_set<std::string>           m_rootSRVHints;       // names to promote to root SRV
};
