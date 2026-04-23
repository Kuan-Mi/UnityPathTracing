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
    ComputeBindingType type;            // SRV / UAV / CBV / SRV_ARRAY / TLAS
    uint32_t          space;            // register space
    uint32_t          registerIndex;    // tn / un / bn number
    uint32_t          heapOffset;       // offset within the shared SRV/UAV alloc range
    uint32_t          rootParam;        // root parameter index (SRV_ARRAY: own; others: shared table)
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
    None            = 0,
    AccelStruct     = 1,
    BindlessTexture = 2,
    BindlessBuffer  = 3,
};

struct CS_BindingSlot
{
    uint64_t             resourcePtr;   // ID3D12Resource* (may be 0)
    uint64_t             objectPtr;     // AccelerationStructure* | BindlessTexture* | BindlessBuffer*
    uint32_t             count;         // element count  (StructuredBuffer)
    uint32_t             stride;        // element stride (StructuredBuffer; 0 = raw)
    CS_BindingObjectKind objectKind;    // what objectPtr points to
    uint32_t             _pad;
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

    // --- Binding metadata queries (called from C# on main thread to build slot arrays) ---

    // Total number of reflected bindings (excluding samplers).
    uint32_t    GetBindingCount() const;
    // Returns the slot index for a given HLSL variable name, or UINT32_MAX if not found.
    uint32_t    GetSlotIndex(const char* name) const;
    // Returns the HLSL variable name for a given slot index, or nullptr if out of range.
    const char* GetBindingName(uint32_t index) const;

    // --- Dispatch ---
    // slots[i] carries the binding for the i-th reflected binding (indexed by GetSlotIndex).
    // slotCount must equal GetBindingCount().
    void Dispatch(ID3D12GraphicsCommandList* cmdList,
                  UINT threadGroupX, UINT threadGroupY, UINT threadGroupZ,
                  const CS_BindingSlot* slots, uint32_t slotCount);

private:
    bool ReflectBindings(IDxcBlob* shaderBlob);
    bool BuildRootSignature();
    bool BuildPipeline(IDxcBlob* shaderBlob);

    // Allocate global-heap slots and write all descriptors.
    // Called once (first Dispatch) or after shader reload.
    bool AllocateAndWriteDescriptors(const CS_BindingSlot* slots, uint32_t slotCount);

    // Re-write all mutable descriptors every frame.
    void UpdateDescriptors(const CS_BindingSlot* slots, uint32_t slotCount);

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
