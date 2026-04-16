#pragma once
#include <cstdint>
#include <d3d12.h>
#include <dxcapi.h>
#include <wrl/client.h>
#include <string>
#include <vector>
#include <unordered_map>
#include "IUnityLog.h"
#include "DescriptorHeapAllocator.h"

using Microsoft::WRL::ComPtr;

class BindlessTexture;
class BindlessBuffer;

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

enum class ComputeBindingType
{
    SRV,        // single StructuredBuffer<T> or ByteAddressBuffer or Texture (SRV)
    UAV,        // single RWTexture2D / RWStructuredBuffer / RWBuffer (UAV)
    CBV,        // ConstantBuffer<T>
    SRV_ARRAY,  // unbounded array[] bound via BindlessTexture or BindlessBuffer
};

struct ComputeBinding
{
    std::string       name;
    ComputeBindingType type;
    uint32_t          space;
    uint32_t          registerIndex;
    uint32_t          heapOffset;       // offset within SRV/UAV alloc range
    uint32_t          rootParam;        // root parameter index (SRV_ARRAY: own; others: shared table)
    ID3D12Resource*   boundResource;    // currently bound resource (non-owning)
    BindlessTexture*  boundBT;
    BindlessBuffer*   boundBB;
    UINT              boundCount;       // element count for StructuredBuffer SRVs
    UINT              boundStride;      // element stride (0 = raw ByteAddressBuffer)
};

class ComputeShader
{
public:
    ComputeShader();
    ~ComputeShader();

    bool Initialize(ID3D12Device* device, IUnityLog* log, DescriptorHeapAllocator* allocator);

    // Build pipeline from pre-compiled DXIL bytes (compiled as cs_6_x).
    bool LoadShaderFromBytes(const uint8_t* dxilBytes, uint32_t size);

    // --- Resource binding (all spaces) ---
    bool SetBuffer          (const char* name, ID3D12Resource* resource);
    bool SetStructuredBuffer(const char* name, ID3D12Resource* resource, UINT elementCount, UINT elementStride);
    bool SetTexture         (const char* name, ID3D12Resource* resource);
    bool SetRWBuffer        (const char* name, ID3D12Resource* resource);
    bool SetRWTexture       (const char* name, ID3D12Resource* resource);
    bool SetConstantBuffer  (const char* name, ID3D12Resource* resource);
    bool SetBindlessTexture (const char* name, BindlessTexture* bt);
    bool SetBindlessBuffer  (const char* name, BindlessBuffer*  bb);

    // --- Dispatch ---
    // threadGroupX/Y/Z are the number of thread groups (not threads) to dispatch.
    void Dispatch(ID3D12GraphicsCommandList* cmdList,
                  UINT threadGroupX, UINT threadGroupY, UINT threadGroupZ);

private:
    bool ReflectBindings(IDxcBlob* shaderBlob);
    bool BuildRootSignature();
    bool BuildPipeline(IDxcBlob* shaderBlob);

    bool AllocateAndWriteDescriptors();
    void UpdateDescriptors();
    void FreeAllAllocations();

    void Log (UnityLogType type, const char* msg) const;
    void Logf(UnityLogType type, const char* fmt, ...) const;

    IUnityLog*               m_log       = nullptr;
    ComPtr<ID3D12Device>     m_device;
    DescriptorHeapAllocator* m_allocator = nullptr;

    ComPtr<ID3D12PipelineState> m_pso;
    ComPtr<ID3D12RootSignature> m_rootSig;

    static constexpr uint32_t kInvalidAlloc = UINT32_MAX;
    uint32_t m_srvAllocBase = kInvalidAlloc;
    uint32_t m_uavAllocBase = kInvalidAlloc;

    std::vector<ComputeBinding>             m_bindings;
    std::unordered_map<std::string, size_t> m_bindingIndex;

    uint32_t m_rootParamSRV     = kInvalidAlloc;
    uint32_t m_rootParamUAV     = kInvalidAlloc;
    uint32_t m_rootParamCBVBase = kInvalidAlloc;

    uint32_t m_numSRV      = 0;
    uint32_t m_numUAV      = 0;
    uint32_t m_numCBV      = 0;
    uint32_t m_numSRVArray = 0;
};
