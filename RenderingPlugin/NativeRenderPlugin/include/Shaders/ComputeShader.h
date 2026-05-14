#pragma once
#include "ShaderBase.h"  // pulls in all D3D12/DXC/Unity/ShaderBindings.h headers

class BindlessTexture;
class BindlessBuffer;
class BindlessUAVTexture;
class AccelerationStructure;
class NativeBuffer;

// ComputeBindingType, ComputeBinding, CS_BindingSlot, CS_BindingObjectKind
// are defined in ShaderBindings.h (included transitively via ShaderBase.h).

// ---------------------------------------------------------------------------
// ComputeShader
//   One self-contained compute shader object.  Common binding metadata,
//   root-signature build, logging, and hints are provided by ShaderBase.
//
// Root parameter layout (built dynamically from reflection):
//   0   - SRV descriptor table (one range per SRV binding)           optional
//   1   - UAV descriptor table (one range per UAV binding)            optional
//   2+  - one descriptor table per SRV_ARRAY (unbounded SRV) binding
//   N+  - one descriptor table per UAV_ARRAY (unbounded UAV) binding
//   M+  - one root CBV descriptor per CBV binding
//   P+  - one root 32-bit constants slot per ROOT_CONSTANTS binding
// ---------------------------------------------------------------------------
class ComputeShader : public ShaderBase
{
public:
    ComputeShader()  = default;
    ~ComputeShader() = default;

    bool Initialize(ID3D12Device5* device, IUnityLog* log,
                    DescriptorHeapAllocator* allocator, IUnityGraphicsD3D12v8* d3d12v8);

    // Build pipeline from pre-compiled DXIL bytes (compiled as cs_6_x).
    bool LoadShaderFromBytes(const uint8_t* dxilBytes, uint32_t size, const char* name = nullptr);

    // --- Accessors for ComputeDescriptorSet ---
    ID3D12PipelineState* GetPSO()         const { return m_pso.Get(); }
    uint32_t GetNumUAVArray()             const { return m_numUAVArray; }
    uint32_t GetNumRootConstants()        const { return m_numRootConstants; }
    uint32_t GetNumRootSRV()              const { return m_numRootSRV; }

private:
    bool ReflectBindings(IDxcBlob* shaderBlob);
    bool BuildPipeline  (IDxcBlob* shaderBlob);

    ComPtr<ID3D12PipelineState> m_pso;
};