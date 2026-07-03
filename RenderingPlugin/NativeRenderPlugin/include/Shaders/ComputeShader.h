#pragma once
#include "ShaderBase.h"  // pulls in all D3D12/DXC/Unity/ShaderBindings.h headers

class BindlessTexture;
class BindlessBuffer;
class BindlessUAVTexture;
class AccelerationStructure;
class NativeBuffer;

// BindingType, Binding, BindingSlot, BindingObjectKind are defined in
// ShaderBindings.h (included transitively via ShaderBase.h).

// ---------------------------------------------------------------------------
// ComputeShader
//   One self-contained compute shader object.  Root signature and binding
//   table are derived entirely from the explicit binding layout declared by
//   the caller before load (nvrhi BindingLayout model — see ShaderBase); the
//   DXIL is never reflected.
// ---------------------------------------------------------------------------
class ComputeShader : public ShaderBase
{
public:
    ComputeShader()  = default;
    ~ComputeShader() = default;

    bool Initialize(ID3D12Device5* device, IUnityLog* log, IUnityGraphicsD3D12v8* d3d12v8);

    // Build pipeline from pre-compiled DXIL bytes (compiled as cs_6_x).
    bool LoadShaderFromBytes(const uint8_t* dxilBytes, uint32_t size, const char* name = nullptr);

    // --- Accessors for ComputeDescriptorSet ---
    ID3D12PipelineState* GetPSO()         const { return m_pso.Get(); }
    uint32_t GetNumUAVArray()             const { return m_numUAVArray; }
    uint32_t GetNumRootConstants()        const { return m_numRootConstants; }
    uint32_t GetNumRootSRV()              const { return m_numRootSRV; }

private:
    bool BuildPipeline  (IDxcBlob* shaderBlob);

    ComPtr<ID3D12PipelineState> m_pso;
};