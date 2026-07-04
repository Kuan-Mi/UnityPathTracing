#pragma once
#include "ShaderBase.h"
#include "ComputeShader.h"

// ---------------------------------------------------------------------------
// ComputePipeline
//   NVRHI-style compute pipeline object. The caller supplies a ComputeShader
//   handle plus binding layout handles; this object builds the root signature,
//   layout slot table, and D3D12 compute PSO.
// ---------------------------------------------------------------------------
class ComputePipeline : public ShaderBase
{
public:
    ComputePipeline()  = default;
    ~ComputePipeline() = default;

    bool Initialize(ID3D12Device5* device, IUnityLog* log, IUnityGraphicsD3D12v8* d3d12v8);
    bool BuildFromShader(ComputeShader* shader, const char* debugName = nullptr);

    ID3D12PipelineState* GetPSO() const { return m_pso.Get(); }
    ComputeShader* GetShader() const { return m_shader; }

    uint32_t GetNumUAVArray()      const { return m_numUAVArray; }
    uint32_t GetNumRootConstants() const { return m_numRootConstants; }
    uint32_t GetNumRootSRV()       const { return m_numRootSRV; }

private:
    bool BuildPipeline(IDxcBlob* shaderBlob);

    ComputeShader* m_shader = nullptr;
    ComPtr<ID3D12PipelineState> m_pso;
};
