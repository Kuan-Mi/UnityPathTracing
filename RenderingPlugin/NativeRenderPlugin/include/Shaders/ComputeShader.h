#pragma once
#include "ShaderBase.h"

// ---------------------------------------------------------------------------
// ComputeShader
//   NVRHI-style shader object: owns ShaderDesc metadata and pre-compiled DXIL
//   bytecode only. Root signatures and PSOs are created later by
//   ComputePipeline from a ComputePipelineDesc.
// ---------------------------------------------------------------------------
class ComputeShader
{
public:
    ComputeShader()  = default;
    ~ComputeShader() = default;

    bool Initialize(ID3D12Device5* device, IUnityLog* log, IUnityGraphicsD3D12v8* d3d12v8);

    bool LoadShaderFromBytes(const uint8_t* dxilBytes, uint32_t size,
                             const char* debugName = nullptr,
                             const char* entryName = nullptr);

    IDxcBlob*   GetBlob()      const { return m_shaderBlob.Get(); }
    const char* GetName()      const { return m_name.c_str(); }
    const char* GetEntryName() const { return m_entryName.c_str(); }

private:
    void Log (UnityLogType type, const char* msg)      const;
    void Logf(UnityLogType type, const char* fmt, ...) const;

    IUnityLog*             m_log     = nullptr;
    ComPtr<ID3D12Device5>  m_device;
    IUnityGraphicsD3D12v8* m_d3d12v8 = nullptr;
    std::string            m_name;
    std::string            m_entryName = "main";
    ComPtr<IDxcBlob>       m_shaderBlob;
};
