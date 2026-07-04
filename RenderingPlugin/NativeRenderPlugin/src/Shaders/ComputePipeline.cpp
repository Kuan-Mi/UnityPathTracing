#include "ComputePipeline.h"

bool ComputePipeline::Initialize(ID3D12Device5* device, IUnityLog* log,
                                 IUnityGraphicsD3D12v8* d3d12v8)
{
    m_log     = log;
    m_device  = device;
    m_d3d12v8 = d3d12v8;
    return true;
}

bool ComputePipeline::BuildFromShader(ComputeShader* shader, const char* debugName)
{
    if (!shader || !shader->GetBlob())
    {
        Log(kUnityLogTypeError, "ComputePipeline::BuildFromShader: invalid shader handle");
        return false;
    }

    m_shader = shader;
    m_name = (debugName && debugName[0]) ? debugName : shader->GetName();

    m_pso.Reset();
    m_rootSig.Reset();
    m_bindings.clear();

    if (!BuildSharedRootSignature()) return false;
    BuildBindingsFromLayout();
    if (!BuildPipeline(shader->GetBlob())) return false;

    Logf(kUnityLogTypeLog,
         "ComputePipeline '%s': pipeline ready (%u layout slots: %u SRV, %u UAV, %u CBV, %u SRV_ARRAY, %u UAV_ARRAY)",
         m_name.c_str(), GetBindingCount(),
         m_numSRV, m_numUAV, m_numCBV, m_numSRVArray, m_numUAVArray);
    return true;
}

bool ComputePipeline::BuildPipeline(IDxcBlob* shaderBlob)
{
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature     = m_rootSig.Get();
    psoDesc.CS.pShaderBytecode = shaderBlob->GetBufferPointer();
    psoDesc.CS.BytecodeLength  = shaderBlob->GetBufferSize();

    HRESULT hr = m_device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_pso));
    if (FAILED(hr))
    {
        Logf(kUnityLogTypeError, "ComputePipeline: CreateComputePipelineState failed (hr=0x%08X)", hr);
        return false;
    }

    std::wstring wname(m_name.begin(), m_name.end());
    wname += L"_PSO";
    m_pso->SetName(wname.c_str());
    return true;
}
