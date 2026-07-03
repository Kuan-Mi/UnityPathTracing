#include "ComputeShader.h"
#include <d3d12shader.h>
#include <cstdio>
#include <cstdarg>

// ---------------------------------------------------------------------------

bool ComputeShader::Initialize(ID3D12Device5* device, IUnityLog* log,
                                IUnityGraphicsD3D12v8*   d3d12v8)
{
    m_log       = log;
    m_device    = device;
    m_d3d12v8   = d3d12v8;
    return true;
}

// ---------------------------------------------------------------------------
// LoadShaderFromBytes
// ---------------------------------------------------------------------------
bool ComputeShader::LoadShaderFromBytes(const uint8_t* dxilBytes, uint32_t size, const char* name)
{
    m_name = (name && name[0]) ? name : "ComputeShader";
    if (!dxilBytes || size == 0)
    {
        Log(kUnityLogTypeError, "ComputeShader::LoadShaderFromBytes: empty input");
        return false;
    }

    ComPtr<IDxcUtils> utils;
    if (FAILED(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils))))
    {
        Log(kUnityLogTypeError, "ComputeShader::LoadShaderFromBytes: failed to create IDxcUtils");
        return false;
    }

    ComPtr<IDxcBlobEncoding> blobEnc;
    if (FAILED(utils->CreateBlob(dxilBytes, size, DXC_CP_ACP, &blobEnc)))
    {
        Log(kUnityLogTypeError, "ComputeShader::LoadShaderFromBytes: failed to create blob");
        return false;
    }
    ComPtr<IDxcBlob> shaderBlob = blobEnc;

    // Reset old pipeline
    m_pso.Reset();
    m_rootSig.Reset();
    m_bindings.clear();

    // Explicit layout is the whole binding contract — no DXIL reflection.
    if (!BuildSharedRootSignature())      return false;
    BuildBindingsFromLayout();
    if (!BuildPipeline(shaderBlob.Get())) return false;

    Logf(kUnityLogTypeLog,
         "ComputeShader '%s': pipeline ready (%u layout slots: %u SRV, %u UAV, %u CBV, %u SRV_ARRAY, %u UAV_ARRAY)",
         m_name.c_str(), GetBindingCount(),
         m_numSRV, m_numUAV, m_numCBV, m_numSRVArray, m_numUAVArray);
    return true;
}

// ---------------------------------------------------------------------------
// BuildPipeline
// ---------------------------------------------------------------------------
bool ComputeShader::BuildPipeline(IDxcBlob* shaderBlob)
{
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature     = m_rootSig.Get();
    psoDesc.CS.pShaderBytecode = shaderBlob->GetBufferPointer();
    psoDesc.CS.BytecodeLength  = shaderBlob->GetBufferSize();

    HRESULT hr = m_device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_pso));
    if (FAILED(hr))
    {
        Logf(kUnityLogTypeError, "ComputeShader: CreateComputePipelineState failed (hr=0x%08X)", hr);
        return false;
    }
    {
        std::wstring wname(m_name.begin(), m_name.end());
        wname += L"_PSO";
        m_pso->SetName(wname.c_str());
    }
    return true;
}
