#include "ComputeShader.h"
#include <d3d12shader.h>
#include <cstdio>
#include <cstdarg>

// ---------------------------------------------------------------------------

bool ComputeShader::Initialize(ID3D12Device5* device, IUnityLog* log,
                                DescriptorHeapAllocator* allocator,
                                IUnityGraphicsD3D12v8*   d3d12v8)
{
    m_log       = log;
    m_device    = device;
    m_allocator = allocator;
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
    m_bindingIndex.clear();
    m_samplerBindings.clear();
    m_numSRV = m_numUAV = m_numCBV = m_numSRVArray = m_numUAVArray = m_numRootConstants = m_numRootSRV = 0;
    m_rootParamSRV = m_rootParamUAV = m_rootParamCBVBase = m_rootParamRootSRVBase = kInvalidAlloc;

    if (!ReflectBindings(shaderBlob.Get())) return false;
    if (!BuildRootSignature())              return false;
    if (!BuildPipeline(shaderBlob.Get()))   return false;

    Logf(kUnityLogTypeLog,
         "ComputeShader '%s': pipeline ready (%u SRV, %u UAV, %u CBV, %u SRV_ARRAY, %u UAV_ARRAY)",
         m_name.c_str(), m_numSRV, m_numUAV, m_numCBV, m_numSRVArray, m_numUAVArray);
    return true;
}

// ---------------------------------------------------------------------------
// ReflectBindings
//   Single-entrypoint compute shader — uses ID3D12ShaderReflection.
// ---------------------------------------------------------------------------
bool ComputeShader::ReflectBindings(IDxcBlob* shaderBlob)
{
    ComPtr<IDxcUtils> utils;
    if (FAILED(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils))))
    {
        Log(kUnityLogTypeError, "ComputeShader: failed to create IDxcUtils for reflection");
        return false;
    }

    DxcBuffer buf;
    buf.Ptr      = shaderBlob->GetBufferPointer();
    buf.Size     = shaderBlob->GetBufferSize();
    buf.Encoding = 0;

    ComPtr<ID3D12ShaderReflection> refl;
    HRESULT hr = utils->CreateReflection(&buf, IID_PPV_ARGS(&refl));
    if (FAILED(hr))
    {
        Logf(kUnityLogTypeWarning,
             "ComputeShader: CreateReflection failed (hr=0x%08X) - no bindings reflected", hr);
        return true; // not fatal
    }

    D3D12_SHADER_DESC shDesc = {};
    refl->GetDesc(&shDesc);
    Logf(kUnityLogTypeLog, "ComputeShader::ReflectBindings: %u bound resources", shDesc.BoundResources);

    for (UINT ri = 0; ri < shDesc.BoundResources; ++ri)
    {
        D3D12_SHADER_INPUT_BIND_DESC bind = {};
        if (FAILED(refl->GetResourceBindingDesc(ri, &bind))) continue;

        if (bind.Type == D3D_SIT_SAMPLER)
        {
            const std::string sname(bind.Name);
            bool found = false;
            for (const auto& s : m_samplerBindings)
                if (s.name == sname) { found = true; break; }
            if (!found)
                m_samplerBindings.push_back({ sname, bind.BindPoint, bind.Space });
            continue;
        }

        const std::string bname(bind.Name);
        if (m_bindingIndex.count(bname)) continue;

        Binding cb = {};
        if (!ClassifyBinding(bind, bname, cb)) continue;

        m_bindingIndex[bname] = m_bindings.size();
        m_bindings.push_back(std::move(cb));
    }

    AssignHeapOffsets();
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