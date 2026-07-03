#include "RasterShader.h"
#include <d3d12shader.h>

// ---------------------------------------------------------------------------

bool RasterShader::Initialize(ID3D12Device5* device, IUnityLog* log,
                              IUnityGraphicsD3D12v8* d3d12v8)
{
    m_log     = log;
    m_device  = device;
    m_d3d12v8 = d3d12v8;
    return true;
}

// ---------------------------------------------------------------------------
// LoadShaderFromBlobs
//   VS + PS DXIL → reflected (merged) bindings → root signature → graphics PSO.
// ---------------------------------------------------------------------------
bool RasterShader::LoadShaderFromBlobs(const uint8_t* vsDxil, uint32_t vsSize,
                                       const uint8_t* psDxil, uint32_t psSize,
                                       const RasterPipelineStateDesc& state,
                                       const char* name)
{
    m_name  = (name && name[0]) ? name : "RasterShader";
    m_state = state;

    if (!vsDxil || vsSize == 0 || !psDxil || psSize == 0)
    {
        Log(kUnityLogTypeError, "RasterShader::LoadShaderFromBlobs: empty VS or PS input");
        return false;
    }

    ComPtr<IDxcUtils> utils;
    if (FAILED(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils))))
    {
        Log(kUnityLogTypeError, "RasterShader::LoadShaderFromBlobs: failed to create IDxcUtils");
        return false;
    }

    ComPtr<IDxcBlobEncoding> vsEnc, psEnc;
    if (FAILED(utils->CreateBlob(vsDxil, vsSize, DXC_CP_ACP, &vsEnc)) ||
        FAILED(utils->CreateBlob(psDxil, psSize, DXC_CP_ACP, &psEnc)))
    {
        Log(kUnityLogTypeError, "RasterShader::LoadShaderFromBlobs: failed to create blobs");
        return false;
    }
    ComPtr<IDxcBlob> vsBlob = vsEnc;
    ComPtr<IDxcBlob> psBlob = psEnc;

    // Reset old pipeline state (mirrors ComputeShader::LoadShaderFromBytes).
    m_pso.Reset();
    m_rootSig.Reset();
    m_bindings.clear();
    m_bindingIndex.clear();
    m_samplerBindings.clear();
    m_numSRV = m_numUAV = m_numCBV = m_numSRVArray = m_numUAVArray = m_numRootConstants = m_numRootSRV = 0;
    m_numSRVSlots = m_numUAVSlots = 0;
    m_rootParamSRV = m_rootParamUAV = m_rootParamCBVBase = m_rootParamRootSRVBase = kInvalidAlloc;

    // Reflect both stages into the shared binding table (deduped by name), then
    // assign heap offsets once over the merged set.
    if (!ReflectBlobInto(vsBlob.Get())) return false;
    if (!ReflectBlobInto(psBlob.Get())) return false;
    AssignHeapOffsets();

    if (!BuildRootSignature())           return false;
    if (!BuildPipeline(vsBlob.Get(), psBlob.Get())) return false;

    Logf(kUnityLogTypeLog,
         "RasterShader '%s': pipeline ready (%u SRV, %u UAV, %u CBV, %u RT)",
         m_name.c_str(), m_numSRV, m_numUAV, m_numCBV, m_state.numRenderTargets);
    return true;
}

// ---------------------------------------------------------------------------
// ReflectBlobInto
//   Single-blob reflection (ID3D12ShaderReflection), additive: appends bindings
//   not already present (deduped by HLSL variable name) so VS + PS merge.
// ---------------------------------------------------------------------------
bool RasterShader::ReflectBlobInto(IDxcBlob* shaderBlob)
{
    ComPtr<IDxcUtils> utils;
    if (FAILED(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils))))
    {
        Log(kUnityLogTypeError, "RasterShader: failed to create IDxcUtils for reflection");
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
             "RasterShader: CreateReflection failed (hr=0x%08X) - stage skipped", hr);
        return true; // not fatal: a stage may legitimately bind no resources
    }

    D3D12_SHADER_DESC shDesc = {};
    refl->GetDesc(&shDesc);

    for (UINT ri = 0; ri < shDesc.BoundResources; ++ri)
    {
        D3D12_SHADER_INPUT_BIND_DESC bind = {};
        if (FAILED(refl->GetResourceBindingDesc(ri, &bind))) continue;

        if (bind.Type == D3D_SIT_SAMPLER && !UsesSharedSamplerTable())
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
        if (m_bindingIndex.count(bname)) continue; // already contributed by the other stage

        Binding cb = {};
        if (!ClassifyBinding(bind, bname, cb)) continue;

        m_bindingIndex[bname] = m_bindings.size();
        m_bindings.push_back(std::move(cb));
    }
    return true;
}

// ---------------------------------------------------------------------------
// BuildPipeline
//   Build the D3D12 graphics PSO from the merged root signature + VS/PS blobs
//   and the fixed-function state in m_state.
// ---------------------------------------------------------------------------
static D3D12_RENDER_TARGET_BLEND_DESC MakeBlend(uint32_t mode)
{
    D3D12_RENDER_TARGET_BLEND_DESC b = {};
    b.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    b.LogicOp               = D3D12_LOGIC_OP_NOOP;
    b.BlendOp               = D3D12_BLEND_OP_ADD;
    b.BlendOpAlpha          = D3D12_BLEND_OP_ADD;
    switch (mode)
    {
    case 1: // alpha
        b.BlendEnable = TRUE;
        b.SrcBlend = D3D12_BLEND_SRC_ALPHA;  b.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
        b.SrcBlendAlpha = D3D12_BLEND_ONE;   b.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
        break;
    case 2: // additive
        b.BlendEnable = TRUE;
        b.SrcBlend = D3D12_BLEND_ONE; b.DestBlend = D3D12_BLEND_ONE;
        b.SrcBlendAlpha = D3D12_BLEND_ONE; b.DestBlendAlpha = D3D12_BLEND_ONE;
        break;
    case 3: // premultiplied alpha
        b.BlendEnable = TRUE;
        b.SrcBlend = D3D12_BLEND_ONE; b.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
        b.SrcBlendAlpha = D3D12_BLEND_ONE; b.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
        break;
    case 4: // constant-color lerp — donut BloomPass composite (SrcBlend=ConstantColor,
            // DestBlend=InvConstantColor, alpha Src=Zero/Dest=One). The blend constant is set
            // per-draw via OMSetBlendFactor (RAS_RenderEventData.blendFactor = bloom intensity),
            // giving result = src*C + dst*(1-C) = lerp(dst, src, C).
        b.BlendEnable = TRUE;
        b.SrcBlend = D3D12_BLEND_BLEND_FACTOR; b.DestBlend = D3D12_BLEND_INV_BLEND_FACTOR;
        b.SrcBlendAlpha = D3D12_BLEND_ZERO;    b.DestBlendAlpha = D3D12_BLEND_ONE;
        break;
    default: // 0 = opaque
        b.BlendEnable = FALSE;
        b.SrcBlend = D3D12_BLEND_ONE; b.DestBlend = D3D12_BLEND_ZERO;
        b.SrcBlendAlpha = D3D12_BLEND_ONE; b.DestBlendAlpha = D3D12_BLEND_ZERO;
        break;
    }
    return b;
}

bool RasterShader::BuildPipeline(IDxcBlob* vsBlob, IDxcBlob* psBlob)
{
    // The caller picks the actual IA primitive topology (list/strip/…); the PSO's coarser
    // PrimitiveTopologyType (point/line/triangle) is derived from it.
    m_primTopology = m_state.primitiveTopology
        ? static_cast<D3D12_PRIMITIVE_TOPOLOGY>(m_state.primitiveTopology)
        : D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

    D3D12_PRIMITIVE_TOPOLOGY_TYPE topoType;
    switch (m_primTopology)
    {
    case D3D_PRIMITIVE_TOPOLOGY_POINTLIST:
        topoType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT; break;
    case D3D_PRIMITIVE_TOPOLOGY_LINELIST:
    case D3D_PRIMITIVE_TOPOLOGY_LINESTRIP:
        topoType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE; break;
    default: // triangle list / strip (and anything else) → triangle
        topoType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE; break;
    }

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = m_rootSig.Get();
    psoDesc.VS = { vsBlob->GetBufferPointer(), vsBlob->GetBufferSize() };
    psoDesc.PS = { psBlob->GetBufferPointer(), psBlob->GetBufferSize() };

    // --- Blend ---
    psoDesc.BlendState.AlphaToCoverageEnable  = FALSE;
    psoDesc.BlendState.IndependentBlendEnable = FALSE;
    const D3D12_RENDER_TARGET_BLEND_DESC rtBlend = MakeBlend(m_state.blendMode);
    for (UINT i = 0; i < 8; ++i)
        psoDesc.BlendState.RenderTarget[i] = rtBlend;

    psoDesc.SampleMask = UINT_MAX;

    // --- Rasterizer ---
    psoDesc.RasterizerState.FillMode = m_state.fillMode ? static_cast<D3D12_FILL_MODE>(m_state.fillMode)
                                                        : D3D12_FILL_MODE_SOLID;
    psoDesc.RasterizerState.CullMode = m_state.cullMode ? static_cast<D3D12_CULL_MODE>(m_state.cullMode)
                                                        : D3D12_CULL_MODE_NONE;
    psoDesc.RasterizerState.FrontCounterClockwise = m_state.frontCounterClockwise ? TRUE : FALSE;
    psoDesc.RasterizerState.DepthClipEnable       = TRUE;

    // --- Depth / stencil ---
    psoDesc.DepthStencilState.DepthEnable    = m_state.depthTestEnable ? TRUE : FALSE;
    psoDesc.DepthStencilState.DepthWriteMask = m_state.depthWriteEnable ? D3D12_DEPTH_WRITE_MASK_ALL
                                                                        : D3D12_DEPTH_WRITE_MASK_ZERO;
    psoDesc.DepthStencilState.DepthFunc = m_state.depthFunc ? static_cast<D3D12_COMPARISON_FUNC>(m_state.depthFunc)
                                                            : D3D12_COMPARISON_FUNC_LESS_EQUAL;
    psoDesc.DepthStencilState.StencilEnable = FALSE;

    // --- IA (vertex pulling: no input layout) ---
    psoDesc.InputLayout = { nullptr, 0 };
    psoDesc.PrimitiveTopologyType = topoType;

    // --- Render targets ---
    psoDesc.NumRenderTargets = m_state.numRenderTargets;
    for (UINT i = 0; i < 8; ++i)
        psoDesc.RTVFormats[i] = (i < m_state.numRenderTargets)
            ? static_cast<DXGI_FORMAT>(m_state.rtvFormats[i]) : DXGI_FORMAT_UNKNOWN;
    psoDesc.DSVFormat = static_cast<DXGI_FORMAT>(m_state.dsvFormat);

    psoDesc.SampleDesc.Count   = m_state.sampleCount ? m_state.sampleCount : 1;
    psoDesc.SampleDesc.Quality = 0;

    HRESULT hr = m_device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_pso));
    if (FAILED(hr))
    {
        Logf(kUnityLogTypeError, "RasterShader: CreateGraphicsPipelineState failed (hr=0x%08X)", hr);
        return false;
    }
    {
        std::wstring wname(m_name.begin(), m_name.end());
        wname += L"_PSO";
        m_pso->SetName(wname.c_str());
    }
    return true;
}
