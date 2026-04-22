#include "ComputeShader.h"
#include "AccelerationStructure.h"
#include "BindlessTexture.h"
#include "BindlessBuffer.h"
#include <d3d12shader.h>
#include <cstdio>
#include <cstdarg>
#include <algorithm>
#include <windows.h>

// ---------------------------------------------------------------------------

ComputeShader::ComputeShader() = default;

ComputeShader::~ComputeShader()
{
    FreeAllAllocations();
}

void ComputeShader::Log(UnityLogType type, const char* msg) const
{
    if (m_log) m_log->Log(type, msg, __FILE__, __LINE__);
    else        printf("[ComputeShader] %s\n", msg);
}

void ComputeShader::Logf(UnityLogType type, const char* fmt, ...) const
{
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    Log(type, buf);
}

bool ComputeShader::Initialize(ID3D12Device* device, IUnityLog* log, DescriptorHeapAllocator* allocator, IUnityGraphicsD3D12v8* d3d12v8)
{
    m_log       = log;
    m_device    = device;
    m_allocator = allocator;
    m_d3d12v8   = d3d12v8;
    return true;
}

// ---------------------------------------------------------------------------
// LoadShaderFromBytes
//   Build the pipeline from pre-compiled DXIL bytes.
// ---------------------------------------------------------------------------
bool ComputeShader::LoadShaderFromBytes(const uint8_t* dxilBytes, uint32_t size)
{
    if (!dxilBytes || size == 0)
    {
        Log(kUnityLogTypeError, "ComputeShader::LoadShaderFromBytes: empty input");
        return false;
    }

    // Wrap the raw bytes in an IDxcBlob via IDxcUtils
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
    m_numSRV = m_numUAV = m_numCBV = m_numSRVArray = 0;
    m_rootParamSRV = m_rootParamUAV = m_rootParamCBVBase = kInvalidAlloc;

    if (m_allocator)
    {
        if (m_srvAllocBase != kInvalidAlloc)
        {
            m_allocator->Free(m_srvAllocBase, m_numSRV);
            m_srvAllocBase = kInvalidAlloc;
        }
        if (m_uavAllocBase != kInvalidAlloc)
        {
            m_allocator->Free(m_uavAllocBase, m_numUAV);
            m_uavAllocBase = kInvalidAlloc;
        }
    }

    if (!ReflectBindings(shaderBlob.Get())) return false;
    if (!BuildRootSignature())              return false;
    if (!BuildPipeline(shaderBlob.Get()))   return false;

    Logf(kUnityLogTypeLog, "ComputeShader: pipeline ready from bytes (%u SRV, %u UAV, %u CBV, %u SRV_ARRAY)",
         m_numSRV, m_numUAV, m_numCBV, m_numSRVArray);
    return true;
}

// ---------------------------------------------------------------------------
// ReflectBindings
//   Collects ALL resource bindings from HLSL (all registers, all spaces).
//   Uses ID3D12ShaderReflection for a single-entrypoint compute shader.
//   No registers are skipped — C# controls everything via SetXxx by name.
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

    Logf(kUnityLogTypeLog, "ComputeShader::ReflectBindings: %u bound resources",
         shDesc.BoundResources);

    for (UINT ri = 0; ri < shDesc.BoundResources; ++ri)
    {
        D3D12_SHADER_INPUT_BIND_DESC bind = {};
        if (FAILED(refl->GetResourceBindingDesc(ri, &bind))) continue;

        if (bind.Type == D3D_SIT_SAMPLER)
        {
            // Collect sampler by name (de-duplicate)
            const std::string sname(bind.Name);
            bool found = false;
            for (const auto& s : m_samplerBindings)
                if (s.name == sname) { found = true; break; }
            if (!found)
                m_samplerBindings.push_back({ sname, bind.BindPoint, bind.Space });
            continue;
        }

        const std::string name(bind.Name);
        if (m_bindingIndex.count(name)) continue; // de-duplicate

        ComputeBinding cb;
        cb.name          = name;
        cb.space         = bind.Space;
        cb.registerIndex = bind.BindPoint;
        cb.boundResource = nullptr;
        cb.boundAS       = nullptr;
        cb.boundBT       = nullptr;
        cb.boundBB       = nullptr;
        cb.heapOffset    = 0;
        cb.rootParam     = kInvalidAlloc;
        cb.boundCount    = 0;
        cb.boundStride   = 0;

        switch (bind.Type)
        {
        case D3D_SIT_RTACCELERATIONSTRUCTURE:
            cb.type = ComputeBindingType::TLAS;
            ++m_numSRV; // TLAS shares the SRV descriptor table
            break;
        case D3D_SIT_CBUFFER:
            cb.type = ComputeBindingType::CBV;
            ++m_numCBV;
            break;
        case D3D_SIT_TBUFFER:
        case D3D_SIT_TEXTURE:
        case D3D_SIT_STRUCTURED:
        case D3D_SIT_BYTEADDRESS:
            if (bind.BindCount == 0)
            {
                cb.type = ComputeBindingType::SRV_ARRAY;
                ++m_numSRVArray;
            }
            else
            {
                cb.type = ComputeBindingType::SRV;
                ++m_numSRV;
            }
            break;
        default: // UAV variants
            cb.type = ComputeBindingType::UAV;
            ++m_numUAV;
            break;
        }

        m_bindingIndex[name] = m_bindings.size();
        m_bindings.push_back(std::move(cb));
    }

    // Assign consecutive heap offsets per type group
    // TLAS shares the SRV descriptor table (same as RayTraceShader)
    uint32_t srvOff = 0, uavOff = 0, cbvOff = 0;
    for (auto& b : m_bindings)
    {
        if      (b.type == ComputeBindingType::SRV || b.type == ComputeBindingType::TLAS)  b.heapOffset = srvOff++;
        else if (b.type == ComputeBindingType::UAV)  b.heapOffset = uavOff++;
        else if (b.type == ComputeBindingType::CBV)  b.heapOffset = cbvOff++;
    }

    return true;
}

// ---------------------------------------------------------------------------
// BuildRootSignature
//   Fully dynamic — no hardcoded registers or spaces.
//   Param 0: SRV table (one range per SRV binding)          optional
//   Param 1: UAV table (one range per UAV binding)           optional
//   Params+: one table per SRV_ARRAY binding
//   Params+: one root CBV per CBV binding
// ---------------------------------------------------------------------------
bool ComputeShader::BuildRootSignature()
{
    std::vector<D3D12_DESCRIPTOR_RANGE1> allRanges;
    allRanges.reserve(m_numSRV + m_numUAV + m_numSRVArray);
    Logf(kUnityLogTypeLog, "ComputeShader::BuildRootSignature: %u SRV, %u UAV, %u SRV_ARRAY bindings",
         m_numSRV, m_numUAV, m_numSRVArray);

    // --- SRV descriptor ranges (one per SRV/TLAS binding) ---
    const size_t srvRangesOffset = allRanges.size();
    for (const auto& b : m_bindings)
    {
        if (b.type != ComputeBindingType::SRV && b.type != ComputeBindingType::TLAS) continue;
        D3D12_DESCRIPTOR_RANGE1 r = {};
        r.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        r.NumDescriptors                    = 1;
        r.BaseShaderRegister                = b.registerIndex;
        r.RegisterSpace                     = b.space;
        r.Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
        r.OffsetInDescriptorsFromTableStart = b.heapOffset;
        allRanges.push_back(r);
        Logf(kUnityLogTypeLog, "  SRV binding: name='%s' t%u space%u heapOffset=%u",
             b.name.c_str(), b.registerIndex, b.space, b.heapOffset);
    }

    // --- UAV descriptor ranges (one per UAV binding) ---
    const size_t uavRangesOffset = allRanges.size();
    for (const auto& b : m_bindings)
    {
        if (b.type != ComputeBindingType::UAV) continue;
        D3D12_DESCRIPTOR_RANGE1 r = {};
        r.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        r.NumDescriptors                    = 1;
        r.BaseShaderRegister                = b.registerIndex;
        r.RegisterSpace                     = b.space;
        r.Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
        r.OffsetInDescriptorsFromTableStart = b.heapOffset;
        allRanges.push_back(r);
        Logf(kUnityLogTypeLog, "  UAV binding: name='%s' u%u space%u heapOffset=%u",
             b.name.c_str(), b.registerIndex, b.space, b.heapOffset);
    }

    // --- SRV_ARRAY descriptor ranges (one per unbounded array binding) ---
    const size_t srvArrayRangesOffset = allRanges.size();
    for (const auto& b : m_bindings)
    {
        if (b.type != ComputeBindingType::SRV_ARRAY) continue;
        D3D12_DESCRIPTOR_RANGE1 r = {};
        r.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        r.NumDescriptors                    = UINT_MAX;
        r.BaseShaderRegister                = b.registerIndex;
        r.RegisterSpace                     = b.space;
        r.Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
        r.OffsetInDescriptorsFromTableStart = 0;
        allRanges.push_back(r);
        Logf(kUnityLogTypeLog, "  SRV_ARRAY binding: name='%s' t%u space%u",
             b.name.c_str(), b.registerIndex, b.space);
    }

    std::vector<D3D12_ROOT_PARAMETER1> params;
    params.reserve((m_numSRV ? 1 : 0) + (m_numUAV ? 1 : 0) + m_numSRVArray + m_numCBV);

    // Optional - SRV table
    if (m_numSRV > 0)
    {
        m_rootParamSRV = static_cast<uint32_t>(params.size());
        D3D12_ROOT_PARAMETER1 p = {};
        p.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p.DescriptorTable.NumDescriptorRanges = m_numSRV;
        p.DescriptorTable.pDescriptorRanges   = &allRanges[srvRangesOffset];
        p.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
        params.push_back(p);
        Logf(kUnityLogTypeLog, "  Root param %u: SRV table with %u descriptors", m_rootParamSRV, m_numSRV);
    }
    // Optional - UAV table
    if (m_numUAV > 0)
    {
        m_rootParamUAV = static_cast<uint32_t>(params.size());
        D3D12_ROOT_PARAMETER1 p = {};
        p.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p.DescriptorTable.NumDescriptorRanges = m_numUAV;
        p.DescriptorTable.pDescriptorRanges   = &allRanges[uavRangesOffset];
        p.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
        params.push_back(p);
        Logf(kUnityLogTypeLog, "  Root param %u: UAV table with %u descriptors", m_rootParamUAV, m_numUAV);
    }
    // One table per SRV_ARRAY
    {
        uint32_t arrayIdx = 0;
        for (auto& b : m_bindings)
        {
            if (b.type != ComputeBindingType::SRV_ARRAY) continue;
            b.rootParam = static_cast<uint32_t>(params.size());
            D3D12_ROOT_PARAMETER1 p = {};
            p.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            p.DescriptorTable.NumDescriptorRanges = 1;
            p.DescriptorTable.pDescriptorRanges   = &allRanges[srvArrayRangesOffset + arrayIdx++];
            p.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
            params.push_back(p);
            Logf(kUnityLogTypeLog, "  Root param %u: SRV_ARRAY '%s' with unbounded descriptors",
                 b.rootParam, b.name.c_str());
        }
    }
    // One root CBV per CBV binding
    if (m_numCBV > 0)
    {
        m_rootParamCBVBase = static_cast<uint32_t>(params.size());
        for (auto& b : m_bindings)
        {
            if (b.type != ComputeBindingType::CBV) continue;
            b.rootParam = static_cast<uint32_t>(params.size());
            D3D12_ROOT_PARAMETER1 p = {};
            p.ParameterType             = D3D12_ROOT_PARAMETER_TYPE_CBV;
            p.Descriptor.ShaderRegister = b.registerIndex;
            p.Descriptor.RegisterSpace  = b.space;
            p.ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;
            params.push_back(p);
            Logf(kUnityLogTypeLog, "  Root param %u: CBV '%s' b%u space%u",
                 b.rootParam, b.name.c_str(), b.registerIndex, b.space);
        }
    }

    // Static samplers — built dynamically from reflected SamplerState names.
    // Properties are parsed from the variable name using Unity's inline sampler convention:
    //   Filter  : "point" → POINT  |  "trilinear"/"linear" → LINEAR (default)  |  "aniso" → ANISOTROPIC
    //   Address : "clamp" → CLAMP  |  "repeat" → WRAP  |  "mirroronce" → MIRROR_ONCE  |  "mirror" → MIRROR
    //             default → WRAP
    auto ToLower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return (char)tolower(c); });
        return s;
    };
    auto Contains = [](const std::string& haystack, const char* needle) {
        return haystack.find(needle) != std::string::npos;
    };

    std::vector<D3D12_STATIC_SAMPLER_DESC> samplers;
    samplers.reserve(m_samplerBindings.size());
    for (const auto& sr : m_samplerBindings)
    {
        const std::string lower = ToLower(sr.name);

        D3D12_FILTER filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR; // default
        if      (Contains(lower, "point")) filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        else if (Contains(lower, "aniso")) filter = D3D12_FILTER_ANISOTROPIC;

        D3D12_TEXTURE_ADDRESS_MODE addr = D3D12_TEXTURE_ADDRESS_MODE_WRAP; // default
        if      (Contains(lower, "mirroronce")) addr = D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
        else if (Contains(lower, "mirror"))     addr = D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
        else if (Contains(lower, "clamp"))      addr = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        else if (Contains(lower, "repeat"))     addr = D3D12_TEXTURE_ADDRESS_MODE_WRAP;

        D3D12_STATIC_SAMPLER_DESC sd = {};
        sd.Filter           = filter;
        sd.AddressU = sd.AddressV = sd.AddressW = addr;
        sd.MaxAnisotropy    = (filter == D3D12_FILTER_ANISOTROPIC) ? 16 : 1;
        sd.ComparisonFunc   = D3D12_COMPARISON_FUNC_NEVER;
        sd.BorderColor      = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE;
        sd.MaxLOD           = D3D12_FLOAT32_MAX;
        sd.ShaderRegister   = sr.reg;
        sd.RegisterSpace    = sr.space;
        sd.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        samplers.push_back(sd);
        Logf(kUnityLogTypeLog, "  Static sampler: name='%s' filter=%u address=%u reg%u space%u",
             sr.name.c_str(), filter, addr, sr.reg, sr.space);
    }

    D3D12_ROOT_SIGNATURE_DESC1 rsDesc1 = {};
    rsDesc1.NumParameters     = static_cast<UINT>(params.size());
    rsDesc1.pParameters       = params.empty() ? nullptr : params.data();
    rsDesc1.NumStaticSamplers = static_cast<UINT>(samplers.size());
    rsDesc1.pStaticSamplers   = samplers.empty() ? nullptr : samplers.data();
    rsDesc1.Flags             = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC vrsDesc = {};
    vrsDesc.Version  = D3D_ROOT_SIGNATURE_VERSION_1_1;
    vrsDesc.Desc_1_1 = rsDesc1;

    Logf(kUnityLogTypeLog, "BuildRootSignature: %u root param(s), %u static sampler(s)",
         rsDesc1.NumParameters, rsDesc1.NumStaticSamplers);

    // Validate all register spaces before serialization to catch HLSL bindings missing explicit register() decorations.
    bool spaceValid = true;
    for (const auto& range : allRanges)
    {
        if (range.RegisterSpace >= 0xfffffff0)
        {
            const char* bindName = "?";
            for (const auto& b : m_bindings)
            {
                if (b.registerIndex == range.BaseShaderRegister && b.space == range.RegisterSpace)
                    { bindName = b.name.c_str(); break; }
            }
            Logf(kUnityLogTypeError,
                 "ComputeShader: binding '%s' has invalid RegisterSpace=0x%08X -- add an explicit register(xN, spaceM) decoration in HLSL",
                 bindName, range.RegisterSpace);
            spaceValid = false;
        }
    }
    for (const auto& p : params)
    {
        if (p.ParameterType == D3D12_ROOT_PARAMETER_TYPE_CBV ||
            p.ParameterType == D3D12_ROOT_PARAMETER_TYPE_SRV ||
            p.ParameterType == D3D12_ROOT_PARAMETER_TYPE_UAV)
        {
            if (p.Descriptor.RegisterSpace >= 0xfffffff0)
            {
                Logf(kUnityLogTypeError,
                     "ComputeShader: root descriptor has invalid RegisterSpace=0x%08X -- add an explicit register() decoration in HLSL",
                     p.Descriptor.RegisterSpace);
                spaceValid = false;
            }
        }
    }
    for (const auto& sd : samplers)
    {
        if (sd.RegisterSpace >= 0xfffffff0)
        {
            Logf(kUnityLogTypeError,
                 "ComputeShader: static sampler s%u has invalid RegisterSpace=0x%08X -- add an explicit register() decoration in HLSL",
                 sd.ShaderRegister, sd.RegisterSpace);
            spaceValid = false;
        }
    }
    if (!spaceValid)
        return false;

    ComPtr<ID3DBlob> sigBlob, errBlob;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&vrsDesc, &sigBlob, &errBlob);
    if (FAILED(hr))
    {
        Logf(kUnityLogTypeError, "ComputeShader: D3D12SerializeVersionedRootSignature failed (hr=0x%08X): %s",
             hr, errBlob ? (char*)errBlob->GetBufferPointer() : "");
        return false;
    }

    hr = m_device->CreateRootSignature(0, sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(), IID_PPV_ARGS(&m_rootSig));
    if (FAILED(hr))
    {
        Logf(kUnityLogTypeError, "ComputeShader: CreateRootSignature failed (hr=0x%08X)", hr);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// BuildPipeline
// ---------------------------------------------------------------------------
bool ComputeShader::BuildPipeline(IDxcBlob* shaderBlob)
{
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature      = m_rootSig.Get();
    psoDesc.CS.pShaderBytecode  = shaderBlob->GetBufferPointer();
    psoDesc.CS.BytecodeLength   = shaderBlob->GetBufferSize();

    HRESULT hr = m_device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_pso));
    if (FAILED(hr))
    {
        Logf(kUnityLogTypeError, "ComputeShader: CreateComputePipelineState failed (hr=0x%08X)", hr);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Resource setters
// ---------------------------------------------------------------------------
bool ComputeShader::SetBuffer(const char* name, ID3D12Resource* res)
{
    if (!name) return false;
    auto it = m_bindingIndex.find(name);
    if (it == m_bindingIndex.end()) return false;
    ComputeBinding& b = m_bindings[it->second];
    if (b.type != ComputeBindingType::SRV) return false;
    b.boundResource = res;
    b.boundStride   = 0; // raw ByteAddressBuffer
    return true;
}

bool ComputeShader::SetStructuredBuffer(const char* name, ID3D12Resource* res, UINT count, UINT stride)
{
    if (!name) return false;
    auto it = m_bindingIndex.find(name);
    if (it == m_bindingIndex.end()) return false;
    ComputeBinding& b = m_bindings[it->second];
    if (b.type != ComputeBindingType::SRV) return false;
    b.boundResource = res;
    b.boundCount    = count;
    b.boundStride   = stride;
    return true;
}

bool ComputeShader::SetAccelerationStructure(const char* name, ID3D12Resource* tlas)
{
    if (!name) return false;
    auto it = m_bindingIndex.find(name);
    if (it == m_bindingIndex.end()) return false;
    ComputeBinding& b = m_bindings[it->second];
    if (b.type != ComputeBindingType::TLAS) return false;
    b.boundResource = tlas;
    b.boundAS       = nullptr; // clear dynamic binding if explicitly setting raw ptr
    return true;
}

bool ComputeShader::SetAccelerationStructureObject(const char* name, AccelerationStructure* as)
{
    if (!name) return false;
    auto it = m_bindingIndex.find(name);
    if (it == m_bindingIndex.end()) return false;
    ComputeBinding& b = m_bindings[it->second];
    if (b.type != ComputeBindingType::TLAS) return false;
    b.boundAS       = as;     // store object pointer; TLAS will be read dynamically at Dispatch
    b.boundResource = nullptr;
    return true;
}

bool ComputeShader::SetTexture(const char* name, ID3D12Resource* res)
{
    if (!name) return false;
    auto it = m_bindingIndex.find(name);
    if (it == m_bindingIndex.end()) return false;
    ComputeBinding& b = m_bindings[it->second];
    if (b.type != ComputeBindingType::SRV) return false;
    b.boundResource = res;
    b.boundStride   = 0;
    return true;
}

bool ComputeShader::SetRWBuffer(const char* name, ID3D12Resource* res)
{
    if (!name) return false;
    auto it = m_bindingIndex.find(name);
    if (it == m_bindingIndex.end()) return false;
    ComputeBinding& b = m_bindings[it->second];
    if (b.type != ComputeBindingType::UAV) return false;
    b.boundResource = res;
    return true;
}

bool ComputeShader::SetRWTexture(const char* name, ID3D12Resource* res)
{
    if (!name) return false;
    auto it = m_bindingIndex.find(name);
    if (it == m_bindingIndex.end()) return false;
    ComputeBinding& b = m_bindings[it->second];
    if (b.type != ComputeBindingType::UAV) return false;
    b.boundResource = res;
    return true;
}

bool ComputeShader::SetConstantBuffer(const char* name, ID3D12Resource* res)
{
    if (!name) return false;
    auto it = m_bindingIndex.find(name);
    if (it == m_bindingIndex.end()) return false;
    ComputeBinding& b = m_bindings[it->second];
    if (b.type != ComputeBindingType::CBV) return false;
    b.boundResource = res;
    return true;
}

bool ComputeShader::SetBindlessTexture(const char* name, BindlessTexture* bt)
{
    if (!name) return false;
    auto it = m_bindingIndex.find(name);
    if (it == m_bindingIndex.end()) return false;
    ComputeBinding& b = m_bindings[it->second];
    if (b.type != ComputeBindingType::SRV_ARRAY) return false;
    b.boundBT = bt;
    return true;
}

bool ComputeShader::SetBindlessBuffer(const char* name, BindlessBuffer* bb)
{
    if (!name) return false;
    auto it = m_bindingIndex.find(name);
    if (it == m_bindingIndex.end()) return false;
    ComputeBinding& b = m_bindings[it->second];
    if (b.type != ComputeBindingType::SRV_ARRAY) return false;
    b.boundBB = bb;
    return true;
}

// ---------------------------------------------------------------------------
// FreeAllAllocations
// ---------------------------------------------------------------------------
void ComputeShader::FreeAllAllocations()
{
    if (!m_allocator) return;
    if (m_srvAllocBase != kInvalidAlloc && m_numSRV > 0) { m_allocator->Free(m_srvAllocBase, m_numSRV); m_srvAllocBase = kInvalidAlloc; }
    if (m_uavAllocBase != kInvalidAlloc && m_numUAV > 0) { m_allocator->Free(m_uavAllocBase, m_numUAV); m_uavAllocBase = kInvalidAlloc; }
}

// ---------------------------------------------------------------------------
// AllocateAndWriteDescriptors
// ---------------------------------------------------------------------------
bool ComputeShader::AllocateAndWriteDescriptors()
{
    if (!m_allocator) return false;

    if (m_srvAllocBase == kInvalidAlloc && m_numSRV > 0)
        m_srvAllocBase = m_allocator->Allocate(m_numSRV);
    if (m_uavAllocBase == kInvalidAlloc && m_numUAV > 0)
        m_uavAllocBase = m_allocator->Allocate(m_numUAV);

    UpdateDescriptors();
    return true;
}

// ---------------------------------------------------------------------------
// UpdateDescriptors
//   Writes all SRV/UAV descriptors every frame.
//   CBVs are bound as inline root descriptors in Dispatch.
//   SRV_ARRAY bindings own their heap slots in BindlessTexture/BindlessBuffer.
// ---------------------------------------------------------------------------
void ComputeShader::UpdateDescriptors()
{
    // --- SRV / TLAS ---
    if (m_srvAllocBase != kInvalidAlloc)
    {
        for (const auto& b : m_bindings)
        {
            if (b.type == ComputeBindingType::TLAS)
            {
                D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
                s.ViewDimension                            = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
                s.Shader4ComponentMapping                  = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                ID3D12Resource* tlas = b.boundAS ? b.boundAS->GetTLAS() : b.boundResource;
                s.RaytracingAccelerationStructure.Location = tlas
                    ? tlas->GetGPUVirtualAddress() : 0;
                m_device->CreateShaderResourceView(nullptr, &s,
                    m_allocator->GetCPUHandle(m_srvAllocBase + b.heapOffset));
            }
            else if (b.type == ComputeBindingType::SRV)
            {
                D3D12_CPU_DESCRIPTOR_HANDLE h = m_allocator->GetCPUHandle(m_srvAllocBase + b.heapOffset);
                D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
                s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                if (b.boundResource)
                {
                    auto rd = b.boundResource->GetDesc();
                    if (rd.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
                    {
                        if (b.boundStride > 0 && b.boundCount > 0)
                        {
                            s.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
                            s.Format                     = DXGI_FORMAT_UNKNOWN;
                            s.Buffer.NumElements         = b.boundCount;
                            s.Buffer.StructureByteStride = b.boundStride;
                        }
                        else
                        {
                            s.ViewDimension      = D3D12_SRV_DIMENSION_BUFFER;
                            s.Format             = DXGI_FORMAT_R32_TYPELESS;
                            s.Buffer.Flags       = D3D12_BUFFER_SRV_FLAG_RAW;
                            s.Buffer.NumElements = static_cast<UINT>(rd.Width / 4);
                        }
                    }
                    else
                    {
                        s.ViewDimension       = D3D12_SRV_DIMENSION_TEXTURE2D;
                        s.Format              = rd.Format;
                        s.Texture2D.MipLevels = rd.MipLevels;
                    }
                    m_device->CreateShaderResourceView(b.boundResource, &s, h);
                }
                else
                {
                    s.ViewDimension      = D3D12_SRV_DIMENSION_BUFFER;
                    s.Format             = DXGI_FORMAT_R32_TYPELESS;
                    s.Buffer.Flags       = D3D12_BUFFER_SRV_FLAG_RAW;
                    s.Buffer.NumElements = 1;
                    m_device->CreateShaderResourceView(nullptr, &s, h);
                }
            }
        }
    }

    // --- UAV ---
    if (m_uavAllocBase != kInvalidAlloc)
    {
        for (const auto& b : m_bindings)
        {
            if (b.type != ComputeBindingType::UAV) continue;
            D3D12_CPU_DESCRIPTOR_HANDLE h = m_allocator->GetCPUHandle(m_uavAllocBase + b.heapOffset);
            D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
            if (b.boundResource)
            {
                auto rd = b.boundResource->GetDesc();
                if (rd.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
                {
                    u.ViewDimension      = D3D12_UAV_DIMENSION_BUFFER;
                    u.Format             = DXGI_FORMAT_R32_TYPELESS;
                    u.Buffer.Flags       = D3D12_BUFFER_UAV_FLAG_RAW;
                    u.Buffer.NumElements = static_cast<UINT>(rd.Width / 4);
                }
                else
                {
                    u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                    u.Format        = rd.Format;
                }
                m_device->CreateUnorderedAccessView(b.boundResource, nullptr, &u, h);
            }
            else
            {
                u.ViewDimension      = D3D12_UAV_DIMENSION_BUFFER;
                u.Format             = DXGI_FORMAT_R32_TYPELESS;
                u.Buffer.Flags       = D3D12_BUFFER_UAV_FLAG_RAW;
                u.Buffer.NumElements = 1;
                m_device->CreateUnorderedAccessView(nullptr, nullptr, &u, h);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch
//   Caller is responsible for resource state transitions.
//   No internal intermediate texture – C# owns all input/output resources.
// ---------------------------------------------------------------------------
void ComputeShader::Dispatch(
    ID3D12GraphicsCommandList* cmdList,
    UINT threadGroupX, UINT threadGroupY, UINT threadGroupZ)
{
    if (!m_pso || !m_rootSig || !m_allocator) return;

    // --- Validate all bindings are set from C# ---
    {
        bool anyMissing = false;
        for (const auto& b : m_bindings)
        {
            bool ok = false;
            const char* kind = "?";
            switch (b.type)
            {
            case ComputeBindingType::TLAS:
                kind = "TLAS";
                ok = (b.boundAS != nullptr) || (b.boundResource != nullptr);
                break;
            case ComputeBindingType::SRV:
                kind = "SRV";
                ok = (b.boundResource != nullptr);
                break;
            case ComputeBindingType::UAV:
                kind = "UAV";
                ok = (b.boundResource != nullptr);
                break;
            case ComputeBindingType::CBV:
                kind = "CBV";
                ok = (b.boundResource != nullptr);
                break;
            case ComputeBindingType::SRV_ARRAY:
                kind = "SRV_ARRAY";
                ok = (b.boundBT != nullptr) || (b.boundBB != nullptr);
                break;
            }
            if (!ok)
            {
                Logf(kUnityLogTypeError,
                     "ComputeShader::Dispatch: binding '%s' (%s, space%u, reg%u) is not set - "
                     "did you forget a SetXxx call from C#?",
                     b.name.c_str(), kind, b.space, b.registerIndex);
                anyMissing = true;
            }
        }
        if (anyMissing) return;
    }

    // Allocate heap slots on first call, then write all descriptors every frame
    if ((m_numSRV > 0 && m_srvAllocBase == kInvalidAlloc) ||
        (m_numUAV > 0 && m_uavAllocBase == kInvalidAlloc))
    {
        if (!AllocateAndWriteDescriptors()) return;
    }
    else
    {
        UpdateDescriptors();
    }

    // Bind the global shared heap (D3D12 allows only one GPU-visible heap at a time)
    ID3D12DescriptorHeap* heapsToBind[1] = { m_allocator->GetHeap() };
    cmdList->SetDescriptorHeaps(1, heapsToBind);

    cmdList->SetPipelineState(m_pso.Get());
    cmdList->SetComputeRootSignature(m_rootSig.Get());

    // SRV table
    if (m_rootParamSRV != kInvalidAlloc && m_srvAllocBase != kInvalidAlloc)
        cmdList->SetComputeRootDescriptorTable(m_rootParamSRV,
            m_allocator->GetGPUHandle(m_srvAllocBase));

    // UAV table
    if (m_rootParamUAV != kInvalidAlloc && m_uavAllocBase != kInvalidAlloc)
        cmdList->SetComputeRootDescriptorTable(m_rootParamUAV,
            m_allocator->GetGPUHandle(m_uavAllocBase));

    // SRV_ARRAY bindings – each has its own root parameter
    for (const auto& b : m_bindings)
    {
        if (b.type != ComputeBindingType::SRV_ARRAY) continue;
        if (b.rootParam == kInvalidAlloc) continue;
        if (b.boundBT)
            cmdList->SetComputeRootDescriptorTable(b.rootParam, b.boundBT->GetGPUHandle());
        else if (b.boundBB)
            cmdList->SetComputeRootDescriptorTable(b.rootParam, b.boundBB->GetGPUHandle());
    }

    // Root CBV per CBV binding
    if (m_rootParamCBVBase != kInvalidAlloc)
    {
        for (const auto& b : m_bindings)
        {
            if (b.type != ComputeBindingType::CBV) continue;
            D3D12_GPU_VIRTUAL_ADDRESS addr = b.boundResource
                ? b.boundResource->GetGPUVirtualAddress() : 0;
            cmdList->SetComputeRootConstantBufferView(m_rootParamCBVBase + b.heapOffset, addr);
        }
    }

    // --- Request resource states ---
    // SRV_ARRAY is managed by BindlessTexture/BindlessBuffer; CBV stays in GENERIC_READ.
    if (m_d3d12v8)
    {
        for (const auto& b : m_bindings)
        {
            if (!b.boundResource) continue;
            if (b.type == ComputeBindingType::SRV)
            {
                m_d3d12v8->RequestResourceState(b.boundResource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
            else if (b.type == ComputeBindingType::UAV)
            {
                m_d3d12v8->RequestResourceState(b.boundResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }
            else if (b.type == ComputeBindingType::CBV)
            {
                m_d3d12v8->RequestResourceState(b.boundResource,
                    D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
            }
        }
    }

    cmdList->Dispatch(threadGroupX, threadGroupY, threadGroupZ);
}
