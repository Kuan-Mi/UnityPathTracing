#include "ShaderBase.h"
#include <d3d12shader.h>
#include <cstdio>
#include <cstdarg>
#include <algorithm>

// ===========================================================================
// Logging
// ===========================================================================

void ShaderBase::Log(UnityLogType type, const char* msg) const
{
    if (m_log) m_log->Log(type, msg, __FILE__, __LINE__);
    else        printf("[ShaderBase] %s\n", msg);
}

void ShaderBase::Logf(UnityLogType type, const char* fmt, ...) const
{
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    Log(type, buf);
}

// ===========================================================================
// Pre-load hints
// ===========================================================================

void ShaderBase::SetRootConstantsHint(const char* name, uint32_t num32BitValues)
{
    if (name) m_rootConstantsHints[name] = num32BitValues;
}

void ShaderBase::SetRootSRVHint(const char* name)
{
    if (name) m_rootSRVHints.insert(name);
}

// ===========================================================================
// Binding metadata queries
// ===========================================================================

uint32_t ShaderBase::GetBindingCount() const
{
    return static_cast<uint32_t>(m_bindings.size());
}

uint32_t ShaderBase::GetSlotIndex(const char* name) const
{
    if (!name) return kInvalidAlloc;
    auto it = m_bindingIndex.find(name);
    return it != m_bindingIndex.end()
        ? static_cast<uint32_t>(it->second)
        : kInvalidAlloc;
}

const char* ShaderBase::GetBindingName(uint32_t index) const
{
    if (index >= m_bindings.size()) return nullptr;
    return m_bindings[index].name.c_str();
}

// ===========================================================================
// ClassifyBinding
//   Fills type / num32BitValues / m_num* on a freshly zeroed ComputeBinding
//   using the D3D12 reflection descriptor and pre-load hints.
//   Returns false for samplers (caller should handle separately).
// ===========================================================================
bool ShaderBase::ClassifyBinding(const D3D12_SHADER_INPUT_BIND_DESC& bind,
                                 const std::string& name,
                                 Binding& b)
{
    b.name           = name;
    b.space          = bind.Space;
    b.registerIndex  = bind.BindPoint;
    b.heapOffset     = 0;
    b.rootParam      = kInvalidAlloc;
    b.num32BitValues = 0;

    switch (bind.Type)
    {
    case D3D_SIT_SAMPLER:
        return false; // caller handles samplers

    case D3D_SIT_RTACCELERATIONSTRUCTURE:
        if (m_rootSRVHints.count(name))
        {
            b.type = BindingType::ROOT_SRV;
            ++m_numRootSRV;
        }
        else
        {
            b.type = BindingType::TLAS;
            ++m_numSRV;
        }
        break;

    case D3D_SIT_CBUFFER:
    {
        auto hint = m_rootConstantsHints.find(name);
        if (hint != m_rootConstantsHints.end())
        {
            b.type           = BindingType::ROOT_CONSTANTS;
            b.num32BitValues = hint->second;
            ++m_numRootConstants;
        }
        else
        {
            b.type = BindingType::CBV;
            ++m_numCBV;
        }
        break;
    }

    case D3D_SIT_TBUFFER:
    case D3D_SIT_TEXTURE:
    case D3D_SIT_STRUCTURED:
    case D3D_SIT_BYTEADDRESS:
        if (bind.BindCount == 0)
        {
            b.type = BindingType::SRV_ARRAY;
            ++m_numSRVArray;
        }
        else if (m_rootSRVHints.count(name))
        {
            b.type = BindingType::ROOT_SRV;
            ++m_numRootSRV;
        }
        else
        {
            b.type = BindingType::SRV;
            ++m_numSRV;
        }
        break;

    default: // UAV variants (D3D_SIT_UAV_RWTYPED / RWSTRUCTURED / etc.)
        if (bind.BindCount == 0)
        {
            b.type = BindingType::UAV_ARRAY;
            ++m_numUAVArray;
        }
        else
        {
            b.type = BindingType::UAV;
            ++m_numUAV;
        }
        break;
    }
    return true;
}

// ===========================================================================
// AssignHeapOffsets
//   Assigns consecutive heapOffset values within each type group.
//   SRV/TLAS → SRV descriptor table slot
//   UAV       → UAV descriptor table slot
//   CBV       → inline root CBV index
//   Others (ROOT_SRV, ROOT_CONSTANTS, SRV_ARRAY, UAV_ARRAY) keep heapOffset=0
// ===========================================================================
void ShaderBase::AssignHeapOffsets()
{
    uint32_t srvOff = 0, uavOff = 0, cbvOff = 0;
    for (auto& b : m_bindings)
    {
        if      (b.type == BindingType::SRV || b.type == BindingType::TLAS)
            b.heapOffset = srvOff++;
        else if (b.type == BindingType::UAV)
            b.heapOffset = uavOff++;
        else if (b.type == BindingType::CBV)
            b.heapOffset = cbvOff++;
    }
}

// ===========================================================================
// BuildRootSignature
//   Fully dynamic — no hardcoded registers or spaces.
//   Uses D3D_ROOT_SIGNATURE_VERSION_1_2 (requires Device5/D3D12Agility SDK).
//   Called by both ComputeShader and RayTraceShader after ReflectBindings().
// ===========================================================================
bool ShaderBase::BuildRootSignature()
{
    std::vector<D3D12_DESCRIPTOR_RANGE1> allRanges;
    allRanges.reserve(m_numSRV + m_numUAV + m_numSRVArray + m_numUAVArray);

    Logf(kUnityLogTypeLog,
         "ShaderBase::BuildRootSignature [%s]: %u SRV, %u UAV, %u CBV, "
         "%u SRV_ARRAY, %u UAV_ARRAY, %u ROOT_SRV, %u ROOT_CONSTANTS",
         m_name.c_str(),
         m_numSRV, m_numUAV, m_numCBV,
         m_numSRVArray, m_numUAVArray, m_numRootSRV, m_numRootConstants);

    // --- SRV descriptor ranges (SRV + TLAS) ---
    const size_t srvRangesOffset = allRanges.size();
    for (const auto& b : m_bindings)
    {
        if (b.type != BindingType::SRV && b.type != BindingType::TLAS) continue;
        D3D12_DESCRIPTOR_RANGE1 r = {};
        r.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        r.NumDescriptors                    = 1;
        r.BaseShaderRegister                = b.registerIndex;
        r.RegisterSpace                     = b.space;
        r.Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE |
                                              D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
        r.OffsetInDescriptorsFromTableStart = b.heapOffset;
        allRanges.push_back(r);
        Logf(kUnityLogTypeLog, "  SRV: '%s' t%u space%u heapOffset=%u",
             b.name.c_str(), b.registerIndex, b.space, b.heapOffset);
    }

    // --- UAV descriptor ranges ---
    const size_t uavRangesOffset = allRanges.size();
    for (const auto& b : m_bindings)
    {
        if (b.type != BindingType::UAV) continue;
        D3D12_DESCRIPTOR_RANGE1 r = {};
        r.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        r.NumDescriptors                    = 1;
        r.BaseShaderRegister                = b.registerIndex;
        r.RegisterSpace                     = b.space;
        r.Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE |
                                              D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
        r.OffsetInDescriptorsFromTableStart = b.heapOffset;
        allRanges.push_back(r);
        Logf(kUnityLogTypeLog, "  UAV: '%s' u%u space%u heapOffset=%u",
             b.name.c_str(), b.registerIndex, b.space, b.heapOffset);
    }

    // --- SRV_ARRAY descriptor ranges (unbounded) ---
    const size_t srvArrayRangesOffset = allRanges.size();
    for (const auto& b : m_bindings)
    {
        if (b.type != BindingType::SRV_ARRAY) continue;
        D3D12_DESCRIPTOR_RANGE1 r = {};
        r.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        r.NumDescriptors                    = UINT_MAX;
        r.BaseShaderRegister                = b.registerIndex;
        r.RegisterSpace                     = b.space;
        r.Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE |
                                              D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
        r.OffsetInDescriptorsFromTableStart = 0;
        allRanges.push_back(r);
        Logf(kUnityLogTypeLog, "  SRV_ARRAY: '%s' t%u space%u", b.name.c_str(), b.registerIndex, b.space);
    }

    // --- UAV_ARRAY descriptor ranges (unbounded) ---
    const size_t uavArrayRangesOffset = allRanges.size();
    for (const auto& b : m_bindings)
    {
        if (b.type != BindingType::UAV_ARRAY) continue;
        D3D12_DESCRIPTOR_RANGE1 r = {};
        r.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        r.NumDescriptors                    = UINT_MAX;
        r.BaseShaderRegister                = b.registerIndex;
        r.RegisterSpace                     = b.space;
        r.Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE |
                                              D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
        r.OffsetInDescriptorsFromTableStart = 0;
        allRanges.push_back(r);
        Logf(kUnityLogTypeLog, "  UAV_ARRAY: '%s' u%u space%u", b.name.c_str(), b.registerIndex, b.space);
    }

    std::vector<D3D12_ROOT_PARAMETER1> params;
    params.reserve(
        (m_numSRV ? 1 : 0) + (m_numUAV ? 1 : 0) +
        m_numSRVArray + m_numUAVArray + m_numCBV + m_numRootSRV + m_numRootConstants);

    // Optional — SRV descriptor table
    if (m_numSRV > 0)
    {
        m_rootParamSRV = static_cast<uint32_t>(params.size());
        D3D12_ROOT_PARAMETER1 p = {};
        p.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p.DescriptorTable.NumDescriptorRanges = m_numSRV;
        p.DescriptorTable.pDescriptorRanges   = &allRanges[srvRangesOffset];
        p.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
        params.push_back(p);
        Logf(kUnityLogTypeLog, "  Root param %u: SRV table (%u)", m_rootParamSRV, m_numSRV);
    }

    // Optional — UAV descriptor table
    if (m_numUAV > 0)
    {
        m_rootParamUAV = static_cast<uint32_t>(params.size());
        D3D12_ROOT_PARAMETER1 p = {};
        p.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p.DescriptorTable.NumDescriptorRanges = m_numUAV;
        p.DescriptorTable.pDescriptorRanges   = &allRanges[uavRangesOffset];
        p.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
        params.push_back(p);
        Logf(kUnityLogTypeLog, "  Root param %u: UAV table (%u)", m_rootParamUAV, m_numUAV);
    }

    // One table per SRV_ARRAY
    {
        uint32_t arrayIdx = 0;
        for (auto& b : m_bindings)
        {
            if (b.type != BindingType::SRV_ARRAY) continue;
            b.rootParam = static_cast<uint32_t>(params.size());
            D3D12_ROOT_PARAMETER1 p = {};
            p.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            p.DescriptorTable.NumDescriptorRanges = 1;
            p.DescriptorTable.pDescriptorRanges   = &allRanges[srvArrayRangesOffset + arrayIdx++];
            p.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
            params.push_back(p);
            Logf(kUnityLogTypeLog, "  Root param %u: SRV_ARRAY '%s'", b.rootParam, b.name.c_str());
        }
    }

    // One table per UAV_ARRAY
    {
        uint32_t arrayIdx = 0;
        for (auto& b : m_bindings)
        {
            if (b.type != BindingType::UAV_ARRAY) continue;
            b.rootParam = static_cast<uint32_t>(params.size());
            D3D12_ROOT_PARAMETER1 p = {};
            p.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            p.DescriptorTable.NumDescriptorRanges = 1;
            p.DescriptorTable.pDescriptorRanges   = &allRanges[uavArrayRangesOffset + arrayIdx++];
            p.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
            params.push_back(p);
            Logf(kUnityLogTypeLog, "  Root param %u: UAV_ARRAY '%s'", b.rootParam, b.name.c_str());
        }
    }

    // One root CBV per CBV binding
    if (m_numCBV > 0)
    {
        m_rootParamCBVBase = static_cast<uint32_t>(params.size());
        for (auto& b : m_bindings)
        {
            if (b.type != BindingType::CBV) continue;
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

    // One inline root SRV per ROOT_SRV binding
    if (m_numRootSRV > 0)
    {
        m_rootParamRootSRVBase = static_cast<uint32_t>(params.size());
        for (auto& b : m_bindings)
        {
            if (b.type != BindingType::ROOT_SRV) continue;
            b.rootParam = static_cast<uint32_t>(params.size());
            D3D12_ROOT_PARAMETER1 p = {};
            p.ParameterType             = D3D12_ROOT_PARAMETER_TYPE_SRV;
            p.Descriptor.ShaderRegister = b.registerIndex;
            p.Descriptor.RegisterSpace  = b.space;
            p.Descriptor.Flags          = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE;
            p.ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;
            params.push_back(p);
            Logf(kUnityLogTypeLog, "  Root param %u: ROOT_SRV '%s' t%u space%u",
                 b.rootParam, b.name.c_str(), b.registerIndex, b.space);
        }
    }

    // One root 32-bit constants slot per ROOT_CONSTANTS binding
    for (auto& b : m_bindings)
    {
        if (b.type != BindingType::ROOT_CONSTANTS) continue;
        b.rootParam = static_cast<uint32_t>(params.size());
        D3D12_ROOT_PARAMETER1 p = {};
        p.ParameterType            = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        p.Constants.ShaderRegister = b.registerIndex;
        p.Constants.RegisterSpace  = b.space;
        p.Constants.Num32BitValues = b.num32BitValues;
        p.ShaderVisibility         = D3D12_SHADER_VISIBILITY_ALL;
        params.push_back(p);
        Logf(kUnityLogTypeLog, "  Root param %u: ROOT_CONSTANTS '%s' b%u space%u num32=%u",
             b.rootParam, b.name.c_str(), b.registerIndex, b.space, b.num32BitValues);
    }

    // --- Static samplers (Unity inline sampler naming convention) ---
    auto ToLower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return (char)tolower(c); });
        return s;
    };
    auto Contains = [](const std::string& haystack, const char* needle) {
        return haystack.find(needle) != std::string::npos;
    };

    std::vector<D3D12_STATIC_SAMPLER_DESC1> samplers;
    samplers.reserve(m_samplerBindings.size());
    for (const auto& sr : m_samplerBindings)
    {
        const std::string lower = ToLower(sr.name);

        D3D12_FILTER filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        if      (Contains(lower, "point"))   filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        else if (Contains(lower, "nearest")) filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        else if (Contains(lower, "aniso"))   filter = D3D12_FILTER_ANISOTROPIC;
        else if (Contains(lower, "linear"))  filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;

        D3D12_TEXTURE_ADDRESS_MODE addr = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        if      (Contains(lower, "mirroronce")) addr = D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
        else if (Contains(lower, "mirror"))     addr = D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
        else if (Contains(lower, "clamp"))      addr = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        else if (Contains(lower, "repeat"))     addr = D3D12_TEXTURE_ADDRESS_MODE_WRAP;

        FLOAT maxLod = Contains(lower, "mipmap") ? 16.0f : 0.0f;

        D3D12_STATIC_SAMPLER_DESC1 sd = {};
        sd.Filter           = filter;
        sd.AddressU = sd.AddressV = sd.AddressW = addr;
        sd.MaxAnisotropy    = (filter == D3D12_FILTER_ANISOTROPIC) ? 16 : 0;
        sd.ComparisonFunc   = D3D12_COMPARISON_FUNC_NONE;
        sd.BorderColor      = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
        sd.MaxLOD           = maxLod;
        sd.ShaderRegister   = sr.reg;
        sd.RegisterSpace    = sr.space;
        sd.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        samplers.push_back(sd);
        Logf(kUnityLogTypeLog, "  Static sampler: '%s' filter=%u addr=%u s%u space%u maxLod=%g",
             sr.name.c_str(), filter, addr, sr.reg, sr.space, maxLod);
    }

    // --- Validate register spaces ---
    bool spaceValid = true;
    for (const auto& range : allRanges)
    {
        if (range.RegisterSpace >= 0xfffffff0)
        {
            const char* bindName = "?";
            for (const auto& b : m_bindings)
                if (b.registerIndex == range.BaseShaderRegister && b.space == range.RegisterSpace)
                    { bindName = b.name.c_str(); break; }
            Logf(kUnityLogTypeError,
                 "ShaderBase [%s]: binding '%s' has invalid RegisterSpace=0x%08X"
                 " — add explicit register(xN, spaceM) in HLSL",
                 m_name.c_str(), bindName, range.RegisterSpace);
            spaceValid = false;
        }
    }
    for (const auto& p : params)
    {
        if ((p.ParameterType == D3D12_ROOT_PARAMETER_TYPE_CBV ||
             p.ParameterType == D3D12_ROOT_PARAMETER_TYPE_SRV ||
             p.ParameterType == D3D12_ROOT_PARAMETER_TYPE_UAV) &&
             p.Descriptor.RegisterSpace >= 0xfffffff0)
        {
            Logf(kUnityLogTypeError,
                 "ShaderBase [%s]: root descriptor has invalid RegisterSpace=0x%08X"
                 " — add explicit register() in HLSL",
                 m_name.c_str(), p.Descriptor.RegisterSpace);
            spaceValid = false;
        }
    }
    if (!spaceValid) return false;

    // --- Serialize & create ---
    D3D12_ROOT_SIGNATURE_DESC2 rsDesc2 = {};
    rsDesc2.NumParameters     = static_cast<UINT>(params.size());
    rsDesc2.pParameters       = params.empty()   ? nullptr : params.data();
    rsDesc2.NumStaticSamplers = static_cast<UINT>(samplers.size());
    rsDesc2.pStaticSamplers   = samplers.empty() ? nullptr : samplers.data();
    rsDesc2.Flags             = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC vrsDesc = {};
    vrsDesc.Version  = D3D_ROOT_SIGNATURE_VERSION_1_2;
    vrsDesc.Desc_1_2 = rsDesc2;

    Logf(kUnityLogTypeLog, "  %u root param(s), %u static sampler(s)",
         rsDesc2.NumParameters, rsDesc2.NumStaticSamplers);

    ComPtr<ID3DBlob> sigBlob, errBlob;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&vrsDesc, &sigBlob, &errBlob);
    if (FAILED(hr))
    {
        Logf(kUnityLogTypeError,
             "ShaderBase [%s]: D3D12SerializeVersionedRootSignature failed (hr=0x%08X): %s",
             m_name.c_str(), hr, errBlob ? (char*)errBlob->GetBufferPointer() : "");
        return false;
    }

    hr = m_device->CreateRootSignature(0, sigBlob->GetBufferPointer(),
                                        sigBlob->GetBufferSize(), IID_PPV_ARGS(&m_rootSig));
    if (FAILED(hr))
    {
        Logf(kUnityLogTypeError, "ShaderBase [%s]: CreateRootSignature failed (hr=0x%08X)",
             m_name.c_str(), hr);
        return false;
    }
    {
        std::wstring wname(m_name.begin(), m_name.end());
        wname += L"_RootSig";
        m_rootSig->SetName(wname.c_str());
    }
    return true;
}
