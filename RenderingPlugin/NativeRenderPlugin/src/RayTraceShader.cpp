#include "RayTraceShader.h"
#include "BindlessTexture.h"
#include "BindlessBuffer.h"
#include "BindlessUAVTexture.h"
#include "AccelerationStructure.h"
#include "ComputeShader.h"   // CS_BindingSlot, CS_BindingObjectKind
#include <d3d12shader.h>
#include <cstdio>
#include <cstdarg>
#include <algorithm>
#include <windows.h>

// ---------------------------------------------------------------------------
// Heap / buffer helpers
// ---------------------------------------------------------------------------
namespace
{
    static D3D12_HEAP_PROPERTIES UploadHeapProps()
    {
        D3D12_HEAP_PROPERTIES p = {};
        p.Type = D3D12_HEAP_TYPE_UPLOAD;
        return p;
    }

    static ComPtr<ID3D12Resource> CreateUploadBuffer(ID3D12Device* device, UINT64 size)
    {
        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width            = size ? size : 1;
        desc.Height           = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels        = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        auto hp = UploadHeapProps();
        ComPtr<ID3D12Resource> buf;
        device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&buf));
        return buf;
    }
} // anonymous namespace

// ---------------------------------------------------------------------------

RayTraceShader::RayTraceShader() = default;

RayTraceShader::~RayTraceShader() = default;

void RayTraceShader::Log(UnityLogType type, const char* msg) const
{
    if (m_log) m_log->Log(type, msg, __FILE__, __LINE__);
    else        printf("[RayTraceShader] %s\n", msg);
}

void RayTraceShader::Logf(UnityLogType type, const char* fmt, ...) const
{
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    Log(type, buf);
}

bool RayTraceShader::Initialize(ID3D12Device5* device, IUnityLog* log,
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
// Pre-load hints
// ---------------------------------------------------------------------------
void RayTraceShader::SetRootConstantsHint(const char* name, uint32_t num32BitValues)
{
    if (name) m_rootConstantsHints[name] = num32BitValues;
}

void RayTraceShader::SetRootSRVHint(const char* name)
{
    if (name) m_rootSRVHints.insert(name);
}

// ---------------------------------------------------------------------------
// Slot-layout queries
// ---------------------------------------------------------------------------
uint32_t RayTraceShader::GetBindingCount() const
{
    return static_cast<uint32_t>(m_bindings.size());
}

uint32_t RayTraceShader::GetSlotIndex(const char* name) const
{
    if (!name) return kInvalidAlloc;
    auto it = m_bindingIndex.find(name);
    return it != m_bindingIndex.end()
        ? static_cast<uint32_t>(it->second)
        : kInvalidAlloc;
}

const char* RayTraceShader::GetBindingName(uint32_t index) const
{
    if (index >= m_bindings.size()) return nullptr;
    return m_bindings[index].name.c_str();
}

// ---------------------------------------------------------------------------
// LoadShaderFromBytes
// ---------------------------------------------------------------------------
bool RayTraceShader::LoadShaderFromBytes(const uint8_t* dxilBytes, uint32_t size, const char* name)
{
    m_name = (name && name[0]) ? name : "RayTraceShader";
    if (!dxilBytes || size == 0)
    {
        Log(kUnityLogTypeError, "RayTraceShader::LoadShaderFromBytes: empty input");
        return false;
    }

    ComPtr<IDxcUtils> utils;
    if (FAILED(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils))))
    {
        Log(kUnityLogTypeError, "RayTraceShader::LoadShaderFromBytes: failed to create IDxcUtils");
        return false;
    }

    ComPtr<IDxcBlobEncoding> blobEnc;
    if (FAILED(utils->CreateBlob(dxilBytes, size, DXC_CP_ACP, &blobEnc)))
    {
        Log(kUnityLogTypeError, "RayTraceShader::LoadShaderFromBytes: failed to create blob");
        return false;
    }
    ComPtr<IDxcBlob> shaderLib = blobEnc;

    // Reset old pipeline state
    m_pso.Reset();
    m_rootSig.Reset();
    m_rayGenTable.Reset();
    m_missTable.Reset();
    m_hitGroupTable.Reset();
    m_bindings.clear();
    m_bindingIndex.clear();
    m_samplerBindings.clear();
    m_rayGenShaders.clear();
    m_missShaders.clear();
    m_hitGroups.clear();
    m_hitGroupIndex.clear();
    m_numSRV = m_numUAV = m_numCBV = m_numSRVArray = m_numUAVArray = m_numRootConstants = m_numRootSRV = 0;
    m_rootParamSRV = m_rootParamUAV = m_rootParamCBVBase = m_rootParamRootSRVBase = kInvalidAlloc;

    if (!ReflectBindings(shaderLib.Get()))   return false;
    if (!BuildRootSignature())               return false;
    if (!BuildPipeline(shaderLib.Get()))     return false;
    if (!BuildShaderTable())                 return false;

    Logf(kUnityLogTypeLog,
         "RayTraceShader '%s': pipeline ready (%u SRV, %u UAV, %u CBV, %u SRV_ARRAY, %u UAV_ARRAY, %u ROOT_SRV, %u ROOT_CONSTANTS)",
         m_name.c_str(),
         m_numSRV, m_numUAV, m_numCBV, m_numSRVArray, m_numUAVArray, m_numRootSRV, m_numRootConstants);
    return true;
}

// ---------------------------------------------------------------------------
// ReflectBindings
//   Collects resource bindings from all functions in the library.
//   Classifies shader entry points (raygen, miss, closesthit, anyhit).
//   Deduplicates bindings across functions.
//   Supports ROOT_CONSTANTS and ROOT_SRV via pre-load hints.
// ---------------------------------------------------------------------------
bool RayTraceShader::ReflectBindings(IDxcBlob* shaderLib)
{
    ComPtr<IDxcUtils> utils;
    if (FAILED(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils))))
    {
        Log(kUnityLogTypeError, "RayTraceShader: failed to create IDxcUtils for reflection");
        return false;
    }

    DxcBuffer buf;
    buf.Ptr      = shaderLib->GetBufferPointer();
    buf.Size     = shaderLib->GetBufferSize();
    buf.Encoding = 0;

    ComPtr<ID3D12LibraryReflection> libRefl;
    HRESULT hr = utils->CreateReflection(&buf, IID_PPV_ARGS(&libRefl));
    if (FAILED(hr))
    {
        Logf(kUnityLogTypeWarning,
             "RayTraceShader: CreateReflection failed (hr=0x%08X) — no bindings", hr);
        return true; // not fatal
    }

    D3D12_LIBRARY_DESC libDesc = {};
    libRefl->GetDesc(&libDesc);
    Logf(kUnityLogTypeLog, "RayTraceShader::ReflectBindings: library has %u function(s)",
         libDesc.FunctionCount);

    for (UINT fi = 0; fi < libDesc.FunctionCount; ++fi)
    {
        ID3D12FunctionReflection* func = libRefl->GetFunctionByIndex(static_cast<INT>(fi));
        if (!func) continue;

        D3D12_FUNCTION_DESC funcDesc = {};
        if (FAILED(func->GetDesc(&funcDesc))) continue;
        if (!funcDesc.Name) continue;

        // --- Demangle DXC name ---
        // DXC may emit ?FunctionName@@<sig>, optionally prefixed with \x01.
        std::string nameA(funcDesc.Name);
        std::string realName = nameA;
        {
            const char* p = nameA.c_str();
            while (*p && (unsigned char)*p < 0x20) ++p;
            if (*p == '?')
            {
                ++p;
                const char* atAt = strstr(p, "@@");
                if (atAt) realName = std::string(p, atAt);
            }
            else if (p != nameA.c_str())
            {
                realName = std::string(p);
            }
        }
        std::wstring nameW(realName.begin(), realName.end());
        const UINT shaderType = (funcDesc.Version >> 16) & 0xFFFF;

        Logf(kUnityLogTypeLog,
             "RayTraceShader: function[%u] realName='%s' version=0x%08X shaderType=%u",
             fi, realName.c_str(), funcDesc.Version, shaderType);

        // --- Strip type prefix + optional '_' from name and return the key ---
        auto TryStrip = [&](const wchar_t* prefix, std::wstring& outKey) -> bool
        {
            const size_t plen = wcslen(prefix);
            if (nameW.size() >= plen && nameW.compare(0, plen, prefix) == 0)
            {
                outKey = nameW.substr(plen);
                if (!outKey.empty() && outKey[0] == L'_') outKey = outKey.substr(1);
                return true;
            }
            return false;
        };

        // --- Classify entry point ---
        if (shaderType == D3D12_SHVER_RAY_GENERATION_SHADER)
        {
            m_rayGenShaders.push_back(nameW);
        }
        else if (shaderType == D3D12_SHVER_MISS_SHADER)
        {
            m_missShaders.push_back(nameW);
        }
        else if (shaderType == D3D12_SHVER_CLOSEST_HIT_SHADER)
        {
            std::wstring groupKey;
            if (!TryStrip(L"ClosestHit", groupKey)) groupKey = nameW;
            std::wstring groupExport = groupKey.empty() ? L"HitGroup" : L"HitGroup_" + groupKey;
            auto it = m_hitGroupIndex.find(groupKey);
            if (it == m_hitGroupIndex.end())
            {
                m_hitGroupIndex[groupKey] = m_hitGroups.size();
                m_hitGroups.push_back({ groupExport, nameW, L"" });
            }
            else
            {
                m_hitGroups[it->second].closestHitExport = nameW;
            }
        }
        else if (shaderType == D3D12_SHVER_ANY_HIT_SHADER)
        {
            std::wstring groupKey;
            if (!TryStrip(L"AnyHit", groupKey)) groupKey = nameW;
            std::wstring groupExport = groupKey.empty() ? L"HitGroup" : L"HitGroup_" + groupKey;
            auto it = m_hitGroupIndex.find(groupKey);
            if (it == m_hitGroupIndex.end())
            {
                m_hitGroupIndex[groupKey] = m_hitGroups.size();
                m_hitGroups.push_back({ groupExport, L"", nameW });
            }
            else
            {
                m_hitGroups[it->second].anyHitExport = nameW;
            }
        }

        // --- Collect resource bindings ---
        for (UINT ri = 0; ri < funcDesc.BoundResources; ++ri)
        {
            D3D12_SHADER_INPUT_BIND_DESC bind = {};
            if (FAILED(func->GetResourceBindingDesc(ri, &bind))) continue;

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
            if (m_bindingIndex.count(bname)) continue; // de-duplicate across functions

            RayTraceBinding rb;
            rb.name           = bname;
            rb.space          = bind.Space;
            rb.registerIndex  = bind.BindPoint;
            rb.heapOffset     = 0;
            rb.rootParam      = kInvalidAlloc;
            rb.num32BitValues = 0;

            switch (bind.Type)
            {
            case D3D_SIT_RTACCELERATIONSTRUCTURE:
                if (m_rootSRVHints.count(bname))
                {
                    rb.type = RayTraceBindingType::ROOT_SRV;
                    ++m_numRootSRV;
                }
                else
                {
                    rb.type = RayTraceBindingType::TLAS;
                    ++m_numSRV;
                }
                break;
            case D3D_SIT_CBUFFER:
            {
                auto hint = m_rootConstantsHints.find(bname);
                if (hint != m_rootConstantsHints.end())
                {
                    rb.type           = RayTraceBindingType::ROOT_CONSTANTS;
                    rb.num32BitValues = hint->second;
                    ++m_numRootConstants;
                }
                else
                {
                    rb.type = RayTraceBindingType::CBV;
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
                    rb.type = RayTraceBindingType::SRV_ARRAY;
                    ++m_numSRVArray;
                }
                else if (m_rootSRVHints.count(bname))
                {
                    rb.type = RayTraceBindingType::ROOT_SRV;
                    ++m_numRootSRV;
                }
                else
                {
                    rb.type = RayTraceBindingType::SRV;
                    ++m_numSRV;
                }
                break;
            default: // UAV variants (D3D_SIT_UAV_RWTYPED / D3D_SIT_UAV_RWSTRUCTURED / etc.)
                if (bind.BindCount == 0)
                {
                    rb.type = RayTraceBindingType::UAV_ARRAY;
                    ++m_numUAVArray;
                }
                else
                {
                    rb.type = RayTraceBindingType::UAV;
                    ++m_numUAV;
                }
                break;
            }

            m_bindingIndex[bname] = m_bindings.size();
            m_bindings.push_back(std::move(rb));
        }
    }

    // Assign consecutive heap offsets per type group
    // SRV table: SRV + TLAS; UAV table: UAV; ROOT_SRV / ROOT_CONSTANTS / SRV_ARRAY / UAV_ARRAY: no slot here.
    {
        uint32_t srvOff = 0, uavOff = 0, cbvOff = 0;
        for (auto& b : m_bindings)
        {
            if      (b.type == RayTraceBindingType::SRV  || b.type == RayTraceBindingType::TLAS)
                b.heapOffset = srvOff++;
            else if (b.type == RayTraceBindingType::UAV)
                b.heapOffset = uavOff++;
            else if (b.type == RayTraceBindingType::CBV)
                b.heapOffset = cbvOff++;
        }
    }

    // Sort miss shaders and hit groups for deterministic shader-table ordering.
    std::sort(m_missShaders.begin(), m_missShaders.end());
    std::sort(m_hitGroups.begin(), m_hitGroups.end(),
        [](const HitGroupInfo& a, const HitGroupInfo& b) { return a.groupExport < b.groupExport; });
    // Rebuild hit-group index after sort.
    m_hitGroupIndex.clear();
    for (size_t i = 0; i < m_hitGroups.size(); ++i)
    {
        const std::wstring& exp = m_hitGroups[i].groupExport;
        static const std::wstring kPrefix = L"HitGroup_";
        std::wstring key = (exp.size() > kPrefix.size() && exp.compare(0, kPrefix.size(), kPrefix) == 0)
            ? exp.substr(kPrefix.size())
            : (exp == L"HitGroup" ? L"" : exp);
        m_hitGroupIndex[key] = i;
    }

    return true;
}

// ---------------------------------------------------------------------------
// BuildRootSignature
//   Fully dynamic — no hardcoded registers or spaces.
//   Mirrors ComputeShader::BuildRootSignature but uses D3D12_SHADER_VISIBILITY_ALL
//   (required for global root signatures in DXR).
// ---------------------------------------------------------------------------
bool RayTraceShader::BuildRootSignature()
{
    std::vector<D3D12_DESCRIPTOR_RANGE1> allRanges;
    allRanges.reserve(m_numSRV + m_numUAV + m_numSRVArray + m_numUAVArray);

    Logf(kUnityLogTypeLog,
         "RayTraceShader::BuildRootSignature: %u SRV, %u UAV, %u CBV, %u SRV_ARRAY, %u UAV_ARRAY, %u ROOT_SRV, %u ROOT_CONSTANTS",
         m_numSRV, m_numUAV, m_numCBV, m_numSRVArray, m_numUAVArray, m_numRootSRV, m_numRootConstants);

    // --- SRV descriptor ranges (SRV + TLAS) ---
    const size_t srvRangesOffset = allRanges.size();
    for (const auto& b : m_bindings)
    {
        if (b.type != RayTraceBindingType::SRV && b.type != RayTraceBindingType::TLAS) continue;
        D3D12_DESCRIPTOR_RANGE1 r = {};
        r.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        r.NumDescriptors                    = 1;
        r.BaseShaderRegister                = b.registerIndex;
        r.RegisterSpace                     = b.space;
        r.Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
        r.OffsetInDescriptorsFromTableStart = b.heapOffset;
        allRanges.push_back(r);
    }

    // --- UAV descriptor ranges ---
    const size_t uavRangesOffset = allRanges.size();
    for (const auto& b : m_bindings)
    {
        if (b.type != RayTraceBindingType::UAV) continue;
        D3D12_DESCRIPTOR_RANGE1 r = {};
        r.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        r.NumDescriptors                    = 1;
        r.BaseShaderRegister                = b.registerIndex;
        r.RegisterSpace                     = b.space;
        r.Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
        r.OffsetInDescriptorsFromTableStart = b.heapOffset;
        allRanges.push_back(r);
    }

    // --- SRV_ARRAY descriptor ranges (unbounded) ---
    const size_t srvArrayRangesOffset = allRanges.size();
    for (const auto& b : m_bindings)
    {
        if (b.type != RayTraceBindingType::SRV_ARRAY) continue;
        D3D12_DESCRIPTOR_RANGE1 r = {};
        r.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        r.NumDescriptors                    = UINT_MAX;
        r.BaseShaderRegister                = b.registerIndex;
        r.RegisterSpace                     = b.space;
        r.Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
        r.OffsetInDescriptorsFromTableStart = 0;
        allRanges.push_back(r);
    }

    // --- UAV_ARRAY descriptor ranges (unbounded) ---
    const size_t uavArrayRangesOffset = allRanges.size();
    for (const auto& b : m_bindings)
    {
        if (b.type != RayTraceBindingType::UAV_ARRAY) continue;
        D3D12_DESCRIPTOR_RANGE1 r = {};
        r.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        r.NumDescriptors                    = UINT_MAX;
        r.BaseShaderRegister                = b.registerIndex;
        r.RegisterSpace                     = b.space;
        r.Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
        r.OffsetInDescriptorsFromTableStart = 0;
        allRanges.push_back(r);
    }

    std::vector<D3D12_ROOT_PARAMETER1> params;
    params.reserve(
        (m_numSRV      ? 1 : 0) +
        (m_numUAV      ? 1 : 0) +
        m_numSRVArray  +
        m_numUAVArray  +
        m_numCBV       +
        m_numRootSRV   +
        m_numRootConstants);

    // Optional — SRV table (SRV + TLAS)
    if (m_numSRV > 0)
    {
        m_rootParamSRV = static_cast<uint32_t>(params.size());
        D3D12_ROOT_PARAMETER1 p = {};
        p.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p.DescriptorTable.NumDescriptorRanges = m_numSRV;
        p.DescriptorTable.pDescriptorRanges   = &allRanges[srvRangesOffset];
        p.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
        params.push_back(p);
        Logf(kUnityLogTypeLog, "  Root param %u: SRV table (%u descriptors)", m_rootParamSRV, m_numSRV);
    }

    // Optional — UAV table
    if (m_numUAV > 0)
    {
        m_rootParamUAV = static_cast<uint32_t>(params.size());
        D3D12_ROOT_PARAMETER1 p = {};
        p.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p.DescriptorTable.NumDescriptorRanges = m_numUAV;
        p.DescriptorTable.pDescriptorRanges   = &allRanges[uavRangesOffset];
        p.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
        params.push_back(p);
        Logf(kUnityLogTypeLog, "  Root param %u: UAV table (%u descriptors)", m_rootParamUAV, m_numUAV);
    }

    // One table per SRV_ARRAY
    {
        uint32_t arrayIdx = 0;
        for (auto& b : m_bindings)
        {
            if (b.type != RayTraceBindingType::SRV_ARRAY) continue;
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
            if (b.type != RayTraceBindingType::UAV_ARRAY) continue;
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
            if (b.type != RayTraceBindingType::CBV) continue;
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
            if (b.type != RayTraceBindingType::ROOT_SRV) continue;
            b.rootParam = static_cast<uint32_t>(params.size());
            D3D12_ROOT_PARAMETER1 p = {};
            p.ParameterType             = D3D12_ROOT_PARAMETER_TYPE_SRV;
            p.Descriptor.ShaderRegister = b.registerIndex;
            p.Descriptor.RegisterSpace  = b.space;
            p.ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;
            params.push_back(p);
            Logf(kUnityLogTypeLog, "  Root param %u: ROOT_SRV '%s' t%u space%u",
                 b.rootParam, b.name.c_str(), b.registerIndex, b.space);
        }
    }

    // One root 32-bit constants slot per ROOT_CONSTANTS binding
    for (auto& b : m_bindings)
    {
        if (b.type != RayTraceBindingType::ROOT_CONSTANTS) continue;
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

    // Static samplers — parsed from Unity inline sampler naming convention.
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
        D3D12_FILTER filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        if      (Contains(lower, "point")) filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        else if (Contains(lower, "aniso")) filter = D3D12_FILTER_ANISOTROPIC;

        D3D12_TEXTURE_ADDRESS_MODE addr = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
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
    }

    // Validate register spaces
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
                 "RayTraceShader: binding '%s' has invalid RegisterSpace=0x%08X — add explicit register() in HLSL",
                 bindName, range.RegisterSpace);
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
                 "RayTraceShader: root descriptor has invalid RegisterSpace=0x%08X",
                 p.Descriptor.RegisterSpace);
            spaceValid = false;
        }
    }
    if (!spaceValid) return false;

    D3D12_ROOT_SIGNATURE_DESC1 rsDesc1 = {};
    rsDesc1.NumParameters     = static_cast<UINT>(params.size());
    rsDesc1.pParameters       = params.empty()   ? nullptr : params.data();
    rsDesc1.NumStaticSamplers = static_cast<UINT>(samplers.size());
    rsDesc1.pStaticSamplers   = samplers.empty() ? nullptr : samplers.data();
    rsDesc1.Flags             = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC vrsDesc = {};
    vrsDesc.Version  = D3D_ROOT_SIGNATURE_VERSION_1_1;
    vrsDesc.Desc_1_1 = rsDesc1;

    ComPtr<ID3DBlob> sigBlob, errBlob;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&vrsDesc, &sigBlob, &errBlob);
    if (FAILED(hr))
    {
        Logf(kUnityLogTypeError,
             "RayTraceShader: D3D12SerializeVersionedRootSignature failed (hr=0x%08X): %s",
             hr, errBlob ? (char*)errBlob->GetBufferPointer() : "");
        return false;
    }

    hr = m_device->CreateRootSignature(0, sigBlob->GetBufferPointer(),
                                        sigBlob->GetBufferSize(), IID_PPV_ARGS(&m_rootSig));
    if (FAILED(hr))
    {
        Logf(kUnityLogTypeError, "RayTraceShader: CreateRootSignature failed (hr=0x%08X)", hr);
        return false;
    }
    {
        std::wstring wname(m_name.begin(), m_name.end());
        wname += L"_RootSig";
        m_rootSig->SetName(wname.c_str());
    }
    Logf(kUnityLogTypeLog, "RayTraceShader: root signature OK (%u params)", rsDesc1.NumParameters);
    return true;
}

// ---------------------------------------------------------------------------
// BuildPipeline
//   Creates the DXR RTPSO from the reflected shader entry points.
//   Hit groups are auto-discovered from ClosestHit*/AnyHit* naming convention.
// ---------------------------------------------------------------------------
bool RayTraceShader::BuildPipeline(IDxcBlob* shaderLib)
{
    if (m_rayGenShaders.empty())
    {
        Log(kUnityLogTypeError, "RayTraceShader::BuildPipeline: no ray generation shader found");
        return false;
    }
    if (m_missShaders.empty())
    {
        Log(kUnityLogTypeError, "RayTraceShader::BuildPipeline: no miss shader found");
        return false;
    }
    if (m_hitGroups.empty())
    {
        Log(kUnityLogTypeError, "RayTraceShader::BuildPipeline: no hit group found");
        return false;
    }

    const UINT hitGroupCount    = static_cast<UINT>(m_hitGroups.size());
    const UINT totalSubobjects  = 4 + hitGroupCount; // lib, N×hitgroups, shaderCfg, rootSig, pipelineCfg

    std::vector<D3D12_STATE_SUBOBJECT> subObjects(totalSubobjects);
    std::vector<D3D12_HIT_GROUP_DESC>  hitGroupDescs(hitGroupCount);
    UINT si = 0;

    // 1. DXIL library (export all)
    D3D12_DXIL_LIBRARY_DESC libDesc = {};
    libDesc.DXILLibrary.pShaderBytecode = shaderLib->GetBufferPointer();
    libDesc.DXILLibrary.BytecodeLength  = shaderLib->GetBufferSize();
    libDesc.NumExports                  = 0;
    subObjects[si++] = { D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &libDesc };

    // 2. Hit groups
    for (UINT i = 0; i < hitGroupCount; ++i)
    {
        const HitGroupInfo& hg = m_hitGroups[i];
        D3D12_HIT_GROUP_DESC& desc = hitGroupDescs[i];
        desc = {};
        desc.HitGroupExport           = hg.groupExport.c_str();
        desc.Type                     = D3D12_HIT_GROUP_TYPE_TRIANGLES;
        desc.ClosestHitShaderImport   = hg.closestHitExport.empty() ? nullptr : hg.closestHitExport.c_str();
        desc.AnyHitShaderImport       = hg.anyHitExport.empty()     ? nullptr : hg.anyHitExport.c_str();
        desc.IntersectionShaderImport = nullptr;
        subObjects[si++] = { D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP, &desc };
    }

    // 3. Shader config (payload + attribute sizes)
    D3D12_RAYTRACING_SHADER_CONFIG shaderCfg = {};
    shaderCfg.MaxPayloadSizeInBytes   = sizeof(float) * 6;
    shaderCfg.MaxAttributeSizeInBytes = sizeof(float) * 2;
    subObjects[si++] = { D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG, &shaderCfg };

    // 4. Global root signature
    ID3D12RootSignature* pRS = m_rootSig.Get();
    subObjects[si++] = { D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &pRS };

    // 5. Pipeline config (supports opacity micromaps)
    D3D12_RAYTRACING_PIPELINE_CONFIG1 pipeCfg = {};
    pipeCfg.MaxTraceRecursionDepth = 1;
    pipeCfg.Flags = D3D12_RAYTRACING_PIPELINE_FLAG_ALLOW_OPACITY_MICROMAPS;
    subObjects[si++] = { D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG1, &pipeCfg };

    D3D12_STATE_OBJECT_DESC soDesc = {};
    soDesc.Type          = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
    soDesc.NumSubobjects = si;
    soDesc.pSubobjects   = subObjects.data();

    HRESULT hr = m_device->CreateStateObject(&soDesc, IID_PPV_ARGS(&m_pso));
    if (FAILED(hr))
    {
        Logf(kUnityLogTypeError, "RayTraceShader: CreateStateObject failed (hr=0x%08X)", hr);
        return false;
    }
    {
        std::wstring wname(m_name.begin(), m_name.end());
        wname += L"_PSO";
        m_pso->SetName(wname.c_str());
    }
    Logf(kUnityLogTypeLog,
         "RayTraceShader: pipeline built (%zu raygen, %zu miss, %zu hitgroup(s))",
         m_rayGenShaders.size(), m_missShaders.size(), m_hitGroups.size());
    return true;
}

// ---------------------------------------------------------------------------
// BuildShaderTable
// ---------------------------------------------------------------------------
bool RayTraceShader::BuildShaderTable()
{
    if (!m_pso) { Log(kUnityLogTypeError, "RayTraceShader::BuildShaderTable: m_pso is null"); return false; }

    ComPtr<ID3D12StateObjectProperties> props;
    if (FAILED(m_pso->QueryInterface(IID_PPV_ARGS(&props))) || !props)
    {
        Log(kUnityLogTypeError, "RayTraceShader::BuildShaderTable: QueryInterface for props failed");
        return false;
    }

    const UINT stride = D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT;
    const UINT idSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;

    auto MakeTable = [&](const char* label, const std::vector<std::wstring>& names) -> ComPtr<ID3D12Resource>
    {
        const UINT totalSize = stride * static_cast<UINT>(names.size());
        auto buf = CreateUploadBuffer(m_device.Get(), totalSize);
        if (!buf) { Logf(kUnityLogTypeError, "RayTraceShader: CreateUploadBuffer failed for '%s'", label); return nullptr; }
        uint8_t* p = nullptr;
        if (FAILED(buf->Map(0, nullptr, reinterpret_cast<void**>(&p))) || !p)
        {
            Logf(kUnityLogTypeError, "RayTraceShader: Map failed for table '%s'", label);
            return nullptr;
        }
        for (const auto& wname : names)
        {
            void* id = props->GetShaderIdentifier(wname.c_str());
            if (!id)
            {
                char nameA[256] = {};
                WideCharToMultiByte(CP_UTF8, 0, wname.c_str(), -1, nameA, sizeof(nameA)-1, nullptr, nullptr);
                Logf(kUnityLogTypeError, "RayTraceShader: GetShaderIdentifier null for '%s' in '%s'", nameA, label);
                buf->Unmap(0, nullptr);
                return nullptr;
            }
            memcpy(p, id, idSize);
            p += stride;
        }
        buf->Unmap(0, nullptr);
        return buf;
    };

    m_rayGenTable   = MakeTable("RayGen",   { m_rayGenShaders[0] });
    m_missTable     = MakeTable("Miss",     m_missShaders);
    std::vector<std::wstring> hgNames;
    hgNames.reserve(m_hitGroups.size());
    for (const auto& hg : m_hitGroups) hgNames.push_back(hg.groupExport);
    m_hitGroupTable = MakeTable("HitGroup", hgNames);

    return m_rayGenTable && m_missTable && m_hitGroupTable;
}
