#include "RayTraceShader.h"
#include "BindlessTexture.h"
#include "BindlessBuffer.h"
#include "AccelerationStructure.h"
#include <d3d12shader.h>
#include <cstdio>
#include <cstdarg>
#include <algorithm>
#include <windows.h>

// -------------------------------------------------------------------------
// Heap / buffer helpers (duplicated locally to avoid coupling to Renderer)
// -------------------------------------------------------------------------
namespace
{
    static D3D12_HEAP_PROPERTIES UploadHeapProps()
    {
        D3D12_HEAP_PROPERTIES p = {};
        p.Type = D3D12_HEAP_TYPE_UPLOAD;
        return p;
    }

    static D3D12_HEAP_PROPERTIES DefaultHeapProps()
    {
        D3D12_HEAP_PROPERTIES p = {};
        p.Type = D3D12_HEAP_TYPE_DEFAULT;
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

// -------------------------------------------------------------------------

RayTraceShader::RayTraceShader() = default;
RayTraceShader::~RayTraceShader()
{
    FreeAllAllocations();
}

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

bool RayTraceShader::Initialize(ID3D12Device5* device, IUnityLog* log, DescriptorHeapAllocator* allocator, IUnityGraphicsD3D12v8* d3d12v8)
{
    m_log       = log;
    m_device    = device;
    m_allocator = allocator;
    m_d3d12v8   = d3d12v8;
    return true;
}

// -------------------------------------------------------------------------
// LoadShaderFromBytes
//   Build the pipeline from pre-compiled DXIL bytes.
//   Identical to the back half of LoadShaderFile (post-compilation).
// -------------------------------------------------------------------------
bool RayTraceShader::LoadShaderFromBytes(const uint8_t* dxilBytes, uint32_t size)
{
    if (!dxilBytes || size == 0)
    {
        Log(kUnityLogTypeError, "RayTraceShader::LoadShaderFromBytes: empty input");
        return false;
    }

    // Wrap the raw bytes in an IDxcBlob via IDxcUtils
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

    // Reset old pipeline
    m_pso.Reset();
    m_rootSig.Reset();
    m_rayGenTable.Reset();
    m_missTable.Reset();
    m_hitGroupTable.Reset();
    m_userBindings.clear();
    m_bindingIndex.clear();
    m_samplerBindings.clear();
    m_rayGenShaders.clear();
    m_missShaders.clear();
    m_hitGroups.clear();
    m_hitGroupIndex.clear();
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

    if (!ReflectUserBindings(shaderLib.Get())) return false;
    if (!BuildRootSignature())                 return false;
    if (!BuildPipeline(shaderLib.Get()))       return false;
    if (!BuildShaderTable())                   return false;

    Logf(kUnityLogTypeLog, "RayTraceShader: pipeline ready from bytes (%u SRV, %u UAV, %u CBV, %u SRV_ARRAY)",
         m_numSRV, m_numUAV, m_numCBV, m_numSRVArray);
    return true;
}

// -------------------------------------------------------------------------
// ReflectUserBindings
//   Collects ALL resource bindings from HLSL (all registers, all spaces).
//   No registers are skipped — C# controls everything via SetXxx by name.
// -------------------------------------------------------------------------
bool RayTraceShader::ReflectUserBindings(IDxcBlob* shaderLib)
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
             "RayTraceShader: CreateReflection failed (hr=0x%08X) - no bindings", hr);
        return true; // not fatal
    }

    D3D12_LIBRARY_DESC libDesc = {};
    libRefl->GetDesc(&libDesc);

    Logf(kUnityLogTypeLog, "RayTraceShader::ReflectUserBindings: library has %u function(s)",
         libDesc.FunctionCount);

    for (UINT fi = 0; fi < libDesc.FunctionCount; ++fi)
    {
        ID3D12FunctionReflection* func = libRefl->GetFunctionByIndex(static_cast<INT>(fi));
        if (!func)
        {
            Logf(kUnityLogTypeWarning, "RayTraceShader: GetFunctionByIndex(%u) returned null", fi);
            continue;
        }

        D3D12_FUNCTION_DESC funcDesc = {};
        HRESULT hrDesc = func->GetDesc(&funcDesc);
        if (FAILED(hrDesc))
        {
            Logf(kUnityLogTypeWarning, "RayTraceShader: GetDesc failed for function %u (hr=0x%08X)", fi, hrDesc);
            continue;
        }

        // --- Classify shader stage and populate entry point lists ---
        // Version high-16 bits encode the shader type (D3D12_SHADER_VERSION_TYPE)
        if (funcDesc.Name)
        {
            const UINT shaderType = (funcDesc.Version >> 16) & 0xFFFF;
            std::string nameA(funcDesc.Name);

            // DXC compiles lib_6_x entry points with C++ name mangling.
            // Mangled names may have a leading \x01 byte (MSVC extern-C marker),
            // followed by ?FunctionName@@<signature>.
            // Extract the real (unmangled) function name.
            std::string realName = nameA;
            {
                const char* p = nameA.c_str();
                // Skip leading \x01 (and any other leading non-printable bytes)
                while (*p && (unsigned char)*p < 0x20) ++p;
                if (*p == '?')
                {
                    ++p; // skip '?'
                    const char* atAt = strstr(p, "@@");
                    if (atAt)
                        realName = std::string(p, atAt);
                    // else: no '@@' found — keep full name as-is
                }
                else if (p != nameA.c_str())
                {
                    // Had a \x01 prefix but no '?' — use the rest after \x01
                    realName = std::string(p);
                }
            }

            std::wstring nameW(realName.begin(), realName.end());

            Logf(kUnityLogTypeLog,
                 "RayTraceShader: function[%u] rawName='%s' realName='%s' version=0x%08X shaderType=%u",
                 fi, nameA.c_str(), realName.c_str(), funcDesc.Version, shaderType);

            // Helper: strip a type prefix and optional leading '_' from nameW.
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

            if (shaderType == D3D12_SHVER_RAY_GENERATION_SHADER)
            {
                Logf(kUnityLogTypeLog, "RayTraceShader:   -> RayGen: '%s'", nameA.c_str());
                m_rayGenShaders.push_back(nameW);
            }
            else if (shaderType == D3D12_SHVER_MISS_SHADER)
            {
                Logf(kUnityLogTypeLog, "RayTraceShader:   -> Miss: '%s'", nameA.c_str());
                m_missShaders.push_back(nameW);
            }
            else if (shaderType == D3D12_SHVER_CLOSEST_HIT_SHADER)
            {
                std::wstring groupKey;
                if (!TryStrip(L"ClosestHit", groupKey)) groupKey = nameW;
                std::wstring groupExport = groupKey.empty() ? L"HitGroup" : L"HitGroup_" + groupKey;
                char groupExportA[256] = {};
                WideCharToMultiByte(CP_UTF8,0,groupExport.c_str(),-1,groupExportA,sizeof(groupExportA)-1,nullptr,nullptr);
                Logf(kUnityLogTypeLog, "RayTraceShader:   -> ClosestHit: '%s' -> group '%s'",
                     nameA.c_str(), groupExportA);
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
                char groupExportA[256] = {};
                WideCharToMultiByte(CP_UTF8,0,groupExport.c_str(),-1,groupExportA,sizeof(groupExportA)-1,nullptr,nullptr);
                Logf(kUnityLogTypeLog, "RayTraceShader:   -> AnyHit: '%s' -> group '%s'",
                     nameA.c_str(), groupExportA);
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
            else
            {
                Logf(kUnityLogTypeLog, "RayTraceShader:   -> unrecognized stage type %u for '%s'",
                     shaderType, nameA.c_str());
            }
        }
        else
        {
            Logf(kUnityLogTypeWarning, "RayTraceShader: function[%u] has null Name", fi);
        }

        collect_bindings:

        // --- Collect resource bindings ---
        for (UINT ri = 0; ri < funcDesc.BoundResources; ++ri)
        {
            D3D12_SHADER_INPUT_BIND_DESC bind = {};
            if (FAILED(func->GetResourceBindingDesc(ri, &bind))) continue;

            if (bind.Type == D3D_SIT_SAMPLER)
            {
                // Collect sampler by name (de-duplicate across functions)
                const std::string sname(bind.Name);
                bool found = false;
                for (const auto& s : m_samplerBindings)
                    if (s.name == sname) { found = true; break; }
                if (!found)
                    m_samplerBindings.push_back({ sname, bind.BindPoint, bind.Space });
                continue;
            }

            const std::string name(bind.Name);
            if (m_bindingIndex.count(name)) continue; // de-duplicate across functions

            UserBinding ub;
            ub.name          = name;
            ub.space         = bind.Space;
            ub.registerIndex = bind.BindPoint;
            ub.boundResource = nullptr;
            ub.boundAS       = nullptr;
            ub.boundBT       = nullptr;
            ub.boundBB       = nullptr;
            ub.heapOffset    = 0;
            ub.rootParam     = kInvalidAlloc;
            ub.boundCount    = 0;
            ub.boundStride   = 0;

            switch (bind.Type)
            {
            case D3D_SIT_RTACCELERATIONSTRUCTURE:
                ub.type = UserBindingType::TLAS;
                ++m_numSRV; // TLAS shares the SRV descriptor table
                break;
            case D3D_SIT_CBUFFER:
                ub.type = UserBindingType::CBV;
                ++m_numCBV;
                break;
            case D3D_SIT_TBUFFER:
            case D3D_SIT_TEXTURE:
            case D3D_SIT_STRUCTURED:
            case D3D_SIT_BYTEADDRESS:
                if (bind.BindCount == 0)
                {
                    ub.type = UserBindingType::SRV_ARRAY;
                    ++m_numSRVArray;
                }
                else
                {
                    ub.type = UserBindingType::SRV;
                    ++m_numSRV;
                }
                break;
            default: // UAV variants
                ub.type = UserBindingType::UAV;
                ++m_numUAV;
                break;
            }

            m_bindingIndex[name] = m_userBindings.size();
            m_userBindings.push_back(std::move(ub));
        }
    }

    // Assign consecutive heap offsets per type group
    // Assign consecutive heap offsets per type group
    uint32_t srvOff = 0, uavOff = 0, cbvOff = 0;
    for (auto& b : m_userBindings)
    {
        if      (b.type == UserBindingType::SRV || b.type == UserBindingType::TLAS)
            b.heapOffset = srvOff++;
        else if (b.type == UserBindingType::UAV)
            b.heapOffset = uavOff++;
        else if (b.type == UserBindingType::CBV)
            b.heapOffset = cbvOff++;
    }

    // Sort miss shaders and hit groups by their exported name for deterministic shader table ordering.
    // Users can control the index by naming shaders with a sortable prefix, e.g. Miss_0_Primary, Miss_1_Shadow.
    std::sort(m_missShaders.begin(), m_missShaders.end());
    std::sort(m_hitGroups.begin(), m_hitGroups.end(),
        [](const HitGroupInfo& a, const HitGroupInfo& b) { return a.groupExport < b.groupExport; });
    // Rebuild the index map after sorting since positions changed.
    m_hitGroupIndex.clear();
    for (size_t i = 0; i < m_hitGroups.size(); ++i)
    {
        // Key = groupExport stripped of "HitGroup_" prefix (the original groupKey).
        const std::wstring& exp = m_hitGroups[i].groupExport;
        static const std::wstring kPrefix = L"HitGroup_";
        std::wstring key = (exp.size() > kPrefix.size() && exp.compare(0, kPrefix.size(), kPrefix) == 0)
            ? exp.substr(kPrefix.size())
            : (exp == L"HitGroup" ? L"" : exp);
        m_hitGroupIndex[key] = i;
    }

    // Log final ordering
    for (size_t i = 0; i < m_missShaders.size(); ++i)
    {
        char buf[256] = {};
        WideCharToMultiByte(CP_UTF8, 0, m_missShaders[i].c_str(), -1, buf, sizeof(buf)-1, nullptr, nullptr);
        Logf(kUnityLogTypeLog, "RayTraceShader: missShader[%zu] = '%s'", i, buf);
    }
    for (size_t i = 0; i < m_hitGroups.size(); ++i)
    {
        char buf[256] = {};
        WideCharToMultiByte(CP_UTF8, 0, m_hitGroups[i].groupExport.c_str(), -1, buf, sizeof(buf)-1, nullptr, nullptr);
        Logf(kUnityLogTypeLog, "RayTraceShader: hitGroup[%zu] = '%s'", i, buf);
    }

    return true;
}

// -------------------------------------------------------------------------
// BuildRootSignature
//   Fully dynamic — no hardcoded registers or spaces.
//   Param 0: SRV table (one range per SRV/TLAS binding)    optional
//   Param 1: UAV table (one range per UAV binding)          optional
//   Params+: one table per SRV_ARRAY binding
//   Params+: one root CBV per CBV binding
// -------------------------------------------------------------------------
bool RayTraceShader::BuildRootSignature()
{
    std::vector<D3D12_DESCRIPTOR_RANGE1> allRanges;
    allRanges.reserve(m_numSRV + m_numUAV + m_numSRVArray);
    Logf(kUnityLogTypeLog, "RayTraceShader::BuildRootSignature: %u SRV, %u UAV, %u SRV_ARRAY bindings",
         m_numSRV, m_numUAV, m_numSRVArray);


    // --- SRV descriptor ranges (one per SRV/TLAS binding) ---
    const size_t srvRangesOffset = allRanges.size();
    for (const auto& b : m_userBindings)
    {
        if (b.type != UserBindingType::SRV && b.type != UserBindingType::TLAS) continue;
        D3D12_DESCRIPTOR_RANGE1 r = {};
        r.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        r.NumDescriptors                    = 1;
        r.BaseShaderRegister                = b.registerIndex;
        r.RegisterSpace                     = b.space;
        r.Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
        r.OffsetInDescriptorsFromTableStart = b.heapOffset;
        allRanges.push_back(r);
        Logf(kUnityLogTypeLog, "  SRV/TLAS binding: name='%s' t%u space%u heapOffset=%u",
             b.name.c_str(), b.registerIndex, b.space, b.heapOffset);
    }

    // --- UAV descriptor ranges (one per UAV binding) ---
    const size_t uavRangesOffset = allRanges.size();
    for (const auto& b : m_userBindings)
    {
        if (b.type != UserBindingType::UAV) continue;
        D3D12_DESCRIPTOR_RANGE1 r = {};
        r.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        r.NumDescriptors                    = 1;
        r.BaseShaderRegister                = b.registerIndex;
        r.RegisterSpace                     = b.space;
        r.Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
        r.OffsetInDescriptorsFromTableStart = b.heapOffset;
        allRanges.push_back(r);
        Logf(kUnityLogTypeLog, "  UAV binding: name='%s' t%u space%u heapOffset=%u",
             b.name.c_str(), b.registerIndex, b.space, b.heapOffset);
    }

    // --- SRV_ARRAY descriptor ranges (one per unbounded array binding) ---
    const size_t srvArrayRangesOffset = allRanges.size();
    for (const auto& b : m_userBindings)
    {
        if (b.type != UserBindingType::SRV_ARRAY) continue;
        D3D12_DESCRIPTOR_RANGE1 r = {};
        r.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        r.NumDescriptors                    = UINT_MAX;
        r.BaseShaderRegister                = b.registerIndex;
        r.RegisterSpace                     = b.space;
        r.Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
        r.OffsetInDescriptorsFromTableStart = 0;
        allRanges.push_back(r);
        Logf(kUnityLogTypeLog, "  SRV_ARRAY binding: name='%s' t%u space%u heapOffset=%u",
             b.name.c_str(), b.registerIndex, b.space, b.heapOffset);
    }

    std::vector<D3D12_ROOT_PARAMETER1> params;
    params.reserve((m_numSRV ? 1 : 0) + (m_numUAV ? 1 : 0) + m_numSRVArray + m_numCBV);

    // Optional - SRV table (all SRV + TLAS)
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
        for (auto& b : m_userBindings)
        {
            if (b.type != UserBindingType::SRV_ARRAY) continue;
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
        for (auto& b : m_userBindings)
        {
            if (b.type != UserBindingType::CBV) continue;
            b.rootParam = static_cast<uint32_t>(params.size());
            D3D12_ROOT_PARAMETER1 p = {};
            p.ParameterType             = D3D12_ROOT_PARAMETER_TYPE_CBV;
            p.Descriptor.ShaderRegister = b.registerIndex;
            p.Descriptor.RegisterSpace  = b.space;
            p.ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;
            params.push_back(p);
            Logf(kUnityLogTypeLog, "  Root param %u: CBV '%s' b%u space%u", b.rootParam, b.name.c_str(), b.registerIndex, b.space);
        }
    }

    // Static samplers — built dynamically from reflected SamplerState names.
    // Properties are parsed from the variable name using Unity's inline sampler convention:
    //   Filter  : "point" → POINT  |  "trilinear"/"linear" → LINEAR (default)  |  "aniso" → ANISOTROPIC
    //   Address : "clamp" → CLAMP  |  "repeat" → WRAP  |  "mirroronce" → MIRROR_ONCE  |  "mirror" → MIRROR
    //             default → WRAP
    // Example: SamplerState sampler_linear_clamp;  →  linear filter, clamp address
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

    // --- Debug: dump all root parameters and their descriptor ranges ---
    Logf(kUnityLogTypeLog, "BuildRootSignature: %u root param(s), %u static sampler(s)",
         rsDesc1.NumParameters, rsDesc1.NumStaticSamplers);
    // for (UINT pi = 0; pi < rsDesc1.NumParameters; ++pi)
    // {
    //     const D3D12_ROOT_PARAMETER1& rp = rsDesc1.pParameters[pi];
    //     if (rp.ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
    //     {
    //         const size_t tableOffset = static_cast<size_t>(rp.DescriptorTable.pDescriptorRanges - allRanges.data());
    //         Logf(kUnityLogTypeLog, "  Param[%u] DESCRIPTOR_TABLE, %u range(s)", pi, rp.DescriptorTable.NumDescriptorRanges);
    //         for (UINT ri = 0; ri < rp.DescriptorTable.NumDescriptorRanges; ++ri)
    //         {
    //             const D3D12_DESCRIPTOR_RANGE1& dr = rp.DescriptorTable.pDescriptorRanges[ri];
    //             const char* typeName =
    //                 (dr.RangeType == D3D12_DESCRIPTOR_RANGE_TYPE_SRV) ? "SRV" :
    //                 (dr.RangeType == D3D12_DESCRIPTOR_RANGE_TYPE_UAV) ? "UAV" :
    //                 (dr.RangeType == D3D12_DESCRIPTOR_RANGE_TYPE_CBV) ? "CBV" : "SAMPLER";
    //             // Resolve binding name from parallel name vectors
    //             const char* bindName = "?";
    //             if (tableOffset == srvRangesOffset && ri < srvRangeNames.size())
    //                 bindName = srvRangeNames[ri].c_str();
    //             else if (tableOffset == uavRangesOffset && ri < uavRangeNames.size())
    //                 bindName = uavRangeNames[ri].c_str();
    //             Logf(kUnityLogTypeLog,
    //                  "    Range[%u] type=%s t/u/b%u space%u numDesc=%u heapOffset=%u  name='%s'",
    //                  ri, typeName, dr.BaseShaderRegister, dr.RegisterSpace,
    //                  dr.NumDescriptors, dr.OffsetInDescriptorsFromTableStart, bindName);
    //         }
    //     }
    //     else if (rp.ParameterType == D3D12_ROOT_PARAMETER_TYPE_CBV)
    //     {
    //         Logf(kUnityLogTypeLog, "  Param[%u] ROOT_CBV b%u space%u", pi,
    //              rp.Descriptor.ShaderRegister, rp.Descriptor.RegisterSpace);
    //     }
    //     else
    //     {
    //         Logf(kUnityLogTypeLog, "  Param[%u] type=%u", pi, (UINT)rp.ParameterType);
    //     }
    // }

    // // --- Debug: check for duplicate (register, space) pairs within SRV ranges ---
    // {
    //     struct RegKey { UINT reg; UINT space; UINT rangeIdx; std::string name; };
    //     std::vector<RegKey> srvSeen;
    //     for (UINT pi = 0; pi < rsDesc1.NumParameters; ++pi)
    //     {
    //         const D3D12_ROOT_PARAMETER1& rp = rsDesc1.pParameters[pi];
    //         if (rp.ParameterType != D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE) continue;
    //         const size_t tableOffset = static_cast<size_t>(rp.DescriptorTable.pDescriptorRanges - allRanges.data());
    //         for (UINT ri = 0; ri < rp.DescriptorTable.NumDescriptorRanges; ++ri)
    //         {
    //             const D3D12_DESCRIPTOR_RANGE1& dr = rp.DescriptorTable.pDescriptorRanges[ri];
    //             if (dr.RangeType != D3D12_DESCRIPTOR_RANGE_TYPE_SRV) continue;
    //             const char* bindName = "?";
    //             if (tableOffset == srvRangesOffset && ri < srvRangeNames.size())
    //                 bindName = srvRangeNames[ri].c_str();
    //             UINT count = (dr.NumDescriptors == UINT_MAX) ? 1 : dr.NumDescriptors;
    //             for (UINT k = 0; k < count; ++k)
    //             {
    //                 UINT reg = dr.BaseShaderRegister + k;
    //                 for (const auto& seen : srvSeen)
    //                 {
    //                     if (seen.reg == reg && seen.space == dr.RegisterSpace)
    //                         Logf(kUnityLogTypeError,
    //                              "BuildRootSignature: DUPLICATE SRV t%u space%u: range[%u]='%s' conflicts with range[%u]='%s' -- missing register() decoration?",
    //                              reg, dr.RegisterSpace, pi, bindName, seen.rangeIdx, seen.name.c_str());
    //                 }
    //                 srvSeen.push_back({ reg, dr.RegisterSpace, ri, bindName });
    //             }
    //         }
    //     }
    // }

    // Validate all register spaces before serialization to catch HLSL bindings missing explicit register() decorations.
    bool spaceValid = true;
    for (const auto& range : allRanges)
    {
        if (range.RegisterSpace >= 0xfffffff0)
        {
            // Find the binding name for a better error message
            const char* bindName = "?";
            for (const auto& b : m_userBindings)
            {
                if (b.registerIndex == range.BaseShaderRegister && b.space == range.RegisterSpace)
                    { bindName = b.name.c_str(); break; }
            }
            Logf(kUnityLogTypeError,
                 "RayTraceShader: binding '%s' has invalid RegisterSpace=0x%08X -- add an explicit register(xN, spaceM) decoration in HLSL",
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
                     "RayTraceShader: root descriptor has invalid RegisterSpace=0x%08X -- add an explicit register() decoration in HLSL",
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
                 "RayTraceShader: static sampler s%u has invalid RegisterSpace=0x%08X -- add an explicit register() decoration in HLSL",
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
        Logf(kUnityLogTypeError, "RayTraceShader: D3D12SerializeVersionedRootSignature failed (hr=0x%08X): %s",
             hr, errBlob ? (char*)errBlob->GetBufferPointer() : "");
        return false;
    }

    hr = m_device->CreateRootSignature(0, sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(), IID_PPV_ARGS(&m_rootSig));
    if (FAILED(hr))
    {
        Logf(kUnityLogTypeError, "RayTraceShader: CreateRootSignature failed (hr=0x%08X)", hr);
        return false;
    }
    return true;
}

// -------------------------------------------------------------------------
// BuildPipeline — RTPSO dynamically built from reflected shader entries.
// Hit groups are auto-discovered: ClosestHit*/AnyHit* pairs are matched by
// the suffix after stripping the type prefix (and optional '_').
// -------------------------------------------------------------------------
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
        Log(kUnityLogTypeError, "RayTraceShader::BuildPipeline: no hit group found (need ClosestHit* and/or AnyHit*)");
        return false;
    }

    // Fixed subobjects: DXIL_LIBRARY, SHADER_CONFIG, GLOBAL_ROOT_SIGNATURE, PIPELINE_CONFIG1
    // Plus one HIT_GROUP subobject per discovered hit group.
    const UINT hitGroupCount = static_cast<UINT>(m_hitGroups.size());
    const UINT totalSubobjects = 4 + hitGroupCount;

    std::vector<D3D12_STATE_SUBOBJECT>  subObjects(totalSubobjects);
    std::vector<D3D12_HIT_GROUP_DESC>   hitGroupDescs(hitGroupCount);
    UINT si = 0;

    // 1. DXIL library — export all symbols automatically
    D3D12_DXIL_LIBRARY_DESC libDesc = {};
    libDesc.DXILLibrary.pShaderBytecode = shaderLib->GetBufferPointer();
    libDesc.DXILLibrary.BytecodeLength  = shaderLib->GetBufferSize();
    libDesc.NumExports                  = 0;
    subObjects[si++] = { D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &libDesc };

    // 2. Hit groups (one per discovered pair/single entry)
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

    // 5. Pipeline config with OMM support
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

    Logf(kUnityLogTypeLog,
         "RayTraceShader: pipeline built (%zu raygen, %zu miss, %zu hitgroup(s))",
         m_rayGenShaders.size(), m_missShaders.size(), m_hitGroups.size());
    for (size_t i = 0; i < m_hitGroups.size(); ++i)
    {
        const HitGroupInfo& hg = m_hitGroups[i];
        char exp[256]={}, ch[256]={}, ah[256]={};
        WideCharToMultiByte(CP_UTF8,0,hg.groupExport.c_str(),-1,exp,sizeof(exp)-1,nullptr,nullptr);
        WideCharToMultiByte(CP_UTF8,0,hg.closestHitExport.c_str(),-1,ch,sizeof(ch)-1,nullptr,nullptr);
        WideCharToMultiByte(CP_UTF8,0,hg.anyHitExport.c_str(),-1,ah,sizeof(ah)-1,nullptr,nullptr);
        Logf(kUnityLogTypeLog,
             "RayTraceShader:   hitgroup[%zu] export='%s' closestHit='%s' anyHit='%s'",
             i, exp, ch, ah);
    }
    return true;
}

bool RayTraceShader::BuildShaderTable()
{
    if (!m_pso)
    {
        Log(kUnityLogTypeError, "RayTraceShader::BuildShaderTable: m_pso is null");
        return false;
    }

    Log(kUnityLogTypeLog, "RayTraceShader::BuildShaderTable: querying shader identifiers");

    ComPtr<ID3D12StateObjectProperties> props;
    HRESULT hrProps = m_pso->QueryInterface(IID_PPV_ARGS(&props));
    if (FAILED(hrProps) || !props)
    {
        Logf(kUnityLogTypeError, "RayTraceShader::BuildShaderTable: QueryInterface for props failed (hr=0x%08X)", hrProps);
        return false;
    }

    // Each shader record is padded to D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT (64 bytes).
    // This allows multiple records per table with proper alignment.
    const UINT stride  = D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT;
    const UINT idSize  = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES; // 32 bytes

    // Helper: allocate and fill a table for a list of exported shader names.
    auto MakeTable = [&](const char* tableLabel, const std::vector<std::wstring>& names) -> ComPtr<ID3D12Resource>
    {
        Logf(kUnityLogTypeLog, "RayTraceShader::BuildShaderTable: building '%s' table (%zu entry/%u stride)",
             tableLabel, names.size(), stride);
        const UINT totalSize = stride * static_cast<UINT>(names.size());
        auto buf = CreateUploadBuffer(m_device.Get(), totalSize);
        if (!buf)
        {
            Logf(kUnityLogTypeError, "RayTraceShader::BuildShaderTable: CreateUploadBuffer failed for '%s'", tableLabel);
            return nullptr;
        }
        uint8_t* p = nullptr;
        HRESULT hrMap = buf->Map(0, nullptr, reinterpret_cast<void**>(&p));
        if (FAILED(hrMap) || !p)
        {
            Logf(kUnityLogTypeError, "RayTraceShader::BuildShaderTable: Map failed (hr=0x%08X) for '%s'", hrMap, tableLabel);
            return nullptr;
        }
        for (const auto& name : names)
        {
            char nameA[256] = {};
            WideCharToMultiByte(CP_UTF8, 0, name.c_str(), -1, nameA, static_cast<int>(sizeof(nameA)-1), nullptr, nullptr);
            void* id = props->GetShaderIdentifier(name.c_str());
            if (!id)
            {
                Logf(kUnityLogTypeError,
                     "RayTraceShader: GetShaderIdentifier returned null for '%s' in table '%s'",
                     nameA, tableLabel);
                buf->Unmap(0, nullptr);
                return nullptr;
            }
            Logf(kUnityLogTypeLog, "RayTraceShader::BuildShaderTable:   '%s' identifier OK", nameA);
            memcpy(p, id, idSize);
            p += stride;
        }
        buf->Unmap(0, nullptr);
        return buf;
    };

    // Ray generation — only the first shader is used per DispatchRays call.
    m_rayGenTable = MakeTable("RayGen", { m_rayGenShaders[0] });

    // Miss shaders
    m_missTable = MakeTable("Miss", m_missShaders);

    // Hit groups — use the auto-generated export names
    std::vector<std::wstring> hgNames;
    hgNames.reserve(m_hitGroups.size());
    for (const auto& hg : m_hitGroups)
        hgNames.push_back(hg.groupExport);
    m_hitGroupTable = MakeTable("HitGroup", hgNames);

    return m_rayGenTable && m_missTable && m_hitGroupTable;
}

// -------------------------------------------------------------------------
// Resource setters
// -------------------------------------------------------------------------
bool RayTraceShader::SetBuffer(const char* name, ID3D12Resource* res)
{
    if (!name) return false;
    auto it = m_bindingIndex.find(name);
    if (it == m_bindingIndex.end()) return false;
    UserBinding& b = m_userBindings[it->second];
    if (b.type != UserBindingType::SRV) return false;
    b.boundResource = res;
    b.boundStride   = 0; // raw ByteAddressBuffer
    return true;
}

bool RayTraceShader::SetStructuredBuffer(const char* name, ID3D12Resource* res, UINT count, UINT stride)
{
    if (!name) return false;
    auto it = m_bindingIndex.find(name);
    if (it == m_bindingIndex.end()) return false;
    UserBinding& b = m_userBindings[it->second];
    if (b.type != UserBindingType::SRV) return false;
    b.boundResource = res;
    b.boundCount    = count;
    b.boundStride   = stride;
    return true;
}

bool RayTraceShader::SetAccelerationStructure(const char* name, ID3D12Resource* tlas)
{
    if (!name) return false;
    auto it = m_bindingIndex.find(name);
    if (it == m_bindingIndex.end()) return false;
    UserBinding& b = m_userBindings[it->second];
    if (b.type != UserBindingType::TLAS) return false;
    b.boundResource = tlas;
    b.boundAS       = nullptr; // clear dynamic binding if explicitly setting raw ptr
    return true;
}

bool RayTraceShader::SetAccelerationStructureObject(const char* name, AccelerationStructure* as)
{
    if (!name) return false;
    auto it = m_bindingIndex.find(name);
    if (it == m_bindingIndex.end()) return false;
    UserBinding& b = m_userBindings[it->second];
    if (b.type != UserBindingType::TLAS) return false;
    b.boundAS       = as;     // store object pointer; TLAS will be read dynamically at Dispatch
    b.boundResource = nullptr;
    return true;
}

bool RayTraceShader::SetRWBuffer(const char* name, ID3D12Resource* res)
{
    if (!name) return false;
    auto it = m_bindingIndex.find(name);
    if (it == m_bindingIndex.end()) return false;
    UserBinding& b = m_userBindings[it->second];
    if (b.type != UserBindingType::UAV) return false;
    b.boundResource = res;
    return true;
}

bool RayTraceShader::SetTexture(const char* name, ID3D12Resource* res)
{
    if (!name) return false;
    auto it = m_bindingIndex.find(name);
    if (it == m_bindingIndex.end()) return false;
    UserBinding& b = m_userBindings[it->second];
    if (b.type != UserBindingType::SRV) return false;
    b.boundResource = res;
    b.boundStride   = 0;
    return true;
}

bool RayTraceShader::SetRWTexture(const char* name, ID3D12Resource* res)
{
    if (!name) return false;
    auto it = m_bindingIndex.find(name);
    if (it == m_bindingIndex.end()) return false;
    UserBinding& b = m_userBindings[it->second];
    if (b.type != UserBindingType::UAV) return false;
    b.boundResource = res;
    return true;
}

bool RayTraceShader::SetConstantBuffer(const char* name, ID3D12Resource* res)
{
    if (!name) return false;
    auto it = m_bindingIndex.find(name);
    if (it == m_bindingIndex.end()) return false;
    UserBinding& b = m_userBindings[it->second];
    if (b.type != UserBindingType::CBV) return false;
    b.boundResource = res;
    return true;
}

bool RayTraceShader::SetBindlessTexture(const char* name, BindlessTexture* bt)
{
    if (!name) return false;
    auto it = m_bindingIndex.find(name);
    if (it == m_bindingIndex.end()) return false;
    UserBinding& b = m_userBindings[it->second];
    if (b.type != UserBindingType::SRV_ARRAY) return false;
    b.boundBT = bt;
    return true;
}

bool RayTraceShader::SetBindlessBuffer(const char* name, BindlessBuffer* bb)
{
    if (!name) return false;
    auto it = m_bindingIndex.find(name);
    if (it == m_bindingIndex.end()) return false;
    UserBinding& b = m_userBindings[it->second];
    if (b.type != UserBindingType::SRV_ARRAY) return false;
    b.boundBB = bb;
    return true;
}

// -------------------------------------------------------------------------
// FreeAllAllocations
// -------------------------------------------------------------------------
void RayTraceShader::FreeAllAllocations()
{
    if (!m_allocator) return;
    if (m_srvAllocBase != kInvalidAlloc && m_numSRV > 0) { m_allocator->Free(m_srvAllocBase, m_numSRV); m_srvAllocBase = kInvalidAlloc; }
    if (m_uavAllocBase != kInvalidAlloc && m_numUAV > 0) { m_allocator->Free(m_uavAllocBase, m_numUAV); m_uavAllocBase = kInvalidAlloc; }
}

// -------------------------------------------------------------------------
// AllocateAndWriteDescriptors
// -------------------------------------------------------------------------
bool RayTraceShader::AllocateAndWriteDescriptors()
{
    if (!m_allocator) return false;

    if (m_srvAllocBase == kInvalidAlloc && m_numSRV > 0)
        m_srvAllocBase = m_allocator->Allocate(m_numSRV);
    if (m_uavAllocBase == kInvalidAlloc && m_numUAV > 0)
        m_uavAllocBase = m_allocator->Allocate(m_numUAV);

    UpdateUserDescriptors();
    return true;
}

// -------------------------------------------------------------------------
// UpdateUserDescriptors
//   Writes all SRV/TLAS/UAV descriptors every frame.
//   CBVs are bound as inline root descriptors in Dispatch.
//   SRV_ARRAY bindings own their heap slots in BindlessTexture/BindlessBuffer.
// -------------------------------------------------------------------------
void RayTraceShader::UpdateUserDescriptors()
{
    // --- SRV / TLAS ---
    if (m_srvAllocBase != kInvalidAlloc)
    {
        for (const auto& b : m_userBindings)
        {
            if (b.type == UserBindingType::TLAS)
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
            else if (b.type == UserBindingType::SRV)
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
        for (const auto& b : m_userBindings)
        {
            if (b.type != UserBindingType::UAV) continue;
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

// -------------------------------------------------------------------------
// Dispatch
//   Caller is responsible for resource state transitions.
//   No internal intermediate texture – C# owns all input/output resources.
// -------------------------------------------------------------------------
void RayTraceShader::Dispatch(
    ID3D12GraphicsCommandList4* cmdList,
    UINT width, UINT height)
{
    if (!m_pso || !m_rootSig || !m_allocator) return;

    // --- Validate all user bindings are set from C# ---
    // Catches forgotten SetTexture/SetBuffer/SetCBV/SetAccelerationStructure/Set*Array calls
    // before we issue DispatchRays with null descriptors / unset root parameters.
    {
        bool anyMissing = false;
        for (const auto& b : m_userBindings)
        {
            bool ok = false;
            const char* kind = "?";
            switch (b.type)
            {
            case UserBindingType::TLAS:
                kind = "TLAS";
                ok = (b.boundAS != nullptr) || (b.boundResource != nullptr);
                break;
            case UserBindingType::SRV:
                kind = "SRV";
                ok = (b.boundResource != nullptr);
                break;
            case UserBindingType::UAV:
                kind = "UAV";
                ok = (b.boundResource != nullptr);
                break;
            case UserBindingType::CBV:
                kind = "CBV";
                ok = (b.boundResource != nullptr);
                break;
            case UserBindingType::SRV_ARRAY:
                kind = "SRV_ARRAY";
                ok = (b.boundBT != nullptr) || (b.boundBB != nullptr);
                break;
            }
            if (!ok)
            {
                Logf(kUnityLogTypeError,
                     "RayTraceShader::Dispatch: binding '%s' (%s, space%u, reg%u) is not set - "
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
        UpdateUserDescriptors();
    }

    // Bind the global shared heap (D3D12 allows only one GPU-visible heap at a time)
    ID3D12DescriptorHeap* heapsToBind[1] = { m_allocator->GetHeap() };
    cmdList->SetDescriptorHeaps(1, heapsToBind);

    cmdList->SetPipelineState1(m_pso.Get());
    cmdList->SetComputeRootSignature(m_rootSig.Get());

    // SRV table (all SRV + TLAS bindings)
    if (m_rootParamSRV != kInvalidAlloc && m_srvAllocBase != kInvalidAlloc)
        cmdList->SetComputeRootDescriptorTable(m_rootParamSRV,
            m_allocator->GetGPUHandle(m_srvAllocBase));

    // UAV table (all UAV bindings)
    if (m_rootParamUAV != kInvalidAlloc && m_uavAllocBase != kInvalidAlloc)
        cmdList->SetComputeRootDescriptorTable(m_rootParamUAV,
            m_allocator->GetGPUHandle(m_uavAllocBase));

    // SRV_ARRAY bindings – each has its own root parameter
    for (const auto& b : m_userBindings)
    {
        if (b.type != UserBindingType::SRV_ARRAY) continue;
        if (b.rootParam == kInvalidAlloc) continue;
        if (b.boundBT)
            cmdList->SetComputeRootDescriptorTable(b.rootParam, b.boundBT->GetGPUHandle());
        else if (b.boundBB)
            cmdList->SetComputeRootDescriptorTable(b.rootParam, b.boundBB->GetGPUHandle());
    }

    // Root CBV per CBV binding
    if (m_rootParamCBVBase != kInvalidAlloc)
    {
        for (const auto& b : m_userBindings)
        {
            if (b.type != UserBindingType::CBV) continue;
            D3D12_GPU_VIRTUAL_ADDRESS addr = b.boundResource
                ? b.boundResource->GetGPUVirtualAddress() : 0;
            cmdList->SetComputeRootConstantBufferView(m_rootParamCBVBase + b.heapOffset, addr);
        }
    }

    // --- Request resource states ---
    // TLAS is managed externally (built with the correct state); SRV_ARRAY is managed
    // by BindlessTexture/BindlessBuffer; CBV stays in GENERIC_READ.
    for (const auto& b : m_userBindings)
    {
        if (!b.boundResource) continue;
        if (b.type == UserBindingType::SRV)
        {
            m_d3d12v8->RequestResourceState(b.boundResource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            // Logf(kUnityLogTypeLog, "RayTraceShader::Dispatch: RequestResourceState '%s' SRV -> NON_PIXEL_SHADER_RESOURCE (res=%p)", b.name.c_str(), b.boundResource);
        }
        else if (b.type == UserBindingType::UAV)
        {
            m_d3d12v8->RequestResourceState(b.boundResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            // Logf(kUnityLogTypeLog, "RayTraceShader::Dispatch: RequestResourceState '%s' UAV -> UNORDERED_ACCESS (res=%p)", b.name.c_str(), b.boundResource);
        }
        else if (b.type == UserBindingType::CBV)
        {
            m_d3d12v8->RequestResourceState(b.boundResource, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
        }
    }

    // DispatchRays
    const UINT shaderTableStride = D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT;
    D3D12_DISPATCH_RAYS_DESC drd = {};
    drd.RayGenerationShaderRecord.StartAddress = m_rayGenTable->GetGPUVirtualAddress();
    drd.RayGenerationShaderRecord.SizeInBytes  = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    drd.MissShaderTable.StartAddress           = m_missTable->GetGPUVirtualAddress();
    drd.MissShaderTable.SizeInBytes            = shaderTableStride * static_cast<UINT>(m_missShaders.size());
    drd.MissShaderTable.StrideInBytes          = shaderTableStride;
    drd.HitGroupTable.StartAddress             = m_hitGroupTable->GetGPUVirtualAddress();
    drd.HitGroupTable.SizeInBytes              = shaderTableStride * static_cast<UINT>(m_hitGroups.size());
    drd.HitGroupTable.StrideInBytes            = shaderTableStride;
    drd.Width  = width;
    drd.Height = height;
    drd.Depth  = 1;
    cmdList->DispatchRays(&drd);
}
