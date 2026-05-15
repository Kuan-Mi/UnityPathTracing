#include "RayTraceShader.h"
#include <d3d12shader.h>
#include <d3d12sdklayers.h>
#include <cstdio>
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
// LoadShaderFromBytes
// ---------------------------------------------------------------------------
bool RayTraceShader::LoadShaderFromBytes(const uint8_t* dxilBytes, uint32_t size,
                                          const char* name, uint32_t flags,
                                          uint32_t maxPayloadSizeInBytes,
                                          const char* rayGenName)
{
    m_name = (name && name[0]) ? name : "RayTraceShader";
    m_allowOpacityMicromaps = (flags & 1u) != 0;
    m_maxPayloadSizeInBytes = maxPayloadSizeInBytes > 0 ? maxPayloadSizeInBytes : 4;
    m_rayGenName = (rayGenName && rayGenName[0])
        ? std::wstring(rayGenName, rayGenName + strlen(rayGenName))
        : L"";

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

    if (!ReflectBindings(shaderLib.Get()))  return false;
    if (!BuildRootSignature())              return false;
    if (!BuildPipeline(shaderLib.Get()))    return false;
    if (!BuildShaderTable())                return false;

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
//   Resource classification is delegated to ShaderBase::ClassifyBinding().
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
             "RayTraceShader: CreateReflection failed (hr=0x%08X) - no bindings", hr);
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

        // --- Collect resource bindings (delegate classification to ShaderBase) ---
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

            Binding cb = {};
            if (!ClassifyBinding(bind, bname, cb)) continue;

            m_bindingIndex[bname] = m_bindings.size();
            m_bindings.push_back(std::move(cb));
        }
    }

    AssignHeapOffsets();

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
// BuildPipeline
//   Creates the DXR RTPSO from the reflected shader entry points.
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

    const UINT hitGroupCount   = static_cast<UINT>(m_hitGroups.size());
    const UINT totalSubobjects = 4 + hitGroupCount;

    std::vector<D3D12_STATE_SUBOBJECT> subObjects(totalSubobjects);
    std::vector<D3D12_HIT_GROUP_DESC>  hitGroupDescs(hitGroupCount);
    UINT si = 0;

    // 1. DXIL library (export all)
    D3D12_DXIL_LIBRARY_DESC libDescSub = {};
    libDescSub.DXILLibrary.pShaderBytecode = shaderLib->GetBufferPointer();
    libDescSub.DXILLibrary.BytecodeLength  = shaderLib->GetBufferSize();
    libDescSub.NumExports                  = 0;
    subObjects[si++] = { D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &libDescSub };

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

    // 3. Shader config
    D3D12_RAYTRACING_SHADER_CONFIG shaderCfg = {};
    shaderCfg.MaxPayloadSizeInBytes   = m_maxPayloadSizeInBytes;
    shaderCfg.MaxAttributeSizeInBytes = sizeof(float) * 2;
    subObjects[si++] = { D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG, &shaderCfg };

    // 4. Global root signature
    ID3D12RootSignature* pRS = m_rootSig.Get();
    subObjects[si++] = { D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &pRS };

    // 5. Pipeline config
    D3D12_RAYTRACING_PIPELINE_CONFIG1 pipeCfg = {};
    pipeCfg.MaxTraceRecursionDepth = 1;
    pipeCfg.Flags = m_allowOpacityMicromaps
        ? D3D12_RAYTRACING_PIPELINE_FLAG_ALLOW_OPACITY_MICROMAPS
        : D3D12_RAYTRACING_PIPELINE_FLAG_NONE;
    subObjects[si++] = { D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG1, &pipeCfg };

    D3D12_STATE_OBJECT_DESC soDesc = {};
    soDesc.Type          = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
    soDesc.NumSubobjects = si;
    soDesc.pSubobjects   = subObjects.data();

    HRESULT hr = m_device->CreateStateObject(&soDesc, IID_PPV_ARGS(&m_pso));
    if (FAILED(hr))
    {
        Logf(kUnityLogTypeError, "RayTraceShader: CreateStateObject failed (hr=0x%08X)", hr);

        ComPtr<ID3D12InfoQueue> infoQueue;
        if (SUCCEEDED(m_device->QueryInterface(IID_PPV_ARGS(&infoQueue))))
        {
            const UINT64 msgCount = infoQueue->GetNumStoredMessages();
            for (UINT64 i = 0; i < msgCount; ++i)
            {
                SIZE_T msgSize = 0;
                if (FAILED(infoQueue->GetMessage(i, nullptr, &msgSize)) || msgSize == 0) continue;
                std::vector<uint8_t> msgBuf(msgSize);
                auto* msg = reinterpret_cast<D3D12_MESSAGE*>(msgBuf.data());
                if (SUCCEEDED(infoQueue->GetMessage(i, msg, &msgSize)) && msg->pDescription)
                    Logf(kUnityLogTypeError, "  D3D12[%llu]: %s", (unsigned long long)i, msg->pDescription);
            }
            infoQueue->ClearStoredMessages();
        }
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

    // Select raygen by name if specified, otherwise fall back to [0]
    std::wstring selectedRayGen = m_rayGenShaders[0];
    if (!m_rayGenName.empty())
    {
        bool found = false;
        for (const auto& rg : m_rayGenShaders)
        {
            if (rg == m_rayGenName)
            {
                selectedRayGen = rg;
                found = true;
                break;
            }
        }
        if (!found)
        {
            char nameA[256] = {};
            WideCharToMultiByte(CP_UTF8, 0, m_rayGenName.c_str(), -1, nameA, sizeof(nameA)-1, nullptr, nullptr);
            // List available raygen shaders to help diagnose the problem
            std::string available;
            for (const auto& rg : m_rayGenShaders)
            {
                char tmp[256] = {};
                WideCharToMultiByte(CP_UTF8, 0, rg.c_str(), -1, tmp, sizeof(tmp)-1, nullptr, nullptr);
                if (!available.empty()) available += ", ";
                available += tmp;
            }
            Logf(kUnityLogTypeError,
                 "RayTraceShader: requested rayGenName '%s' not found in shader library. Available: [%s]",
                 nameA, available.c_str());
            return false;
        }
    }
    m_rayGenTable = MakeTable("RayGen", { selectedRayGen });
    m_missTable   = MakeTable("Miss",   m_missShaders);
    std::vector<std::wstring> hgNames;
    hgNames.reserve(m_hitGroups.size());
    for (const auto& hg : m_hitGroups) hgNames.push_back(hg.groupExport);
    m_hitGroupTable = MakeTable("HitGroup", hgNames);

    return m_rayGenTable && m_missTable && m_hitGroupTable;
}