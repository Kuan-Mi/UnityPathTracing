#include "ShaderBase.h"
#include <cstdio>
#include <cstdarg>
#include <algorithm>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace
{
std::mutex g_sharedRootSigMutex;
std::unordered_map<std::string, Microsoft::WRL::ComPtr<ID3D12RootSignature>> g_sharedRootSigCache;
}

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
// Layout declaration
// ===========================================================================

void ShaderBase::ClearSharedLayout()
{
    m_sharedLayout.clear();
    m_staticSamplers.clear();
}

void ShaderBase::AddSharedLayoutItem(SharedLayoutKind kind, uint32_t shaderRegister,
                                     uint32_t space, uint32_t count, uint32_t num32BitValues,
                                     bool groupWithPrev)
{
    SharedLayoutItem item;
    item.kind           = kind;
    item.shaderRegister = shaderRegister;
    item.space          = space;
    item.count          = (count == 0) ? 1 : count;
    item.num32BitValues = num32BitValues;
    item.groupWithPrev  = groupWithPrev;
    m_sharedLayout.push_back(item);
}

void ShaderBase::AddStaticSampler(const StaticSamplerDef& def)
{
    m_staticSamplers.push_back(def);
}

// ===========================================================================
// BuildSharedRootSignature
//   Explicit-layout mode, replicating nvrhi::d3d12::BindingLayout /
//   Device::buildRootSignature (RTXPT) exactly, so Unity pipelines produce
//   GPU-identical root signatures to the reference app:
//     [push constants] [volatile-CBV root descriptors (DATA_STATIC)]
//     [sampler table] [one combined SRV/UAV/static-CBV table]
//     [bindless table per consecutive Bindless* run, unbounded ranges]
//   Ranges in the combined table follow declaration order; adjacent registers
//   of the same range class merge into one range (nvrhi
//   AreResourceTypesCompatible collapses to range-class equality).
//   RootSRV is a plugin extension (nvrhi has none): root descriptors placed
//   after the volatile CBVs.
//   The declared layout is the ONLY binding contract — like nvrhi, no shader
//   reflection is consulted; register mismatches with HLSL surface via the
//   C#-side import-time validation or the D3D12 debug layer.
// ===========================================================================
bool ShaderBase::BuildSharedRootSignature()
{
    if (m_sharedLayout.empty())
    {
        Logf(kUnityLogTypeError,
             "ShaderBase [%s]: explicit binding layout is mandatory — create the "
             "pipeline with a NativeBindingLayout (auto-generated or hand-authored)",
             m_name.c_str());
        return false;
    }

    // Rebuild-safe: all layout-derived state is recomputed below.
    m_rootParamSRV         = kInvalidAlloc;
    m_rootParamUAV         = kInvalidAlloc;
    m_rootParamSampler     = kInvalidAlloc;
    m_rootParamCBVBase     = kInvalidAlloc;
    m_rootParamRootSRVBase = kInvalidAlloc;

    // Range class in the combined CBV_SRV_UAV table (nvrhi's SRVetc table).
    enum class RangeClass : uint32_t { SRV, UAV, CBV, Sampler, None };
    auto ClassOf = [](SharedLayoutKind k) {
        switch (k)
        {
        case SharedLayoutKind::SRV:
        case SharedLayoutKind::TLAS:    return RangeClass::SRV;
        case SharedLayoutKind::UAV:     return RangeClass::UAV;
        case SharedLayoutKind::CBV:     return RangeClass::CBV;
        case SharedLayoutKind::Sampler: return RangeClass::Sampler;
        default:                        return RangeClass::None;
        }
    };

    std::string buildLog;
    auto AppendLogf = [&buildLog](const char* fmt, ...) {
        char buf[512];
        va_list args;
        va_start(args, fmt);
        vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
        buildLog += buf;
        buildLog += '\n';
    };

    AppendLogf("ShaderBase::BuildSharedRootSignature [%s]: %u layout item(s), %u static sampler(s)",
               m_name.c_str(), static_cast<uint32_t>(m_sharedLayout.size()),
               static_cast<uint32_t>(m_staticSamplers.size()));

    // --- Build descriptor ranges (nvrhi::d3d12::BindingLayout ctor) ---
    std::vector<D3D12_DESCRIPTOR_RANGE1> srvEtcRanges;   // combined SRV/UAV/CBV table
    std::vector<D3D12_DESCRIPTOR_RANGE1> samplerRanges;
    std::vector<D3D12_DESCRIPTOR_RANGE1> bindlessRanges;
    srvEtcRanges.reserve(m_sharedLayout.size());
    samplerRanges.reserve(m_sharedLayout.size());
    bindlessRanges.reserve(m_sharedLayout.size());

    // Bindless groups: a run of consecutive Bindless* items maps to ONE root
    // parameter whose table holds one unbounded range per item (nvrhi bindless
    // layout with several registerSpaces). first = index into bindlessRanges.
    struct BindlessGroup { uint32_t first, count; };
    std::vector<BindlessGroup> bindlessGroups;
    std::vector<uint32_t> bindlessGroupOfItem(m_sharedLayout.size(), kInvalidAlloc);

    uint32_t srvEtcSlots  = 0;
    uint32_t samplerSlots = 0;

    RangeClass prevClass  = RangeClass::None;
    uint32_t prevEndReg   = 0;    // one past the last register of the previous table item
    uint32_t prevSpace    = ~0u;
    bool prevWasBindless  = false;

    for (size_t itemIdx = 0; itemIdx < m_sharedLayout.size(); ++itemIdx)
    {
        auto& item = m_sharedLayout[itemIdx];
        const RangeClass cls = ClassOf(item.kind);

        if (item.kind == SharedLayoutKind::BindlessSRV ||
            item.kind == SharedLayoutKind::BindlessUAV)
        {
            // A bindless item joins the previous item's root parameter only when
            // explicitly marked (all ranges of a group alias ONE descriptor
            // table); otherwise it gets its own table.
            if (!prevWasBindless || !item.groupWithPrev)
                bindlessGroups.push_back({ static_cast<uint32_t>(bindlessRanges.size()), 0 });
            bindlessGroupOfItem[itemIdx] = static_cast<uint32_t>(bindlessGroups.size() - 1);
            ++bindlessGroups.back().count;

            D3D12_DESCRIPTOR_RANGE1 r = {};
            r.RangeType                         = (item.kind == SharedLayoutKind::BindlessSRV)
                                                ? D3D12_DESCRIPTOR_RANGE_TYPE_SRV
                                                : D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            r.NumDescriptors                    = UINT_MAX;    // unbounded
            r.BaseShaderRegister                = item.shaderRegister;
            r.RegisterSpace                     = item.space;
            r.Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE;
            r.OffsetInDescriptorsFromTableStart = 0;
            bindlessRanges.push_back(r);

            prevWasBindless = true;
            prevClass = RangeClass::None;
            continue;
        }
        prevWasBindless = false;

        if (cls == RangeClass::Sampler)
        {
            item.tableOffset = samplerSlots;
            if (prevClass == RangeClass::Sampler && item.space == prevSpace &&
                item.shaderRegister == prevEndReg && !samplerRanges.empty())
            {
                samplerRanges.back().NumDescriptors += item.count;
            }
            else
            {
                D3D12_DESCRIPTOR_RANGE1 r = {};
                r.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
                r.NumDescriptors                    = item.count;
                r.BaseShaderRegister                = item.shaderRegister;
                r.RegisterSpace                     = item.space;
                r.Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_NONE;
                r.OffsetInDescriptorsFromTableStart = item.tableOffset;
                samplerRanges.push_back(r);
            }
            samplerSlots += item.count;
            prevClass  = cls;
            prevEndReg = item.shaderRegister + item.count;
            prevSpace  = item.space;
        }
        else if (cls == RangeClass::SRV || cls == RangeClass::UAV || cls == RangeClass::CBV)
        {
            item.tableOffset = srvEtcSlots;
            if (cls == prevClass && item.space == prevSpace &&
                item.shaderRegister == prevEndReg && !srvEtcRanges.empty())
            {
                srvEtcRanges.back().NumDescriptors += item.count;
            }
            else
            {
                D3D12_DESCRIPTOR_RANGE1 r = {};
                r.RangeType = (cls == RangeClass::SRV) ? D3D12_DESCRIPTOR_RANGE_TYPE_SRV
                            : (cls == RangeClass::UAV) ? D3D12_DESCRIPTOR_RANGE_TYPE_UAV
                                                       : D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
                r.NumDescriptors                    = item.count;
                r.BaseShaderRegister                = item.shaderRegister;
                r.RegisterSpace                     = item.space;
                r.Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
                r.OffsetInDescriptorsFromTableStart = item.tableOffset;
                srvEtcRanges.push_back(r);
            }
            srvEtcSlots += item.count;
            prevClass  = cls;
            prevEndReg = item.shaderRegister + item.count;
            prevSpace  = item.space;
        }
        else
        {
            // Root-bound kinds don't participate in table-range merging.
            prevClass = RangeClass::None;
        }
    }

    // --- Assemble root parameters in nvrhi order ---
    std::vector<D3D12_ROOT_PARAMETER1> params;
    params.reserve(m_sharedLayout.size() + 3);

    // 1) Push constants
    for (auto& item : m_sharedLayout)
    {
        if (item.kind != SharedLayoutKind::PushConstants) continue;
        item.rootParam = static_cast<uint32_t>(params.size());
        D3D12_ROOT_PARAMETER1 p = {};
        p.ParameterType            = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        p.Constants.ShaderRegister = item.shaderRegister;
        p.Constants.RegisterSpace  = item.space;
        p.Constants.Num32BitValues = item.num32BitValues
                                    ? item.num32BitValues
                                    : ((item.count >= 4) ? (item.count / 4) : item.count);
        p.ShaderVisibility         = D3D12_SHADER_VISIBILITY_ALL;
        params.push_back(p);
    }

    // 2) Volatile CBVs — root descriptors, DATA_STATIC (nvrhi: the versioned
    //    upload allocation is immutable once bound; our VolatileConstantBuffer
    //    follows the same write-then-bind discipline).
    for (auto& item : m_sharedLayout)
    {
        if (item.kind != SharedLayoutKind::VolatileCBV) continue;
        if (m_rootParamCBVBase == kInvalidAlloc)
            m_rootParamCBVBase = static_cast<uint32_t>(params.size());
        item.rootParam = static_cast<uint32_t>(params.size());
        D3D12_ROOT_PARAMETER1 p = {};
        p.ParameterType             = D3D12_ROOT_PARAMETER_TYPE_CBV;
        p.Descriptor.ShaderRegister = item.shaderRegister;
        p.Descriptor.RegisterSpace  = item.space;
        p.Descriptor.Flags          = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC;
        p.ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;
        params.push_back(p);
    }

    // 3) Root SRVs (plugin extension, absent from RTXPT layouts)
    for (auto& item : m_sharedLayout)
    {
        if (item.kind != SharedLayoutKind::RootSRV) continue;
        if (m_rootParamRootSRVBase == kInvalidAlloc)
            m_rootParamRootSRVBase = static_cast<uint32_t>(params.size());
        item.rootParam = static_cast<uint32_t>(params.size());
        D3D12_ROOT_PARAMETER1 p = {};
        p.ParameterType             = D3D12_ROOT_PARAMETER_TYPE_SRV;
        p.Descriptor.ShaderRegister = item.shaderRegister;
        p.Descriptor.RegisterSpace  = item.space;
        p.Descriptor.Flags          = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE;
        p.ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;
        params.push_back(p);
    }

    // 4) Sampler table (nvrhi puts it before the SRVetc table)
    if (!samplerRanges.empty())
    {
        m_rootParamSampler = static_cast<uint32_t>(params.size());
        D3D12_ROOT_PARAMETER1 p = {};
        p.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p.DescriptorTable.NumDescriptorRanges = static_cast<UINT>(samplerRanges.size());
        p.DescriptorTable.pDescriptorRanges   = samplerRanges.data();
        p.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
        params.push_back(p);
    }

    // 5) Combined SRV/UAV/static-CBV table. m_rootParamSRV doubles as "the"
    //    table param; m_rootParamUAV stays invalid so the descriptor sets
    //    bind exactly one table.
    if (!srvEtcRanges.empty())
    {
        m_rootParamSRV = static_cast<uint32_t>(params.size());
        D3D12_ROOT_PARAMETER1 p = {};
        p.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p.DescriptorTable.NumDescriptorRanges = static_cast<UINT>(srvEtcRanges.size());
        p.DescriptorTable.pDescriptorRanges   = srvEtcRanges.data();
        p.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
        params.push_back(p);
    }

    // 6) Bindless groups — one root param per consecutive run of Bindless items
    //    (nvrhi appends each bindless layout as one table of unbounded ranges).
    {
        std::vector<uint32_t> groupRootParam(bindlessGroups.size(), kInvalidAlloc);
        for (size_t g = 0; g < bindlessGroups.size(); ++g)
        {
            groupRootParam[g] = static_cast<uint32_t>(params.size());
            D3D12_ROOT_PARAMETER1 p = {};
            p.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            p.DescriptorTable.NumDescriptorRanges = bindlessGroups[g].count;
            p.DescriptorTable.pDescriptorRanges   = &bindlessRanges[bindlessGroups[g].first];
            p.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
            params.push_back(p);
        }
        for (size_t itemIdx = 0; itemIdx < m_sharedLayout.size(); ++itemIdx)
            if (bindlessGroupOfItem[itemIdx] != kInvalidAlloc)
                m_sharedLayout[itemIdx].rootParam = groupRootParam[bindlessGroupOfItem[itemIdx]];
    }

    // The combined table is exposed through the "SRV slots" total: descriptor
    // sets allocate/copy one contiguous block for the whole SRV/UAV/CBV table
    // and bind it via m_rootParamSRV. UAV slots stay 0 (combined table).
    m_numSRVSlots     = srvEtcSlots;
    m_numUAVSlots     = 0;
    m_numSamplerSlots = samplerSlots;

    // --- Static samplers: explicit defs supplied with the layout ---
    std::vector<D3D12_STATIC_SAMPLER_DESC> staticSamplers;
    staticSamplers.reserve(m_staticSamplers.size());
    for (const auto& def : m_staticSamplers)
    {
        auto Addr = [](uint32_t a) {
            switch (a)
            {
            case 1: return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            case 2: return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
            case 3: return D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
            case 4: return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
            default: return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            }
        };
        D3D12_STATIC_SAMPLER_DESC sd = {};
        sd.Filter           = (def.filter == 0) ? D3D12_FILTER_MIN_MAG_MIP_POINT :
                              (def.filter == 2) ? D3D12_FILTER_ANISOTROPIC :
                                                  D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        sd.AddressU         = Addr(def.addressU);
        sd.AddressV         = Addr(def.addressV);
        sd.AddressW         = Addr(def.addressW);
        sd.MaxAnisotropy    = (sd.Filter == D3D12_FILTER_ANISOTROPIC) ? def.maxAnisotropy : 0;
        sd.ComparisonFunc   = D3D12_COMPARISON_FUNC_NONE;
        sd.BorderColor      = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
        sd.MaxLOD           = def.mips ? 16.0f : 0.0f;
        sd.ShaderRegister   = def.reg;
        sd.RegisterSpace    = def.space;
        sd.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        staticSamplers.push_back(sd);
        AppendLogf("  Static sampler: s%u space%u filter=%u addr=%u/%u/%u maxLod=%g",
                   def.reg, def.space, def.filter, def.addressU, def.addressV, def.addressW,
                   static_cast<double>(sd.MaxLOD));
    }

    std::ostringstream key;
    key << "rs1_1|";
    for (const auto& item : m_sharedLayout)
    {
        key << static_cast<uint32_t>(item.kind) << ':'
            << item.shaderRegister << ':'
            << item.space << ':'
            << item.count << ':'
            << item.num32BitValues << ':'
            << (item.groupWithPrev ? 1 : 0) << ';';
    }
    key << "|static_samp|";
    for (const auto& s : staticSamplers)
    {
        key << s.ShaderRegister << ':' << s.RegisterSpace << ':'
            << static_cast<uint32_t>(s.Filter) << ':'
            << static_cast<uint32_t>(s.AddressU) << ':'
            << static_cast<uint32_t>(s.AddressV) << ':'
            << static_cast<uint32_t>(s.AddressW) << ':'
            << s.MaxAnisotropy << ':' << s.MaxLOD << ';';
    }
    const std::string cacheKey = key.str();
    {
        std::lock_guard<std::mutex> lock(g_sharedRootSigMutex);
        auto it = g_sharedRootSigCache.find(cacheKey);
        if (it != g_sharedRootSigCache.end())
        {
            m_rootSig = it->second;
            AppendLogf("  reused cached shared root signature");
            Log(kUnityLogTypeLog, buildLog.c_str());
            return true;
        }
    }

    D3D12_ROOT_SIGNATURE_DESC1 rsDesc1 = {};
    rsDesc1.NumParameters     = static_cast<UINT>(params.size());
    rsDesc1.pParameters       = params.empty() ? nullptr : params.data();
    rsDesc1.NumStaticSamplers = static_cast<UINT>(staticSamplers.size());
    rsDesc1.pStaticSamplers   = staticSamplers.empty() ? nullptr : staticSamplers.data();
    rsDesc1.Flags             = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    // 1.1, not 1.2: PIX "Export Capture as C++ Project" cannot serialize a root
    // signature containing a 1.2 D3D12_STATIC_SAMPLER_DESC1 static sampler. No
    // 1.2-only sampler features are used, so 1.1 is lossless here.
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC vrsDesc = {};
    vrsDesc.Version  = D3D_ROOT_SIGNATURE_VERSION_1_1;
    vrsDesc.Desc_1_1 = rsDesc1;

    AppendLogf("  shared root params=%u, srvEtcSlots=%u (ranges=%u), samplerSlots=%u, bindlessGroups=%u",
               static_cast<uint32_t>(params.size()), srvEtcSlots,
               static_cast<uint32_t>(srvEtcRanges.size()), samplerSlots,
               static_cast<uint32_t>(bindlessGroups.size()));
    Log(kUnityLogTypeLog, buildLog.c_str());

    ComPtr<ID3DBlob> sigBlob, errBlob;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&vrsDesc, &sigBlob, &errBlob);
    if (FAILED(hr))
    {
        Logf(kUnityLogTypeError,
             "ShaderBase [%s]: shared D3D12SerializeVersionedRootSignature failed "
             "(hr=0x%08X): %s",
             m_name.c_str(), hr, errBlob ? (char*)errBlob->GetBufferPointer() : "");
        return false;
    }

    hr = m_device->CreateRootSignature(0, sigBlob->GetBufferPointer(),
                                       sigBlob->GetBufferSize(), IID_PPV_ARGS(&m_rootSig));
    if (FAILED(hr))
    {
        Logf(kUnityLogTypeError,
             "ShaderBase [%s]: shared CreateRootSignature failed (hr=0x%08X)",
             m_name.c_str(), hr);
        return false;
    }
    std::wstring wname(m_name.begin(), m_name.end());
    wname += L"_SharedRootSig";
    m_rootSig->SetName(wname.c_str());
    {
        std::lock_guard<std::mutex> lock(g_sharedRootSigMutex);
        g_sharedRootSigCache[cacheKey] = m_rootSig;
    }
    return true;
}

// ===========================================================================
// BuildBindingsFromLayout
//   Derives the per-slot binding table straight from the declared layout —
//   one Binding per layout item, in item order. This order IS the dispatch
//   contract: BindingSlot payload index i addresses layout item i (the C#
//   side resolves names / nvrhi-style set items to the same indices from its
//   own copy of the layout). Requires BuildSharedRootSignature to have filled
//   tableOffset / rootParam on each item first.
// ===========================================================================
void ShaderBase::BuildBindingsFromLayout()
{
    m_bindings.clear();
    m_bindings.reserve(m_sharedLayout.size());
    m_numSRV = m_numUAV = m_numCBV = m_numSRVArray = m_numUAVArray =
        m_numRootConstants = m_numRootSRV = m_numSampler = 0;

    char nameBuf[64];
    for (const auto& item : m_sharedLayout)
    {
        Binding b = {};
        b.registerIndex  = item.shaderRegister;
        b.space          = item.space;
        b.arrayCount     = 1;
        b.heapOffset     = 0;
        b.rootParam      = kInvalidAlloc;
        b.num32BitValues = 0;
        const char* prefix = "?";

        switch (item.kind)
        {
        case SharedLayoutKind::SRV:
            b.type       = BindingType::SRV;
            b.arrayCount = item.count;
            b.heapOffset = item.tableOffset;
            prefix       = "t";
            ++m_numSRV;
            break;
        case SharedLayoutKind::TLAS:
            b.type       = BindingType::TLAS;
            b.heapOffset = item.tableOffset;
            prefix       = "t";
            ++m_numSRV;
            break;
        case SharedLayoutKind::UAV:
            b.type       = BindingType::UAV;
            b.arrayCount = item.count;
            b.heapOffset = item.tableOffset;
            prefix       = "u";
            ++m_numUAV;
            break;
        case SharedLayoutKind::CBV:
            // Static CBV — occupies a slot in the combined table (nvrhi
            // ConstantBuffer); rootParam stays invalid to mark table residency.
            b.type       = BindingType::CBV;
            b.heapOffset = item.tableOffset;
            prefix       = "b";
            ++m_numCBV;
            break;
        case SharedLayoutKind::VolatileCBV:
            // Root CBV — bound by GPU VA via its own root parameter.
            b.type      = BindingType::CBV;
            b.rootParam = item.rootParam;
            prefix      = "b";
            ++m_numCBV;
            break;
        case SharedLayoutKind::PushConstants:
            b.type           = BindingType::ROOT_CONSTANTS;
            b.rootParam      = item.rootParam;
            b.num32BitValues = item.num32BitValues
                              ? item.num32BitValues
                              : ((item.count >= 4) ? (item.count / 4) : item.count);
            prefix           = "b";
            ++m_numRootConstants;
            break;
        case SharedLayoutKind::RootSRV:
            b.type      = BindingType::ROOT_SRV;
            b.rootParam = item.rootParam;
            prefix      = "t";
            ++m_numRootSRV;
            break;
        case SharedLayoutKind::BindlessSRV:
            b.type      = BindingType::SRV_ARRAY;
            b.rootParam = item.rootParam;
            prefix      = "t";
            ++m_numSRVArray;
            break;
        case SharedLayoutKind::BindlessUAV:
            b.type      = BindingType::UAV_ARRAY;
            b.rootParam = item.rootParam;
            prefix      = "u";
            ++m_numUAVArray;
            break;
        case SharedLayoutKind::Sampler:
            b.type       = BindingType::SAMPLER;
            b.arrayCount = item.count;
            b.heapOffset = item.tableOffset;
            prefix       = "s";
            ++m_numSampler;
            break;
        }

        // Synthesized register name — used only for log messages (the layout
        // has no HLSL variable names; C# owns the name→slot mapping).
        snprintf(nameBuf, sizeof(nameBuf), "%s%u_space%u",
                 prefix, item.shaderRegister, item.space);
        b.name = nameBuf;
        m_bindings.push_back(std::move(b));
    }
}
