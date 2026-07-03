#include "ShaderBase.h"
#include <d3d12shader.h>
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

void ShaderBase::SetSamplerHint(const char* name, uint32_t filter,
                                uint32_t addressU, uint32_t addressV, uint32_t addressW,
                                bool mips, uint32_t maxAnisotropy)
{
    if (!name) return;
    SamplerHint h;
    h.filter        = filter;
    h.addressU      = addressU;
    h.addressV      = addressV;
    h.addressW      = addressW;
    h.mips          = mips;
    h.maxAnisotropy = maxAnisotropy;
    m_samplerHints[name] = h;
}

void ShaderBase::ClearSharedLayout()
{
    m_sharedLayout.clear();
}

void ShaderBase::AddSharedLayoutItem(SharedLayoutKind kind, uint32_t shaderRegister,
                                     uint32_t space, uint32_t count, uint32_t num32BitValues)
{
    SharedLayoutItem item;
    item.kind           = kind;
    item.shaderRegister = shaderRegister;
    item.space          = space;
    item.count          = (count == 0) ? 1 : count;
    item.num32BitValues = num32BitValues;
    m_sharedLayout.push_back(item);
}

bool ShaderBase::UsesSharedSamplerTable() const
{
    for (const auto& item : m_sharedLayout)
        if (item.kind == SharedLayoutKind::Sampler)
            return true;
    return false;
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
//   Fills type / num32BitValues / m_num* on a freshly zeroed Binding
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
    b.arrayCount     = 1;

    switch (bind.Type)
    {
    case D3D_SIT_SAMPLER:
        if (!UsesSharedSamplerTable())
            return false; // caller handles legacy static samplers
        b.type       = BindingType::SAMPLER;
        b.arrayCount = bind.BindCount ? bind.BindCount : 1;
        ++m_numSampler;
        m_numSamplerSlots += b.arrayCount;
        break;

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
            ++m_numSRVSlots;
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
            // BindCount == 1 → single; BindCount > 1 → bounded array (e.g. Texture2D t[4]),
            // which occupies BindCount contiguous descriptors in the SRV table.
            b.type       = BindingType::SRV;
            b.arrayCount = bind.BindCount;
            ++m_numSRV;
            m_numSRVSlots += bind.BindCount;
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
            // BindCount > 1 → bounded array (e.g. RWTexture2D u[NUM_LODS]).
            b.type       = BindingType::UAV;
            b.arrayCount = bind.BindCount;
            ++m_numUAV;
            m_numUAVSlots += bind.BindCount;
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
        // SRV/UAV bindings reserve arrayCount contiguous slots (1 for singles); the binding's
        // heapOffset is the first, so a bounded array fills [heapOffset, heapOffset+arrayCount).
        if      (b.type == BindingType::SRV || b.type == BindingType::TLAS)
            { b.heapOffset = srvOff; srvOff += b.arrayCount; }
        else if (b.type == BindingType::UAV)
            { b.heapOffset = uavOff; uavOff += b.arrayCount; }
        else if (b.type == BindingType::CBV)
            b.heapOffset = cbvOff++;
    }
}

// ===========================================================================
// BuildSharedRootSignature
//   Explicit-layout mode, modelled after RTXPT/NVRHI's shared BindingLayout idea:
//   multiple shaders may declare the same global register layout, while reflection
//   still supplies the per-shader binding names used by C# descriptor sets.
// ===========================================================================
bool ShaderBase::BuildSharedRootSignature()
{
    if (m_sharedLayout.empty())
        return BuildRootSignature();

    auto IsSrvLike = [](SharedLayoutKind k) {
        return k == SharedLayoutKind::SRV || k == SharedLayoutKind::TLAS;
    };
    auto IsUavLike = [](SharedLayoutKind k) {
        return k == SharedLayoutKind::UAV;
    };
    auto IsSamplerLike = [](SharedLayoutKind k) {
        return k == SharedLayoutKind::Sampler;
    };
    auto BindingMatches = [&](const Binding& b, const SharedLayoutItem& item) {
        if (b.registerIndex != item.shaderRegister || b.space != item.space)
            return false;
        switch (b.type)
        {
        case BindingType::SRV:
            return item.kind == SharedLayoutKind::SRV || item.kind == SharedLayoutKind::RootSRV;
        case BindingType::TLAS:
            return item.kind == SharedLayoutKind::TLAS || item.kind == SharedLayoutKind::SRV ||
                   item.kind == SharedLayoutKind::RootSRV;
        case BindingType::UAV:
            return item.kind == SharedLayoutKind::UAV;
        case BindingType::CBV:
            return item.kind == SharedLayoutKind::CBV ||
                   item.kind == SharedLayoutKind::VolatileCBV ||
                   item.kind == SharedLayoutKind::PushConstants;
        case BindingType::ROOT_CONSTANTS:
            return item.kind == SharedLayoutKind::PushConstants;
        case BindingType::ROOT_SRV:
            return item.kind == SharedLayoutKind::RootSRV;
        case BindingType::SRV_ARRAY:
            return item.kind == SharedLayoutKind::BindlessSRV;
        case BindingType::UAV_ARRAY:
            return item.kind == SharedLayoutKind::BindlessUAV;
        case BindingType::SAMPLER:
            return item.kind == SharedLayoutKind::Sampler;
        default:
            return false;
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

    AppendLogf("ShaderBase::BuildSharedRootSignature [%s]: %u reflected binding(s), %u layout item(s)",
               m_name.c_str(), static_cast<uint32_t>(m_bindings.size()),
               static_cast<uint32_t>(m_sharedLayout.size()));

    std::vector<D3D12_DESCRIPTOR_RANGE1> srvRanges;
    std::vector<D3D12_DESCRIPTOR_RANGE1> uavRanges;
    std::vector<D3D12_DESCRIPTOR_RANGE1> samplerRanges;
    std::vector<D3D12_DESCRIPTOR_RANGE1> bindlessRanges;
    srvRanges.reserve(m_sharedLayout.size());
    uavRanges.reserve(m_sharedLayout.size());
    samplerRanges.reserve(m_sharedLayout.size());
    bindlessRanges.reserve(m_sharedLayout.size());

    uint32_t srvSlots = 0;
    uint32_t uavSlots = 0;
    uint32_t samplerSlots = 0;
    uint32_t cbvRootCount = 0;
    uint32_t rootSrvCount = 0;

    for (auto& item : m_sharedLayout)
    {
        if (IsSrvLike(item.kind))
        {
            item.tableOffset = srvSlots;
            D3D12_DESCRIPTOR_RANGE1 r = {};
            r.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            r.NumDescriptors                    = item.count;
            r.BaseShaderRegister                = item.shaderRegister;
            r.RegisterSpace                     = item.space;
            r.Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE |
                                                  D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
            r.OffsetInDescriptorsFromTableStart = item.tableOffset;
            srvRanges.push_back(r);
            srvSlots += item.count;
        }
        else if (IsUavLike(item.kind))
        {
            item.tableOffset = uavSlots;
            D3D12_DESCRIPTOR_RANGE1 r = {};
            r.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            r.NumDescriptors                    = item.count;
            r.BaseShaderRegister                = item.shaderRegister;
            r.RegisterSpace                     = item.space;
            r.Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE |
                                                  D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
            r.OffsetInDescriptorsFromTableStart = item.tableOffset;
            uavRanges.push_back(r);
            uavSlots += item.count;
        }
        else if (IsSamplerLike(item.kind))
        {
            item.tableOffset = samplerSlots;
            D3D12_DESCRIPTOR_RANGE1 r = {};
            r.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
            r.NumDescriptors                    = item.count;
            r.BaseShaderRegister                = item.shaderRegister;
            r.RegisterSpace                     = item.space;
            r.Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE;
            r.OffsetInDescriptorsFromTableStart = item.tableOffset;
            samplerRanges.push_back(r);
            samplerSlots += item.count;
        }
        else if (item.kind == SharedLayoutKind::BindlessSRV ||
                 item.kind == SharedLayoutKind::BindlessUAV)
        {
            D3D12_DESCRIPTOR_RANGE1 r = {};
            r.RangeType                         = (item.kind == SharedLayoutKind::BindlessSRV)
                                                ? D3D12_DESCRIPTOR_RANGE_TYPE_SRV
                                                : D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            r.NumDescriptors                    = UINT_MAX;
            r.BaseShaderRegister                = item.shaderRegister;
            r.RegisterSpace                     = item.space;
            r.Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE |
                                                  D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
            r.OffsetInDescriptorsFromTableStart = 0;
            bindlessRanges.push_back(r);
        }
    }

    std::vector<D3D12_ROOT_PARAMETER1> params;
    params.reserve(m_sharedLayout.size() + 2);

    // Push constants first, then root CBVs, then root SRVs, then descriptor tables.
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

    for (auto& item : m_sharedLayout)
    {
        if (item.kind != SharedLayoutKind::CBV && item.kind != SharedLayoutKind::VolatileCBV)
            continue;
        if (m_rootParamCBVBase == kInvalidAlloc)
            m_rootParamCBVBase = static_cast<uint32_t>(params.size());
        item.rootParam = static_cast<uint32_t>(params.size());
        item.tableOffset = cbvRootCount++;
        D3D12_ROOT_PARAMETER1 p = {};
        p.ParameterType             = D3D12_ROOT_PARAMETER_TYPE_CBV;
        p.Descriptor.ShaderRegister = item.shaderRegister;
        p.Descriptor.RegisterSpace  = item.space;
        p.Descriptor.Flags          = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE;
        p.ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;
        params.push_back(p);
    }

    for (auto& item : m_sharedLayout)
    {
        if (item.kind != SharedLayoutKind::RootSRV) continue;
        if (m_rootParamRootSRVBase == kInvalidAlloc)
            m_rootParamRootSRVBase = static_cast<uint32_t>(params.size());
        item.rootParam = static_cast<uint32_t>(params.size());
        item.tableOffset = rootSrvCount++;
        D3D12_ROOT_PARAMETER1 p = {};
        p.ParameterType             = D3D12_ROOT_PARAMETER_TYPE_SRV;
        p.Descriptor.ShaderRegister = item.shaderRegister;
        p.Descriptor.RegisterSpace  = item.space;
        p.Descriptor.Flags          = D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE;
        p.ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;
        params.push_back(p);
    }

    if (!srvRanges.empty())
    {
        m_rootParamSRV = static_cast<uint32_t>(params.size());
        D3D12_ROOT_PARAMETER1 p = {};
        p.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p.DescriptorTable.NumDescriptorRanges = static_cast<UINT>(srvRanges.size());
        p.DescriptorTable.pDescriptorRanges   = srvRanges.data();
        p.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
        params.push_back(p);
    }

    if (!uavRanges.empty())
    {
        m_rootParamUAV = static_cast<uint32_t>(params.size());
        D3D12_ROOT_PARAMETER1 p = {};
        p.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p.DescriptorTable.NumDescriptorRanges = static_cast<UINT>(uavRanges.size());
        p.DescriptorTable.pDescriptorRanges   = uavRanges.data();
        p.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
        params.push_back(p);
    }

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

    uint32_t bindlessRangeIndex = 0;
    for (auto& item : m_sharedLayout)
    {
        if (item.kind != SharedLayoutKind::BindlessSRV &&
            item.kind != SharedLayoutKind::BindlessUAV)
            continue;
        item.rootParam = static_cast<uint32_t>(params.size());
        D3D12_ROOT_PARAMETER1 p = {};
        p.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p.DescriptorTable.NumDescriptorRanges = 1;
        p.DescriptorTable.pDescriptorRanges   = &bindlessRanges[bindlessRangeIndex++];
        p.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
        params.push_back(p);
    }

    bool ok = true;
    for (auto& b : m_bindings)
    {
        SharedLayoutItem* match = nullptr;
        for (auto& item : m_sharedLayout)
        {
            if (BindingMatches(b, item))
            {
                match = &item;
                break;
            }
        }
        if (!match)
        {
            Logf(kUnityLogTypeError,
                 "ShaderBase [%s]: shared root layout has no item for reflected binding '%s' "
                 "(reg=%u space=%u type=%u)",
                 m_name.c_str(), b.name.c_str(), b.registerIndex, b.space,
                 static_cast<uint32_t>(b.type));
            ok = false;
            continue;
        }

        if (match->kind == SharedLayoutKind::PushConstants)
        {
            b.type           = BindingType::ROOT_CONSTANTS;
            b.num32BitValues = match->num32BitValues
                              ? match->num32BitValues
                              : ((match->count >= 4) ? (match->count / 4) : match->count);
            b.rootParam      = match->rootParam;
        }
        else if (match->kind == SharedLayoutKind::RootSRV)
        {
            b.type       = BindingType::ROOT_SRV;
            b.rootParam  = match->rootParam;
            b.heapOffset = match->tableOffset;
        }
        else if (match->kind == SharedLayoutKind::CBV || match->kind == SharedLayoutKind::VolatileCBV)
        {
            b.type       = BindingType::CBV;
            b.rootParam  = match->rootParam;
            b.heapOffset = match->tableOffset;
        }
        else if (match->kind == SharedLayoutKind::BindlessSRV)
        {
            b.type      = BindingType::SRV_ARRAY;
            b.rootParam = match->rootParam;
        }
        else if (match->kind == SharedLayoutKind::BindlessUAV)
        {
            b.type      = BindingType::UAV_ARRAY;
            b.rootParam = match->rootParam;
        }
        else if (match->kind == SharedLayoutKind::Sampler)
        {
            b.type       = BindingType::SAMPLER;
            b.heapOffset = match->tableOffset;
            b.rootParam  = kInvalidAlloc;
        }
        else
        {
            b.heapOffset = match->tableOffset;
            b.rootParam  = kInvalidAlloc;
        }
    }
    if (!ok) return false;

    // Recompute counts/category totals from the shared-layout interpretation.
    m_numSRV = m_numUAV = m_numCBV = m_numSRVArray = m_numUAVArray = m_numRootConstants = m_numRootSRV = m_numSampler = 0;
    for (const auto& b : m_bindings)
    {
        switch (b.type)
        {
        case BindingType::SRV:
        case BindingType::TLAS: ++m_numSRV; break;
        case BindingType::UAV: ++m_numUAV; break;
        case BindingType::CBV: ++m_numCBV; break;
        case BindingType::SRV_ARRAY: ++m_numSRVArray; break;
        case BindingType::UAV_ARRAY: ++m_numUAVArray; break;
        case BindingType::ROOT_CONSTANTS: ++m_numRootConstants; break;
        case BindingType::ROOT_SRV: ++m_numRootSRV; break;
        case BindingType::SAMPLER: ++m_numSampler; break;
        }
    }
    m_numSRVSlots = srvSlots;
    m_numUAVSlots = uavSlots;
    m_numSamplerSlots = samplerSlots;

    std::vector<D3D12_STATIC_SAMPLER_DESC> staticSamplers;
    staticSamplers.reserve(m_samplerBindings.size());
    for (const auto& sr : m_samplerBindings)
    {
        D3D12_STATIC_SAMPLER_DESC sd = {};
        sd.Filter           = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        sd.AddressU         = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sd.AddressV         = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sd.AddressW         = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sd.MaxAnisotropy    = 0;
        sd.ComparisonFunc   = D3D12_COMPARISON_FUNC_NONE;
        sd.BorderColor      = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
        sd.MaxLOD           = 16.0f;
        sd.ShaderRegister   = sr.reg;
        sd.RegisterSpace    = sr.space;
        sd.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        if (auto it = m_samplerHints.find(sr.name); it != m_samplerHints.end())
        {
            const SamplerHint& h = it->second;
            sd.Filter = (h.filter == 0) ? D3D12_FILTER_MIN_MAG_MIP_POINT :
                        (h.filter == 2) ? D3D12_FILTER_ANISOTROPIC :
                                          D3D12_FILTER_MIN_MAG_MIP_LINEAR;
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
            sd.AddressU = Addr(h.addressU);
            sd.AddressV = Addr(h.addressV);
            sd.AddressW = Addr(h.addressW);
            sd.MaxLOD = h.mips ? 16.0f : 0.0f;
            sd.MaxAnisotropy = (sd.Filter == D3D12_FILTER_ANISOTROPIC) ? h.maxAnisotropy : 0;
        }
        staticSamplers.push_back(sd);
    }

    std::ostringstream key;
    key << "rs1_1|";
    for (const auto& item : m_sharedLayout)
    {
        key << static_cast<uint32_t>(item.kind) << ':'
            << item.shaderRegister << ':'
            << item.space << ':'
            << item.count << ':'
            << item.num32BitValues << ';';
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

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC vrsDesc = {};
    vrsDesc.Version  = D3D_ROOT_SIGNATURE_VERSION_1_1;
    vrsDesc.Desc_1_1 = rsDesc1;

    AppendLogf("  shared root params=%u, srvSlots=%u, uavSlots=%u, samplerSlots=%u",
               static_cast<uint32_t>(params.size()), srvSlots, uavSlots, samplerSlots);
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
// BuildRootSignature
//   Fully dynamic — no hardcoded registers or spaces.
//   Uses D3D_ROOT_SIGNATURE_VERSION_1_1. (1.1, not 1.2: PIX "Export Capture as C++
//   Project" cannot serialize a root signature containing a 1.2 D3D12_STATIC_SAMPLER_DESC1
//   static sampler. No 1.2-only sampler features are used, so 1.1 is lossless here.)
//   Called by both ComputeShader and RayTraceShader after ReflectBindings().
// ===========================================================================
bool ShaderBase::BuildRootSignature()
{
    if (!m_sharedLayout.empty())
        return BuildSharedRootSignature();

    std::vector<D3D12_DESCRIPTOR_RANGE1> allRanges;
    allRanges.reserve(m_numSRV + m_numUAV + m_numSRVArray + m_numUAVArray);

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

    AppendLogf("ShaderBase::BuildRootSignature [%s]: %u SRV, %u UAV, %u CBV, "
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
        r.NumDescriptors                    = b.arrayCount; // 1 for singles, N for bounded arrays
        r.BaseShaderRegister                = b.registerIndex;
        r.RegisterSpace                     = b.space;
        r.Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE |
                                              D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
        r.OffsetInDescriptorsFromTableStart = b.heapOffset;
        allRanges.push_back(r);
        AppendLogf("  SRV: '%s' t%u space%u heapOffset=%u count=%u",
                   b.name.c_str(), b.registerIndex, b.space, b.heapOffset, b.arrayCount);
    }

    // --- UAV descriptor ranges ---
    const size_t uavRangesOffset = allRanges.size();
    for (const auto& b : m_bindings)
    {
        if (b.type != BindingType::UAV) continue;
        D3D12_DESCRIPTOR_RANGE1 r = {};
        r.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        r.NumDescriptors                    = b.arrayCount; // 1 for singles, N for bounded arrays
        r.BaseShaderRegister                = b.registerIndex;
        r.RegisterSpace                     = b.space;
        r.Flags                             = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE |
                                              D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
        r.OffsetInDescriptorsFromTableStart = b.heapOffset;
        allRanges.push_back(r);
        AppendLogf("  UAV: '%s' u%u space%u heapOffset=%u count=%u",
                   b.name.c_str(), b.registerIndex, b.space, b.heapOffset, b.arrayCount);
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
        AppendLogf("  SRV_ARRAY: '%s' t%u space%u", b.name.c_str(), b.registerIndex, b.space);
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
        AppendLogf("  UAV_ARRAY: '%s' u%u space%u", b.name.c_str(), b.registerIndex, b.space);
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
        AppendLogf("  Root param %u: SRV table (%u)", m_rootParamSRV, m_numSRV);
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
        AppendLogf("  Root param %u: UAV table (%u)", m_rootParamUAV, m_numUAV);
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
            AppendLogf("  Root param %u: SRV_ARRAY '%s'", b.rootParam, b.name.c_str());
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
            AppendLogf("  Root param %u: UAV_ARRAY '%s'", b.rootParam, b.name.c_str());
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
            AppendLogf("  Root param %u: CBV '%s' b%u space%u",
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
            AppendLogf("  Root param %u: ROOT_SRV '%s' t%u space%u",
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
        AppendLogf("  Root param %u: ROOT_CONSTANTS '%s' b%u space%u num32=%u",
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

    // Maps the editor-authored enum values (see ShaderBase::SetSamplerHint) to D3D12.
    auto FilterFromHint = [](uint32_t f) -> D3D12_FILTER {
        switch (f)
        {
            case 0:  return D3D12_FILTER_MIN_MAG_MIP_POINT;
            case 2:  return D3D12_FILTER_ANISOTROPIC;
            default: return D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        }
    };
    auto AddressFromHint = [](uint32_t a) -> D3D12_TEXTURE_ADDRESS_MODE {
        switch (a)
        {
            case 1:  return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            case 2:  return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
            case 3:  return D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
            case 4:  return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
            default: return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        }
    };

    std::vector<D3D12_STATIC_SAMPLER_DESC> samplers;
    samplers.reserve(m_samplerBindings.size());
    for (const auto& sr : m_samplerBindings)
    {
        D3D12_FILTER               filter;
        D3D12_TEXTURE_ADDRESS_MODE addrU, addrV, addrW;
        FLOAT                      maxLod;
        UINT                       maxAniso;

        auto hintIt = m_samplerHints.find(sr.name);
        if (hintIt != m_samplerHints.end())
        {
            // Editor override takes precedence over name inference.
            const SamplerHint& h = hintIt->second;
            filter   = FilterFromHint(h.filter);
            addrU    = AddressFromHint(h.addressU);
            addrV    = AddressFromHint(h.addressV);
            addrW    = AddressFromHint(h.addressW);
            maxLod   = h.mips ? 16.0f : 0.0f;
            maxAniso = (filter == D3D12_FILTER_ANISOTROPIC) ? h.maxAnisotropy : 0;
        }
        else
        {
            // Fallback: infer from the Unity inline sampler naming convention.
            const std::string lower = ToLower(sr.name);

            filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
            if      (Contains(lower, "point"))   filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
            else if (Contains(lower, "nearest")) filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
            else if (Contains(lower, "aniso"))   filter = D3D12_FILTER_ANISOTROPIC;
            else if (Contains(lower, "linear"))  filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;

            D3D12_TEXTURE_ADDRESS_MODE addr = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            if      (Contains(lower, "mirroronce")) addr = D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
            else if (Contains(lower, "mirror"))     addr = D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
            else if (Contains(lower, "clamp"))      addr = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            else if (Contains(lower, "repeat"))     addr = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            addrU = addrV = addrW = addr;

            maxLod   = Contains(lower, "mipmap") ? 16.0f : 0.0f;
            maxAniso = (filter == D3D12_FILTER_ANISOTROPIC) ? 16 : 0;

            if (!Contains(lower, "point") && !Contains(lower, "nearest") &&
                !Contains(lower, "aniso") && !Contains(lower, "linear"))
            {
                AppendLogf("  WARNING: sampler '%s' matches no known filter token and no "
                           "editor override; defaulting to linear/wrap", sr.name.c_str());
            }
        }

        D3D12_STATIC_SAMPLER_DESC sd = {};
        sd.Filter           = filter;
        sd.AddressU         = addrU;
        sd.AddressV         = addrV;
        sd.AddressW         = addrW;
        sd.MaxAnisotropy    = maxAniso;
        sd.ComparisonFunc   = D3D12_COMPARISON_FUNC_NONE;
        sd.BorderColor      = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
        sd.MaxLOD           = maxLod;
        sd.ShaderRegister   = sr.reg;
        sd.RegisterSpace    = sr.space;
        sd.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        samplers.push_back(sd);
        AppendLogf("  Static sampler: '%s' filter=%u addr=%u/%u/%u s%u space%u maxLod=%g%s",
                   sr.name.c_str(), filter, addrU, addrV, addrW, sr.reg, sr.space, maxLod,
                   (m_samplerHints.find(sr.name) != m_samplerHints.end()) ? " (override)" : "");
    }

    AppendLogf("  %u root param(s), %u static sampler(s)",
               static_cast<UINT>(params.size()), static_cast<UINT>(samplers.size()));
    Log(kUnityLogTypeLog, buildLog.c_str());

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
