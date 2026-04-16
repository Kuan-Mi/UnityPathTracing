#include "ComputeShader.h"
#include "BindlessTexture.h"
#include "BindlessBuffer.h"
#include <d3d12shader.h>
#include <cstdio>
#include <cstdarg>
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

bool ComputeShader::Initialize(ID3D12Device* device, IUnityLog* log, DescriptorHeapAllocator* allocator)
{
    m_log       = log;
    m_device    = device;
    m_allocator = allocator;
    return true;
}

// ---------------------------------------------------------------------------
// LoadShaderFromBytes
// ---------------------------------------------------------------------------
bool ComputeShader::LoadShaderFromBytes(const uint8_t* dxilBytes, uint32_t size)
{
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
    m_numSRV = m_numUAV = m_numCBV = m_numSRVArray = 0;
    m_rootParamSRV = m_rootParamUAV = m_rootParamCBVBase = kInvalidAlloc;

    if (m_allocator)
    {
        if (m_srvAllocBase != kInvalidAlloc) { m_allocator->Free(m_srvAllocBase, m_numSRV); m_srvAllocBase = kInvalidAlloc; }
        if (m_uavAllocBase != kInvalidAlloc) { m_allocator->Free(m_uavAllocBase, m_numUAV); m_uavAllocBase = kInvalidAlloc; }
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
//   Uses ID3D12ShaderReflection for a single-entrypoint compute shader.
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

    for (UINT ri = 0; ri < shDesc.BoundResources; ++ri)
    {
        D3D12_SHADER_INPUT_BIND_DESC bind = {};
        if (FAILED(refl->GetResourceBindingDesc(ri, &bind))) continue;
        if (bind.Type == D3D_SIT_SAMPLER) continue;

        const std::string name(bind.Name);
        if (m_bindingIndex.count(name)) continue;

        ComputeBinding cb;
        cb.name          = name;
        cb.space         = bind.Space;
        cb.registerIndex = bind.BindPoint;
        cb.boundResource = nullptr;
        cb.boundBT       = nullptr;
        cb.boundBB       = nullptr;
        cb.heapOffset    = 0;
        cb.rootParam     = kInvalidAlloc;
        cb.boundCount    = 0;
        cb.boundStride   = 0;

        switch (bind.Type)
        {
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
    uint32_t srvOff = 0, uavOff = 0, cbvOff = 0;
    for (auto& b : m_bindings)
    {
        if      (b.type == ComputeBindingType::SRV)  b.heapOffset = srvOff++;
        else if (b.type == ComputeBindingType::UAV)  b.heapOffset = uavOff++;
        else if (b.type == ComputeBindingType::CBV)  b.heapOffset = cbvOff++;
    }

    return true;
}

// ---------------------------------------------------------------------------
// BuildRootSignature
// ---------------------------------------------------------------------------
bool ComputeShader::BuildRootSignature()
{
    std::vector<D3D12_DESCRIPTOR_RANGE> srvRanges, uavRanges, srvArrayRanges;
    srvRanges.reserve(m_numSRV);
    uavRanges.reserve(m_numUAV);
    srvArrayRanges.reserve(m_numSRVArray);

    for (const auto& b : m_bindings)
    {
        if (b.type == ComputeBindingType::SRV)
        {
            D3D12_DESCRIPTOR_RANGE r = {};
            r.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            r.NumDescriptors                    = 1;
            r.BaseShaderRegister                = b.registerIndex;
            r.RegisterSpace                     = b.space;
            r.OffsetInDescriptorsFromTableStart = b.heapOffset;
            srvRanges.push_back(r);
        }
        else if (b.type == ComputeBindingType::UAV)
        {
            D3D12_DESCRIPTOR_RANGE r = {};
            r.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            r.NumDescriptors                    = 1;
            r.BaseShaderRegister                = b.registerIndex;
            r.RegisterSpace                     = b.space;
            r.OffsetInDescriptorsFromTableStart = b.heapOffset;
            uavRanges.push_back(r);
        }
        else if (b.type == ComputeBindingType::SRV_ARRAY)
        {
            D3D12_DESCRIPTOR_RANGE r = {};
            r.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            r.NumDescriptors                    = UINT_MAX;
            r.BaseShaderRegister                = b.registerIndex;
            r.RegisterSpace                     = b.space;
            r.OffsetInDescriptorsFromTableStart = 0;
            srvArrayRanges.push_back(r);
        }
    }

    std::vector<D3D12_ROOT_PARAMETER> params;
    params.reserve((m_numSRV ? 1 : 0) + (m_numUAV ? 1 : 0) + m_numSRVArray + m_numCBV);

    if (m_numSRV > 0)
    {
        m_rootParamSRV = static_cast<uint32_t>(params.size());
        D3D12_ROOT_PARAMETER p = {};
        p.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p.DescriptorTable.NumDescriptorRanges = static_cast<UINT>(srvRanges.size());
        p.DescriptorTable.pDescriptorRanges   = srvRanges.data();
        p.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
        params.push_back(p);
    }

    if (m_numUAV > 0)
    {
        m_rootParamUAV = static_cast<uint32_t>(params.size());
        D3D12_ROOT_PARAMETER p = {};
        p.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p.DescriptorTable.NumDescriptorRanges = static_cast<UINT>(uavRanges.size());
        p.DescriptorTable.pDescriptorRanges   = uavRanges.data();
        p.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
        params.push_back(p);
    }

    {
        uint32_t arrayIdx = 0;
        for (auto& b : m_bindings)
        {
            if (b.type != ComputeBindingType::SRV_ARRAY) continue;
            b.rootParam = static_cast<uint32_t>(params.size());
            D3D12_ROOT_PARAMETER p = {};
            p.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            p.DescriptorTable.NumDescriptorRanges = 1;
            p.DescriptorTable.pDescriptorRanges   = &srvArrayRanges[arrayIdx++];
            p.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
            params.push_back(p);
        }
    }

    if (m_numCBV > 0)
    {
        m_rootParamCBVBase = static_cast<uint32_t>(params.size());
        for (auto& b : m_bindings)
        {
            if (b.type != ComputeBindingType::CBV) continue;
            D3D12_ROOT_PARAMETER p = {};
            p.ParameterType             = D3D12_ROOT_PARAMETER_TYPE_CBV;
            p.Descriptor.ShaderRegister = b.registerIndex;
            p.Descriptor.RegisterSpace  = b.space;
            p.ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;
            params.push_back(p);
        }
    }

    // Static samplers
    D3D12_STATIC_SAMPLER_DESC samplers[2] = {};
    samplers[0].Filter           = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    samplers[0].AddressU = samplers[0].AddressV = samplers[0].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplers[0].MaxAnisotropy    = 1;
    samplers[0].ComparisonFunc   = D3D12_COMPARISON_FUNC_NEVER;
    samplers[0].BorderColor      = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE;
    samplers[0].MaxLOD           = D3D12_FLOAT32_MAX;
    samplers[0].ShaderRegister   = 0;
    samplers[0].RegisterSpace    = 0;
    samplers[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    samplers[1]                = samplers[0];
    samplers[1].AddressU = samplers[1].AddressV = samplers[1].AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samplers[1].ShaderRegister = 1;

    D3D12_ROOT_SIGNATURE_DESC rsDesc = {};
    rsDesc.NumParameters     = static_cast<UINT>(params.size());
    rsDesc.pParameters       = params.empty() ? nullptr : params.data();
    rsDesc.NumStaticSamplers = 2;
    rsDesc.pStaticSamplers   = samplers;
    rsDesc.Flags             = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> sigBlob, errBlob;
    HRESULT hr = D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sigBlob, &errBlob);
    if (FAILED(hr))
    {
        Logf(kUnityLogTypeError, "ComputeShader: D3D12SerializeRootSignature failed (hr=0x%08X): %s",
             hr, errBlob ? (char*)errBlob->GetBufferPointer() : "");
        return false;
    }

    hr = m_device->CreateRootSignature(0,
        sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(), IID_PPV_ARGS(&m_rootSig));
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
    b.boundStride   = 0;
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
// Descriptor management
// ---------------------------------------------------------------------------
void ComputeShader::FreeAllAllocations()
{
    if (!m_allocator) return;
    if (m_srvAllocBase != kInvalidAlloc && m_numSRV > 0) { m_allocator->Free(m_srvAllocBase, m_numSRV); m_srvAllocBase = kInvalidAlloc; }
    if (m_uavAllocBase != kInvalidAlloc && m_numUAV > 0) { m_allocator->Free(m_uavAllocBase, m_numUAV); m_uavAllocBase = kInvalidAlloc; }
}

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

void ComputeShader::UpdateDescriptors()
{
    // SRV descriptors
    if (m_srvAllocBase != kInvalidAlloc)
    {
        for (const auto& b : m_bindings)
        {
            if (b.type != ComputeBindingType::SRV) continue;
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

    // UAV descriptors
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
// ---------------------------------------------------------------------------
void ComputeShader::Dispatch(
    ID3D12GraphicsCommandList* cmdList,
    UINT threadGroupX, UINT threadGroupY, UINT threadGroupZ)
{
    if (!m_pso || !m_rootSig || !m_allocator) return;

    if ((m_numSRV > 0 && m_srvAllocBase == kInvalidAlloc) ||
        (m_numUAV > 0 && m_uavAllocBase == kInvalidAlloc))
    {
        if (!AllocateAndWriteDescriptors()) return;
    }
    else
    {
        UpdateDescriptors();
    }

    ID3D12DescriptorHeap* heaps[1] = { m_allocator->GetHeap() };
    cmdList->SetDescriptorHeaps(1, heaps);

    cmdList->SetPipelineState(m_pso.Get());
    cmdList->SetComputeRootSignature(m_rootSig.Get());

    if (m_rootParamSRV != kInvalidAlloc && m_srvAllocBase != kInvalidAlloc)
        cmdList->SetComputeRootDescriptorTable(m_rootParamSRV,
            m_allocator->GetGPUHandle(m_srvAllocBase));

    if (m_rootParamUAV != kInvalidAlloc && m_uavAllocBase != kInvalidAlloc)
        cmdList->SetComputeRootDescriptorTable(m_rootParamUAV,
            m_allocator->GetGPUHandle(m_uavAllocBase));

    for (const auto& b : m_bindings)
    {
        if (b.type != ComputeBindingType::SRV_ARRAY) continue;
        if (b.rootParam == kInvalidAlloc) continue;
        if (b.boundBT)
            cmdList->SetComputeRootDescriptorTable(b.rootParam, b.boundBT->GetGPUHandle());
        else if (b.boundBB)
            cmdList->SetComputeRootDescriptorTable(b.rootParam, b.boundBB->GetGPUHandle());
    }

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

    cmdList->Dispatch(threadGroupX, threadGroupY, threadGroupZ);
}
