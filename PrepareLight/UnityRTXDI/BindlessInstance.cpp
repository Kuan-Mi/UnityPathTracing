#include "BindlessInstance.h"
#include "RenderSystem.h"

using namespace nri;

#undef  max
#undef  min

#define LOG(msg) UNITY_LOG(s_Log, msg)

const char* g_ComputeShaderSource = R"(
struct PrimitiveData
{
    float2 uv0;
    float2 uv1;
    float2 uv2;

    float3 pos0; 
    float3 pos1; 
    float3 pos2; 

    uint instanceId;
};


struct InstanceData
{
    float4x4 transform;
    float3 emissiveColor;
    uint emissiveTextureIndex;
};


struct TriangleLight
{
    float3 base;
    float3 edge1;
    float3 edge2;
    float3 radiance;
    float3 normal;
    float surfaceArea;
};


struct RAB_LightInfo
{
    // uint4[0]
    float3 center;
    uint colorTypeAndFlags; // RGB8 + uint8 (see the kPolymorphicLight... constants above)

    // uint4[1]
    uint direction1; // oct-encoded
    uint direction2; // oct-encoded
    uint scalars; // 2x float16
    uint logRadiance; // uint16

    // uint4[2] -- optional, contains only shaping data
    uint iesProfileIndex;
    uint primaryAxis; // oct-encoded
    uint cosConeAngleAndSoftness; // 2x float16
    uint padding;
};

struct PrepareLightsConstants
{
    uint numTasks;
};

// Push Constants

ConstantBuffer<PrepareLightsConstants> g_Const : register(b0);

// Outputs
RWStructuredBuffer<RAB_LightInfo> u_LightDataBuffer : register(u0);

// Inputs (Adapted to C# buffers)

StructuredBuffer<PrimitiveData> t_PrimitiveData : register(t0); // 绑定到 slot t0
StructuredBuffer<InstanceData> t_InstanceData : register(t2); // 绑定到 slot t2

// Textures

Texture2D t_BindlessTextures[] : register(t0, space1);

SamplerState s_MaterialSampler : register(s0);



float2 octWrap(float2 v)
{
    #if __HLSL_VERSION >= 2021 || __SLANG__
    return (1.f - abs(v.yx)) * select(v.xy >= 0.f, 1.f, -1.f);
    #else
    return (1.f - abs(v.yx)) * (v.xy >= 0.f ? 1.f : -1.f);
    #endif
}

float2 ndirToOctSigned(float3 n)
{
    // Project the sphere onto the octahedron (|x|+|y|+|z| = 1) and then onto the xy-plane
    float2 p = n.xy * (1.f / (abs(n.x) + abs(n.y) + abs(n.z)));
    return (n.z < 0.f) ? octWrap(p) : p;
}

uint ndirToOctUnorm32(float3 n)
{
    float2 p = ndirToOctSigned(n);
    p = saturate(p.xy * 0.5 + 0.5);
    return uint(p.x * 0xfffe) | (uint(p.y * 0xfffe) << 16);
}

enum PolymorphicLightType
{
    kSphere = 0,
    kCylinder,
    kDisk,
    kRect,
    kTriangle,
    kDirectional,
    kEnvironment,
    kPoint
};

static const uint kPolymorphicLightTypeShift = 24;
static const float kPolymorphicLightMinLog2Radiance = -8.f;
static const float kPolymorphicLightMaxLog2Radiance = 40.f;
#define uint32_t uint


float unpackLightRadiance(uint logRadiance)
{
    return (logRadiance == 0) ? 0 : exp2((float(logRadiance - 1) / 65534.0) * (kPolymorphicLightMaxLog2Radiance - kPolymorphicLightMinLog2Radiance) + kPolymorphicLightMinLog2Radiance);
}

// Pack [0.0, 1.0] float to a uint of a given bit depth
#define PACK_UFLOAT_TEMPLATE(size)                      \
uint Pack_R ## size ## _UFLOAT(float r, float d = 0.5f) \
{                                                       \
const uint mask = (1U << size) - 1U;                \
\
return (uint)floor(r * mask + d) & mask;            \
}                                                       \
\
float Unpack_R ## size ## _UFLOAT(uint r)               \
{                                                       \
const uint mask = (1U << size) - 1U;                \
\
return (float)(r & mask) / (float)mask;             \
}

PACK_UFLOAT_TEMPLATE(8)

uint Pack_R8G8B8_UFLOAT(float3 rgb, float3 d = float3(0.5f, 0.5f, 0.5f))
{
    uint r = Pack_R8_UFLOAT(rgb.r, d.r);
    uint g = Pack_R8_UFLOAT(rgb.g, d.g) << 8;
    uint b = Pack_R8_UFLOAT(rgb.b, d.b) << 16;
    return r | g | b;
}


void packLightColor(float3 radiance, inout RAB_LightInfo lightInfo)
{   
    float intensity = max(radiance.r, max(radiance.g, radiance.b));

    if (intensity > 0.0)
    {
        float logRadiance = saturate((log2(intensity) - kPolymorphicLightMinLog2Radiance) 
            / (kPolymorphicLightMaxLog2Radiance - kPolymorphicLightMinLog2Radiance));
        uint packedRadiance = min(uint32_t(ceil(logRadiance * 65534.0)) + 1, 0xffffu);
        float unpackedRadiance = unpackLightRadiance(packedRadiance);

        float3 normalizedRadiance = saturate(radiance.rgb / unpackedRadiance.xxx);

        lightInfo.logRadiance |= packedRadiance;
        lightInfo.colorTypeAndFlags |= Pack_R8G8B8_UFLOAT(normalizedRadiance);
    }
}


RAB_LightInfo Store(TriangleLight triLight)
{
    RAB_LightInfo lightInfo = (RAB_LightInfo)0;


    packLightColor(triLight.radiance, lightInfo);
    lightInfo.center = triLight.base + (triLight.edge1 + triLight.edge2) / 3.0;
    lightInfo.direction1 = ndirToOctUnorm32(normalize(triLight.edge1));
    lightInfo.direction2 = ndirToOctUnorm32(normalize(triLight.edge2));
    lightInfo.scalars = f32tof16(length(triLight.edge1)) | (f32tof16(length(triLight.edge2)) << 16);
    lightInfo.colorTypeAndFlags |= uint(PolymorphicLightType::kTriangle) << kPolymorphicLightTypeShift;
        
    return lightInfo;
}

[numthreads(256, 1, 1)]
void main(uint dispatchThreadId : SV_DispatchThreadID)
{
    uint triangleIdx = dispatchThreadId;

    if (triangleIdx >= g_Const.numTasks)
        return;

    PrimitiveData prim = t_PrimitiveData[triangleIdx];

    InstanceData instance = t_InstanceData[prim.instanceId];

    float3 positions[3];
    positions[0] = mul(instance.transform, float4(prim.pos0, 1.0)).xyz;
    positions[1] = mul(instance.transform, float4(prim.pos1, 1.0)).xyz;
    positions[2] = mul(instance.transform, float4(prim.pos2, 1.0)).xyz;

    float3 radiance = instance.emissiveColor;

    {
        Texture2D emissiveTexture = t_BindlessTextures[NonUniformResourceIndex(instance.emissiveTextureIndex)];

        float2 uvs[3];
        uvs[0] = float2(prim.uv0);
        uvs[1] = float2(prim.uv1);
        uvs[2] = float2(prim.uv2);


        // Calculate the triangle edges and edge lengths in UV space
        float2 edges[3];
        edges[0] = uvs[1] - uvs[0];
        edges[1] = uvs[2] - uvs[1];
        edges[2] = uvs[0] - uvs[2];

        float3 edgeLengths;
        edgeLengths[0] = length(edges[0]);
        edgeLengths[1] = length(edges[1]);
        edgeLengths[2] = length(edges[2]);

        // Find the shortest edge and the other two (longer) edges
        float2 shortEdge;
        float2 longEdge1;
        float2 longEdge2;

        if (edgeLengths[0] < edgeLengths[1] && edgeLengths[0] < edgeLengths[2])
        {
            shortEdge = edges[0];
            longEdge1 = edges[1];
            longEdge2 = edges[2];
        }
        else if (edgeLengths[1] < edgeLengths[2])
        {
            shortEdge = edges[1];
            longEdge1 = edges[2];
            longEdge2 = edges[0];
        }
        else
        {
            shortEdge = edges[2];
            longEdge1 = edges[0];
            longEdge2 = edges[1];
        }

        // Use anisotropic sampling
        float2 shortGradient = shortEdge * (2.0 / 3.0);
        float2 longGradient = (longEdge1 + longEdge2) / 3.0;

        // Sample
        float2 centerUV = (uvs[0] + uvs[1] + uvs[2]) / 3.0;
        float3 emissiveMask = emissiveTexture.SampleGrad(s_MaterialSampler, centerUV, shortGradient, longGradient).rgb;

        //emissiveMask.xy = centerUV;
        //emissiveMask.z = 0;
        radiance *= emissiveMask;
    }

    radiance.rgb = max(0, radiance.rgb);

    TriangleLight triLight;
    triLight.base = positions[0];
    triLight.edge1 = positions[1] - positions[0];
    triLight.edge2 = positions[2] - positions[0];
    triLight.radiance = radiance;

    RAB_LightInfo lightInfo = Store(triLight);

    u_LightDataBuffer[triangleIdx] = lightInfo;
}

)";


void BindlessInstance::CompileShaderRuntime(const char* source, const wchar_t* entryPoint, const wchar_t* targetProfile)
{
    ComPtr<IDxcUtils> pUtils;
    ComPtr<IDxcCompiler3> pCompiler;
    DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&pUtils));
    DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&pCompiler));

    DxcBuffer sourceBuffer;
    sourceBuffer.Ptr = source;
    sourceBuffer.Size = strlen(source);
    sourceBuffer.Encoding = DXC_CP_UTF8;

    // 编译参数
    std::vector<const wchar_t*> args;
    args.push_back(L"-E");
    args.push_back(entryPoint);
    args.push_back(L"-T");
    args.push_back(targetProfile);
    args.push_back(L"-enable-16bit-types"); 

    // 如果是 Vulkan 后端，需要加上 -spirv
    // args.push_back(L"-spirv"); 
    // args.push_back(L"-fspv-target-env=vulkan1.2");

    ComPtr<IDxcResult> pResults;
    pCompiler->Compile(&sourceBuffer, args.data(), (uint32_t)args.size(), nullptr, IID_PPV_ARGS(&pResults));

    // 检查错误
    ComPtr<IDxcBlobUtf8> pErrors = nullptr;
    pResults->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&pErrors), nullptr);
    if (pErrors && pErrors->GetStringLength() > 0)
    {
        // printf("Shader Compile Error: %s\n", pErrors->GetStringPointer());
        LOG(("[Bindless] id:" + std::to_string(id) + " - Shader Compile Error: " + std::string(pErrors->GetStringPointer())).c_str());
        return;
    }

    ComPtr<IDxcBlob> pBlob;
    pResults->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&pBlob), nullptr);

    blob = pBlob;
    size = pBlob->GetBufferSize();
    LOG(("[Bindless] id:" + std::to_string(id) + " - Shader compiled successfully, size: " + std::to_string(size) + " bytes.").c_str());

    // CompiledShader result;
    // result.blob = pBlob;
    // result.size = pBlob->GetBufferSize();
    // return result;
}


BindlessInstance::BindlessInstance(IUnityInterfaces* interfaces, int instanceId)
{
    s_d3d12 = interfaces->Get<IUnityGraphicsD3D12v8>();
    s_Log = interfaces->Get<IUnityLog>();
    id = instanceId;

    initialize_and_create_resources();
}

BindlessInstance::~BindlessInstance()
{
    release_resources();
}

void BindlessInstance::DispatchCompute(FrameData* data)
{
    if (data == nullptr || data->numPrimitives  == 0  ) return;
    if (data->primitiveBuffer == nullptr || data->instanceBuffer == nullptr || data->lightDataBuffer == nullptr) return;

    UnityGraphicsD3D12RecordingState recording_state;
    if (!s_d3d12->CommandRecordingState(&recording_state)) return;

    CommandBufferD3D12Desc cmdDesc;
    cmdDesc.d3d12CommandList = recording_state.commandList;
    cmdDesc.d3d12CommandAllocator = nullptr;

    CommandBuffer* nriCmdBuffer = nullptr;
    RenderSystem::Get().GetNriWrapper().CreateCommandBufferD3D12(*RenderSystem::Get().GetNriDevice(), cmdDesc, nriCmdBuffer);

    // Initial Setup (Run once)
    if (m_Pipeline == nullptr)
    {
        CreateDescriptors(); // Creates Pool & Output View
        CreatePipelineAndLayout(); // Compile Shader & Pipeline

        // If UpdateResources happened BEFORE Dispatch (likely), we need to apply those pending views now
        if (!m_InputTextureViews.empty() && m_BindlessDescSet)
        {
            UpdateDescriptorRangeDesc bindlessUpdate = {
                m_BindlessDescSet, 0, 0, m_InputTextureViews.data(), (uint32_t)m_InputTextureViews.size()
            };
            m_NRI->UpdateDescriptorRanges(&bindlessUpdate, 1);
        }
    }
    
    
    // --------------------------------------------------------------------------------
    // 1. Dynamic Buffer Views Creation (每一帧可能不同，或者在此处更新)
    // --------------------------------------------------------------------------------
    // 清理旧的 View (简单起见每帧创建，优化可缓存)
    if (m_PrimitiveBufferView) m_NRI->DestroyDescriptor(m_PrimitiveBufferView);
    if (m_InstanceBufferView) m_NRI->DestroyDescriptor(m_InstanceBufferView);
    if (m_LightOutputView) m_NRI->DestroyDescriptor(m_LightOutputView);
    
    
    // 获取 Native 资源指针
    nri::Buffer* pPrimBuffer =  data->primitiveBuffer; // Unity Native Pointer cast
    nri::Buffer* pInstBuffer =  data->instanceBuffer;
    nri::Buffer* pOutBuffer  =  data->lightDataBuffer;
    
    
    // 创建 Primitive Buffer View (t0) - Struct size 64
    BufferViewDesc primViewDesc = {};
    primViewDesc.buffer = pPrimBuffer;
    primViewDesc.viewType = BufferViewType::SHADER_RESOURCE;
    primViewDesc.format = Format::UNKNOWN; // Structured Buffer
    primViewDesc.offset = 0;
    primViewDesc.size = data->numPrimitives * 64; // Size in bytes
    primViewDesc.structureStride = 64; 
    m_NRI->CreateBufferView(primViewDesc, m_PrimitiveBufferView);
    
    
    // 创建 Instance Buffer View (t2) - Struct size 80
    // 注意：假设 Instance 数量足够覆盖所有 triangle 的 instanceId 引用
    // 这里简单假设 size 足够大，实际应由 FrameData 传入 InstanceCount
    BufferViewDesc instViewDesc = {};
    instViewDesc.buffer = pInstBuffer;
    instViewDesc.viewType = BufferViewType::SHADER_RESOURCE;
    instViewDesc.format = Format::UNKNOWN;
    instViewDesc.offset = 0;
    instViewDesc.size = (data->InstanceCount * 80);
    instViewDesc.structureStride = 80; 
    m_NRI->CreateBufferView(instViewDesc, m_InstanceBufferView);
    
    // 创建 Output Buffer View (u0) - Struct size 48
    BufferViewDesc outViewDesc = {};
    outViewDesc.buffer = pOutBuffer;
    outViewDesc.viewType = BufferViewType::SHADER_RESOURCE_STORAGE;
    outViewDesc.format = Format::UNKNOWN;
    outViewDesc.offset = 0;
    outViewDesc.size = data->numPrimitives * 48;
    outViewDesc.structureStride = 48;
    m_NRI->CreateBufferView(outViewDesc, m_LightOutputView);

    // 1. Transition Input Resources (All 100 textures)
    for (auto* pRes : m_InputNativeResources)
    {
        if (pRes) s_d3d12->RequestResourceState(pRes, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }

    // 2. Transition Output Resource
    uint64_t nativePrim = RenderSystem::Get().GetNriCore().GetBufferNativeObject(data->primitiveBuffer);
    uint64_t nativeInst = RenderSystem::Get().GetNriCore().GetBufferNativeObject(data->instanceBuffer);
    uint64_t nativeLight = RenderSystem::Get().GetNriCore().GetBufferNativeObject(data->lightDataBuffer);
    
    ID3D12Resource* pD3DPrim = reinterpret_cast<ID3D12Resource*>(nativePrim);
    ID3D12Resource* pD3DInst = reinterpret_cast<ID3D12Resource*>(nativeInst);
    ID3D12Resource* pOutputResource = reinterpret_cast<ID3D12Resource*>(nativeLight);
    
    
    s_d3d12->RequestResourceState(pD3DPrim, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    s_d3d12->RequestResourceState(pD3DInst, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    s_d3d12->RequestResourceState(pOutputResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    
    
    // --------------------------------------------------------------------------------
    // 3. Update Descriptor Set 0 (Global) with new Buffer Views
    // --------------------------------------------------------------------------------
    // 顺序必须与 CreatePipelineAndLayout 中的 Range 定义一致
    // Range 0: u0 (Output)
    // Range 1: t0 (Primitive)
    // Range 2: t2 (Instance)
    // Range 3: s0 (Sampler)
    
    
    Descriptor* descriptors[] = { m_LightOutputView, m_PrimitiveBufferView, m_InstanceBufferView, m_Sampler };

    frameIndex ++;
    
    uint32_t setIndex = frameIndex % 3;
    DescriptorSet* currentSet = m_GlobalDescSets[setIndex];
    
    UpdateDescriptorRangeDesc set0Updates[] = {
        {currentSet, 0, 0, &descriptors[0], 1}, // u0
        {currentSet, 1, 0, &descriptors[1], 1}, // t0
        {currentSet, 2, 0, &descriptors[2], 1}, // t2
        {currentSet, 3, 0, &descriptors[3], 1}  // s0
    };
    
    m_NRI->UpdateDescriptorRanges(set0Updates, 4);
    
    
    // 3. Setup Commands
    m_NRI->CmdSetDescriptorPool(*nriCmdBuffer, *m_DescriptorPool);
    m_NRI->CmdSetPipelineLayout(*nriCmdBuffer, nri::BindPoint::COMPUTE, *m_PipelineLayout);
    m_NRI->CmdSetPipeline(*nriCmdBuffer, *m_Pipeline);
    
    
    m_NRI->CmdSetRootConstants(*nriCmdBuffer, { 0, &data->numPrimitives, 4 }); // Push Constants if any

    m_NRI->CmdSetDescriptorSet(*nriCmdBuffer, {0, currentSet}); // Output + Sampler
    m_NRI->CmdSetDescriptorSet(*nriCmdBuffer, {1, m_BindlessDescSet}); // Inputs Array

    uint32_t groupCountX = (data->numPrimitives + 255) / 256;
    // Dispatch (Grid size 512x512 assumption from original code, adjusted for 8x8 threadgroup)
    m_NRI->CmdDispatch(*nriCmdBuffer, {groupCountX, 1, 1});

    // 4. Notify Unity of Output State
    s_d3d12->NotifyResourceState(pOutputResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);

    RenderSystem::Get().GetNriCore().DestroyCommandBuffer(nriCmdBuffer);
    
    // LOG(("[Bindless] id:" + std::to_string(id) + " - Dispatched compute for " + std::to_string(data->numPrimitives) + " primitives.").c_str());
}

void BindlessInstance::UpdateResources(const ResourceInput* resources, int count)
{
    if (!m_are_resources_initialized) return;

    // 1. Clean up old views
    for (auto* view : m_InputTextureViews)
    {
        if (view) m_NRI->DestroyDescriptor(view);
    }
    m_InputTextureViews.clear();
    m_InputNativeResources.clear();

    if (!resources || count <= 0) return;

    // Cap at 100 textures (10x10)
    int processCount = (count > 100) ? 100 : count;

    m_InputTextureViews.reserve(processCount);
    m_InputNativeResources.reserve(processCount);

    LOG(("[Bindless] id:" + std::to_string(id) + " - Updating " + std::to_string(processCount) + " input textures.").c_str());

    // 2. Create Views & Cache Resources
    for (int i = 0; i < processCount; i++)
    {
        nri::Texture* tex = resources[i].texture;
        if (!tex)
        {
            LOG(("[Bindless] id:" + std::to_string(id) + " - Warning: Input texture at index " + std::to_string(i) + " is null. Skipping.").c_str());
            m_InputTextureViews.push_back(nullptr);
            m_InputNativeResources.push_back(nullptr);
            continue;
        }

        Texture2DViewDesc srvDesc = {};
        srvDesc.texture = tex;
        srvDesc.viewType = Texture2DViewType::SHADER_RESOURCE_2D;
        srvDesc.format = resources[i].format;

        nri::Descriptor* view = nullptr;
        m_NRI->CreateTexture2DView(srvDesc, view);

        m_InputTextureViews.push_back(view);

        // Cache for barriers in Dispatch
        uint64_t nativeHandle = RenderSystem::Get().GetNriCore().GetTextureNativeObject(tex);
        m_InputNativeResources.push_back(reinterpret_cast<ID3D12Resource*>(nativeHandle));
    }

    // 3. Update the Bindless Descriptor Set IMMEDIATELY
    // Note: Assuming m_BindlessDescSet is created in Initialize/CreateDescriptors. 
    // If CreateDescriptors hasn't run yet (lazy init), this update will act as "pending" state 
    // or you must ensure Init runs before this. 
    // Here we check if set exists.
    if (m_BindlessDescSet && !m_InputTextureViews.empty())
    {
        UpdateDescriptorRangeDesc bindlessUpdate = {
            m_BindlessDescSet, 0, 0, m_InputTextureViews.data(), (uint32_t)m_InputTextureViews.size()
        };
        m_NRI->UpdateDescriptorRanges(&bindlessUpdate, 1);
        LOG(("[Bindless] id:" + std::to_string(id) + " - Updated " + std::to_string(processCount) + " input textures.").c_str());
    }
}


void BindlessInstance::CreateDescriptors()
{
    // // 输出纹理的 UAV
    // Texture2DViewDesc uavDesc = {};
    // uavDesc.texture = m_OutputTexture;
    // uavDesc.viewType = Texture2DViewType::SHADER_RESOURCE_STORAGE_2D;
    // uavDesc.format = Format::RGBA8_UNORM;
    // m_NRI->CreateTexture2DView(uavDesc, m_OutputTextureView);
    //
    // 采样器
    SamplerDesc samplerDesc = {};
    samplerDesc.filters.min = Filter::LINEAR;
    samplerDesc.filters.mag = Filter::LINEAR;
    samplerDesc.filters.mip = Filter::LINEAR; 
    
    samplerDesc.anisotropy = 16; 
    
    samplerDesc.addressModes = {AddressMode::REPEAT, AddressMode::REPEAT, AddressMode::REPEAT};
    m_NRI->CreateSampler(*m_Device, samplerDesc, m_Sampler);

    // 2. 创建 Descriptor Pool
    DescriptorPoolDesc poolDesc = {};
    poolDesc.descriptorSetMaxNum = 4;
    poolDesc.textureMaxNum = MAX_BINDLESS_TEXTURES; // 为 Bindless 数组预留空间
    poolDesc.storageBufferMaxNum = 3;
    poolDesc.bufferMaxNum = 6;
    poolDesc.samplerMaxNum = 3;
    m_NRI->CreateDescriptorPool(*m_Device, poolDesc, m_DescriptorPool);

    LOG(("[Bindless] id:" + std::to_string(id) + " - Descriptors created successfully.").c_str());
}

void BindlessInstance::CreatePipelineAndLayout()
{
    // ---------------- Descriptor Set 0: Static (UAV + Sampler) ----------------
    DescriptorRangeDesc rangeSet0[4];
    // Range 0: u0 (RWStructuredBuffer)
    rangeSet0[0] = {0, 1, DescriptorType::STORAGE_BUFFER, StageBits::COMPUTE_SHADER}; 
    // Range 1: t0 (StructuredBuffer - Primitives)
    rangeSet0[1] = {0, 1, DescriptorType::BUFFER, StageBits::COMPUTE_SHADER}; 
    // Range 2: t2 (StructuredBuffer - Instances) -> Notice baseRegisterIndex = 2
    rangeSet0[2] = {2, 1, DescriptorType::BUFFER, StageBits::COMPUTE_SHADER}; 
    // Range 3: s0 (Sampler)
    rangeSet0[3] = {0, 1, DescriptorType::SAMPLER, StageBits::COMPUTE_SHADER};

    // ---------------- Descriptor Set 1: Bindless (Texture Array) ----------------
    DescriptorRangeDesc rangeSet1[1];
    rangeSet1[0] = {
        0, MAX_BINDLESS_TEXTURES, DescriptorType::TEXTURE, StageBits::COMPUTE_SHADER,
        DescriptorRangeBits::VARIABLE_SIZED_ARRAY | DescriptorRangeBits::PARTIALLY_BOUND
    };

    DescriptorSetDesc setDescs[] = {
        {0, rangeSet0, 4},
        {1, rangeSet1, 1},
    };

    // --------------------------------------------------------------------------------
    // 2. 定义 Root Constants (Push Constants)
    // --------------------------------------------------------------------------------
    RootConstantDesc rootConstant = {};
    rootConstant.registerIndex = 0; // b0
    rootConstant.size = 4; // sizeof(uint)
    rootConstant.shaderStages = StageBits::COMPUTE_SHADER;

    // --------------------------------------------------------------------------------
    // 3. 创建 Pipeline Layout
    // --------------------------------------------------------------------------------
    PipelineLayoutDesc layoutDesc = {};
    layoutDesc.descriptorSets = setDescs;
    layoutDesc.descriptorSetNum = 2;
    layoutDesc.rootConstants = &rootConstant;
    layoutDesc.rootConstantNum = 1;
    layoutDesc.shaderStages = StageBits::COMPUTE_SHADER;

    if (m_NRI->CreatePipelineLayout(*m_Device, layoutDesc, m_PipelineLayout) != Result::SUCCESS)
    {
        throw std::runtime_error("Failed to create pipeline layout");
    }

    CompileShaderRuntime(g_ComputeShaderSource, L"main", L"cs_6_2");

    ShaderDesc computeShader = {};
    computeShader.stage = StageBits::COMPUTE_SHADER;
    computeShader.bytecode = blob->GetBufferPointer();
    computeShader.size = blob->GetBufferSize();

    // 创建计算管线
    ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.pipelineLayout = m_PipelineLayout;
    pipelineDesc.shader = computeShader;
    m_NRI->CreateComputePipeline(*m_Device, pipelineDesc, m_Pipeline);

    // 更新 Descriptor Sets
    for(int i=0; i<3; ++i) {
        m_NRI->AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 0, &m_GlobalDescSets[i], 1, 0);
    }
    // m_NRI->AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 0, &m_GlobalDescSet, 1, 0);
    m_NRI->AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &m_BindlessDescSet, 1, MAX_BINDLESS_TEXTURES);

    // Descriptor* set0Descriptors[] = {m_OutputTextureView, m_Sampler};
    // UpdateDescriptorRangeDesc set0Updates[] = {
    //     {m_GlobalDescSet, 0, 0, &set0Descriptors[0], 1}, // UAV
    //     {m_GlobalDescSet, 1, 0, &set0Descriptors[1], 1} // Sampler
    // };

    // m_NRI->UpdateDescriptorRanges(set0Updates, 2);

    LOG(("[Bindless] id:" + std::to_string(id) + " - Pipeline and Layout created successfully.").c_str());
}

void BindlessInstance::initialize_and_create_resources()
{
    if (m_are_resources_initialized)
        return;

    m_Device = RenderSystem::Get().GetNriDevice();
    m_NRI = &RenderSystem::Get().GetNriCore();

    m_are_resources_initialized = true;
}

void BindlessInstance::release_resources()
{
    if (!m_are_resources_initialized)
        return;

    for (auto* v : m_InputTextureViews)
    {
        if (v) m_NRI->DestroyDescriptor(v);
    }
    m_InputTextureViews.clear();
    m_InputNativeResources.clear();

    m_are_resources_initialized = false;

    LOG(("[Bindless] id:" + std::to_string(id) + " - Instance Released.").c_str());
}
