#include "NrdInstance.h"
#include "RenderSystem.h"

#undef  max
#undef  min
#include "NRDIntegration.hpp"

#define LOG(msg) UNITY_LOG(s_Log, msg)

NrdInstance::NrdInstance(IUnityInterfaces* interfaces)
{
    s_d3d12 = interfaces->Get<IUnityGraphicsD3D12v7>();
    s_d3d121 = interfaces->Get<IUnityGraphicsD3D12>();
    s_Log = interfaces->Get<IUnityLog>();
    initialize_and_create_resources();
}

NrdInstance::~NrdInstance()
{
    release_resources();
}

void NrdInstance::DispatchCompute(const FrameData* data)
{
    if (data == nullptr)
        return;
    if (data->width == 0 || data->height == 0)
    {
        LOG("Invalid texture dimensions");
        return;
    }

    UnityGraphicsD3D12RecordingState recording_state;
    if (!s_d3d12->CommandRecordingState(&recording_state))
        return;

    if (TextureWidth != data->width || TextureHeight != data->height)
    {
        if (TextureWidth == 0 || TextureHeight == 0)
        {
            LOG("Creating NRD instance.");
        }
        else
        {
            LOG("Input texture size changed, recreating NRD instance.");
        }

        if (data->commonSettings.frameIndex != 0)
        {
            LOG("Warning: data->commonSettings.frameIndex != 0");
        }

        TextureWidth = data->width;
        TextureHeight = data->height;

        CreateNrd();
    }

    nri::CommandBufferD3D12Desc cmdDesc;
    cmdDesc.d3d12CommandList = recording_state.commandList;
    cmdDesc.d3d12CommandAllocator = nullptr;

    nri::CommandBuffer* nriCmdBuffer = nullptr;
    RenderSystem::Get().GetNriWrapper().CreateCommandBufferD3D12(*RenderSystem::Get().GetNriDevice(), cmdDesc,
                                                                 nriCmdBuffer);

    nri::Texture* nriMv = WrapD3D12Texture(data->mvPointer, DXGI_FORMAT_R16G16B16A16_FLOAT);
    nri::Texture* nriNormal = WrapD3D12Texture(data->normalRoughnessPointer, DXGI_FORMAT_R10G10B10A2_UNORM);
    nri::Texture* nriViewZ = WrapD3D12Texture(data->viewZPointer, DXGI_FORMAT_R32_FLOAT);
    nri::Texture* nriPENUMBRA = WrapD3D12Texture(data->penumbraPointer, DXGI_FORMAT_R16_FLOAT);
    nri::Texture* nriSHADOW_TRANSLUCENCY = WrapD3D12Texture(data->shadowTranslucencyPointer, DXGI_FORMAT_R16_FLOAT);

    nri::Texture* nriDiffRadiance = WrapD3D12Texture(data->diffRadiancePointer, DXGI_FORMAT_R16G16B16A16_FLOAT);
    nri::Texture* nriOutDiffRadiance = WrapD3D12Texture(data->outDiffRadiancePointer, DXGI_FORMAT_R16G16B16A16_FLOAT);
    nri::Texture* nriValidation = WrapD3D12Texture(data->validationPointer, DXGI_FORMAT_R8G8B8A8_UNORM);
    
    
    
    m_NrdIntegration.SetCommonSettings(data->commonSettings);
    m_NrdIntegration.SetDenoiserSettings(m_SigmaId, &data->sigmaSettings);
    m_NrdIntegration.SetDenoiserSettings(m_ReblurId, &data->reblurSettings);

    m_NrdIntegration.NewFrame();

    nrd::ResourceSnapshot snapshot = {};

    // 定义辅助 lambda 来填充 pool
    auto AddResource = [&](nrd::ResourceType type, nri::Texture* tex, nri::AccessLayoutStage state)
    {
        if (!tex) return;
        nrd::Resource r = {};
        r.nri.texture = tex;
        r.state = state; // 指定资源进入 NRD 之前的状态
        snapshot.SetResource(type, r);
    };

    nri::AccessLayoutStage srvState = {
        nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE, nri::StageBits::FRAGMENT_SHADER
    };
    nri::AccessLayoutStage uavState = {
        nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER
    };

    // D3D12_RESOURCE_STATES* outState = nullptr;
    // s_d3d121->GetResourceState(data->mvPointer, outState);
    //
    // LOG(("MV Resource State: " + std::to_string(static_cast<uint32_t>(*outState))).c_str());

    AddResource(nrd::ResourceType::IN_MV, nriMv, data->commonSettings.frameIndex == 0 ? uavState : srvState);
    AddResource(nrd::ResourceType::IN_NORMAL_ROUGHNESS, nriNormal,data->commonSettings.frameIndex == 0 ? uavState : srvState);
    AddResource(nrd::ResourceType::IN_VIEWZ, nriViewZ, data->commonSettings.frameIndex == 0 ? uavState : srvState);
    AddResource(nrd::ResourceType::IN_PENUMBRA, nriPENUMBRA,data->commonSettings.frameIndex == 0 ? uavState : srvState);
    AddResource(nrd::ResourceType::OUT_SHADOW_TRANSLUCENCY, nriSHADOW_TRANSLUCENCY, uavState);

    // [NEW] Reblur Inputs/Outputs
    // REBLUR 的 Noisy Input 必须绑定 (SRV)
    AddResource(nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST, nriDiffRadiance, data->commonSettings.frameIndex == 0 ? uavState : srvState);
    // REBLUR 的 Denoised Output (UAV)
    AddResource(nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST, nriOutDiffRadiance, uavState);
    
    AddResource(nrd::ResourceType::OUT_VALIDATION, nriValidation, uavState);
    
    
    const nrd::Identifier denoisers[] = {m_SigmaId, m_ReblurId};

    m_NrdIntegration.Denoise(denoisers, 2, *nriCmdBuffer, snapshot);

    RenderSystem::Get().GetNriCore().DestroyCommandBuffer(nriCmdBuffer);
}

void NrdInstance::CreateNrd()
{
    m_NrdIntegration.Destroy();

    // 1. 配置 NRD Integration
    nrd::IntegrationCreationDesc integrationDesc = {};
    integrationDesc.resourceWidth = static_cast<uint16_t>(TextureWidth);
    integrationDesc.resourceHeight = static_cast<uint16_t>(TextureHeight);
    integrationDesc.queuedFrameNum = kMaxFramesInFlight;
    integrationDesc.demoteFloat32to16 = true; // 可选优化
    integrationDesc.autoWaitForIdle = false;
    integrationDesc.enableWholeLifetimeDescriptorCaching = true; // 推荐开启以提高性能

    // 2. 配置 NRD Denoiser
    nrd::DenoiserDesc denoisers[] = {
        {0, nrd::Denoiser::SIGMA_SHADOW},
        {1, nrd::Denoiser::REBLUR_DIFFUSE} // Identifier设为1，类型为 REBLUR_DIFFUSE
    };
    m_SigmaId = denoisers[0].identifier;
    m_ReblurId = denoisers[1].identifier; // [NEW] 保存 Reblur ID

    nrd::InstanceCreationDesc instanceDesc = {};
    instanceDesc.denoisers = denoisers;
    instanceDesc.denoisersNum = 2;

    nrd::Result result = m_NrdIntegration.Recreate(integrationDesc, instanceDesc, RenderSystem::Get().GetNriDevice());

    if (result != nrd::Result::SUCCESS)
    {
        LOG("Failed to initialize NRD Integration with NRI Wrapper");
        throw std::runtime_error("NRD Integration Init Failed");
    }

    LOG("NRD Integration created successfully.");
}

nri::Texture* NrdInstance::WrapD3D12Texture(ID3D12Resource* resource, DXGI_FORMAT format)
{
    if (!resource) return nullptr;

    // 检查缓存
    auto it = m_NriTextureCache.find(resource);
    if (it != m_NriTextureCache.end())
    {
        return it->second;
    }

    // 使用 Wrapper 接口创建 NRI Texture
    nri::TextureD3D12Desc desc;
    desc.d3d12Resource = resource;
    desc.format = format;

    nri::Texture* nriTexture = nullptr;
    RenderSystem::Get().GetNriWrapper().CreateTextureD3D12(*RenderSystem::Get().GetNriDevice(), desc, nriTexture);

    if (nriTexture)
    {
        m_NriTextureCache[resource] = nriTexture;
    }
    else
    {
        LOG("Failed to wrap D3D12 texture into NRI texture");
    }

    return nriTexture;
}

void NrdInstance::initialize_and_create_resources()
{
    if (m_are_resources_initialized)
        return;
    m_are_resources_initialized = true;
}

void NrdInstance::release_resources()
{
    if (!m_are_resources_initialized)
        return;

    for (auto& pair : m_NriTextureCache)
    {
        if (nri::Texture* nri_texture = pair.second)
        {
            RenderSystem::Get().GetNriCore().DestroyTexture(nri_texture);
        }
    }

    m_NriTextureCache.clear();

    m_NrdIntegration.Destroy();

    m_are_resources_initialized = false;

    LOG("NrdInstance resources released.");
}
