#include "NrdInstance.h"
#include "RenderSystem.h"

#undef  max
#undef  min
#include "NRDIntegration.hpp"

#define LOG(msg) UNITY_LOG(s_Log, msg)

NrdInstance::NrdInstance(IUnityInterfaces* interfaces)
{
    s_d3d12 = interfaces->Get<IUnityGraphicsD3D12v7>();
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

    LOG(("Dispatching Frame : " + std::to_string(data->commonSettings.frameIndex)).c_str());

    nri::CommandBufferD3D12Desc cmdDesc;
    cmdDesc.d3d12CommandList = recording_state.commandList;
    cmdDesc.d3d12CommandAllocator = nullptr;

    nri::CommandBuffer* nriCmdBuffer = nullptr;
    RenderSystem::Get().GetNriWrapper().CreateCommandBufferD3D12(*RenderSystem::Get().GetNriDevice(), cmdDesc, nriCmdBuffer);

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
    nri::AccessLayoutStage rtState = {
        nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::UNDEFINED, nri::StageBits::FRAGMENT_SHADER
    };
    nri::AccessLayoutStage commonState = {
        nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE
    };

    AddResource(nrd::ResourceType::IN_MV, data->nriMv, data->commonSettings.frameIndex == 0 ? uavState : srvState);
    AddResource(nrd::ResourceType::IN_NORMAL_ROUGHNESS, data->nriNormalRoughness, data->commonSettings.frameIndex == 0 ? uavState : srvState);
    AddResource(nrd::ResourceType::IN_VIEWZ, data->nriViewZ, data->commonSettings.frameIndex == 0 ? uavState : srvState);
    AddResource(nrd::ResourceType::IN_PENUMBRA, data->nriPenumbra, data->commonSettings.frameIndex == 0 ? uavState : srvState);
    AddResource(nrd::ResourceType::OUT_SHADOW_TRANSLUCENCY, data->nriShadowTranslucency, uavState);
    AddResource(nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST, data->nriDiffRadiance, data->commonSettings.frameIndex == 0 ? rtState : srvState);
    AddResource(nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST, data->nriOutDiffRadiance, data->commonSettings.frameIndex == 0 ? rtState : uavState);
    AddResource(nrd::ResourceType::OUT_VALIDATION, data->nriValidation, commonState);

    const nrd::Identifier denoisers[] = {m_SigmaId, m_ReblurId};

    D3D12_RESOURCE_BARRIER barrier;
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = data->validationPointer;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;

    recording_state.commandList->ResourceBarrier(1, &barrier);

    m_NrdIntegration.Denoise(denoisers, 2, *nriCmdBuffer, snapshot);

    D3D12_RESOURCE_BARRIER barrier2;
    barrier2.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier2.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier2.Transition.pResource = data->validationPointer;
    barrier2.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier2.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier2.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

    recording_state.commandList->ResourceBarrier(1, &barrier2);

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

    m_NrdIntegration.Destroy();

    m_are_resources_initialized = false;

    LOG("NrdInstance resources released.");
}
