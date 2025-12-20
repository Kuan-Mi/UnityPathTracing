#include "NrdInstance.h"
#include "RenderSystem.h"

#undef  max
#undef  min
#include "NRDIntegration.hpp"

#define LOG(msg) UNITY_LOG(s_Log, msg)

NrdInstance::NrdInstance(IUnityInterfaces* interfaces)
{
    s_d3d12 = interfaces->Get<IUnityGraphicsD3D12v8>();
    s_Log = interfaces->Get<IUnityLog>();

    initialize_and_create_resources();
}

NrdInstance::~NrdInstance()
{
    release_resources();
}
inline bool IsUAVAccess(nri::AccessBits access) {
    // 定义所有映射到 D3D12_RESOURCE_STATE_UNORDERED_ACCESS 的 NRI 位
    const nri::AccessBits uavBits = 
        nri::AccessBits::SHADER_RESOURCE_STORAGE | 
        nri::AccessBits::SCRATCH_BUFFER | 
        nri::AccessBits::CLEAR_STORAGE |
        nri::AccessBits::ACCELERATION_STRUCTURE_READ |
        nri::AccessBits::ACCELERATION_STRUCTURE_WRITE |
        nri::AccessBits::MICROMAP_READ |
        nri::AccessBits::MICROMAP_WRITE;

    return (access & uavBits) != 0;
}

static inline D3D12_RESOURCE_STATES GetResourceStates(nri::AccessBits accessBits, D3D12_COMMAND_LIST_TYPE commandListType)
{
    D3D12_RESOURCE_STATES resourceStates = D3D12_RESOURCE_STATE_COMMON;

    if (accessBits & nri::AccessBits::INDEX_BUFFER)
        resourceStates |= D3D12_RESOURCE_STATE_INDEX_BUFFER;

    if (accessBits & (nri::AccessBits::CONSTANT_BUFFER | nri::AccessBits::VERTEX_BUFFER))
        resourceStates |= D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;

    if (accessBits & nri::AccessBits::ARGUMENT_BUFFER)
        resourceStates |= D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;

    if (accessBits & nri::AccessBits::COLOR_ATTACHMENT)
        resourceStates |= D3D12_RESOURCE_STATE_RENDER_TARGET;

    if (accessBits & nri::AccessBits::SHADING_RATE_ATTACHMENT)
        resourceStates |= D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE;

    if (accessBits & nri::AccessBits::DEPTH_STENCIL_ATTACHMENT_READ)
        resourceStates |= D3D12_RESOURCE_STATE_DEPTH_READ;

    if (accessBits & nri::AccessBits::DEPTH_STENCIL_ATTACHMENT_WRITE)
        resourceStates |= D3D12_RESOURCE_STATE_DEPTH_WRITE;

    if (accessBits & (nri::AccessBits::ACCELERATION_STRUCTURE_READ | nri::AccessBits::MICROMAP_READ))
        resourceStates |= D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

    if (accessBits & (nri::AccessBits::ACCELERATION_STRUCTURE_WRITE | nri::AccessBits::MICROMAP_WRITE))
        resourceStates |= D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

    if (accessBits & nri::AccessBits::SHADER_RESOURCE)
    {
        resourceStates |= D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        if (commandListType == D3D12_COMMAND_LIST_TYPE_DIRECT)
            resourceStates |= D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    }

    if (accessBits & nri::AccessBits::SHADER_BINDING_TABLE)
        resourceStates |= D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

    if (accessBits & (nri::AccessBits::SHADER_RESOURCE_STORAGE | nri::AccessBits::SCRATCH_BUFFER | nri::AccessBits::CLEAR_STORAGE))
        resourceStates |= D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

    if (accessBits & nri::AccessBits::COPY_SOURCE)
        resourceStates |= D3D12_RESOURCE_STATE_COPY_SOURCE;

    if (accessBits & nri::AccessBits::COPY_DESTINATION)
        resourceStates |= D3D12_RESOURCE_STATE_COPY_DEST;

    if (accessBits & nri::AccessBits::RESOLVE_SOURCE)
        resourceStates |= D3D12_RESOURCE_STATE_RESOLVE_SOURCE;

    if (accessBits & nri::AccessBits::RESOLVE_DESTINATION)
        resourceStates |= D3D12_RESOURCE_STATE_RESOLVE_DEST;

    return resourceStates;
}


void NrdInstance::DispatchCompute(const FrameData* data)
{
    if (data == nullptr)
        return;
    if (data->width == 0 || data->height == 0)
    {
        LOG("[NRD Native] Invalid texture dimensions");
        return;
    }

    UnityGraphicsD3D12RecordingState recording_state;
    if (!s_d3d12->CommandRecordingState(&recording_state))
        return;

    if (TextureWidth != data->width || TextureHeight != data->height)
    {
        if (TextureWidth == 0 || TextureHeight == 0)
        {
            LOG("[NRD Native] Creating NRD instance.");
        }
        else
        {
            LOG("[NRD Native] Input texture size changed, recreating NRD instance.");
        }

        if (data->commonSettings.frameIndex != 0)
        {
            LOG("[NRD Native] Warning: data->commonSettings.frameIndex != 0");
        }

        TextureWidth = data->width;
        TextureHeight = data->height;

        CreateNrd();
    }

    // LOG(("[NRD Native] Dispatching Frame : " + std::to_string(data->commonSettings.frameIndex)).c_str());

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

    for (const auto& input : m_CachedResources)
    {
        if (input.texture == nullptr) continue;

        uint64_t nativeHandle = RenderSystem::Get().GetNriCore().GetTextureNativeObject(input.texture);
        ID3D12Resource* rawResource = reinterpret_cast<ID3D12Resource*>(nativeHandle);
        auto state = GetResourceStates(input.state.accessBits, D3D12_COMMAND_LIST_TYPE_DIRECT);
        s_d3d12->RequestResourceState(rawResource, state);

        nrd::Resource r = {};
        r.nri.texture = input.texture;
        r.state.access = input.state.accessBits;
        r.state.layout = (nri::Layout)input.state.layout;
        r.state.stages = input.state.stageBits;

        snapshot.SetResource(input.type, r);
    }

    const nrd::Identifier denoisers[] = {m_SigmaId, m_ReblurId};

    m_NrdIntegration.Denoise(denoisers, 2, *nriCmdBuffer, snapshot);

    for (size_t i = 0; i < snapshot.uniqueNum; i++)
    {
        nrd::Resource& res = snapshot.unique[i];
        uint64_t nativeHandle = RenderSystem::Get().GetNriCore().GetTextureNativeObject(res.nri.texture);
        ID3D12Resource* rawResource = reinterpret_cast<ID3D12Resource*>(nativeHandle);
        auto state = GetResourceStates(res.state.access, D3D12_COMMAND_LIST_TYPE_DIRECT);
        
        bool isUAV = IsUAVAccess (res.state.access);
        
        s_d3d12->NotifyResourceState(rawResource, state, isUAV);
    }

    RenderSystem::Get().GetNriCore().DestroyCommandBuffer(nriCmdBuffer);
}

void NrdInstance::UpdateResources(NrdResourceInput* resources, int count)
{
    if (!resources || count <= 0)
    {
        m_CachedResources.clear();
        return;
    }

    m_CachedResources.resize(count);

    memcpy(m_CachedResources.data(), resources, count * sizeof(NrdResourceInput));

    LOG(("[NRD Native] Updated NRD Resources. Count: " + std::to_string(count)).c_str());
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
        LOG("[NRD Native] Failed to initialize NRD Integration with NRI Wrapper");
        throw std::runtime_error("NRD Integration Init Failed");
    }

    LOG("[NRD Native] NRD Integration created successfully.");
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

    LOG("[NRD Native] NrdInstance resources released.");
}
