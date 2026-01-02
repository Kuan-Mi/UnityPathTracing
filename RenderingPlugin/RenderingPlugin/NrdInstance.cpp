#include "NrdInstance.h"
#include "RenderSystem.h"

#undef  max
#undef  min
#include "NRDIntegration.hpp"

#define LOG(msg) UNITY_LOG(s_Log, msg)

NrdInstance::NrdInstance(IUnityInterfaces* interfaces, int instanceId)
{
    s_d3d12 = interfaces->Get<IUnityGraphicsD3D12v8>();
    s_Log = interfaces->Get<IUnityLog>();
    id = instanceId;

    initialize_and_create_resources();
}

NrdInstance::~NrdInstance()
{
    release_resources();
}

static bool IsUAVAccess(nri::AccessBits access)
{
    // 定义所有映射到 D3D12_RESOURCE_STATE_UNORDERED_ACCESS 的 NRI 位
    constexpr nri::AccessBits uavBits =
        nri::AccessBits::SHADER_RESOURCE_STORAGE |
        nri::AccessBits::SCRATCH_BUFFER |
        nri::AccessBits::CLEAR_STORAGE |
        nri::AccessBits::ACCELERATION_STRUCTURE_READ |
        nri::AccessBits::ACCELERATION_STRUCTURE_WRITE |
        nri::AccessBits::MICROMAP_READ |
        nri::AccessBits::MICROMAP_WRITE;

    return (access & uavBits) != 0;
}

static D3D12_RESOURCE_STATES GetResourceStates(nri::AccessBits accessBits, D3D12_COMMAND_LIST_TYPE commandListType)
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


void NrdInstance::DispatchCompute(FrameData* data)
{
    if (data == nullptr)
        return;
    if (data->width == 0 || data->height == 0)
    {
        LOG(("[NRD Native] id:" + std::to_string(id) + " - Invalid texture size, skipping dispatch.").c_str());
        return;
    }

    UnityGraphicsD3D12RecordingState recording_state;
    if (!s_d3d12->CommandRecordingState(&recording_state))
        return;

    nri::CommandBufferD3D12Desc cmdDesc;
    cmdDesc.d3d12CommandList = recording_state.commandList;
    cmdDesc.d3d12CommandAllocator = nullptr;

    nri::CommandBuffer* nriCmdBuffer = nullptr;
    RenderSystem::Get().GetNriWrapper().CreateCommandBufferD3D12(*RenderSystem::Get().GetNriDevice(), cmdDesc, nriCmdBuffer);
    
    
    if (TextureWidth != data->width || TextureHeight != data->height)
    {
        if (TextureWidth == 0 || TextureHeight == 0)
        {
            LOG(("[NRD Native] id:" + std::to_string(id) + " - Creating NRD instance for the first time.").c_str());
        }
        else
        {
            LOG(("[NRD Native] id:" + std::to_string(id) + " - Texture size changed, recreating NRD instance.").c_str());
        }

        TextureWidth = data->width;
        TextureHeight = data->height;

        
        RenderSystem& rs = RenderSystem::Get();

        const nri::DeviceDesc& nrideviceDesc = rs.GetNriCore().GetDeviceDesc(*rs.GetNriDevice());

        LOG(("[NRD Native] NRI Device created. VendorID: " + std::to_string((int)nrideviceDesc.adapterDesc.vendor) + ", rayTracing: " + std::to_string(nrideviceDesc.features.rayTracing)).c_str());


        if (rs.GetNriUpScaler().IsUpscalerSupported(*rs.GetNriDevice(), nri::UpscalerType::DLRR))
        {
            LOG("[NRD Native] DLRR Upscaler is supported.");
        }
        else
        {
            LOG("[NRD Native] DLRR Upscaler is NOT supported.");
        }

        nri::UpscalerMode mode = nri::UpscalerMode::NATIVE;

        nri::UpscalerBits upscalerFlags = nri::UpscalerBits::DEPTH_INFINITE;

        upscalerFlags |= nri::UpscalerBits::DEPTH_INVERTED;

        nri::UpscalerDesc upscalerDesc = {};
        upscalerDesc.upscaleResolution = {(nri::Dim_t) TextureWidth, (nri::Dim_t)TextureHeight};
        upscalerDesc.type = nri::UpscalerType::DLRR;
        upscalerDesc.mode = mode;
        upscalerDesc.flags = upscalerFlags;
        upscalerDesc.commandBuffer = nriCmdBuffer;

        nri::Result r = rs.GetNriUpScaler().CreateUpscaler(*rs.GetNriDevice(), upscalerDesc, m_DLRR);
        if (r != nri::Result::SUCCESS)
        {
            LOG(("[NRD Native] Failed to create DLRR Upscaler . Error code: " + std::to_string(static_cast<int>(r))).c_str());
        }else
        {
            LOG("[NRD Native] DLRR Upscaler created successfully.");
        }

        
        
        CreateNrd();
        frameIndex = 0;
    }

    // LOG(("[NRD Native] id:" + std::to_string(id) + " - Dispatching NRD compute for frame index " + std::to_string(data->commonSettings.frameIndex) + ".").c_str());


    data->commonSettings.frameIndex = frameIndex;
    frameIndex++;

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
        r.state.layout = static_cast<nri::Layout>(input.state.layout);
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

        bool isUAV = IsUAVAccess(res.state.access);

        s_d3d12->NotifyResourceState(rawResource, state, isUAV);
    }
    
    
    nri::DispatchUpscaleDesc dispatchUpscaleDesc = {};
    // dispatchUpscaleDesc.output = {Get(Texture::DlssOutput), Get(Descriptor::DlssOutput_StorageTexture)};
    // dispatchUpscaleDesc.input = {Get(Texture::Composed), Get(Descriptor::Composed_Texture)};
    
    
    dispatchUpscaleDesc.currentResolution = {(nri::Dim_t)data->width, (nri::Dim_t)data->height};
    
    dispatchUpscaleDesc.cameraJitter = {-data->commonSettings.cameraJitter[0], -data->commonSettings.cameraJitter[1]};
    dispatchUpscaleDesc.mvScale = {1.0f, 1.0f};
    dispatchUpscaleDesc.flags = nri::DispatchUpscaleBits::NONE;
    
    
    

    RenderSystem::Get().GetNriCore().DestroyCommandBuffer(nriCmdBuffer);
}

void NrdInstance::UpdateResources(const NrdResourceInput* resources, int count)
{
    if (!resources || count <= 0)
    {
        m_CachedResources.clear();
        return;
    }

    m_CachedResources.resize(count);

    memcpy(m_CachedResources.data(), resources, count * sizeof(NrdResourceInput));
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
        {1, nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR} // Identifier设为1，类型为 REBLUR_DIFFUSE
    };
    m_SigmaId = denoisers[0].identifier;
    m_ReblurId = denoisers[1].identifier; // [NEW] 保存 Reblur ID

    nrd::InstanceCreationDesc instanceDesc = {};
    instanceDesc.denoisers = denoisers;
    instanceDesc.denoisersNum = 2;

    nrd::Result result = m_NrdIntegration.Recreate(integrationDesc, instanceDesc, RenderSystem::Get().GetNriDevice());

    if (result != nrd::Result::SUCCESS)
    {
        LOG(("[NRD Native] id:" + std::to_string(id) + " - NRD Integration Init Failed.").c_str());
        throw std::runtime_error("NRD Integration Init Failed");
    }

    LOG(("[NRD Native] id:" + std::to_string(id) + " - NRD Instance Created/Updated.").c_str());
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

    LOG(("[NRD Native] id:" + std::to_string(id) + " - NRD Instance Released.").c_str());
}
