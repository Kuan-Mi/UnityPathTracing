#include "RenderSystem.h"

#define LOG(msg) UNITY_LOG(s_Log, msg)

RenderSystem& RenderSystem::Get()
{
    static RenderSystem instance;
    return instance;
}

void RenderSystem::Initialize(IUnityInterfaces* interfaces)
{
    if (m_are_resources_initialized)
        return;

    m_UnityInterfaces = interfaces;
    s_d3d12 = interfaces->Get<IUnityGraphicsD3D12v7>();
    s_Log = interfaces->Get<IUnityLog>();

    device = s_d3d12->GetDevice();

    if (device == nullptr)
    {
        LOG("[Bindless] D3D12 device is null, skipping NRI initialization.");
        return;
    }

    nri::DeviceCreationD3D12Desc deviceDesc = {};
    deviceDesc.d3d12Device = device;
    deviceDesc.disableD3D12EnhancedBarriers = true;
    deviceDesc.enableNRIValidation = true;


    nri::Result result = nriCreateDeviceFromD3D12Device(deviceDesc, m_NriDevice);
    if (result != nri::Result::SUCCESS)
    {
        LOG("[Bindless] Failed to create NRI device from D3D12");
        return;
    }

    nriGetInterface(*m_NriDevice, NRI_INTERFACE(nri::CoreInterface), &m_NriCore);
    nriGetInterface(*m_NriDevice, NRI_INTERFACE(nri::WrapperD3D12Interface), &m_NriWrapper);
    nriGetInterface(*m_NriDevice, NRI_INTERFACE(nri::UpscalerInterface), &m_NriUpScaler);

    UnityGraphicsD3D12PhysicalVideoMemoryControlValues control_values;
    control_values.reservation = 64000000;
    control_values.systemMemoryThreshold = 64000000;
    control_values.residencyHysteresisThreshold = 128000000;
    control_values.nonEvictableRelativeThreshold = 0.25;
    s_d3d12->SetPhysicalVideoMemoryControlValues(&control_values);

    m_are_resources_initialized = true;

    LOG("[Bindless] RenderSystem Initialized.");
}

void RenderSystem::Shutdown()
{
    if (!m_are_resources_initialized)
        return;

    if (m_NriDevice)
    {
        nriDestroyDevice(m_NriDevice);
        m_NriDevice = nullptr;
    }

    m_NriCore = {};
    m_NriWrapper = {};

    m_are_resources_initialized = false;

    LOG("[Bindless] RenderSystem Shutdown completed.");
}

void RenderSystem::Release(nri::Texture* texture)
{
    if (texture)
    {
        m_NriCore.DestroyTexture(texture);
    }
}

RenderSystem::RenderSystem()
= default;

RenderSystem::~RenderSystem()
{

}

// 处理设备事件
void RenderSystem::ProcessDeviceEvent(UnityGfxDeviceEventType type, IUnityInterfaces* interfaces)
{
    switch (type)
    {
    case kUnityGfxDeviceEventInitialize:
        s_d3d12 = interfaces->Get<IUnityGraphicsD3D12v7>();
        s_Log = interfaces->Get<IUnityLog>();

        LOG("[Bindless] ProcessDeviceEvent kUnityGfxDeviceEventInitialize");

        // 检查D3D12设备是否就绪，Asset Import Worker中设备为null，跳过ConfigureEvent避免崩溃
        if (s_d3d12->GetDevice() == nullptr)
            break;

        UnityD3D12PluginEventConfig config_1;
        config_1.graphicsQueueAccess = kUnityD3D12GraphicsQueueAccess_DontCare;
        
        config_1.flags = kUnityD3D12EventConfigFlag_ModifiesCommandBuffersState ;
        
        config_1.ensureActiveRenderTextureIsBound = false;
        s_d3d12->ConfigureEvent(1, &config_1);


        // initialize_and_create_resources();
        break;
    case kUnityGfxDeviceEventShutdown:
        LOG("[Bindless] ProcessDeviceEvent kUnityGfxDeviceEventShutdown");
        // release_resources();
        break;
    case kUnityGfxDeviceEventBeforeReset:
    case kUnityGfxDeviceEventAfterReset:
        break;
    }
}

nri::Texture* RenderSystem::WrapD3D12Texture(ID3D12Resource* resource, DXGI_FORMAT format) const
{
    nri::TextureD3D12Desc desc;
    desc.d3d12Resource = resource;
    desc.format = format;

    nri::Texture* nriTexture = nullptr;
    m_NriWrapper.CreateTextureD3D12(*RenderSystem::Get().GetNriDevice(), desc, nriTexture);
    return nriTexture;
}

nri::Buffer* RenderSystem::WrapD3D12Buffer(ID3D12Resource* resource, uint32_t structureStride) const
{
    nri::BufferD3D12Desc desc = {};
    desc.d3d12Resource = resource;
    desc.structureStride = structureStride;

    nri::Buffer* nriBuffer = nullptr;
    m_NriWrapper.CreateBufferD3D12(*RenderSystem::Get().GetNriDevice(), desc, nriBuffer);
    return nriBuffer;
}
