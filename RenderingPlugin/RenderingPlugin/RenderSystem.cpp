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
    
    s_d3d12 = interfaces->Get<IUnityGraphicsD3D12v7>();
    s_Log = interfaces->Get<IUnityLog>();

    ID3D12Device* device = s_d3d12->GetDevice();

    nri::DeviceCreationD3D12Desc deviceDesc = {};
    deviceDesc.d3d12Device = device;
    deviceDesc.disableD3D12EnhancedBarriers = true;
    deviceDesc.enableNRIValidation = true;

    nri::Result result = nriCreateDeviceFromD3D12Device(deviceDesc, m_NriDevice);
    if (result != nri::Result::SUCCESS)
    {
        LOG("Failed to create NRI device from D3D12");
        return;
    }

    nriGetInterface(*m_NriDevice, NRI_INTERFACE(nri::CoreInterface), &m_NriCore);
    nriGetInterface(*m_NriDevice, NRI_INTERFACE(nri::WrapperD3D12Interface), &m_NriWrapper);

    UnityGraphicsD3D12PhysicalVideoMemoryControlValues control_values;
    control_values.reservation = 64000000;
    control_values.systemMemoryThreshold = 64000000;
    control_values.residencyHysteresisThreshold = 128000000;
    control_values.nonEvictableRelativeThreshold = 0.25;
    s_d3d12->SetPhysicalVideoMemoryControlValues(&control_values);

    m_are_resources_initialized = true;
    
    LOG("RenderSystem Initialized.");
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

    LOG("RenderSystem Shutdown completed.");
}

RenderSystem::RenderSystem()
= default;

RenderSystem::~RenderSystem()
{
    // DestroyNrd();
}

// 处理设备事件
void RenderSystem::ProcessDeviceEvent(UnityGfxDeviceEventType type, IUnityInterfaces* interfaces)
{
    switch (type)
    {
    case kUnityGfxDeviceEventInitialize:
        s_d3d12 = interfaces->Get<IUnityGraphicsD3D12v7>();
        s_Log = interfaces->Get<IUnityLog>();

        LOG("ProcessDeviceEvent kUnityGfxDeviceEventInitialize");

        UnityD3D12PluginEventConfig config_1;
        config_1.graphicsQueueAccess = kUnityD3D12GraphicsQueueAccess_DontCare;
        config_1.flags = kUnityD3D12EventConfigFlag_SyncWorkerThreads |
            kUnityD3D12EventConfigFlag_ModifiesCommandBuffersState |
            kUnityD3D12EventConfigFlag_EnsurePreviousFrameSubmission;
        config_1.ensureActiveRenderTextureIsBound = true;
        s_d3d12->ConfigureEvent(1, &config_1);

        UnityD3D12PluginEventConfig config_2;
        config_2.graphicsQueueAccess = kUnityD3D12GraphicsQueueAccess_Allow;
        config_2.flags = kUnityD3D12EventConfigFlag_SyncWorkerThreads |
            kUnityD3D12EventConfigFlag_ModifiesCommandBuffersState |
            kUnityD3D12EventConfigFlag_EnsurePreviousFrameSubmission;
        config_2.ensureActiveRenderTextureIsBound = false;
        s_d3d12->ConfigureEvent(2, &config_2);

        // initialize_and_create_resources();
        break;
    case kUnityGfxDeviceEventShutdown:
        LOG("ProcessDeviceEvent kUnityGfxDeviceEventShutdown");
        // release_resources();
        break;
    case kUnityGfxDeviceEventBeforeReset:
    case kUnityGfxDeviceEventAfterReset:
        break;
    }
}
