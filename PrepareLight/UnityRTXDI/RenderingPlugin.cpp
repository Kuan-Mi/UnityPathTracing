#include <cassert>
#include <mutex>

#include "RenderSystem.h"
#include "BindlessInstance.h"
#include "Unity/IUnityLog.h"


#pragma comment(lib, "NRI.lib")
#pragma comment(lib, "dxcompiler.lib")


#define LOG(msg) UNITY_LOG(s_Logger, msg)


namespace
{
    IUnityInterfaces* s_UnityInterfaces = nullptr;
    IUnityGraphics* s_Graphics = nullptr;
    IUnityLog* s_Logger = nullptr;

    std::unordered_map<int32_t, BindlessInstance*> g_Instances;
    std::mutex g_InstanceMutex;
    int32_t g_NextInstanceId = 1;


    // 图形设备事件回调
    void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
    {
        // 初始化时，创建图形API
        if (eventType == kUnityGfxDeviceEventInitialize)
        {
            RenderSystem::Get().Initialize(s_UnityInterfaces);
        }

        // 让图形API处理与设备相关的事件
        RenderSystem::Get().ProcessDeviceEvent(eventType, s_UnityInterfaces);

        // 在关闭时清理图形API
        if (eventType == kUnityGfxDeviceEventShutdown)
        {
            std::scoped_lock lock(g_InstanceMutex);
            for (auto& pair : g_Instances) delete pair.second;
            g_Instances.clear();

            RenderSystem::Get().Shutdown();
        }
    }

    // 渲染事件和数据的回调
    void UNITY_INTERFACE_API OnRenderEventAndData(int eventID, void* data)
    {
        if (eventID == 1)
        {
            FrameData* frameData = static_cast<FrameData*>(data);

            std::scoped_lock lock(g_InstanceMutex);
            auto it = g_Instances.find(frameData->instanceId);
            if (it != g_Instances.end())
            {
                it->second->DispatchCompute(frameData);
            }
        }
    }
}

// 加载Unity插件
extern "C" {
void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces* unityInterfaces)
{
    s_UnityInterfaces = unityInterfaces;
    // 获取IUnityGraphics接口
    s_Graphics = s_UnityInterfaces->Get<IUnityGraphics>();
    s_Logger = s_UnityInterfaces->Get<IUnityLog>();
    // 注册回调以接收图形设备事件
    s_Graphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);

    // 在插件加载时手动运行OnGraphicsDeviceEvent（initialize）
    OnGraphicsDeviceEvent(kUnityGfxDeviceEventInitialize);

    LOG("[Bindless] UnityPluginLoad completed.");
}

// 卸载Unity插件
void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload()
{
    // 取消注册图形设备事件回调
    s_Graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
    LOG("[Bindless] UnityPluginUnload completed.");
}

// 获取渲染事件和数据的函数指针
UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetRenderEventAndDataFunc()
{
    return OnRenderEventAndData;
}

// C# 构造时调用
UNITY_INTERFACE_EXPORT int UNITY_INTERFACE_API CreateDenoiserInstance()
{
    std::scoped_lock lock(g_InstanceMutex);
    int id = g_NextInstanceId++;
    g_Instances[id] = new BindlessInstance(s_UnityInterfaces, id);
    return id;
}


// C# Dispose 时调用
UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API DestroyDenoiserInstance(int id)
{
    std::scoped_lock lock(g_InstanceMutex);
    auto it = g_Instances.find(id);
    if (it != g_Instances.end())
    {
        delete it->second;
        g_Instances.erase(it);
    }
}


// C# Dispose 时调用
UNITY_INTERFACE_EXPORT void* UNITY_INTERFACE_API WrapD3D12Texture(ID3D12Resource* resource, DXGI_FORMAT format)
{
    return RenderSystem::Get().WrapD3D12Texture(resource, format);
}

UNITY_INTERFACE_EXPORT void* UNITY_INTERFACE_API WrapD3D12Buffer(ID3D12Resource* resource, uint32_t structureStride)
{
    return RenderSystem::Get().WrapD3D12Buffer(resource, structureStride);
}

UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API ReleaseTexture(nri::Texture* nriTex)
{
    RenderSystem::Get().Release(nriTex);
}

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UpdateDenoiserResources(
    int instanceId,
    ResourceInput* resources,
    int count)
{
    std::scoped_lock lock(g_InstanceMutex);
    auto it = g_Instances.find(instanceId);
    if (it != g_Instances.end())
    {
        it->second->UpdateResources(resources, count);
    }
}
}
