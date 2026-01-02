#include <cassert>
#include <mutex>

#include "DLRRInstance.h"
#include "RenderSystem.h"
#include "NrdInstance.h"
#include "RRFrameData.h"
#include "Unity/IUnityLog.h"


#pragma comment(lib, "NRD.lib")
#pragma comment(lib, "NRI.lib")


#define LOG(msg) UNITY_LOG(s_Logger, msg)


namespace
{
    IUnityInterfaces* s_UnityInterfaces = nullptr;
    IUnityGraphics* s_Graphics = nullptr;
    IUnityLog* s_Logger = nullptr;

    std::unordered_map<int32_t, NrdInstance*> g_NrdInstances;
    std::mutex g_NrdInstanceMutex;
    int32_t g_NrdNextInstanceId = 1;

    std::unordered_map<int32_t, DLRRInstance*> g_DLRRInstances;
    std::mutex g_DLRRInstanceMutex;
    int32_t g_DLRRNextInstanceId = 1;


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
            std::scoped_lock lock(g_NrdInstanceMutex);
            for (auto& pair : g_NrdInstances) delete pair.second;
            g_NrdInstances.clear();

            RenderSystem::Get().Shutdown();
        }
    }

    // 渲染事件和数据的回调
    void UNITY_INTERFACE_API OnRenderEventAndData(int eventID, void* data)
    {
        if (eventID == 1)
        {
            FrameData* frameData = static_cast<FrameData*>(data);

            std::scoped_lock lock(g_NrdInstanceMutex);
            auto it = g_NrdInstances.find(frameData->instanceId);
            if (it != g_NrdInstances.end())
            {
                it->second->DispatchCompute(frameData);
            }
        }
        else if (eventID == 2)
        {
            // DLRR 事件处理（如果需要）
            RRFrameData* frameData = static_cast<RRFrameData*>(data);

            std::scoped_lock lock(g_DLRRInstanceMutex);
            auto it = g_DLRRInstances.find(frameData->instanceId);
            if (it != g_DLRRInstances.end())
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

    LOG("[NRD Native] UnityPluginLoad completed.");
}

// 卸载Unity插件
void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload()
{
    // 取消注册图形设备事件回调
    s_Graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
    LOG("[NRD Native] UnityPluginUnload completed.");
}

// 获取渲染事件和数据的函数指针
UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetRenderEventAndDataFunc()
{
    return OnRenderEventAndData;
}

// C# 构造时调用
UNITY_INTERFACE_EXPORT int UNITY_INTERFACE_API CreateDenoiserInstance()
{
    std::scoped_lock lock(g_NrdInstanceMutex);
    int id = g_NrdNextInstanceId++;
    g_NrdInstances[id] = new NrdInstance(s_UnityInterfaces, id);
    return id;
}

UNITY_INTERFACE_EXPORT int UNITY_INTERFACE_API CreateDLRRInstance()
{
    std::scoped_lock lock(g_NrdInstanceMutex);
    int id = g_NrdNextInstanceId++;
    g_DLRRInstances[id] = new DLRRInstance(s_UnityInterfaces, id);
    return id;
}

// C# Dispose 时调用
UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API DestroyDenoiserInstance(int id)
{
    std::scoped_lock lock(g_NrdInstanceMutex);
    auto it = g_NrdInstances.find(id);
    if (it != g_NrdInstances.end())
    {
        delete it->second;
        g_NrdInstances.erase(it);
    }
}

UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API DestroyDLRRInstance(int id)
{
    std::scoped_lock lock(g_DLRRInstanceMutex);
    auto it = g_DLRRInstances.find(id);
    if (it != g_DLRRInstances.end())
    {
        delete it->second;
        g_DLRRInstances.erase(it);
    }
}

// C# Dispose 时调用
UNITY_INTERFACE_EXPORT void* UNITY_INTERFACE_API WrapD3D12Texture(ID3D12Resource* resource, DXGI_FORMAT format)
{
    return RenderSystem::Get().WrapD3D12Texture(resource, format);
}

UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API ReleaseTexture(nri::Texture* nriTex)
{
    RenderSystem::Get().Release(nriTex);
}

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UpdateDenoiserResources(
    int instanceId,
    NrdResourceInput* resources,
    int count)
{
    std::scoped_lock lock(g_NrdInstanceMutex);
    auto it = g_NrdInstances.find(instanceId);
    if (it != g_NrdInstances.end())
    {
        it->second->UpdateResources(resources, count);
    }
}
}
