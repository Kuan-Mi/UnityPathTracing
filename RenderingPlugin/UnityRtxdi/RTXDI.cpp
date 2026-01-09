#include <memory>

#include "IUnityLog.h"
#include "IUnityGraphics.h"

#include <Rtxdi/DI/ReSTIRDI.h>


#define LOG(msg) UNITY_LOG(s_Logger, msg)

namespace
{
    IUnityInterfaces* s_UnityInterfaces = nullptr;
    IUnityGraphics* s_Graphics = nullptr;
    IUnityLog* s_Logger = nullptr;


    void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
    {
    }
}

extern "C" {
void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces* unityInterfaces)
{
    s_UnityInterfaces = unityInterfaces;
    // 获取IUnityGraphics接口
    s_Graphics = s_UnityInterfaces->Get<IUnityGraphics>();
    s_Logger = s_UnityInterfaces->Get<IUnityLog>();

    s_Graphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);

    LOG("[UnityRtxdi] UnityPluginLoad completed.");
}


void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload()
{
    // 取消注册图形设备事件回调
    s_Graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
    LOG("[NRD Native] UnityPluginUnload completed.");
}

UNITY_INTERFACE_EXPORT rtxdi::ReSTIRDIContext* UNITY_INTERFACE_API CreateReSTIRDIContext(int width, int height)
{
    rtxdi::ReSTIRDIStaticParameters contextParams;
    contextParams.RenderWidth = width;
    contextParams.RenderHeight = height;

    // 创建并在堆上分配对象，将指针返回给 C#
    // C# 端需要负责保存这个指针，并在不再使用时调用对应的 Destroy 函数（如果有的话）来释放内存
    return new rtxdi::ReSTIRDIContext(contextParams);
}

UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API DestroyReSTIRDIContext(rtxdi::ReSTIRDIContext* context)
{
    if (context)
    {
        delete context;
    }
}

UNITY_INTERFACE_EXPORT const rtxdi::ReSTIRDIStaticParameters* UNITY_INTERFACE_API GetStaticParameters(rtxdi::ReSTIRDIContext* context)
{
    if (context)
    {
        return &context->GetStaticParameters();
    }
    return nullptr;
}
}
