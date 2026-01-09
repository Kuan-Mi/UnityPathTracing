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


    std::unique_ptr<rtxdi::ReSTIRDIContext> m_restirDIContext;


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

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API CreateReSTIRDIContext(int width, int height)
{
    rtxdi::ReSTIRDIStaticParameters contextParams;
    contextParams.RenderWidth = width;
    contextParams.RenderHeight = height;

    // 初始化ReSTIR-DI上下文
    m_restirDIContext = std::make_unique<rtxdi::ReSTIRDIContext>(contextParams);
}

const rtxdi::ReSTIRDIStaticParameters* UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetStaticParameters()
{
    if (m_restirDIContext)
    {
        // 返回内部对象的地址
        return &m_restirDIContext->GetStaticParameters();
    }
    // 如果上下文还没创建，返回 nullptr，防止 C# 端读乱码或崩溃
    return nullptr;
}
}
