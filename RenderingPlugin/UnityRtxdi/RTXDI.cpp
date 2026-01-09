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
    LOG("[UnityRtxdi] UnityPluginUnload completed.");
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
    if (!context) return nullptr;
    return &context->GetStaticParameters();
}

UNITY_INTERFACE_EXPORT RTXDI_ReservoirBufferParameters UNITY_INTERFACE_API GetReservoirBufferParameters(rtxdi::ReSTIRDIContext* context)
{
    if (!context) return {};
    return context->GetReservoirBufferParameters();
}

UNITY_INTERFACE_EXPORT rtxdi::ReSTIRDI_ResamplingMode UNITY_INTERFACE_API GetResamplingMode(rtxdi::ReSTIRDIContext* context)
{
    if (!context) return rtxdi::ReSTIRDI_ResamplingMode::None;
    return context->GetResamplingMode();
}

UNITY_INTERFACE_EXPORT RTXDI_RuntimeParameters UNITY_INTERFACE_API GetRuntimeParameters(rtxdi::ReSTIRDIContext* context)
{
    if (!context) return {};
    return context->GetRuntimeParams();
}

UNITY_INTERFACE_EXPORT ReSTIRDI_BufferIndices UNITY_INTERFACE_API GetBufferIndices(rtxdi::ReSTIRDIContext* context)
{
    if (!context) return {}; // 假设默认构造函数存在
    return context->GetBufferIndices();
}

UNITY_INTERFACE_EXPORT ReSTIRDI_InitialSamplingParameters UNITY_INTERFACE_API GetInitialSamplingParameters(rtxdi::ReSTIRDIContext* context)
{
    if (!context) return {};
    return context->GetInitialSamplingParameters();
}

UNITY_INTERFACE_EXPORT ReSTIRDI_TemporalResamplingParameters UNITY_INTERFACE_API GetTemporalResamplingParameters(rtxdi::ReSTIRDIContext* context)
{
    if (!context) return {};
    return context->GetTemporalResamplingParameters();
}

UNITY_INTERFACE_EXPORT ReSTIRDI_SpatialResamplingParameters UNITY_INTERFACE_API GetSpatialResamplingParameters(rtxdi::ReSTIRDIContext* context)
{
    if (!context) return {};
    return context->GetSpatialResamplingParameters();
}

UNITY_INTERFACE_EXPORT ReSTIRDI_ShadingParameters UNITY_INTERFACE_API GetShadingParameters(rtxdi::ReSTIRDIContext* context)
{
    if (!context) return {};
    return context->GetShadingParameters();
}

// --------------------------------------------------------------------------
// Setters implementation (新增 Set 函数)
// --------------------------------------------------------------------------

UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetFrameIndex(rtxdi::ReSTIRDIContext* context, uint32_t frameIndex)
{
    if (context) context->SetFrameIndex(frameIndex);
}

UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetResamplingMode(rtxdi::ReSTIRDIContext* context, rtxdi::ReSTIRDI_ResamplingMode mode)
{
    if (context) context->SetResamplingMode(mode);
}

UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetInitialSamplingParameters(rtxdi::ReSTIRDIContext* context, ReSTIRDI_InitialSamplingParameters params)
{
    // 注意：这里按值传递 params 到导出函数，再传给 C++ 对象
    // 如果结构体很大，可以改用指针传参：const rtxdi::ReSTIRDI_InitialSamplingParameters* params
    if (context) context->SetInitialSamplingParameters(params);
}

UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetTemporalResamplingParameters(rtxdi::ReSTIRDIContext* context, ReSTIRDI_TemporalResamplingParameters params)
{
    if (context) context->SetTemporalResamplingParameters(params);
}

UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetSpatialResamplingParameters(rtxdi::ReSTIRDIContext* context, ReSTIRDI_SpatialResamplingParameters params)
{
    if (context) context->SetSpatialResamplingParameters(params);
}

UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetShadingParameters(rtxdi::ReSTIRDIContext* context, ReSTIRDI_ShadingParameters params)
{
    if (context) context->SetShadingParameters(params);
}


UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API FillNeighborOffsetBuffer(uint8_t* buffer, uint32_t neighborOffsetCount)
{
    return rtxdi::FillNeighborOffsetBuffer(buffer, neighborOffsetCount);
}
}
