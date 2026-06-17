// SLDenoiserPlugin.cpp
// Unity native plugin entry for DLSS Ray Reconstruction via Streamline (SL).
//
// On-demand plugin (NOT load-on-startup): UnityPluginLoad runs on the first P/Invoke,
// after Unity's D3D12 device exists, so we init SL and set the device immediately
// (mirrors Denoiser/RenderingPlugin.cpp). DLSS-RR evaluates on Unity's command list and
// does not touch the swapchain, so this is editor-safe.
//
// IMPORTANT: keep this plugin and the DLSS-G StreamlineProbePlugin mutually exclusive —
// Streamline allows only one slInit per process.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#include "IUnityInterface.h"
#include "IUnityGraphics.h"
#include "IUnityGraphicsD3D12.h"
#include "IUnityLog.h"

#include "SLDlssrr.h"
#include "SLDlssrrFrameData.h"

namespace
{
    IUnityInterfaces*      s_Interfaces = nullptr;
    IUnityGraphics*        s_Graphics   = nullptr;
    IUnityLog*             s_Log        = nullptr;
    IUnityGraphicsD3D12v7* s_D3D12      = nullptr;

    void LogBridge(int level, const char* msg)
    {
        if (s_Log)
        {
            UnityLogType type = (level == 2) ? kUnityLogTypeError
                              : (level == 1) ? kUnityLogTypeWarning
                                             : kUnityLogTypeLog;
            s_Log->Log(type, msg, __FILE__, __LINE__);
        }
        else { OutputDebugStringA(msg); OutputDebugStringA("\n"); }
    }

    void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
    {
        if (eventType == kUnityGfxDeviceEventInitialize)
        {
            s_D3D12 = s_Interfaces->Get<IUnityGraphicsD3D12v7>();
            if (s_D3D12)
                SLDlssrr::SetDevice(s_D3D12->GetDevice());
            else
                LogBridge(2, "[NR/SLDlssrr] IUnityGraphicsD3D12v7 unavailable.");
        }
        else if (eventType == kUnityGfxDeviceEventShutdown)
        {
            SLDlssrr::Shutdown();
            s_D3D12 = nullptr;
        }
    }

    // Render-thread callback: pull Unity's open command list and run the SL evaluate.
    void UNITY_INTERFACE_API OnRenderEventAndData(int /*eventId*/, void* data)
    {
        if (!data || !s_D3D12) return;

        UnityGraphicsD3D12RecordingState state{};
        if (!s_D3D12->CommandRecordingState(&state) || !state.commandList) return;

        SLDlssrr::Dispatch(reinterpret_cast<SLDlssrrFrameData*>(data), state.commandList);
    }
}

extern "C"
{
    void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces* unityInterfaces)
    {
        s_Interfaces = unityInterfaces;
        s_Graphics   = s_Interfaces->Get<IUnityGraphics>();
        s_Log        = s_Interfaces->Get<IUnityLog>();

        s_Graphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);

        // Init SL before grabbing the device, then run the init event manually (the real
        // kUnityGfxDeviceEventInitialize already fired before this on-demand load).
        SLDlssrr::InitSL(&LogBridge);
        OnGraphicsDeviceEvent(kUnityGfxDeviceEventInitialize);

        LogBridge(0, "[NR/SLDlssrr] SLDenoiser plugin loaded (DLSS-RR via Streamline).");
    }

    void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload()
    {
        if (s_Graphics)
            s_Graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
        SLDlssrr::Shutdown();
        LogBridge(0, "[NR/SLDlssrr] SLDenoiser plugin unloaded.");
    }

    UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
    GetSLDlssrrRenderEventAndDataFunc()
    {
        return OnRenderEventAndData;
    }

    int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API CreateSLDlssrrInstance()
    {
        return SLDlssrr::CreateInstance();
    }

    void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DestroySLDlssrrInstance(int id)
    {
        SLDlssrr::DestroyInstance(id);
    }

    bool UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SLDlssrr_QueryOptimalRenderSize(
        unsigned outputWidth, unsigned outputHeight, unsigned char mode,
        unsigned* outRenderWidth, unsigned* outRenderHeight)
    {
        return SLDlssrr::QueryOptimalRenderSize(outputWidth, outputHeight, mode,
                                                outRenderWidth, outRenderHeight);
    }
}
