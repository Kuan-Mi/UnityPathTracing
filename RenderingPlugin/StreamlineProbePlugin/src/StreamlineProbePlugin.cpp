// StreamlineProbePlugin.cpp
// Standalone diagnostic plugin: initializes Streamline against Unity's D3D12
// device and logs whether DLSS-G (Frame Generation) is supported in-process.
//
// Like SwapChainHookPlugin, enable it via the plugin importer's "Load on
// startup". As a preloaded plugin it loads during PlayerInitEngineNoGraphics —
// BEFORE the graphics device exists — so we must NOT touch the device at load
// time; a worker poller calls the probe once a real device appears (GetDevice()
// is null on the first device-init events, same pitfall as the NGX probe).
//
// Keep only ONE of StreamlineProbePlugin / SwapChainHookPlugin enabled at a time
// so the Streamline runtime and the raw-NGX probe don't both init NGX.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_5.h>
#include <atomic>
#include <thread>

#include "IUnityInterface.h"
#include "IUnityGraphics.h"
#include "IUnityGraphicsD3D12.h"
#include "IUnityLog.h"

#include "StreamlineProbe.h"

namespace
{
    IUnityInterfaces*      s_Interfaces = nullptr;
    IUnityLog*             s_Log        = nullptr;
    IUnityGraphics*        s_Graphics   = nullptr;
    IUnityGraphicsD3D12v8* s_D3D12v8    = nullptr;

    std::thread       s_Poller;
    std::atomic<bool> s_PollerStop{false};

    void LogBridge(int level, const char* msg)
    {
        if (s_Log)
        {
            UnityLogType type = (level == 2) ? kUnityLogTypeError
                              : (level == 1) ? kUnityLogTypeWarning
                                             : kUnityLogTypeLog;
            s_Log->Log(type, msg, __FILE__, __LINE__);
        }
        else
        {
            OutputDebugStringA(msg);
            OutputDebugStringA("\n");
        }
    }

    // GetDevice() is null on the first device-init events; SEH-guarded because it
    // dereferences Unity-internal state that may not exist yet. No C++ objects.
    ID3D12Device* SafeGetDevice()
    {
        if (!s_D3D12v8) return nullptr;
        __try { return s_D3D12v8->GetDevice(); }
        __except (EXCEPTION_EXECUTE_HANDLER) { return nullptr; }
    }

    void PollerMain()
    {
        // Retry until a non-null device appears (then probe once) or give up.
        for (int i = 0; i < 240 && !s_PollerStop.load(std::memory_order_relaxed); ++i)
        {
            if (!StreamlineProbe::InitAttempted())
            {
                if (ID3D12Device* dev = SafeGetDevice())
                    StreamlineProbe::Init(dev, &LogBridge);
            }
            else
            {
                return;   // probe ran; done
            }
            Sleep(250);
        }
        if (!s_PollerStop.load(std::memory_order_relaxed) && !StreamlineProbe::InitAttempted())
            LogBridge(1, "[NR/StreamlineProbe WRN] Poller gave up: GetDevice() never "
                         "returned a device (editor, or non-D3D12 renderer).");
    }

    void StartPoller()
    {
        if (s_Poller.joinable()) return;
        s_PollerStop.store(false, std::memory_order_relaxed);
        s_Poller = std::thread(PollerMain);
    }

    void StopPoller()
    {
        s_PollerStop.store(true, std::memory_order_relaxed);
        if (s_Poller.joinable()) s_Poller.join();
    }

    void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
    {
        if (eventType == kUnityGfxDeviceEventInitialize)
        {
            s_D3D12v8 = s_Interfaces ? s_Interfaces->Get<IUnityGraphicsD3D12v8>() : nullptr;
            LogBridge(s_D3D12v8 ? 0 : 1,
                      s_D3D12v8 ? "[NR/StreamlineProbe] D3D12v8 interface acquired"
                                : "[NR/StreamlineProbe WRN] D3D12v8 unavailable "
                                  "(non-D3D12 renderer?); probe disabled.");
            StartPoller();
        }
        else if (eventType == kUnityGfxDeviceEventShutdown)
        {
            StopPoller();
            StreamlineProbe::Shutdown();
            s_D3D12v8 = nullptr;
        }
    }
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
UnityPluginLoad(IUnityInterfaces* unityInterfaces)
{
    s_Interfaces = unityInterfaces;
    s_Log        = unityInterfaces->Get<IUnityLog>();
    s_Graphics   = unityInterfaces->Get<IUnityGraphics>();

    LogBridge(0, "[NR/StreamlineProbe] StreamlineProbePlugin loaded (standalone diagnostic)");

    // Do NOT touch the graphics device here (preload: it doesn't exist yet).
    if (s_Graphics)
        s_Graphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);
    else
        LogBridge(1, "[NR/StreamlineProbe WRN] IUnityGraphics unavailable; probe disabled.");
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
UnityPluginUnload()
{
    StopPoller();
    StreamlineProbe::Shutdown();
    if (s_Graphics)
        s_Graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
}
