// SwapChainHookPlugin.cpp
// Standalone diagnostic plugin: installs SwapChainHook (DXGI factory + Present
// vtable patches) to probe whether a Unity native plugin can intercept the
// player swapchain's Present — the prerequisite for DLSS Frame Generation.
//
// Deliberately decoupled from NativeRenderPlugin so it can be toggled on its own
// ("Load on startup" in the plugin importer) without touching the shipping
// plugins. Diagnostic-only: every hook logs and calls straight through.
//
// Because a standalone plugin receives no per-frame render callback, the Present
// hook (which needs IUnityGraphicsD3D12v8::GetSwapChain(), valid only in a player
// build and usually a few frames after init) is installed from a worker thread
// that polls GetSwapChain() until it succeeds. Logs route into Unity's player
// log via IUnityLog.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>      // ID3D12* types referenced by IUnityGraphicsD3D12.h
#include <dxgi1_5.h>    // IDXGISwapChain referenced by IUnityGraphicsD3D12.h
#include <atomic>
#include <thread>

#include "IUnityInterface.h"
#include "IUnityGraphics.h"
#include "IUnityGraphicsD3D12.h"
#include "IUnityLog.h"

#include "SwapChainHook.h"
#include "DlssgProbe.h"

namespace
{
    IUnityInterfaces*      s_Interfaces = nullptr;
    IUnityLog*             s_Log        = nullptr;
    IUnityGraphics*        s_Graphics   = nullptr;
    IUnityGraphicsD3D12v8* s_D3D12v8    = nullptr;

    std::thread       s_Poller;
    std::atomic<bool> s_PollerStop{false};

    // SwapChainHook passes already-tagged, level-classified messages here. Map
    // its level to a Unity log channel. Safe to call from any thread.
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

    // GetSwapChain() dereferences Unity-internal state that does not exist until
    // the graphics device is up. Calling it too early (e.g. while preloaded
    // plugins load during PlayerInitEngineNoGraphics) crashes inside
    // GetSwapChainImpl, so wrap it in SEH and only call it after device init.
    // No C++ objects here — required for __try/__except.
    IDXGISwapChain* SafeGetSwapChain()
    {
        if (!s_D3D12v8) return nullptr;
        __try { return s_D3D12v8->GetSwapChain(); }
        __except (EXCEPTION_EXECUTE_HANDLER) { return nullptr; }
    }

    // GetDevice() is likewise null on the first device-init events; SEH-guarded
    // for the same reason as SafeGetSwapChain. No C++ objects here.
    ID3D12Device* SafeGetDevice()
    {
        if (!s_D3D12v8) return nullptr;
        __try { return s_D3D12v8->GetDevice(); }
        __except (EXCEPTION_EXECUTE_HANDLER) { return nullptr; }
    }

    // Fallback Present-hook installer (primary path is the factory CreateSwapChain
    // hook). Only valid after kUnityGfxDeviceEventInitialize. No-op once installed.
    bool TryHookPresent()
    {
        if (SwapChainHook::IsPresentHookInstalled()) return true;
        IDXGISwapChain* sc = SafeGetSwapChain();
        if (!sc) return false;
        return SwapChainHook::TryInstallPresentHook(sc);
    }

    // How many Evaluate smoke tests to run after the FG feature is created.
    constexpr int kDlssgEvalRuns = 8;
    int s_dlssgEvalsDone = 0;

    // Worker poller. The device and swapchain are both null on the first
    // device-init events, so we poll here (after init) toward three goals until
    // all are met or we give up (~60s at 250 ms cadence):
    //   1. Present hook — normally already installed by the factory CreateSwapChain
    //      hook; this is the fallback via GetSwapChain().
    //   2. DLSS-G context Init — once a non-null device appears.
    //   3. DLSS-G Evaluate smoke tests — kDlssgEvalRuns times once the feature is
    //      ready (proves create-once / evaluate-many on plugin-owned resources).
    // In the editor the device/swapchain stay null, so this times out.
    void PollerMain()
    {
        for (int i = 0; i < 240 && !s_PollerStop.load(std::memory_order_relaxed); ++i)
        {
            if (!DlssgProbe::InitAttempted())
            {
                if (ID3D12Device* dev = SafeGetDevice())
                    // 1920x1080 / R8G8B8A8_UNORM(28): representative FG target.
                    DlssgProbe::Init(dev, 1920, 1080, 28, &LogBridge);
            }
            else if (DlssgProbe::IsReady() && s_dlssgEvalsDone < kDlssgEvalRuns)
            {
                DlssgProbe::Evaluate();
                ++s_dlssgEvalsDone;
            }

            const bool presentDone = TryHookPresent();
            const bool dlssgDone = DlssgProbe::InitAttempted() &&
                                   (!DlssgProbe::IsReady() || s_dlssgEvalsDone >= kDlssgEvalRuns);
            if (presentDone && dlssgDone) return;
            Sleep(250);
        }
        if (s_PollerStop.load(std::memory_order_relaxed)) return;
        if (!SwapChainHook::IsPresentHookInstalled())
            LogBridge(1, "[NR/SwapChainHook WRN] Poller gave up: GetSwapChain() never "
                         "returned a swapchain (editor, or non-D3D12 / no player swapchain).");
        if (!DlssgProbe::InitAttempted())
            LogBridge(1, "[NR/SwapChainHook WRN] Poller gave up: GetDevice() never "
                         "returned a device; DLSS-G context did not init.");
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
                      s_D3D12v8 ? "[NR/SwapChainHook] D3D12v8 interface acquired"
                                : "[NR/SwapChainHook WRN] D3D12v8 interface unavailable "
                                  "(non-D3D12 renderer?); GetSwapChain fallback disabled.");
            // Start the worker poller. The device/swapchain are not ready on this
            // first event, so the poller retries until it can install the Present
            // hook and run the DLSS-G probe (the factory hook usually installs the
            // Present hook first, so the poller mainly drives the probe).
            StartPoller();
        }
        else if (eventType == kUnityGfxDeviceEventShutdown)
        {
            StopPoller();
            DlssgProbe::Shutdown();   // release FG feature + NGX before the device dies
            s_D3D12v8 = nullptr;
        }
    }
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
UnityPluginLoad(IUnityInterfaces* unityInterfaces)
{
    s_Interfaces = unityInterfaces;
    s_Log        = unityInterfaces->Get<IUnityLog>();      // may be null on very old Unity
    s_Graphics   = unityInterfaces->Get<IUnityGraphics>();

    SwapChainHook::SetLogger(&LogBridge);
    LogBridge(0, "[NR/SwapChainHook] SwapChainHookPlugin loaded (standalone diagnostic)");

    // Primary path: install the DXGI factory hook now. As a "Load on startup"
    // plugin we run during PlayerInitEngineNoGraphics — before Unity creates its
    // graphics device or swapchain — so this hook is in place in time to catch
    // the CreateSwapChain[ForHwnd] call and self-install the Present hook.
    SwapChainHook::InstallFactoryHook();

    // Register for the device-init event. We must NOT touch the graphics device
    // or call GetSwapChain() here: at preload time it does not exist yet and
    // doing so crashes inside Unity (GetSwapChainImpl null-deref). The real
    // callback fires later, once the device is up, and starts the fallback poller.
    if (s_Graphics)
    {
        s_Graphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);
    }
    else
    {
        LogBridge(1, "[NR/SwapChainHook WRN] IUnityGraphics unavailable; "
                     "GetSwapChain fallback disabled (factory hook still active).");
    }
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
UnityPluginUnload()
{
    StopPoller();
    DlssgProbe::Shutdown();   // safe/idempotent if device-shutdown already ran it
    if (s_Graphics)
        s_Graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
}
