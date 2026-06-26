// SLDenoiserPlugin.cpp
// Unity native plugin entry for DLSS Ray Reconstruction (evaluate-time) AND DLSS-G Frame
// Generation (present-path) via Streamline. One shared slInit via SLCore (loads
// DLSS_RR + DLSS_G + Reflex + PCL). SLCore owns the SL lifecycle + logging so future
// features (DLSS-SR, NIS, …) can be added without touching init/device/shutdown.
//
// LOAD MODEL: this must be a LOAD-ON-STARTUP plugin so the DLSS-G queue/swapchain hooks are
// installed before Unity creates its device/swapchain. The FG hooks are installed ONLY in a
// standalone PLAYER — in the editor they crash, so we auto-detect the host process (the
// editor runs as Unity.exe) and skip them there. DLSS-RR (evaluate-only, no hooks) works in
// both editor and player.
//
// Keep this plugin and the legacy StreamlineProbePlugin mutually exclusive (one slInit/process).

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_5.h>
#include <atomic>
#include <wrl/client.h>

#include "IUnityInterface.h"
#include "IUnityGraphics.h"
#include "IUnityGraphicsD3D12.h"
#include "IUnityLog.h"

#include "SLCore.h"
#include "SLDlssrr.h"
#include "SLDlssrrFrameData.h"
#include "SLDlssg.h"
#include "SLReflex.h"

namespace sl { struct FrameToken; }

using Microsoft::WRL::ComPtr;

namespace
{
    IUnityInterfaces*      s_Interfaces = nullptr;
    IUnityGraphics*        s_Graphics   = nullptr;
    IUnityLog*             s_Log        = nullptr;
    IUnityGraphicsD3D12v7* s_D3D12      = nullptr;
    bool                   s_IsPlayer   = false; // false in the editor (Unity.exe)

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

    // Editor (Unity.exe) vs standalone player (<Project>.exe). The FG present/queue hooks
    // are player-only; the editor gets DLSS-RR evaluate only.
    bool DetectIsPlayer()
    {
        char path[MAX_PATH] = {};
        DWORD n = GetModuleFileNameA(nullptr, path, (DWORD)sizeof(path));
        if (n == 0) return false; // unknown → treat as editor (safe: no hooks)
        const char* slash = strrchr(path, '\\');
        const char* exe = slash ? slash + 1 : path;
        return _stricmp(exe, "Unity.exe") != 0;
    }

    // --- DXGI factory vtable hook (catch Unity's swapchain creation) — PLAYER ONLY ---
    constexpr UINT kCreateSwapChainVTIdx        = 10;
    constexpr UINT kCreateSwapChainForHwndVTIdx = 15;

    using PFN_CreateSwapChain = HRESULT(STDMETHODCALLTYPE*)(
        IDXGIFactory*, IUnknown*, DXGI_SWAP_CHAIN_DESC*, IDXGISwapChain**);
    using PFN_CreateSwapChainForHwnd = HRESULT(STDMETHODCALLTYPE*)(
        IDXGIFactory2*, IUnknown*, HWND, const DXGI_SWAP_CHAIN_DESC1*,
        const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*, IDXGIOutput*, IDXGISwapChain1**);

    std::atomic<bool>          s_FactoryHooked{ false };
    PFN_CreateSwapChain        s_OrigCreateSwapChain        = nullptr;
    PFN_CreateSwapChainForHwnd s_OrigCreateSwapChainForHwnd = nullptr;

    void* PatchSlot(void* objWithVtable, UINT index, void* hook)
    {
        void** vtable = *reinterpret_cast<void***>(objWithVtable);
        void** slot   = vtable + index;
        DWORD old = 0;
        if (!VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &old)) return nullptr;
        void* orig = *slot;
        *slot = hook;
        VirtualProtect(slot, sizeof(void*), old, &old);
        return orig;
    }

    HRESULT STDMETHODCALLTYPE Hooked_CreateSwapChainForHwnd(
        IDXGIFactory2* This, IUnknown* pDevice, HWND hWnd,
        const DXGI_SWAP_CHAIN_DESC1* pDesc, const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* pFS,
        IDXGIOutput* pOut, IDXGISwapChain1** ppSwapChain)
    {
        static thread_local bool t_inProxyCreate = false;
        if (t_inProxyCreate)
        {
            IUnknown* nativeQ = SLDlssg::NativeIfProxy(pDevice);
            return s_OrigCreateSwapChainForHwnd(This, nativeQ, hWnd, pDesc, pFS, pOut, ppSwapChain);
        }

        LogBridge(0, "[NR/SLDlssg] CreateSwapChainForHwnd intercepted");

        if (SLDlssg::IsQueueProxyActive())
        {
            t_inProxyCreate = true;
            HRESULT hr = SLDlssg::CreateSwapChainViaProxyFactory(pDevice, hWnd, pDesc, pFS, pOut, ppSwapChain);
            t_inProxyCreate = false;
            if (SUCCEEDED(hr) && ppSwapChain && *ppSwapChain)
                SLDlssg::AdoptSwapChain(ppSwapChain, pDevice, /*alreadyProxy=*/true);
            return hr;
        }

        HRESULT hr = s_OrigCreateSwapChainForHwnd(This, pDevice, hWnd, pDesc, pFS, pOut, ppSwapChain);
        if (SUCCEEDED(hr) && ppSwapChain && *ppSwapChain)
            SLDlssg::AdoptSwapChain(ppSwapChain, pDevice, /*alreadyProxy=*/false);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE Hooked_CreateSwapChain(
        IDXGIFactory* This, IUnknown* pDevice, DXGI_SWAP_CHAIN_DESC* pDesc, IDXGISwapChain** ppSwapChain)
    {
        LogBridge(0, "[NR/SLDlssg] CreateSwapChain intercepted");
        HRESULT hr = s_OrigCreateSwapChain(This, pDevice, pDesc, ppSwapChain);
        if (SUCCEEDED(hr) && ppSwapChain && *ppSwapChain)
        {
            ComPtr<IDXGISwapChain1> sc1;
            if (SUCCEEDED((*ppSwapChain)->QueryInterface(IID_PPV_ARGS(&sc1))) && sc1)
            {
                IDXGISwapChain1* p = sc1.Get();
                SLDlssg::AdoptSwapChain(&p, pDevice, /*alreadyProxy=*/false);
            }
        }
        return hr;
    }

    void InstallFactoryHook()
    {
        bool expected = false;
        if (!s_FactoryHooked.compare_exchange_strong(expected, true)) return;

        ComPtr<IDXGIFactory2> factory;
        if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))) || !factory)
        {
            LogBridge(2, "[NR/SLDlssg ERR] InstallFactoryHook: CreateDXGIFactory2 failed");
            s_FactoryHooked.store(false);
            return;
        }
        s_OrigCreateSwapChain = reinterpret_cast<PFN_CreateSwapChain>(
            PatchSlot(factory.Get(), kCreateSwapChainVTIdx, &Hooked_CreateSwapChain));
        s_OrigCreateSwapChainForHwnd = reinterpret_cast<PFN_CreateSwapChainForHwnd>(
            PatchSlot(factory.Get(), kCreateSwapChainForHwndVTIdx, &Hooked_CreateSwapChainForHwnd));
        const bool ok = s_OrigCreateSwapChain && s_OrigCreateSwapChainForHwnd;
        LogBridge(ok ? 0 : 2, ok ? "[NR/SLDlssg] Factory hook installed (CreateSwapChain[ForHwnd])"
                                 : "[NR/SLDlssg ERR] Factory hook PARTIAL/FAILED");
    }

    void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
    {
        if (eventType == kUnityGfxDeviceEventInitialize)
        {
            s_D3D12 = s_Interfaces->Get<IUnityGraphicsD3D12v7>();
            if (s_D3D12 && s_D3D12->GetDevice())
                SLCore::SetDevice(s_D3D12->GetDevice());
            // (Device may not exist yet at plugin-load time; the real init event sets it.)
        }
        else if (eventType == kUnityGfxDeviceEventShutdown)
        {
            SLReflex::Shutdown();
            SLDlssg::Shutdown();
            SLDlssrr::Shutdown();
            SLCore::Shutdown();
            s_D3D12 = nullptr;
        }
    }

    // DLSS-RR render-event-and-data (evaluate on Unity's command list).
    void UNITY_INTERFACE_API OnRRRenderEventAndData(int /*eventId*/, void* data)
    {
        if (!data || !s_D3D12) return;
        UnityGraphicsD3D12RecordingState state{};
        if (!s_D3D12->CommandRecordingState(&state) || !state.commandList) return;
        SLDlssrr::Dispatch(reinterpret_cast<SLDlssrrFrameData*>(data), state.commandList);
    }

    // Render-thread frame-begin event: data is the FrameToken* minted on the main thread by
    // SL_GetNewFrameToken, forwarded verbatim via IssuePluginEventAndData. Pins the render/present
    // side to that exact token so DLSS-G tagging + present markers and DLSS-RR evaluate all
    // use the frame actually being rendered. The latency-critical Reflex sleep is NOT here —
    // it runs on the main thread.
    void UNITY_INTERFACE_API OnFGBeginFrame(int /*eventId*/, void* data)
    {
        sl::FrameToken* token = reinterpret_cast<sl::FrameToken*>(data);
        SLCore::SetRenderFrame(token);
        // Start of CPU render submission for this frame — emit eRenderSubmitStart here (Reflex/
        // PCL), not bunched at present. Universal (PCL works on every adapter, FG or not).
        if (token) SLDlssg::MarkRenderSubmitStart(*token);
    }

    // DLSS-G per-frame inputs (render thread): tag depth/mvec + set constants.
    void UNITY_INTERFACE_API OnFGFrameInputs(int /*eventId*/, void* data)
    {
        if (data) SLDlssg::ConsumeFrameInputs(*reinterpret_cast<const SLDlssg::FrameInputs*>(data));
    }
}

extern "C"
{
    void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces* unityInterfaces)
    {
        s_Interfaces = unityInterfaces;
        s_Graphics   = s_Interfaces->Get<IUnityGraphics>();
        s_Log        = s_Interfaces->Get<IUnityLog>();
        s_IsPlayer   = DetectIsPlayer();

        s_Graphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);

        // Shared slInit (RR + G + Reflex + PCL) + log bridge — before the device/swapchain exist.
        SLCore::Init(&LogBridge);

        if (s_IsPlayer)
        {
            // Player: install the present-path hooks before Unity creates its device/queues/
            // swapchain. Queue hook FIRST (the present queue must be SL-proxied). The proxy
            // device/queue/swapchain are SL's common interposer (required so presentCommon() runs
            // every frame under manual hooking) and are installed on every adapter; Frame
            // Generation is only enabled as a *mode* on top when the adapter supports it.
            SLDlssg::InstallDeviceQueueHook();
            InstallFactoryHook();
            LogBridge(0, "[NR/SLDlssg] Player detected: present-path hooks installed (Reflex/PCL on "
                         "every adapter; DLSS-G frame generation enabled only when supported).");
        }
        else
        {
            LogBridge(0, "[NR/SLDlssg] Editor detected: present hooks skipped (DLSS-RR evaluate only).");
        }

        // If the device already exists (on-demand load), set it now; otherwise the registered
        // callback will (load-on-startup, before device creation).
        OnGraphicsDeviceEvent(kUnityGfxDeviceEventInitialize);

        LogBridge(0, "[NR/SLDlssg] SLDenoiser plugin loaded (DLSS-RR + DLSS-G via Streamline).");
    }

    void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload()
    {
        if (s_Graphics)
            s_Graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
        SLDlssg::Shutdown();
        SLDlssrr::Shutdown();
        SLCore::Shutdown();
        LogBridge(0, "[NR/SLDlssg] SLDenoiser plugin unloaded.");
    }

    // ---- DLSS-RR (evaluate) ----
    UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
    GetSLDlssrrRenderEventAndDataFunc() { return OnRRRenderEventAndData; }

    int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API CreateSLDlssrrInstance() { return SLDlssrr::CreateInstance(); }
    void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DestroySLDlssrrInstance(int id) { SLDlssrr::DestroyInstance(id); }

    bool UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SLDlssrr_QueryOptimalRenderSize(
        unsigned outputWidth, unsigned outputHeight, unsigned char mode,
        unsigned* outRenderWidth, unsigned* outRenderHeight)
    {
        return SLDlssrr::QueryOptimalRenderSize(outputWidth, outputHeight, mode, outRenderWidth, outRenderHeight);
    }

    // ---- DLSS-G (frame generation) ----
    // Issue at frame begin with the token from SL_GetNewFrameToken as data:
    //   cmd.IssuePluginEventAndData(GetSLFGBeginFrameFunc(), 0, frameToken).
    UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
    GetSLFGBeginFrameFunc() { return OnFGBeginFrame; }

    // Issue per frame: cmd.IssuePluginEventAndData(GetSLFGFrameInputsFunc(), 0, ptrToFrameInputs).
    UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
    GetSLFGFrameInputsFunc() { return OnFGFrameInputs; }

    void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SL_SetFrameGeneration(int enable)
    {
        SLDlssg::SetFrameGeneration(enable != 0);
    }

    int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SL_IsFrameGenerationOn()
    {
        return SLDlssg::IsFrameGenerationOn() ? 1 : 0;
    }

    // ---- Streamline frame token + Reflex simulation markers (main thread, not plugin events) ----
    typedef void* SLFrameTokenHandle;

    // SL_GetNewFrameToken: mint this frame's shared Streamline token. C# forwards this opaque
    // handle to render-thread events and to every main-thread marker call for the same frame.
    SLFrameTokenHandle UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SL_GetNewFrameToken()
    {
        // Still mint the per-frame token in the editor — DLSS-RR (evaluate-only, editor + player)
        // forwards it to the render thread for tagging.
        return SLCore::BeginFrame();
    }

    // SL_ReflexSleep: main thread, top of frame BEFORE input. Applies Reflex options and calls
    // slReflexSleep for this token. Player-only; no-op in the editor.
    void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SL_ReflexSleep(void* frameToken)
    {
        if (!s_IsPlayer || !frameToken) return;
        SLReflex::Sleep(*reinterpret_cast<sl::FrameToken*>(frameToken));
    }

    // SL_MarkSimulationStart: main thread, immediately after Reflex sleep, before input/sim.
    void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SL_MarkSimulationStart(void* frameToken)
    {
        if (!s_IsPlayer || !frameToken) return;
        SLReflex::MarkSimulationStart(*reinterpret_cast<sl::FrameToken*>(frameToken));
    }

    // SL_MarkSimulationEnd: main thread, after game logic (before rendering). eSimulationEnd.
    // frameToken is this frame's token from SL_GetNewFrameToken. Player-only (Reflex/PCL).
    void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SL_MarkSimulationEnd(void* frameToken)
    {
        if (!s_IsPlayer || !frameToken) return;
        SLReflex::MarkSimulationEnd(*reinterpret_cast<sl::FrameToken*>(frameToken));
    }

    // SL_ConsumePclPingCount: main thread, after SL_GetNewFrameToken minted the token for the
    // frame that is about to sample input. Returns how many PCL stats pings the WndProc saw
    // since the previous consume; C# owns the token attribution and calls SL_MarkPclLatencyPing.
    unsigned UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SL_ConsumePclPingCount()
    {
        if (!s_IsPlayer) return 0;
        return SLReflex::ConsumePclPingCount();
    }

    void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SL_MarkPclLatencyPing(void* frameToken, unsigned count)
    {
        if (!s_IsPlayer || !frameToken || count == 0) return;
        SLReflex::MarkPclLatencyPing(*reinterpret_cast<sl::FrameToken*>(frameToken), count);
    }

    // ---- Reflex (low latency) ----
    // mode: 0 = Off, 1 = On (Low Latency), 2 = On + Boost. fpsCapUs: 0 = uncapped.
    // Player-only: tied to the DLSS-G present path, so it's a no-op in the editor (Unity.exe).
    void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SL_SetReflexMode(int mode, unsigned fpsCapUs)
    {
        if (!s_IsPlayer) return; // Reflex is player-only.
        SLReflex::SetMode(mode, fpsCapUs);
    }

    int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SL_GetReflexMode()
    {
        if (!s_IsPlayer) return 0; // Off in the editor — Reflex never runs there.
        return SLReflex::GetMode();
    }

    int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SL_IsReflexLowLatencyAvailable()
    {
        if (!s_IsPlayer) return 0; // Reflex is player-only.
        return SLReflex::IsLowLatencyAvailable() ? 1 : 0;
    }
}
