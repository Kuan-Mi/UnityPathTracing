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

    // DLSS-G frame-begin tick (render thread): mint token + drive Reflex loop.
    void UNITY_INTERFACE_API OnFGBeginFrame(int /*eventId*/) { SLDlssg::BeginFrame(); }

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
            // Player: install the DLSS-G present-path hooks before Unity creates its
            // device/queues/swapchain. Queue hook FIRST (present queue must be SL-proxied).
            SLDlssg::InstallDeviceQueueHook();
            InstallFactoryHook();
            LogBridge(0, "[NR/SLDlssg] Player detected: DLSS-G present hooks installed.");
        }
        else
        {
            LogBridge(0, "[NR/SLDlssg] Editor detected: DLSS-G hooks skipped (DLSS-RR evaluate only).");
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
    // Issue at frame begin: cmd.IssuePluginEvent(GetSLFGBeginFrameFunc(), 0).
    UnityRenderingEvent UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
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
}
