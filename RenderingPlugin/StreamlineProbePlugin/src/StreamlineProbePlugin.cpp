// StreamlineProbePlugin.cpp
// Step 1: Streamline owns Unity's present path for DLSS-G.
//
// As a "Load on startup" plugin, UnityPluginLoad runs during
// PlayerInitEngineNoGraphics — BEFORE Unity creates its D3D12 device/swapchain —
// which is exactly when Streamline must be initialized (slInit before the
// CreateSwapChain hook can fire). We then vtable-hook the DXGI factory's
// CreateSwapChainForHwnd; when Unity creates its swapchain we let the real call
// run, then upgrade that swapchain into Streamline's FG proxy and hand it back to
// Unity. From then on Unity presents through Streamline.
//
// Keep only ONE of StreamlineProbePlugin / SwapChainHookPlugin enabled at a time.

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

#include "StreamlineProbe.h"

using Microsoft::WRL::ComPtr;

namespace
{
    IUnityInterfaces* s_Interfaces = nullptr;
    IUnityLog*        s_Log        = nullptr;
    IUnityGraphics*   s_Graphics   = nullptr;

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

    // --- DXGI factory vtable hook (catch Unity's swapchain creation) --------
    // IUnknown(0-2)+IDXGIObject(3-6)+IDXGIFactory: EnumAdapters(7),
    // MakeWindowAssociation(8), GetWindowAssociation(9), CreateSwapChain(10).
    constexpr UINT kCreateSwapChainVTIdx = 10;
    // IDXGIFactory1(+2) + IDXGIFactory2: IsWindowedStereoEnabled(14),
    // CreateSwapChainForHwnd(15).
    constexpr UINT kCreateSwapChainForHwndVTIdx = 15;

    using PFN_CreateSwapChain = HRESULT(STDMETHODCALLTYPE*)(
        IDXGIFactory*, IUnknown*, DXGI_SWAP_CHAIN_DESC*, IDXGISwapChain**);
    using PFN_CreateSwapChainForHwnd = HRESULT(STDMETHODCALLTYPE*)(
        IDXGIFactory2*, IUnknown*, HWND, const DXGI_SWAP_CHAIN_DESC1*,
        const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*, IDXGIOutput*, IDXGISwapChain1**);

    std::atomic<bool>          s_FactoryHooked{false};
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
        const DXGI_SWAP_CHAIN_DESC1* pDesc,
        const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* pFS,
        IDXGIOutput* pOut, IDXGISwapChain1** ppSwapChain)
    {
        LogBridge(0, "[NR/StreamlineProbe] CreateSwapChainForHwnd intercepted");
        HRESULT hr = s_OrigCreateSwapChainForHwnd(This, pDevice, hWnd, pDesc, pFS, pOut, ppSwapChain);
        if (SUCCEEDED(hr) && ppSwapChain && *ppSwapChain)
            StreamlineProbe::AdoptSwapChain(ppSwapChain, pDevice);   // upgrade → SL FG proxy
        return hr;
    }

    HRESULT STDMETHODCALLTYPE Hooked_CreateSwapChain(
        IDXGIFactory* This, IUnknown* pDevice,
        DXGI_SWAP_CHAIN_DESC* pDesc, IDXGISwapChain** ppSwapChain)
    {
        LogBridge(0, "[NR/StreamlineProbe] CreateSwapChain intercepted");
        HRESULT hr = s_OrigCreateSwapChain(This, pDevice, pDesc, ppSwapChain);
        if (SUCCEEDED(hr) && ppSwapChain && *ppSwapChain)
        {
            ComPtr<IDXGISwapChain1> sc1;
            if (SUCCEEDED((*ppSwapChain)->QueryInterface(IID_PPV_ARGS(&sc1))) && sc1)
            {
                IDXGISwapChain1* p = sc1.Get();
                StreamlineProbe::AdoptSwapChain(&p, pDevice);
                // base path is unused by Unity; leave *ppSwapChain as-is on adopt.
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
            LogBridge(2, "[NR/StreamlineProbe ERR] InstallFactoryHook: CreateDXGIFactory2 failed");
            s_FactoryHooked.store(false);
            return;
        }
        s_OrigCreateSwapChain = reinterpret_cast<PFN_CreateSwapChain>(
            PatchSlot(factory.Get(), kCreateSwapChainVTIdx, &Hooked_CreateSwapChain));
        s_OrigCreateSwapChainForHwnd = reinterpret_cast<PFN_CreateSwapChainForHwnd>(
            PatchSlot(factory.Get(), kCreateSwapChainForHwndVTIdx, &Hooked_CreateSwapChainForHwnd));

        const bool ok = s_OrigCreateSwapChain && s_OrigCreateSwapChainForHwnd;
        LogBridge(ok ? 0 : 2, ok
            ? "[NR/StreamlineProbe] Factory hook installed (CreateSwapChain[ForHwnd])"
            : "[NR/StreamlineProbe ERR] Factory hook PARTIAL/FAILED");
    }

    void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
    {
        if (eventType == kUnityGfxDeviceEventShutdown)
            StreamlineProbe::Shutdown();   // before the device dies
    }
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
UnityPluginLoad(IUnityInterfaces* unityInterfaces)
{
    s_Interfaces = unityInterfaces;
    s_Log        = unityInterfaces->Get<IUnityLog>();
    s_Graphics   = unityInterfaces->Get<IUnityGraphics>();

    LogBridge(0, "[NR/StreamlineProbe] StreamlineProbePlugin loaded (Step 1: SL present takeover)");

    // Preload (before Unity's device/swapchain exist): init Streamline first, then
    // install the factory hook so it's in place when Unity creates its swapchain.
    StreamlineProbe::InitSL(&LogBridge);
    InstallFactoryHook();

    if (s_Graphics)
        s_Graphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
UnityPluginUnload()
{
    StreamlineProbe::Shutdown();
    if (s_Graphics)
        s_Graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
}
