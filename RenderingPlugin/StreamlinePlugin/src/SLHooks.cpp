// SLHooks.cpp - Streamline manual-hooking plumbing for StreamlinePlugin.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_5.h>
#include <atomic>
#include <wrl/client.h>

#include "sl.h"

#include "SLCore.h"
#include "SLDlssg.h"
#include "SLHooks.h"
#include "SLReflex.h"

using Microsoft::WRL::ComPtr;

#define Logf(level, ...) SLCore::Logf("SLHooks", level, __VA_ARGS__)
#define R(r)             SLCore::ResultStr(r)

namespace
{
    constexpr UINT kCreateCommandQueueVTIdx     = 8;
    constexpr UINT kCreateSwapChainVTIdx        = 10;
    constexpr UINT kCreateSwapChainForHwndVTIdx = 15;
    constexpr UINT kPresentVTIdx                = 8;
    constexpr UINT kPresent1VTIdx               = 22;

    using PFN_CreateCommandQueue = HRESULT(STDMETHODCALLTYPE*)(
        ID3D12Device*, const D3D12_COMMAND_QUEUE_DESC*, REFIID, void**);
    using PFN_CreateSwapChain = HRESULT(STDMETHODCALLTYPE*)(
        IDXGIFactory*, IUnknown*, DXGI_SWAP_CHAIN_DESC*, IDXGISwapChain**);
    using PFN_CreateSwapChainForHwnd = HRESULT(STDMETHODCALLTYPE*)(
        IDXGIFactory2*, IUnknown*, HWND, const DXGI_SWAP_CHAIN_DESC1*,
        const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*, IDXGIOutput*, IDXGISwapChain1**);
    using PFN_Present = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, UINT);
    using PFN_Present1 = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain1*, UINT, UINT,
                                                     const DXGI_PRESENT_PARAMETERS*);

    std::atomic<bool>      g_deviceHooked{ false };
    std::atomic<bool>      g_factoryHooked{ false };
    std::atomic<bool>      g_presentHooked{ false };
    PFN_CreateCommandQueue g_origCreateCommandQueue = nullptr;
    PFN_CreateSwapChain    g_origCreateSwapChain = nullptr;
    PFN_CreateSwapChainForHwnd g_origCreateSwapChainForHwnd = nullptr;
    PFN_Present            g_origPresent = nullptr;
    PFN_Present1           g_origPresent1 = nullptr;

    ComPtr<ID3D12Device>   g_device;
    ComPtr<ID3D12Device>   g_proxyDevice;
    bool                   g_proxyDeviceTried = false;
    ComPtr<IDXGIFactory2>  g_proxyFactory;
    bool                   g_proxyFactoryTried = false;
    IDXGISwapChain1*       g_proxySwapchain = nullptr;
    ComPtr<IDXGISwapChain3> g_proxySC3;

    HRESULT STDMETHODCALLTYPE Hooked_SLPresent(
        IDXGISwapChain* This, UINT sync, UINT flags);
    HRESULT STDMETHODCALLTYPE Hooked_SLPresent1(
        IDXGISwapChain1* This, UINT sync, UINT flags, const DXGI_PRESENT_PARAMETERS* pp);

    void* PatchSlot(void* objWithVtable, UINT index, void* hook)
    {
        void** vtable = *reinterpret_cast<void***>(objWithVtable);
        void** slot = vtable + index;
        DWORD old = 0;
        if (!VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &old)) return nullptr;
        void* orig = *slot;
        *slot = hook;
        VirtualProtect(slot, sizeof(void*), old, &old);
        return orig;
    }

    IUnknown* NativeIfProxy(IUnknown* maybeProxy)
    {
        if (!maybeProxy) return maybeProxy;
        void* native = nullptr;
        sl::Result r = slGetNativeInterface(maybeProxy, &native);
        if (r == sl::Result::eOk && native)
            return reinterpret_cast<IUnknown*>(native);
        return maybeProxy;
    }

    void EnsureDevice(IUnknown* presentQueue)
    {
        if (SLCore::IsDeviceSet() || !presentQueue) return;

        ComPtr<ID3D12CommandQueue> queue;
        if (FAILED(presentQueue->QueryInterface(IID_PPV_ARGS(&queue))) || !queue)
        {
            Logf(2, "AdoptSwapChain: present 'device' arg is not an ID3D12CommandQueue.");
            return;
        }

        queue->GetDevice(IID_PPV_ARGS(&g_device));
        if (!g_device)
        {
            Logf(2, "AdoptSwapChain: queue->GetDevice failed.");
            return;
        }

        SLCore::SetDevice(g_device.Get());
        if (SLCore::IsDeviceSet())
        {
            LUID luid = g_device->GetAdapterLuid();
            sl::AdapterInfo ai{};
            ai.deviceLUID = reinterpret_cast<uint8_t*>(&luid);
            ai.deviceLUIDSizeInBytes = sizeof(luid);
            sl::Result rg = slIsFeatureSupported(sl::kFeatureDLSS_G, ai);
            Logf(rg == sl::Result::eOk ? 0 : 1, "slIsFeatureSupported(DLSS_G) -> %s", R(rg));
        }
    }

    bool EnsureProxyFactory()
    {
        if (g_proxyFactory) return true;
        if (g_proxyFactoryTried) return false;
        g_proxyFactoryTried = true;

        ComPtr<IDXGIFactory2> nativeFactory;
        HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&nativeFactory));
        if (FAILED(hr) || !nativeFactory)
        {
            Logf(2, "EnsureProxyFactory: CreateDXGIFactory2 failed 0x%08lx", (unsigned long)hr);
            return false;
        }

        IDXGIFactory2* f = nativeFactory.Get();
        sl::Result r = slUpgradeInterface(reinterpret_cast<void**>(&f));
        if (r != sl::Result::eOk || !f || f == nativeFactory.Get())
        {
            Logf(2, "EnsureProxyFactory: slUpgradeInterface(factory) -> %s", R(r));
            return false;
        }

        g_proxyFactory.Attach(f);
        Logf(0, "EnsureProxyFactory: SL proxy factory ready.");
        return true;
    }

    HRESULT CreateSwapChainViaProxyFactory(IUnknown* queue, HWND hWnd,
        const DXGI_SWAP_CHAIN_DESC1* desc, const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fs,
        IDXGIOutput* out, IDXGISwapChain1** ppSwapChain)
    {
        if (!EnsureProxyFactory()) return E_FAIL;
        HRESULT hr = g_proxyFactory->CreateSwapChainForHwnd(queue, hWnd, desc, fs, out, ppSwapChain);
        Logf(SUCCEEDED(hr) ? 0 : 2, "CreateSwapChainViaProxyFactory -> hr=0x%08lx", (unsigned long)hr);
        return hr;
    }

    void InstallPresentHookOnProxy(IDXGISwapChain1* proxy)
    {
        bool expected = false;
        if (!g_presentHooked.compare_exchange_strong(expected, true)) return;

        g_origPresent = reinterpret_cast<PFN_Present>(
            PatchSlot(proxy, kPresentVTIdx, &Hooked_SLPresent));
        g_origPresent1 = reinterpret_cast<PFN_Present1>(
            PatchSlot(proxy, kPresent1VTIdx, &Hooked_SLPresent1));

        const bool ok = g_origPresent1 != nullptr;
        Logf(ok ? 0 : 2, "Per-frame tagging hook on SL proxy Present1: %s",
             ok ? "installed" : "FAILED");
        if (!ok) g_presentHooked.store(false);
    }

    void AdoptSwapChain(IDXGISwapChain1** ppSwapChain, IUnknown* presentQueue, bool alreadyProxy)
    {
        if (!ppSwapChain || !*ppSwapChain) return;

        EnsureDevice(presentQueue);
        if (!SLCore::IsDeviceSet())
        {
            Logf(1, "AdoptSwapChain: device not set; leaving native swapchain.");
            return;
        }

        DXGI_SWAP_CHAIN_DESC1 desc{};
        (*ppSwapChain)->GetDesc1(&desc);

        // Always route presentation through the SL common proxy. DLSS-G is only a mode on top
        // of this path; Reflex/PCL and presentCommon bookkeeping need it too.
        if (!alreadyProxy)
        {
            void* iface = *ppSwapChain;
            sl::Result r = slUpgradeInterface(&iface);
            if (r != sl::Result::eOk || !iface)
            {
                Logf(2, "slUpgradeInterface(swapchain) -> %s; no proxy present path.", R(r));
                return;
            }
            *ppSwapChain = reinterpret_cast<IDXGISwapChain1*>(iface);
            Logf(0, "slUpgradeInterface(swapchain) -> eOk; Unity now presents through SL proxy.");
        }
        else
        {
            Logf(0, "Swapchain created via SL proxy factory (device/queue/swapchain links set).");
        }

        g_proxySwapchain = *ppSwapChain;
        g_proxySwapchain->QueryInterface(IID_PPV_ARGS(&g_proxySC3));
        InstallPresentHookOnProxy(*ppSwapChain);

        SLReflex::InstallPclPing(nullptr);

        SLDlssg::OnSwapChainAdopted(desc.Width, desc.Height);
    }

    // Resolve the frame token for the back buffer being presented (read BEFORE g_origPresent, while
    // GetCurrentBackBufferIndex still points at the buffer about to flip) and use it for both the
    // start and end PCL markers — resolving per-marker would consume the slot twice for one present.
    // Null (slot never registered, e.g. pre-roll, or a recycled token) simply skips this present's
    // markers rather than tagging the wrong frame.
    HRESULT STDMETHODCALLTYPE Hooked_SLPresent1(
        IDXGISwapChain1* This, UINT sync, UINT flags, const DXGI_PRESENT_PARAMETERS* pp)
    {
        SLDlssg::OnPresentPre(g_proxySC3.Get());
        sl::FrameToken* token = SLCore::ResolvePresentToken(SLHooks::CurrentBackBufferIndex());
        if (token) SLReflex::MarkPresentStart(*token);
        HRESULT hr = g_origPresent1(This, sync, flags, pp);
        if (token) SLReflex::MarkPresentEnd(*token);
        SLDlssg::OnPresentPost();
        return hr;
    }

    HRESULT STDMETHODCALLTYPE Hooked_SLPresent(IDXGISwapChain* This, UINT sync, UINT flags)
    {
        SLDlssg::OnPresentPre(g_proxySC3.Get());
        sl::FrameToken* token = SLCore::ResolvePresentToken(SLHooks::CurrentBackBufferIndex());
        if (token) SLReflex::MarkPresentStart(*token);
        HRESULT hr = g_origPresent(This, sync, flags);
        if (token) SLReflex::MarkPresentEnd(*token);
        SLDlssg::OnPresentPost();
        return hr;
    }

    HRESULT STDMETHODCALLTYPE Hooked_CreateCommandQueue(
        ID3D12Device* This, const D3D12_COMMAND_QUEUE_DESC* desc, REFIID riid, void** ppQueue)
    {
        static thread_local bool t_inProxyCreate = false;
        if (t_inProxyCreate)
            return g_origCreateCommandQueue(This, desc, riid, ppQueue);

        if (!g_proxyDeviceTried)
        {
            g_proxyDeviceTried = true;
            g_device = This;
            SLCore::SetDevice(This);

            ID3D12Device* dev = This;
            sl::Result ru = slUpgradeInterface(reinterpret_cast<void**>(&dev));
            if (ru == sl::Result::eOk && dev && dev != This)
                g_proxyDevice.Attach(dev);
            Logf(ru == sl::Result::eOk ? 0 : 2,
                 "CreateCommandQueue hook: slUpgradeInterface(device) -> %s%s",
                 R(ru), g_proxyDevice ? " (proxy device cached)" : "");
        }

        if (g_proxyDevice)
        {
            t_inProxyCreate = true;
            HRESULT hr = g_proxyDevice->CreateCommandQueue(desc, riid, ppQueue);
            t_inProxyCreate = false;
            static bool s_logged = false;
            if (!s_logged)
            {
                s_logged = true;
                Logf(SUCCEEDED(hr) ? 0 : 2,
                     "CreateCommandQueue routed via SL proxy device (Type=%d) -> hr=0x%08lx",
                     desc ? (int)desc->Type : -1, (unsigned long)hr);
            }
            return hr;
        }

        return g_origCreateCommandQueue(This, desc, riid, ppQueue);
    }

    HRESULT STDMETHODCALLTYPE Hooked_CreateSwapChainForHwnd(
        IDXGIFactory2* This, IUnknown* pDevice, HWND hWnd,
        const DXGI_SWAP_CHAIN_DESC1* pDesc, const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* pFS,
        IDXGIOutput* pOut, IDXGISwapChain1** ppSwapChain)
    {
        static thread_local bool t_inProxyCreate = false;
        if (t_inProxyCreate)
        {
            IUnknown* nativeQ = NativeIfProxy(pDevice);
            return g_origCreateSwapChainForHwnd(This, nativeQ, hWnd, pDesc, pFS, pOut, ppSwapChain);
        }

        Logf(0, "CreateSwapChainForHwnd intercepted");

        if (SLHooks::IsQueueProxyActive())
        {
            t_inProxyCreate = true;
            HRESULT hr = CreateSwapChainViaProxyFactory(pDevice, hWnd, pDesc, pFS, pOut, ppSwapChain);
            t_inProxyCreate = false;
            if (SUCCEEDED(hr) && ppSwapChain && *ppSwapChain)
                AdoptSwapChain(ppSwapChain, pDevice, true);
            return hr;
        }

        HRESULT hr = g_origCreateSwapChainForHwnd(This, pDevice, hWnd, pDesc, pFS, pOut, ppSwapChain);
        if (SUCCEEDED(hr) && ppSwapChain && *ppSwapChain)
            AdoptSwapChain(ppSwapChain, pDevice, false);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE Hooked_CreateSwapChain(
        IDXGIFactory* This, IUnknown* pDevice, DXGI_SWAP_CHAIN_DESC* pDesc, IDXGISwapChain** ppSwapChain)
    {
        Logf(0, "CreateSwapChain intercepted");
        HRESULT hr = g_origCreateSwapChain(This, pDevice, pDesc, ppSwapChain);
        if (SUCCEEDED(hr) && ppSwapChain && *ppSwapChain)
        {
            ComPtr<IDXGISwapChain1> sc1;
            if (SUCCEEDED((*ppSwapChain)->QueryInterface(IID_PPV_ARGS(&sc1))) && sc1)
            {
                IDXGISwapChain1* p = sc1.Get();
                AdoptSwapChain(&p, pDevice, false);
            }
        }
        return hr;
    }

    void InstallDeviceQueueHook()
    {
        bool expected = false;
        if (!g_deviceHooked.compare_exchange_strong(expected, true)) return;

        ComPtr<ID3D12Device> tmp;
        HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&tmp));
        if (FAILED(hr) || !tmp)
        {
            Logf(2, "InstallDeviceQueueHook: D3D12CreateDevice failed hr=0x%08lx", (unsigned long)hr);
            g_deviceHooked.store(false);
            return;
        }

        g_origCreateCommandQueue = reinterpret_cast<PFN_CreateCommandQueue>(
            PatchSlot(tmp.Get(), kCreateCommandQueueVTIdx, &Hooked_CreateCommandQueue));
        const bool ok = g_origCreateCommandQueue != nullptr;
        Logf(ok ? 0 : 2, "Device CreateCommandQueue hook (vtbl idx %u): %s",
             kCreateCommandQueueVTIdx, ok ? "installed (queues will be SL-proxied)" : "FAILED");
        if (!ok) g_deviceHooked.store(false);
    }

    void InstallFactoryHook()
    {
        bool expected = false;
        if (!g_factoryHooked.compare_exchange_strong(expected, true)) return;

        ComPtr<IDXGIFactory2> factory;
        if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))) || !factory)
        {
            Logf(2, "InstallFactoryHook: CreateDXGIFactory2 failed");
            g_factoryHooked.store(false);
            return;
        }

        g_origCreateSwapChain = reinterpret_cast<PFN_CreateSwapChain>(
            PatchSlot(factory.Get(), kCreateSwapChainVTIdx, &Hooked_CreateSwapChain));
        g_origCreateSwapChainForHwnd = reinterpret_cast<PFN_CreateSwapChainForHwnd>(
            PatchSlot(factory.Get(), kCreateSwapChainForHwndVTIdx, &Hooked_CreateSwapChainForHwnd));
        const bool ok = g_origCreateSwapChain && g_origCreateSwapChainForHwnd;
        Logf(ok ? 0 : 2, ok ? "Factory hook installed (CreateSwapChain[ForHwnd])"
                            : "Factory hook PARTIAL/FAILED");
    }
}

void SLHooks::InstallPresentPathHooks()
{
    InstallDeviceQueueHook();
    InstallFactoryHook();
}

bool SLHooks::IsQueueProxyActive()
{
    return g_proxyDevice != nullptr;
}

uint32_t SLHooks::CurrentBackBufferIndex()
{
    return g_proxySC3 ? g_proxySC3->GetCurrentBackBufferIndex() : 0xFFFFFFFFu;
}

void SLHooks::Shutdown()
{
    g_proxySwapchain = nullptr;
    g_proxySC3.Reset();
    g_device.Reset();
    g_proxyDevice.Reset();
    g_proxyFactory.Reset();
    g_proxyDeviceTried = false;
    g_proxyFactoryTried = false;
}
