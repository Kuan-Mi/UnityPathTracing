// SwapChainHook.cpp — see header.
//
// Diagnostic-only. Patches the DXGI factory + swapchain vtables to log how
// Unity creates and presents its swapchain, then calls through unchanged.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dxgi1_5.h>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <wrl/client.h>

#include "SwapChainHook.h"

using Microsoft::WRL::ComPtr;

namespace
{
    // --- vtable indices ----------------------------------------------------
    // IUnknown(0-2) + IDXGIObject(3-6) + IDXGIFactory: EnumAdapters(7),
    // MakeWindowAssociation(8), GetWindowAssociation(9), CreateSwapChain(10).
    constexpr UINT kCreateSwapChainVTIdx = 10;
    // IDXGIFactory1(+EnumAdapters1=12, IsCurrent=13) + IDXGIFactory2:
    // IsWindowedStereoEnabled(14), CreateSwapChainForHwnd(15).
    constexpr UINT kCreateSwapChainForHwndVTIdx = 15;
    // IUnknown(0-2)+IDXGIObject(3-6)+IDXGIDeviceSubObject(7)+IDXGISwapChain: Present(8).
    constexpr UINT kPresentVTIdx = 8;
    // IDXGISwapChain(8-17) + IDXGISwapChain1(GetDesc1=18, GetFullscreenDesc=19,
    // GetHwnd=20, GetCoreWindow=21, Present1=22).
    constexpr UINT kPresent1VTIdx = 22;

    using PFN_CreateSwapChain = HRESULT(STDMETHODCALLTYPE*)(
        IDXGIFactory*, IUnknown*, DXGI_SWAP_CHAIN_DESC*, IDXGISwapChain**);
    using PFN_CreateSwapChainForHwnd = HRESULT(STDMETHODCALLTYPE*)(
        IDXGIFactory2*, IUnknown*, HWND, const DXGI_SWAP_CHAIN_DESC1*,
        const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*, IDXGIOutput*, IDXGISwapChain1**);
    using PFN_Present = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, UINT);
    using PFN_Present1 = HRESULT(STDMETHODCALLTYPE*)(
        IDXGISwapChain1*, UINT, UINT, const DXGI_PRESENT_PARAMETERS*);

    std::atomic<bool>  g_FactoryHooked{false};
    std::atomic<bool>  g_PresentHooked{false};

    PFN_CreateSwapChain          g_OrigCreateSwapChain        = nullptr;
    PFN_CreateSwapChainForHwnd   g_OrigCreateSwapChainForHwnd = nullptr;
    PFN_Present                  g_OrigPresent                = nullptr;
    PFN_Present1                 g_OrigPresent1               = nullptr;

    SwapChainHook::LogFn  g_Logger = nullptr;

    std::atomic<uint64_t> g_PresentCount{0};

    void Logf(int level, const char* fmt, ...)
    {
        char buf[640];
        va_list ap;
        va_start(ap, fmt);
        int n = _vsnprintf_s(buf, sizeof(buf), _TRUNCATE, fmt, ap);
        va_end(ap);
        if (n < 0) buf[sizeof(buf) - 1] = '\0';

        const char* tag = (level == 2) ? "[NR/SwapChainHook ERR] "
                        : (level == 1) ? "[NR/SwapChainHook WRN] "
                                       : "[NR/SwapChainHook] ";
        char prefixed[704];
        _snprintf_s(prefixed, sizeof(prefixed), _TRUNCATE, "%s%s", tag, buf);

        if (g_Logger)
        {
            g_Logger(level, prefixed);
        }
        else
        {
            char withNl[720];
            _snprintf_s(withNl, sizeof(withNl), _TRUNCATE, "%s\n", prefixed);
            OutputDebugStringA(withNl);
        }
    }

    bool Unprotect(void* slot, DWORD* oldProtect)
    {
        return VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, oldProtect) != 0;
    }

    // Patch one vtable slot, returning the previous function pointer (or null).
    void* PatchSlot(void* objectWithVtable, UINT index, void* hook)
    {
        void** vtable = *reinterpret_cast<void***>(objectWithVtable);
        void** slot   = vtable + index;

        DWORD oldProtect = 0;
        if (!Unprotect(slot, &oldProtect))
        {
            Logf(2, "VirtualProtect failed (idx=%u, GetLastError=%lu)",
                 index, (unsigned long)GetLastError());
            return nullptr;
        }
        void* orig = *slot;
        *slot = hook;
        VirtualProtect(slot, sizeof(void*), oldProtect, &oldProtect);
        return orig;
    }

    // --- hooked implementations -------------------------------------------
    void LogSwapChainDesc(const DXGI_SWAP_CHAIN_DESC* d)
    {
        if (!d) return;
        Logf(0, "  desc: %ux%u fmt=%d buffers=%u usage=0x%lx swapEffect=%d flags=0x%lx "
                "sampleCount=%u windowed=%d",
             d->BufferDesc.Width, d->BufferDesc.Height, (int)d->BufferDesc.Format,
             d->BufferCount, (unsigned long)d->BufferUsage, (int)d->SwapEffect,
             (unsigned long)d->Flags, d->SampleDesc.Count, (int)d->Windowed);
    }

    void LogSwapChainDesc1(const DXGI_SWAP_CHAIN_DESC1* d)
    {
        if (!d) return;
        Logf(0, "  desc1: %ux%u fmt=%d buffers=%u usage=0x%lx swapEffect=%d flags=0x%lx "
                "sampleCount=%u scaling=%d alphaMode=%d",
             d->Width, d->Height, (int)d->Format, d->BufferCount,
             (unsigned long)d->BufferUsage, (int)d->SwapEffect, (unsigned long)d->Flags,
             d->SampleDesc.Count, (int)d->Scaling, (int)d->AlphaMode);
    }

    HRESULT STDMETHODCALLTYPE Hooked_CreateSwapChain(
        IDXGIFactory* This, IUnknown* pDevice,
        DXGI_SWAP_CHAIN_DESC* pDesc, IDXGISwapChain** ppSwapChain)
    {
        Logf(0, "CreateSwapChain called (factory=%p device=%p)", (void*)This, (void*)pDevice);
        LogSwapChainDesc(pDesc);
        HRESULT hr = g_OrigCreateSwapChain(This, pDevice, pDesc, ppSwapChain);
        Logf(0, "CreateSwapChain -> hr=0x%08lx swapChain=%p",
             (unsigned long)hr, (void*)(ppSwapChain ? *ppSwapChain : nullptr));
        if (SUCCEEDED(hr) && ppSwapChain && *ppSwapChain)
            SwapChainHook::TryInstallPresentHook(*ppSwapChain);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE Hooked_CreateSwapChainForHwnd(
        IDXGIFactory2* This, IUnknown* pDevice, HWND hWnd,
        const DXGI_SWAP_CHAIN_DESC1* pDesc,
        const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* pFullscreenDesc,
        IDXGIOutput* pRestrictToOutput, IDXGISwapChain1** ppSwapChain)
    {
        Logf(0, "CreateSwapChainForHwnd called (factory=%p device=%p hwnd=%p)",
             (void*)This, (void*)pDevice, (void*)hWnd);
        LogSwapChainDesc1(pDesc);
        HRESULT hr = g_OrigCreateSwapChainForHwnd(
            This, pDevice, hWnd, pDesc, pFullscreenDesc, pRestrictToOutput, ppSwapChain);
        Logf(0, "CreateSwapChainForHwnd -> hr=0x%08lx swapChain=%p",
             (unsigned long)hr, (void*)(ppSwapChain ? *ppSwapChain : nullptr));
        if (SUCCEEDED(hr) && ppSwapChain && *ppSwapChain)
            SwapChainHook::TryInstallPresentHook(*ppSwapChain);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE Hooked_Present(
        IDXGISwapChain* This, UINT SyncInterval, UINT Flags)
    {
        const uint64_t n = g_PresentCount.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n <= 4 || (n & 0xFF) == 0)
            Logf(0, "Present #%llu (swapChain=%p sync=%u flags=0x%lx)",
                 (unsigned long long)n, (void*)This, SyncInterval, (unsigned long)Flags);
        return g_OrigPresent(This, SyncInterval, Flags);
    }

    HRESULT STDMETHODCALLTYPE Hooked_Present1(
        IDXGISwapChain1* This, UINT SyncInterval, UINT PresentFlags,
        const DXGI_PRESENT_PARAMETERS* pPresentParameters)
    {
        const uint64_t n = g_PresentCount.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n <= 4 || (n & 0xFF) == 0)
        {
            UINT dirty = pPresentParameters ? pPresentParameters->DirtyRectsCount : 0;
            Logf(0, "Present1 #%llu (swapChain=%p sync=%u flags=0x%lx dirtyRects=%u)",
                 (unsigned long long)n, (void*)This, SyncInterval,
                 (unsigned long)PresentFlags, dirty);
        }
        return g_OrigPresent1(This, SyncInterval, PresentFlags, pPresentParameters);
    }
}

namespace SwapChainHook
{
    void SetLogger(LogFn fn)
    {
        g_Logger = fn;
        Logf(0, "Logger attached");
    }

    bool InstallFactoryHook()
    {
        if (g_FactoryHooked.load(std::memory_order_acquire)) return true;

        // Race-guard: only the first caller patches.
        bool expected = false;
        if (!g_FactoryHooked.compare_exchange_strong(expected, true,
                                                     std::memory_order_acq_rel))
            return true;

        // A throwaway factory just to reach the shared vtable. IDXGIFactory2 is
        // available on Win7+ (platform update) / Win8+, which Unity D3D12 requires.
        ComPtr<IDXGIFactory2> factory;
        HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
        if (FAILED(hr) || !factory)
        {
            Logf(2, "InstallFactoryHook: CreateDXGIFactory2 failed hr=0x%08lx",
                 (unsigned long)hr);
            g_FactoryHooked.store(false, std::memory_order_release);
            return false;
        }

        // CreateSwapChain lives on IDXGIFactory; CreateSwapChainForHwnd on
        // IDXGIFactory2. Both share the same concrete vtable here.
        g_OrigCreateSwapChain = reinterpret_cast<PFN_CreateSwapChain>(
            PatchSlot(factory.Get(), kCreateSwapChainVTIdx,
                      reinterpret_cast<void*>(&Hooked_CreateSwapChain)));
        g_OrigCreateSwapChainForHwnd = reinterpret_cast<PFN_CreateSwapChainForHwnd>(
            PatchSlot(factory.Get(), kCreateSwapChainForHwndVTIdx,
                      reinterpret_cast<void*>(&Hooked_CreateSwapChainForHwnd)));

        const bool ok = g_OrigCreateSwapChain && g_OrigCreateSwapChainForHwnd;
        Logf(ok ? 0 : 2,
             "InstallFactoryHook: %s (CreateSwapChain orig=%p, ForHwnd orig=%p)",
             ok ? "SUCCESS" : "PARTIAL/FAILED",
             (void*)g_OrigCreateSwapChain, (void*)g_OrigCreateSwapChainForHwnd);
        return ok;
    }

    bool TryInstallPresentHook(IDXGISwapChain* swapChain)
    {
        if (g_PresentHooked.load(std::memory_order_acquire)) return true;
        if (!swapChain) return false;

        bool expected = false;
        if (!g_PresentHooked.compare_exchange_strong(expected, true,
                                                     std::memory_order_acq_rel))
            return true;

        g_OrigPresent = reinterpret_cast<PFN_Present>(
            PatchSlot(swapChain, kPresentVTIdx,
                      reinterpret_cast<void*>(&Hooked_Present)));

        // Present1 only exists if this object is an IDXGISwapChain1. Querying
        // confirms the vtable is wide enough before we patch index 22.
        ComPtr<IDXGISwapChain1> sc1;
        if (SUCCEEDED(swapChain->QueryInterface(IID_PPV_ARGS(&sc1))) && sc1)
        {
            g_OrigPresent1 = reinterpret_cast<PFN_Present1>(
                PatchSlot(sc1.Get(), kPresent1VTIdx,
                          reinterpret_cast<void*>(&Hooked_Present1)));
        }

        const bool ok = g_OrigPresent != nullptr;
        if (!ok) g_PresentHooked.store(false, std::memory_order_release);
        Logf(ok ? 0 : 2,
             "TryInstallPresentHook: %s (swapChain=%p Present orig=%p Present1 orig=%p)",
             ok ? "SUCCESS" : "FAILED", (void*)swapChain,
             (void*)g_OrigPresent, (void*)g_OrigPresent1);
        return ok;
    }

    bool IsPresentHookInstalled()
    {
        return g_PresentHooked.load(std::memory_order_acquire);
    }
}
