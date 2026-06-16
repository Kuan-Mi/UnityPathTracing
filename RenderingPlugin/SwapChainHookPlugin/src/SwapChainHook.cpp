// SwapChainHook.cpp — see header.
//
// Diagnostic-only. Patches the DXGI factory + swapchain vtables to log how
// Unity creates and presents its swapchain, then calls through unchanged.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_5.h>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <wrl/client.h>

#include "SwapChainHook.h"
#include "SwapChainProxy.h"

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

    // NOTE: we must NOT change the swapchain BufferCount underneath Unity. Unity
    // sizes its RTV array from its OWN requested count (not a GetDesc1 re-query),
    // so a larger count makes GetCurrentBackBufferIndex return indices Unity has
    // no RTV for → out-of-bounds → device removed (observed crash at present #4).
    // The frame-doubler therefore works within Unity's buffer count.

    PFN_CreateSwapChain          g_OrigCreateSwapChain        = nullptr;
    PFN_CreateSwapChainForHwnd   g_OrigCreateSwapChainForHwnd = nullptr;
    PFN_Present                  g_OrigPresent                = nullptr;
    PFN_Present1                 g_OrigPresent1               = nullptr;

    SwapChainHook::LogFn  g_Logger = nullptr;

    std::atomic<uint64_t> g_PresentCount{0};

    // --- present-injection test (R2 present-pacing feasibility) -------------
    // Interception is already proven; this asks the next question: can the
    // plugin push an EXTRA present through Unity's flip-model waitable swapchain
    // (the doubled cadence FG needs) without erroring or hanging? For a bounded
    // window of real presents we issue one additional original Present per real
    // one (duplicate of the just-presented frame — no rendering yet) and log the
    // HRESULT + how long the real vs the injected present each take.
    //
    // Hardcoded toggle (no env var). NOTE: naive present-doubling DEVICE-REMOVES
    // the GPU within 1-2 injected presents — Unity owns the swapchain back-buffer
    // rotation (1 present/frame) and an extra Present desyncs GetCurrentBackBuffer
    // Index, hanging the GPU. Left OFF; proper present-pacing needs a swapchain
    // proxy that owns the buffer bookkeeping, not a passive side hook.
    std::atomic<bool>     g_InjectEnabled{false};
    constexpr uint64_t    g_InjectStart = 80;
    constexpr uint64_t    g_InjectEnd   = 280;
    std::atomic<uint64_t> g_InjectCount{0};
    std::atomic<bool>     g_ConfigRead{false};
    LARGE_INTEGER         g_QpcFreq{};

    void Logf(int level, const char* fmt, ...);   // defined below

    void ReadPresentTestConfig()
    {
        bool expected = false;
        if (!g_ConfigRead.compare_exchange_strong(expected, true)) return;

        QueryPerformanceFrequency(&g_QpcFreq);
        Logf(0, "Present-injection test ENABLED (window presents [%llu,%llu)) "
                "- will issue 1 extra present per real present in that range",
             (unsigned long long)g_InjectStart, (unsigned long long)g_InjectEnd);
    }

    double QpcMs(LARGE_INTEGER a, LARGE_INTEGER b)
    {
        if (g_QpcFreq.QuadPart == 0) return 0.0;
        return (double)(b.QuadPart - a.QuadPart) * 1000.0 / (double)g_QpcFreq.QuadPart;
    }

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

    // Replace Unity's just-created swapchain with our proxy (so we own the
    // present path the proper way). If wrapping fails, fall back to the old
    // passive Present vtable hook. Marks the present path "handled" so the
    // DlssgProbe poller stops trying to install the fallback hook.
    void InstallProxyOrPresentHook(IDXGISwapChain1** ppSwapChain, IUnknown* pDevice)
    {
        // The swapchain's "device" arg is Unity's ID3D12CommandQueue (flip-model
        // present queue); the proxy records its frame-doubler copy onto it.
        ComPtr<ID3D12CommandQueue> queue;
        if (pDevice) pDevice->QueryInterface(IID_PPV_ARGS(&queue));

        IDXGISwapChain1* real  = *ppSwapChain;
        IDXGISwapChain1* proxy = SwapChainProxy::Wrap(real, queue.Get(), g_Logger);
        if (proxy)
        {
            real->Release();              // the proxy now holds its own reference
            *ppSwapChain = proxy;         // hand Unity the proxy instead
            g_PresentHooked.store(true, std::memory_order_release);
            Logf(0, "Returned PROXY swapchain to Unity (present owned by proxy).");
        }
        else
        {
            SwapChainHook::TryInstallPresentHook(real);   // fallback: vtable hook
        }
    }

    HRESULT STDMETHODCALLTYPE Hooked_CreateSwapChain(
        IDXGIFactory* This, IUnknown* pDevice,
        DXGI_SWAP_CHAIN_DESC* pDesc, IDXGISwapChain** ppSwapChain)
    {
        Logf(0, "CreateSwapChain called (factory=%p device=%p)", (void*)This, (void*)pDevice);
        LogSwapChainDesc(pDesc);
        // Unity D3D12 uses CreateSwapChainForHwnd in practice; this base path is
        // kept consistent (no BufferCount change, same proxy helper).
        HRESULT hr = g_OrigCreateSwapChain(This, pDevice, pDesc, ppSwapChain);
        Logf(0, "CreateSwapChain -> hr=0x%08lx swapChain=%p",
             (unsigned long)hr, (void*)(ppSwapChain ? *ppSwapChain : nullptr));
        if (SUCCEEDED(hr) && ppSwapChain && *ppSwapChain)
            InstallProxyOrPresentHook(reinterpret_cast<IDXGISwapChain1**>(ppSwapChain), pDevice);
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
            InstallProxyOrPresentHook(reinterpret_cast<IDXGISwapChain1**>(ppSwapChain), pDevice);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE Hooked_Present(
        IDXGISwapChain* This, UINT SyncInterval, UINT Flags)
    {
        const uint64_t n = g_PresentCount.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n <= 4 || (n & 0xFF) == 0)
            Logf(0, "Present #%llu (swapChain=%p sync=%u flags=0x%lx)",
                 (unsigned long long)n, (void*)This, SyncInterval, (unsigned long)Flags);

        LARGE_INTEGER t0; QueryPerformanceCounter(&t0);
        HRESULT hr = g_OrigPresent(This, SyncInterval, Flags);
        LARGE_INTEGER t1; QueryPerformanceCounter(&t1);

        if (g_InjectEnabled.load(std::memory_order_acquire) &&
            n >= g_InjectStart && n < g_InjectEnd)
        {
            if (n == g_InjectStart) Logf(0, "INJECT window START at present #%llu",
                                         (unsigned long long)n);
            HRESULT hr2 = g_OrigPresent(This, 0, Flags);
            LARGE_INTEGER t2; QueryPerformanceCounter(&t2);
            const uint64_t ic = g_InjectCount.fetch_add(1, std::memory_order_relaxed) + 1;
            if (ic <= 4 || (ic & 0x7F) == 0 || FAILED(hr2))
                Logf(FAILED(hr2) ? 2 : 0,
                     "INJECT #%llu (present #%llu) extra Present hr=0x%08lx | "
                     "real=%.2fms extra=%.2fms",
                     (unsigned long long)ic, (unsigned long long)n,
                     (unsigned long)hr2, QpcMs(t0, t1), QpcMs(t1, t2));
            if (FAILED(hr2))
            {
                g_InjectEnabled.store(false, std::memory_order_release);
                Logf(2, "INJECT disabled after failure (hr=0x%08lx).", (unsigned long)hr2);
            }
        }
        return hr;
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

        LARGE_INTEGER t0; QueryPerformanceCounter(&t0);
        HRESULT hr = g_OrigPresent1(This, SyncInterval, PresentFlags, pPresentParameters);
        LARGE_INTEGER t1; QueryPerformanceCounter(&t1);

        // R2 test: inject one extra present inside the window. Duplicate the just
        // -presented frame (sync=0 so we never block on vsync; flip-model +
        // ALLOW_TEARING lets it tear/flip immediately). This proves we can drive
        // an extra frame through Unity's swapchain and measures its cost.
        if (g_InjectEnabled.load(std::memory_order_acquire) &&
            n >= g_InjectStart && n < g_InjectEnd)
        {
            if (n == g_InjectStart) Logf(0, "INJECT window START at present #%llu",
                                         (unsigned long long)n);
            HRESULT hr2 = g_OrigPresent1(This, 0, PresentFlags, pPresentParameters);
            LARGE_INTEGER t2; QueryPerformanceCounter(&t2);
            const uint64_t ic = g_InjectCount.fetch_add(1, std::memory_order_relaxed) + 1;
            if (ic <= 4 || (ic & 0x7F) == 0 || FAILED(hr2))
                Logf(FAILED(hr2) ? 2 : 0,
                     "INJECT #%llu (present #%llu) extra Present1 hr=0x%08lx | "
                     "real=%.2fms extra=%.2fms",
                     (unsigned long long)ic, (unsigned long long)n,
                     (unsigned long)hr2, QpcMs(t0, t1), QpcMs(t1, t2));
            if (FAILED(hr2))
            {
                g_InjectEnabled.store(false, std::memory_order_release);
                Logf(2, "INJECT disabled after failure (hr=0x%08lx) — extra present "
                        "rejected by the swapchain.", (unsigned long)hr2);
            }
        }
        else if (g_InjectEnabled.load(std::memory_order_acquire) && n == g_InjectEnd)
        {
            Logf(0, "INJECT window END at present #%llu (%llu extra presents issued)",
                 (unsigned long long)n,
                 (unsigned long long)g_InjectCount.load(std::memory_order_relaxed));
        }
        return hr;
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
        ReadPresentTestConfig();   // one-time: read NR_FG_PRESENT_TEST env config
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
