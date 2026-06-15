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

    // --- DLSS-FG present-pacing stub --------------------------------------
    // Stands up the FG present plumbing without AI: on each Unity Present1 it
    // retains the real frame, presents it, paces ~half a frame, then presents a
    // second (duplicate) frame. Content-preserving, so safe on any GPU. The
    // duplicate present is the slot EvaluateFeature's interpolated frame will
    // occupy on Ada hardware. All work runs on Unity's present queue, so GPU
    // ordering relative to Present is preserved without extra cross-queue sync.
    struct PacingStub
    {
        std::atomic<bool>             enabled{false};
        ComPtr<ID3D12Device>          device;
        ComPtr<ID3D12CommandQueue>    queue;        // Unity's present queue
        ComPtr<ID3D12CommandAllocator> alloc;
        ComPtr<ID3D12GraphicsCommandList> list;
        ComPtr<ID3D12Fence>           fence;
        UINT64                        fenceVal = 0;
        HANDLE                        fenceEvt = nullptr;

        ComPtr<ID3D12Resource>        scratch;      // holds the retained real frame
        UINT                          scW = 0, scH = 0;
        DXGI_FORMAT                   scFmt = DXGI_FORMAT_UNKNOWN;

        LARGE_INTEGER                 qpcFreq{};
        LARGE_INTEGER                 lastPresentQpc{};
        std::atomic<uint64_t>         pacedFrames{0};

        bool EnsureObjects()
        {
            if (!device) return false;
            if (!alloc &&
                FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                      IID_PPV_ARGS(&alloc))))
            { Logf(2, "PacingStub: CreateCommandAllocator failed"); return false; }
            if (!list)
            {
                if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                     alloc.Get(), nullptr, IID_PPV_ARGS(&list))))
                { Logf(2, "PacingStub: CreateCommandList failed"); return false; }
                list->Close(); // created in the open state; close so the first Reset is valid
            }
            if (!fence &&
                FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
            { Logf(2, "PacingStub: CreateFence failed"); return false; }
            if (!fenceEvt)
                fenceEvt = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (!qpcFreq.QuadPart) QueryPerformanceFrequency(&qpcFreq);
            return alloc && list && fence && fenceEvt;
        }

        // (Re)create the scratch texture to match the swapchain backbuffer.
        bool EnsureScratch(ID3D12Resource* bb)
        {
            D3D12_RESOURCE_DESC d = bb->GetDesc();
            if (scratch && scW == (UINT)d.Width && scH == d.Height && scFmt == d.Format)
                return true;

            D3D12_HEAP_PROPERTIES hp{};
            hp.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC rd = d;
            rd.Flags = D3D12_RESOURCE_FLAG_NONE; // copy target only
            scratch.Reset();
            HRESULT hr = device->CreateCommittedResource(
                &hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON,
                nullptr, IID_PPV_ARGS(&scratch));
            if (FAILED(hr))
            {
                Logf(2, "PacingStub: scratch CreateCommittedResource failed hr=0x%08lx",
                     (unsigned long)hr);
                return false;
            }
            scW = (UINT)d.Width; scH = d.Height; scFmt = d.Format;
            Logf(0, "PacingStub: scratch (re)created %ux%u fmt=%d", scW, scH, (int)scFmt);
            return true;
        }

        void Barrier(ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
        {
            D3D12_RESOURCE_BARRIER b{};
            b.Type  = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition.pResource   = r;
            b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            b.Transition.StateBefore = from;
            b.Transition.StateAfter  = to;
            list->ResourceBarrier(1, &b);
        }

        // Copy full-resource src->dst, with src/dst entering and leaving PRESENT/
        // COMMON state (Unity leaves backbuffers in PRESENT == COMMON). Records,
        // executes on the present queue, and blocks until complete so the
        // allocator can be reused next call.
        bool CopyBlocking(ID3D12Resource* dst, D3D12_RESOURCE_STATES dstState,
                          ID3D12Resource* src, D3D12_RESOURCE_STATES srcState)
        {
            if (FAILED(alloc->Reset()) || FAILED(list->Reset(alloc.Get(), nullptr)))
            { Logf(2, "PacingStub: cmd reset failed"); return false; }

            Barrier(dst, dstState, D3D12_RESOURCE_STATE_COPY_DEST);
            Barrier(src, srcState, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->CopyResource(dst, src);
            Barrier(dst, D3D12_RESOURCE_STATE_COPY_DEST, dstState);
            Barrier(src, D3D12_RESOURCE_STATE_COPY_SOURCE, srcState);
            if (FAILED(list->Close())) { Logf(2, "PacingStub: list close failed"); return false; }

            ID3D12CommandList* lists[] = { list.Get() };
            queue->ExecuteCommandLists(1, lists);
            const UINT64 v = ++fenceVal;
            if (FAILED(queue->Signal(fence.Get(), v))) return false;
            if (fence->GetCompletedValue() < v)
            {
                fence->SetEventOnCompletion(v, fenceEvt);
                WaitForSingleObject(fenceEvt, INFINITE);
            }
            return true;
        }

        // Pace by sleeping roughly half the last real-frame interval, so the two
        // presents are spread in time rather than back-to-back.
        void Pace()
        {
            LARGE_INTEGER now; QueryPerformanceCounter(&now);
            if (lastPresentQpc.QuadPart && qpcFreq.QuadPart)
            {
                double frameMs = (double)(now.QuadPart - lastPresentQpc.QuadPart)
                                 * 1000.0 / (double)qpcFreq.QuadPart;
                double halfMs = frameMs * 0.5;
                if (halfMs > 0.2 && halfMs < 33.0)
                    Sleep((DWORD)(halfMs + 0.5));
            }
            lastPresentQpc = now;
        }
    };

    PacingStub g_stub;

    // Run the paced double-present for the given swapchain. Returns the HRESULT
    // of the final (real) present. presentFn issues one original Present1.
    HRESULT RunPacingStub(IDXGISwapChain1* sc, UINT sync, UINT flags,
                          const DXGI_PRESENT_PARAMETERS* params,
                          PFN_Present1 origPresent1)
    {
        if (!g_stub.EnsureObjects())
            return origPresent1(sc, sync, flags, params); // fall back to passthrough

        ComPtr<IDXGISwapChain3> sc3;
        if (FAILED(sc->QueryInterface(IID_PPV_ARGS(&sc3))) || !sc3)
            return origPresent1(sc, sync, flags, params);

        UINT i0 = sc3->GetCurrentBackBufferIndex();
        ComPtr<ID3D12Resource> bb0;
        if (FAILED(sc->GetBuffer(i0, IID_PPV_ARGS(&bb0))) || !bb0 ||
            !g_stub.EnsureScratch(bb0.Get()))
            return origPresent1(sc, sync, flags, params);

        // 1. Retain the real frame (backbuffer -> scratch), both in PRESENT state.
        g_stub.CopyBlocking(g_stub.scratch.Get(), D3D12_RESOURCE_STATE_COMMON,
                            bb0.Get(), D3D12_RESOURCE_STATE_PRESENT);

        // 2. Present the real frame (first of the pair). Advances the buffer index.
        HRESULT hr1 = origPresent1(sc, sync, flags, params);

        // 3. Pace ~half a frame.
        g_stub.Pace();

        // 4. Restore the real frame into the new current backbuffer.
        UINT i1 = sc3->GetCurrentBackBufferIndex();
        ComPtr<ID3D12Resource> bb1;
        HRESULT hr2 = hr1;
        if (SUCCEEDED(sc->GetBuffer(i1, IID_PPV_ARGS(&bb1))) && bb1)
        {
            g_stub.CopyBlocking(bb1.Get(), D3D12_RESOURCE_STATE_PRESENT,
                                g_stub.scratch.Get(), D3D12_RESOURCE_STATE_COMMON);
            // 5. Present the duplicate (the FG "generated" frame slot).
            hr2 = origPresent1(sc, sync, flags, params);
        }

        const uint64_t n = g_stub.pacedFrames.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n <= 4 || (n & 0xFF) == 0)
            Logf(0, "PacingStub: paced pair #%llu (idx %u->%u, real hr=0x%08lx, dup hr=0x%08lx)",
                 (unsigned long long)n, i0, i1, (unsigned long)hr1, (unsigned long)hr2);
        return hr2;
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
        // When the FG pacing stub is active, replace the single present with the
        // retain -> present -> pace -> present pair. Otherwise pass through.
        if (g_stub.enabled.load(std::memory_order_acquire))
            return RunPacingStub(This, SyncInterval, PresentFlags, pPresentParameters,
                                 g_OrigPresent1);
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

    void EnablePacingStub(ID3D12Device* device, ID3D12CommandQueue* presentQueue)
    {
        if (g_stub.enabled.load(std::memory_order_acquire)) return;
        if (!device || !presentQueue)
        {
            Logf(2, "EnablePacingStub: null device/queue");
            return;
        }
        g_stub.device = device;
        g_stub.queue  = presentQueue;
        if (!g_stub.EnsureObjects())
        {
            Logf(2, "EnablePacingStub: failed to create D3D12 objects");
            g_stub.device.Reset();
            g_stub.queue.Reset();
            return;
        }
        g_stub.enabled.store(true, std::memory_order_release);
        Logf(0, "PacingStub ENABLED (device=%p queue=%p) — Present1 will be doubled+paced",
             (void*)device, (void*)presentQueue);
    }

    bool IsPacingStubEnabled()
    {
        return g_stub.enabled.load(std::memory_order_acquire);
    }
}
