// SwapChainProxy.cpp — see header.
//
// Stage 1: every IDXGISwapChain4 method forwards straight to the wrapped real
// swapchain. The only non-trivial method is QueryInterface, which must return
// THIS proxy for the swapchain interfaces (so Unity keeps talking to us) while
// forwarding unknown interfaces to the real object for compatibility.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_5.h>
#include <atomic>
#include <cstdio>
#include <wrl/client.h>

#include "SwapChainProxy.h"

using Microsoft::WRL::ComPtr;

namespace
{
    SwapChainProxy::LogFn g_Log = nullptr;

    // Stage 2 frame-doubler tuning. Engage after a short warmup so the log shows
    // pass-through baseline first, then the doubled cadence. Fail-safe: any error
    // disables the doubler and reverts to pass-through (never hard-crash again).
    constexpr uint64_t kDoublerWarmupPresents = 120;
    // Diagnostic: double only every Nth real present so the 3-buffer swapchain
    // fully retires queued buffers between doublings. =1 means double every frame
    // (over-commits a 3-buffer waitable swapchain → faults ~1 frame later).
    constexpr uint64_t kDoubleEveryN = 20;

    void Logf(int level, const char* fmt, ...)
    {
        char buf[512];
        va_list ap;
        va_start(ap, fmt);
        _vsnprintf_s(buf, sizeof(buf), _TRUNCATE, fmt, ap);
        va_end(ap);
        char line[576];
        const char* tag = (level == 2) ? "[NR/SwapChainProxy ERR] "
                        : (level == 1) ? "[NR/SwapChainProxy WRN] "
                                       : "[NR/SwapChainProxy] ";
        _snprintf_s(line, sizeof(line), _TRUNCATE, "%s%s", tag, buf);
        if (g_Log) g_Log(level, line);
        else { OutputDebugStringA(line); OutputDebugStringA("\n"); }
    }

    // Pass-through proxy. Implements IDXGISwapChain4 (the widest interface modern
    // D3D12 apps QI for); every call forwards to m_real.
    class Proxy final : public IDXGISwapChain4
    {
    public:
        Proxy(ComPtr<IDXGISwapChain4> real, ComPtr<ID3D12CommandQueue> queue)
            : m_real(std::move(real)), m_queue(std::move(queue)) {}

        // --- IUnknown ------------------------------------------------------
        HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override
        {
            if (!ppv) return E_POINTER;
            if (riid == __uuidof(IUnknown) ||
                riid == __uuidof(IDXGIObject) ||
                riid == __uuidof(IDXGIDeviceSubObject) ||
                riid == __uuidof(IDXGISwapChain) ||
                riid == __uuidof(IDXGISwapChain1) ||
                riid == __uuidof(IDXGISwapChain2) ||
                riid == __uuidof(IDXGISwapChain3) ||
                riid == __uuidof(IDXGISwapChain4))
            {
                *ppv = static_cast<IDXGISwapChain4*>(this);
                AddRef();
                return S_OK;
            }
            // Unknown/private interface: forward to the real object. This can let
            // a caller reach the unwrapped swapchain — log the first few so we
            // know if Unity bypasses us via some interface we don't proxy.
            HRESULT hr = m_real->QueryInterface(riid, ppv);
            if (SUCCEEDED(hr) && m_qiForwardLogged < 8)
            {
                ++m_qiForwardLogged;
                wchar_t g[64];
                StringFromGUID2(riid, g, 64);
                Logf(1, "QueryInterface forwarded to REAL swapchain for %ls "
                        "(not proxied)", g);
            }
            return hr;
        }

        ULONG STDMETHODCALLTYPE AddRef() override
        {
            return (ULONG)m_ref.fetch_add(1, std::memory_order_relaxed) + 1;
        }

        ULONG STDMETHODCALLTYPE Release() override
        {
            ULONG r = (ULONG)m_ref.fetch_sub(1, std::memory_order_acq_rel) - 1;
            if (r == 0)
            {
                Logf(0, "Proxy released (final). Forwarded %llu Present1 / %llu "
                        "Present over its lifetime.",
                     (unsigned long long)m_present1Count.load(),
                     (unsigned long long)m_presentCount.load());
                delete this;
            }
            return r;
        }

        // --- IDXGIObject ---------------------------------------------------
        HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID Name, UINT DataSize, const void* pData) override
        { return m_real->SetPrivateData(Name, DataSize, pData); }
        HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID Name, const IUnknown* pUnknown) override
        { return m_real->SetPrivateDataInterface(Name, pUnknown); }
        HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID Name, UINT* pDataSize, void* pData) override
        { return m_real->GetPrivateData(Name, pDataSize, pData); }
        HRESULT STDMETHODCALLTYPE GetParent(REFIID riid, void** ppParent) override
        { return m_real->GetParent(riid, ppParent); }

        // --- IDXGIDeviceSubObject -----------------------------------------
        HRESULT STDMETHODCALLTYPE GetDevice(REFIID riid, void** ppDevice) override
        { return m_real->GetDevice(riid, ppDevice); }

        // --- IDXGISwapChain ------------------------------------------------
        HRESULT STDMETHODCALLTYPE Present(UINT SyncInterval, UINT Flags) override
        {
            const uint64_t n = m_presentCount.fetch_add(1, std::memory_order_relaxed) + 1;
            if (n <= 4 || (n & 0xFF) == 0)
                Logf(0, "Present #%llu (sync=%u flags=0x%lx) [pass-through]",
                     (unsigned long long)n, SyncInterval, (unsigned long)Flags);
            return m_real->Present(SyncInterval, Flags);
        }
        HRESULT STDMETHODCALLTYPE GetBuffer(UINT Buffer, REFIID riid, void** ppSurface) override
        { return m_real->GetBuffer(Buffer, riid, ppSurface); }
        HRESULT STDMETHODCALLTYPE SetFullscreenState(BOOL Fullscreen, IDXGIOutput* pTarget) override
        { return m_real->SetFullscreenState(Fullscreen, pTarget); }
        HRESULT STDMETHODCALLTYPE GetFullscreenState(BOOL* pFullscreen, IDXGIOutput** ppTarget) override
        { return m_real->GetFullscreenState(pFullscreen, ppTarget); }
        HRESULT STDMETHODCALLTYPE GetDesc(DXGI_SWAP_CHAIN_DESC* pDesc) override
        { return m_real->GetDesc(pDesc); }
        HRESULT STDMETHODCALLTYPE ResizeBuffers(UINT BufferCount, UINT Width, UINT Height,
                                                DXGI_FORMAT NewFormat, UINT SwapChainFlags) override
        {
            // Forward Unity's request unchanged (changing the count breaks Unity's
            // RTV bookkeeping); just refresh our cached count for the doubler.
            Logf(0, "ResizeBuffers(count=%u %ux%u fmt=%d flags=0x%lx)",
                 BufferCount, Width, Height, (int)NewFormat, (unsigned long)SwapChainFlags);
            HRESULT hr = m_real->ResizeBuffers(BufferCount, Width, Height, NewFormat, SwapChainFlags);
            if (SUCCEEDED(hr))
            {
                DXGI_SWAP_CHAIN_DESC1 d{};
                if (SUCCEEDED(m_real->GetDesc1(&d))) m_bufferCount = d.BufferCount;
            }
            return hr;
        }
        HRESULT STDMETHODCALLTYPE ResizeTarget(const DXGI_MODE_DESC* pNewTargetParameters) override
        { return m_real->ResizeTarget(pNewTargetParameters); }
        HRESULT STDMETHODCALLTYPE GetContainingOutput(IDXGIOutput** ppOutput) override
        { return m_real->GetContainingOutput(ppOutput); }
        HRESULT STDMETHODCALLTYPE GetFrameStatistics(DXGI_FRAME_STATISTICS* pStats) override
        { return m_real->GetFrameStatistics(pStats); }
        HRESULT STDMETHODCALLTYPE GetLastPresentCount(UINT* pLastPresentCount) override
        { return m_real->GetLastPresentCount(pLastPresentCount); }

        // --- IDXGISwapChain1 ----------------------------------------------
        HRESULT STDMETHODCALLTYPE GetDesc1(DXGI_SWAP_CHAIN_DESC1* pDesc) override
        { return m_real->GetDesc1(pDesc); }
        HRESULT STDMETHODCALLTYPE GetFullscreenDesc(DXGI_SWAP_CHAIN_FULLSCREEN_DESC* pDesc) override
        { return m_real->GetFullscreenDesc(pDesc); }
        HRESULT STDMETHODCALLTYPE GetHwnd(HWND* pHwnd) override
        { return m_real->GetHwnd(pHwnd); }
        HRESULT STDMETHODCALLTYPE GetCoreWindow(REFIID refiid, void** ppUnk) override
        { return m_real->GetCoreWindow(refiid, ppUnk); }
        HRESULT STDMETHODCALLTYPE Present1(UINT SyncInterval, UINT PresentFlags,
                                           const DXGI_PRESENT_PARAMETERS* pPresentParameters) override
        {
            const uint64_t n = m_present1Count.fetch_add(1, std::memory_order_relaxed) + 1;

            const bool wantDouble = !m_doublerFailed && n > kDoublerWarmupPresents
                                    && (n % kDoubleEveryN == 0);
            if (wantDouble && !m_doublerInit)
            {
                m_doublerOk = InitDoubler();          // first engage after warmup
                if (!m_doublerOk) m_doublerFailed = true;
            }

            if (wantDouble && m_doublerOk)
            {
                // Phase 1: stash frame N (current buffer) before presenting it.
                if (CaptureCurrentFrame())
                {
                    // Real present of frame N → its buffer goes to the display and
                    // the next buffer becomes current.
                    HRESULT hr = m_real->Present1(SyncInterval, PresentFlags, pPresentParameters);

                    // Phase 2: blit frame N into the now-current buffer, then present
                    // it as the duplicate (sync=0 so the extra never vsync-blocks).
                    HRESULT hr2 = E_FAIL;
                    bool blit = (m_device->GetDeviceRemovedReason() == S_OK) && BlitSavedToCurrent();
                    if (blit)
                        hr2 = m_real->Present1(0, PresentFlags, pPresentParameters);

                    const uint64_t dc = ++m_doubledCount;
                    const HRESULT removed = m_device->GetDeviceRemovedReason();
                    const bool bad = !blit || FAILED(hr2) || removed != S_OK;
                    if (dc <= 4 || (dc & 0x7F) == 0 || bad)
                        Logf(bad ? 2 : 0,
                             "DOUBLED #%llu (present #%llu) real hr=0x%08lx extra hr=0x%08lx "
                             "blit=%d removed=0x%08lx",
                             (unsigned long long)dc, (unsigned long long)n,
                             (unsigned long)hr, (unsigned long)hr2, (int)blit,
                             (unsigned long)removed);
                    if (bad)
                    {
                        m_doublerOk = false; m_doublerFailed = true;
                        Logf(2, "Doubler DISABLED — reverting to pass-through.");
                    }
                    return hr;   // Unity sees its own present's result
                }
                m_doublerOk = false; m_doublerFailed = true;
                Logf(2, "Doubler DISABLED: capture failed (removed=0x%08lx) — "
                        "reverting to pass-through.",
                     (unsigned long)(m_device ? m_device->GetDeviceRemovedReason() : E_FAIL));
                // fall through to a plain present so this frame still shows
            }

            if (n <= 4 || (n & 0xFF) == 0 || n == kDoublerWarmupPresents)
                Logf(0, "Present1 #%llu (sync=%u flags=0x%lx) [pass-through]",
                     (unsigned long long)n, SyncInterval, (unsigned long)PresentFlags);
            return m_real->Present1(SyncInterval, PresentFlags, pPresentParameters);
        }
        BOOL STDMETHODCALLTYPE IsTemporaryMonoSupported(void) override
        { return m_real->IsTemporaryMonoSupported(); }
        HRESULT STDMETHODCALLTYPE GetRestrictToOutput(IDXGIOutput** ppRestrictToOutput) override
        { return m_real->GetRestrictToOutput(ppRestrictToOutput); }
        HRESULT STDMETHODCALLTYPE SetBackgroundColor(const DXGI_RGBA* pColor) override
        { return m_real->SetBackgroundColor(pColor); }
        HRESULT STDMETHODCALLTYPE GetBackgroundColor(DXGI_RGBA* pColor) override
        { return m_real->GetBackgroundColor(pColor); }
        HRESULT STDMETHODCALLTYPE SetRotation(DXGI_MODE_ROTATION Rotation) override
        { return m_real->SetRotation(Rotation); }
        HRESULT STDMETHODCALLTYPE GetRotation(DXGI_MODE_ROTATION* pRotation) override
        { return m_real->GetRotation(pRotation); }

        // --- IDXGISwapChain2 ----------------------------------------------
        HRESULT STDMETHODCALLTYPE SetSourceSize(UINT Width, UINT Height) override
        { return m_real->SetSourceSize(Width, Height); }
        HRESULT STDMETHODCALLTYPE GetSourceSize(UINT* pWidth, UINT* pHeight) override
        { return m_real->GetSourceSize(pWidth, pHeight); }
        HRESULT STDMETHODCALLTYPE SetMaximumFrameLatency(UINT MaxLatency) override
        { return m_real->SetMaximumFrameLatency(MaxLatency); }
        HRESULT STDMETHODCALLTYPE GetMaximumFrameLatency(UINT* pMaxLatency) override
        { return m_real->GetMaximumFrameLatency(pMaxLatency); }
        HANDLE STDMETHODCALLTYPE GetFrameLatencyWaitableObject(void) override
        { return m_real->GetFrameLatencyWaitableObject(); }
        HRESULT STDMETHODCALLTYPE SetMatrixTransform(const DXGI_MATRIX_3X2_F* pMatrix) override
        { return m_real->SetMatrixTransform(pMatrix); }
        HRESULT STDMETHODCALLTYPE GetMatrixTransform(DXGI_MATRIX_3X2_F* pMatrix) override
        { return m_real->GetMatrixTransform(pMatrix); }

        // --- IDXGISwapChain3 ----------------------------------------------
        UINT STDMETHODCALLTYPE GetCurrentBackBufferIndex(void) override
        { return m_real->GetCurrentBackBufferIndex(); }
        HRESULT STDMETHODCALLTYPE CheckColorSpaceSupport(DXGI_COLOR_SPACE_TYPE ColorSpace,
                                                         UINT* pColorSpaceSupport) override
        { return m_real->CheckColorSpaceSupport(ColorSpace, pColorSpaceSupport); }
        HRESULT STDMETHODCALLTYPE SetColorSpace1(DXGI_COLOR_SPACE_TYPE ColorSpace) override
        { return m_real->SetColorSpace1(ColorSpace); }
        HRESULT STDMETHODCALLTYPE ResizeBuffers1(UINT BufferCount, UINT Width, UINT Height,
                                                 DXGI_FORMAT Format, UINT SwapChainFlags,
                                                 const UINT* pCreationNodeMask,
                                                 IUnknown* const* ppPresentQueue) override
        {
            Logf(0, "ResizeBuffers1(count=%u %ux%u fmt=%d flags=0x%lx) [pass-through]",
                 BufferCount, Width, Height, (int)Format, (unsigned long)SwapChainFlags);
            return m_real->ResizeBuffers1(BufferCount, Width, Height, Format, SwapChainFlags,
                                          pCreationNodeMask, ppPresentQueue);
        }

        // --- IDXGISwapChain4 ----------------------------------------------
        HRESULT STDMETHODCALLTYPE SetHDRMetaData(DXGI_HDR_METADATA_TYPE Type, UINT Size,
                                                 void* pMetaData) override
        { return m_real->SetHDRMetaData(Type, Size, pMetaData); }

    private:
        ~Proxy()   // deleted via Release() only
        {
            if (m_fenceEvent) CloseHandle(m_fenceEvent);
        }

        ComPtr<IDXGISwapChain4> m_real;
        std::atomic<ULONG>      m_ref{1};       // born with the caller's reference
        std::atomic<uint64_t>   m_presentCount{0};
        std::atomic<uint64_t>   m_present1Count{0};
        int                     m_qiForwardLogged{0};

        // --- Stage 2 frame-doubler -----------------------------------------
        ComPtr<ID3D12CommandQueue>        m_queue;        // Unity's present queue
        ComPtr<ID3D12Device>              m_device;
        ComPtr<ID3D12CommandAllocator>    m_alloc;
        ComPtr<ID3D12GraphicsCommandList> m_list;
        ComPtr<ID3D12Fence>               m_fence;
        ComPtr<ID3D12Resource>            m_saved;        // offscreen copy of frame N
        HANDLE                            m_fenceEvent{nullptr};
        UINT64                            m_fenceVal{0};
        UINT                              m_bufferCount{0};
        bool                              m_doublerInit{false};
        bool                              m_doublerOk{false};   // ready to double
        bool                              m_doublerFailed{false};
        uint64_t                          m_doubledCount{0};

        // Lazily create the command objects needed to copy a back buffer. Returns
        // false (and latches m_doublerFailed) if anything is missing/unsupported.
        bool InitDoubler()
        {
            m_doublerInit = true;
            if (!m_queue) { Logf(1, "Doubler: no present queue — disabled."); return false; }

            if (FAILED(m_queue->GetDevice(IID_PPV_ARGS(&m_device))) || !m_device)
            { Logf(2, "Doubler: GetDevice from queue failed."); return false; }

            DXGI_SWAP_CHAIN_DESC1 d{};
            if (FAILED(m_real->GetDesc1(&d))) { Logf(2, "Doubler: GetDesc1 failed."); return false; }
            m_bufferCount = d.BufferCount;
            if (m_bufferCount < 2)
            {
                Logf(1, "Doubler: only %u back buffer(s) — disabled.", m_bufferCount);
                return false;
            }

            HRESULT hr = m_device->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_alloc));
            if (SUCCEEDED(hr))
                hr = m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                 m_alloc.Get(), nullptr, IID_PPV_ARGS(&m_list));
            if (SUCCEEDED(hr)) hr = m_list->Close();
            if (SUCCEEDED(hr))
                hr = m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence));
            if (SUCCEEDED(hr))
            {
                m_fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
                if (!m_fenceEvent) hr = HRESULT_FROM_WIN32(GetLastError());
            }
            if (FAILED(hr)) { Logf(2, "Doubler: command-object create failed hr=0x%08lx",
                                   (unsigned long)hr); return false; }

            // Offscreen copy of frame N. We can only write the CURRENT back buffer,
            // and flip-discard discards a buffer's contents on present, so we must
            // stash frame N here before presenting and blit it into the next
            // current buffer afterwards.
            D3D12_HEAP_PROPERTIES heap{};
            heap.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC td{};
            td.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            td.Width            = d.Width;
            td.Height           = d.Height;
            td.DepthOrArraySize = 1;
            td.MipLevels        = 1;
            td.Format           = d.Format;
            td.SampleDesc.Count = 1;
            td.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            hr = m_device->CreateCommittedResource(
                &heap, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr, IID_PPV_ARGS(&m_saved));
            if (FAILED(hr)) { Logf(2, "Doubler: offscreen create failed hr=0x%08lx "
                                    "(%ux%u fmt=%d)", (unsigned long)hr, d.Width, d.Height,
                                    (int)d.Format); return false; }

            Logf(0, "Doubler READY: %u back buffers, %ux%u fmt=%d, copying on Unity's "
                    "queue %p. Will present a DUPLICATE of each real frame.",
                 m_bufferCount, d.Width, d.Height, (int)d.Format, (void*)m_queue.Get());
            return true;
        }

        // Submit `list` on Unity's queue and block until the GPU finishes it.
        // Single-queue ordering puts this strictly after Unity's frame render.
        bool SubmitAndWait()
        {
            ID3D12CommandList* lists[] = { m_list.Get() };
            m_queue->ExecuteCommandLists(1, lists);
            const UINT64 v = ++m_fenceVal;
            if (FAILED(m_queue->Signal(m_fence.Get(), v))) return false;
            if (m_fence->GetCompletedValue() < v)
            {
                m_fence->SetEventOnCompletion(v, m_fenceEvent);
                WaitForSingleObject(m_fenceEvent, 2000);
            }
            return true;
        }

        static D3D12_RESOURCE_BARRIER Transition(ID3D12Resource* r,
                                                 D3D12_RESOURCE_STATES a,
                                                 D3D12_RESOURCE_STATES b)
        {
            D3D12_RESOURCE_BARRIER br{};
            br.Type  = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            br.Transition.pResource   = r;
            br.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            br.Transition.StateBefore = a;
            br.Transition.StateAfter  = b;
            return br;
        }

        // Phase 1 (BEFORE the real present): copy the current back buffer (frame N,
        // which Unity left in PRESENT state) into our offscreen save texture. Only
        // the current buffer is touched. m_saved starts/ends in COPY_DEST except
        // it is left in COPY_SOURCE ready for phase 2.
        bool CaptureCurrentFrame()
        {
            ComPtr<ID3D12Resource> cur;
            if (FAILED(m_real->GetBuffer(m_real->GetCurrentBackBufferIndex(),
                                         IID_PPV_ARGS(&cur))) || !cur) return false;
            if (FAILED(m_alloc->Reset())) return false;
            if (FAILED(m_list->Reset(m_alloc.Get(), nullptr))) return false;

            D3D12_RESOURCE_BARRIER pre[1] = {
                Transition(cur.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_SOURCE),
            };
            m_list->ResourceBarrier(1, pre);                       // m_saved already COPY_DEST
            m_list->CopyResource(m_saved.Get(), cur.Get());
            D3D12_RESOURCE_BARRIER post[2] = {
                Transition(cur.Get(),     D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT),
                Transition(m_saved.Get(), D3D12_RESOURCE_STATE_COPY_DEST,   D3D12_RESOURCE_STATE_COPY_SOURCE),
            };
            m_list->ResourceBarrier(2, post);
            if (FAILED(m_list->Close())) return false;
            if (!SubmitAndWait()) return false;
            return m_device->GetDeviceRemovedReason() == S_OK;
        }

        // Phase 2 (AFTER the real present, so the duplicate's target is now the
        // CURRENT buffer): blit the saved frame N into it and leave it PRESENT-
        // ready for the extra present. m_saved returns to COPY_DEST for next frame.
        bool BlitSavedToCurrent()
        {
            ComPtr<ID3D12Resource> dst;
            if (FAILED(m_real->GetBuffer(m_real->GetCurrentBackBufferIndex(),
                                         IID_PPV_ARGS(&dst))) || !dst) return false;
            if (FAILED(m_alloc->Reset())) return false;
            if (FAILED(m_list->Reset(m_alloc.Get(), nullptr))) return false;

            D3D12_RESOURCE_BARRIER pre[1] = {
                Transition(dst.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST),
            };
            m_list->ResourceBarrier(1, pre);                       // m_saved already COPY_SOURCE
            m_list->CopyResource(dst.Get(), m_saved.Get());
            D3D12_RESOURCE_BARRIER post[2] = {
                Transition(dst.Get(),     D3D12_RESOURCE_STATE_COPY_DEST,   D3D12_RESOURCE_STATE_PRESENT),
                Transition(m_saved.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
            };
            m_list->ResourceBarrier(2, post);
            if (FAILED(m_list->Close())) return false;
            if (!SubmitAndWait()) return false;
            return m_device->GetDeviceRemovedReason() == S_OK;
        }
    };
}

namespace SwapChainProxy
{
    IDXGISwapChain1* Wrap(IDXGISwapChain1* real, ID3D12CommandQueue* presentQueue, LogFn log)
    {
        g_Log = log;
        if (!real) return nullptr;

        ComPtr<IDXGISwapChain4> real4;
        HRESULT hr = real->QueryInterface(IID_PPV_ARGS(&real4));
        if (FAILED(hr) || !real4)
        {
            Logf(1, "Wrap: real swapchain is not IDXGISwapChain4 (hr=0x%08lx); "
                    "NOT wrapping — Unity keeps the real swapchain.",
                 (unsigned long)hr);
            return nullptr;
        }

        ComPtr<ID3D12CommandQueue> queue(presentQueue);   // proxy takes a reference
        Proxy* p = new Proxy(real4, queue);   // real4 holds its own ref on the real object
        Logf(0, "Wrap: proxy created over real swapchain %p (present queue=%p) "
                "[Stage 2 frame-doubler, engages after %llu presents]",
             (void*)real, (void*)presentQueue, (unsigned long long)kDoublerWarmupPresents);
        return static_cast<IDXGISwapChain1*>(static_cast<IDXGISwapChain4*>(p));
    }
}
