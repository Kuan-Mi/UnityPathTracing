// StreamlineProbe.cpp — see header.
//
// Step 2b (frame-loop pacing spike): after SL owns the present path (step 1), prove
// the SL pacer actually GENERATES a frame when driven from Unity's real frame loop.
// Inputs are still DUMMY textures (Depth/MotionVectors/HUDLessColor/UIColorAndAlpha)
// so the output is garbage — we only care that the present cadence doubles
// (DLSSGState.numFramesActuallyPresented -> 2).
//
// The per-frame work is split across the real frame timeline:
//   * BeginFrame() — issued from a render-thread plugin event on Unity's
//     RenderPipelineManager.beginContextRendering (see StreamlineFrameDriver.cs).
//     Mints the frame token, calls slReflexSleep + eSimulationStart, tags the dummy
//     inputs and sets constants. This is the fix over step 2a, which minted the token
//     and fired every marker at present — collapsing the timeline so the pacer had no
//     frame interval and never generated.
//   * EmitPresentMarkersPre()/PostPresentMarker() — in the SL proxy Present hook,
//     close out eRenderSubmit*/ePresent* on that same token.
// DLSS-G tags are eValidUntilPresent, so slSetTagForFrame's command buffer may be
// null (SL copies internally) — no command list needed here.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_5.h>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <wrl/client.h>

#include "sl.h"
#include "sl_consts.h"
#include "sl_dlss_g.h"
#include "sl_reflex.h"
#include "sl_pcl.h"

#include "StreamlineProbe.h"

using Microsoft::WRL::ComPtr;

namespace
{
    StreamlineProbe::LogFn g_log = nullptr;
    std::atomic<bool>      g_inited{false};
    bool                   g_deviceSet = false;
    bool                   g_dlssgEnabled = false;

    // Dummy DLSS-G inputs + per-frame tagging state.
    ComPtr<ID3D12Device>   g_device;
    ComPtr<ID3D12Resource> g_depth, g_mvec, g_hudless, g_ui;
    UINT                   g_w = 0, g_h = 0;
    bool                   g_dummiesReady = false;
    bool                   g_reflexOn = false;
    uint32_t               g_frameIndex = 0;
    uint64_t               g_taggedFrames = 0;
    uint64_t               g_presentCount = 0;
    bool                   g_featuresEnabledOnPresent = false;  // Reflex+DLSS-G enabled on present thread
    IDXGISwapChain1*       g_proxySwapchain = nullptr;   // SL FG proxy (borrowed)
    ComPtr<IDXGISwapChain3> g_proxySC3;                  // for GetCurrentBackBufferIndex
    sl::FrameToken*        g_curToken = nullptr;         // token for the in-flight present

    // SL proxy swapchain Present hook (installed on SL's proxy vtable).
    constexpr UINT kPresentVTIdx  = 8;
    constexpr UINT kPresent1VTIdx = 22;
    using PFN_Present  = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, UINT);
    using PFN_Present1 = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain1*, UINT, UINT,
                                                     const DXGI_PRESENT_PARAMETERS*);
    PFN_Present       g_slOrigPresent  = nullptr;
    PFN_Present1      g_slOrigPresent1 = nullptr;
    std::atomic<bool> g_presentHooked{false};

    void Logf(int level, const char* fmt, ...)
    {
        char buf[768];
        va_list ap; va_start(ap, fmt);
        _vsnprintf_s(buf, sizeof(buf), _TRUNCATE, fmt, ap);
        va_end(ap);
        const char* tag = (level == 2) ? "[NR/StreamlineProbe ERR] "
                        : (level == 1) ? "[NR/StreamlineProbe WRN] "
                                       : "[NR/StreamlineProbe] ";
        char line[864];
        _snprintf_s(line, sizeof(line), _TRUNCATE, "%s%s", tag, buf);
        if (g_log) g_log(level, line);
        else { OutputDebugStringA(line); OutputDebugStringA("\n"); }
    }

    void SLLog(sl::LogType type, const char* msg)
    {
        if (!g_log || !msg) return;
        const int lvl = (type == sl::LogType::eError) ? 2
                      : (type == sl::LogType::eWarn)  ? 1 : 0;
        char line[1024];
        _snprintf_s(line, sizeof(line), _TRUNCATE, "[NR/StreamlineProbe][SL] %s", msg);
        size_t n = strnlen_s(line, sizeof(line));
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
        g_log(lvl, line);
    }

    const char* R(sl::Result r)
    {
        switch (r)
        {
            case sl::Result::eOk:                           return "eOk";
            case sl::Result::eErrorDriverOutOfDate:         return "eErrorDriverOutOfDate";
            case sl::Result::eErrorOSDisabledHWS:           return "eErrorOSDisabledHWS";
            case sl::Result::eErrorNoSupportedAdapterFound: return "eErrorNoSupportedAdapterFound";
            case sl::Result::eErrorAdapterNotSupported:     return "eErrorAdapterNotSupported";
            case sl::Result::eErrorNoPlugins:               return "eErrorNoPlugins";
            case sl::Result::eErrorNotInitialized:          return "eErrorNotInitialized";
            case sl::Result::eErrorInitNotCalled:           return "eErrorInitNotCalled";
            case sl::Result::eErrorFeatureNotSupported:     return "eErrorFeatureNotSupported";
            case sl::Result::eErrorMissingProxy:            return "eErrorMissingProxy";
            case sl::Result::eErrorMissingInputParameter:   return "eErrorMissingInputParameter";
            case sl::Result::eErrorMissingConstants:        return "eErrorMissingConstants";
            case sl::Result::eErrorUnsupportedInterface:    return "eErrorUnsupportedInterface";
            default:                                        return "(other)";
        }
    }

    void EnsureDevice(IUnknown* presentQueue)
    {
        if (g_deviceSet || !presentQueue) return;
        ComPtr<ID3D12CommandQueue> queue;
        if (FAILED(presentQueue->QueryInterface(IID_PPV_ARGS(&queue))) || !queue)
        { Logf(2, "AdoptSwapChain: present 'device' arg is not an ID3D12CommandQueue."); return; }
        queue->GetDevice(IID_PPV_ARGS(&g_device));
        if (!g_device) { Logf(2, "AdoptSwapChain: queue->GetDevice failed."); return; }

        sl::Result rd = slSetD3DDevice(g_device.Get());
        Logf(rd == sl::Result::eOk ? 0 : 2, "slSetD3DDevice -> %s", R(rd));
        g_deviceSet = (rd == sl::Result::eOk);
        if (g_deviceSet)
        {
            LUID luid = g_device->GetAdapterLuid();
            sl::AdapterInfo ai{};
            ai.deviceLUID = reinterpret_cast<uint8_t*>(&luid);
            ai.deviceLUIDSizeInBytes = sizeof(luid);
            sl::Result rg = slIsFeatureSupported(sl::kFeatureDLSS_G, ai);
            Logf(rg == sl::Result::eOk ? 0 : 1, "slIsFeatureSupported(DLSS_G) -> %s", R(rg));
        }
    }

    void EnableDLSSG()
    {
        if (g_dlssgEnabled) return;
        sl::ViewportHandle viewport{ 0 };
        sl::DLSSGOptions opt{};
        opt.mode = sl::DLSSGMode::eOn;
        opt.numFramesToGenerate = 1;
        sl::Result r = slDLSSGSetOptions(viewport, opt);
        Logf(r == sl::Result::eOk ? 0 : 2, "slDLSSGSetOptions(mode=eOn, gen=1) -> %s", R(r));
        g_dlssgEnabled = (r == sl::Result::eOk);
    }

    // DLSS-G requires Reflex active at runtime (eFailReflexNotDetectedAtRuntime).
    void EnableReflex()
    {
        if (g_reflexOn) return;
        sl::ReflexOptions opt{};
        opt.mode = sl::ReflexMode::eLowLatency;
        opt.frameLimitUs = 0;
        sl::Result r = slReflexSetOptions(opt);
        Logf(r == sl::Result::eOk ? 0 : 2, "slReflexSetOptions(eLowLatency) -> %s", R(r));
        g_reflexOn = (r == sl::Result::eOk);
    }

    void LogDlssgState()
    {
        sl::ViewportHandle viewport{ 0 };
        sl::DLSSGState st{};
        sl::Result r = slDLSSGGetState(viewport, st, nullptr);
        if (r != sl::Result::eOk) { Logf(1, "slDLSSGGetState -> %s", R(r)); return; }
        const uint32_t s = (uint32_t)st.status;
        char flags[256] = "";
        if (s == 0) strcpy_s(flags, "eOk");
        else {
            if (s & (1u<<0)) strcat_s(flags, "ResolutionTooLow|");
            if (s & (1u<<1)) strcat_s(flags, "ReflexNotDetected|");
            if (s & (1u<<2)) strcat_s(flags, "HDRFormatNotSupported|");
            if (s & (1u<<3)) strcat_s(flags, "CommonConstantsInvalid|");
            if (s & (1u<<4)) strcat_s(flags, "GetCurrentBackBufferIndexNotCalled|");
        }
        Logf(s == 0 ? 0 : 1,
             "DLSSGState: status=0x%x [%s] framesPresentedSinceLast=%u maxGen=%u minDim=%u",
             s, flags, st.numFramesActuallyPresented, st.numFramesToGenerateMax,
             st.minWidthOrHeight);
    }

    bool CreateTex(DXGI_FORMAT fmt, ComPtr<ID3D12Resource>& out, const char* name)
    {
        D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = g_w; d.Height = g_h; d.DepthOrArraySize = 1; d.MipLevels = 1;
        d.Format = fmt; d.SampleDesc.Count = 1;
        d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        d.Flags  = D3D12_RESOURCE_FLAG_NONE;
        HRESULT hr = g_device->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, nullptr,
            IID_PPV_ARGS(&out));
        if (FAILED(hr)) { Logf(2, "Create dummy %s failed hr=0x%08lx", name, (unsigned long)hr); return false; }
        return true;
    }

    // Fill the dummy motion-vector texture (R16G16_FLOAT) with a CONSTANT non-zero
    // value so DLSS-G's generated (interpolated) frames are visibly WARPED relative to
    // the real frames — a deliberate, unmistakable proof that generated frames are
    // actually being displayed. R = horizontal motion, G = 0. Large value for a big,
    // obvious sideways vibration; bump kHalfVX further for more.
    void FillMvecConstant()
    {
        if (!g_device || !g_mvec || !g_w || !g_h) return;

        const uint16_t kHalfVX = 0x4400;  // ~4.0 (half-float) — large horizontal motion
        const uint16_t kHalfVY = 0x0000;  // 0.0
        const uint32_t texel = (uint32_t)kHalfVX | ((uint32_t)kHalfVY << 16);

        const UINT  rowPitch   = (g_w * 4u + 255u) & ~255u;   // R16G16 = 4 bytes/texel, 256-aligned
        const UINT64 uploadSize = (UINT64)rowPitch * g_h;

        D3D12_HEAP_PROPERTIES up{}; up.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC bd{};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bd.Width = uploadSize; bd.Height = 1; bd.DepthOrArraySize = 1; bd.MipLevels = 1;
        bd.Format = DXGI_FORMAT_UNKNOWN; bd.SampleDesc.Count = 1;
        bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> upload;
        if (FAILED(g_device->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &bd,
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload))))
        { Logf(2, "FillMvec: upload buffer failed"); return; }

        uint8_t* mapped = nullptr;
        D3D12_RANGE noRead{ 0, 0 };
        if (FAILED(upload->Map(0, &noRead, reinterpret_cast<void**>(&mapped))) || !mapped)
        { Logf(2, "FillMvec: Map failed"); return; }
        for (UINT y = 0; y < g_h; ++y)
        {
            uint32_t* row = reinterpret_cast<uint32_t*>(mapped + (UINT64)y * rowPitch);
            for (UINT x = 0; x < g_w; ++x) row[x] = texel;
        }
        upload->Unmap(0, nullptr);

        // One-shot DIRECT queue/list/fence (queue is SL-proxied via our hook; fine for a copy).
        D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        ComPtr<ID3D12CommandQueue>        q;
        ComPtr<ID3D12CommandAllocator>    alloc;
        ComPtr<ID3D12GraphicsCommandList> list;
        ComPtr<ID3D12Fence>               fence;
        if (FAILED(g_device->CreateCommandQueue(&qd, IID_PPV_ARGS(&q))) ||
            FAILED(g_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc))) ||
            FAILED(g_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&list))) ||
            FAILED(g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
        { Logf(2, "FillMvec: temp cmd objects failed"); return; }

        auto barrier = [&](D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
        {
            D3D12_RESOURCE_BARRIER b{};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition.pResource   = g_mvec.Get();
            b.Transition.Subresource = 0;
            b.Transition.StateBefore = before;
            b.Transition.StateAfter  = after;
            list->ResourceBarrier(1, &b);
        };
        barrier(D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);

        D3D12_TEXTURE_COPY_LOCATION dst{}; dst.pResource = g_mvec.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; dst.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION src{}; src.pResource = upload.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint.Offset = 0;
        src.PlacedFootprint.Footprint.Format   = DXGI_FORMAT_R16G16_FLOAT;
        src.PlacedFootprint.Footprint.Width    = g_w;
        src.PlacedFootprint.Footprint.Height   = g_h;
        src.PlacedFootprint.Footprint.Depth    = 1;
        src.PlacedFootprint.Footprint.RowPitch = rowPitch;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

        barrier(D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        list->Close();

        ID3D12CommandList* lists[] = { list.Get() };
        q->ExecuteCommandLists(1, lists);
        q->Signal(fence.Get(), 1);
        if (fence->GetCompletedValue() < 1)
        {
            HANDLE ev = CreateEventA(nullptr, FALSE, FALSE, nullptr);
            if (ev) { fence->SetEventOnCompletion(1, ev); WaitForSingleObject(ev, 2000); CloseHandle(ev); }
        }
        Logf(0, "FillMvec: motion-vector dummy filled (R=0x%04x G=0x%04x) — generated "
                "frames should now be VISIBLY shifted vs real ones.", kHalfVX, kHalfVY);
    }

    void CreateDummies(DXGI_FORMAT bbFmt)
    {
        if (g_dummiesReady) return;
        const bool ok = CreateTex(DXGI_FORMAT_R32_FLOAT,    g_depth,   "depth")
                     && CreateTex(DXGI_FORMAT_R16G16_FLOAT, g_mvec,    "mvec")
                     && CreateTex(bbFmt,                     g_hudless, "hudless")
                     && CreateTex(bbFmt,                     g_ui,      "ui");
        g_dummiesReady = ok;
        Logf(ok ? 0 : 2, "Dummy DLSS-G inputs %s (%ux%u): depth R32F, mvec RG16F, "
                         "hudless+ui bbFmt.", ok ? "created" : "FAILED", g_w, g_h);
        if (ok) FillMvecConstant();   // make generated frames visibly warped (FG proof)
    }

    void identity(sl::float4x4& m)
    {
        m.row[0] = sl::float4(1, 0, 0, 0);
        m.row[1] = sl::float4(0, 1, 0, 0);
        m.row[2] = sl::float4(0, 0, 1, 0);
        m.row[3] = sl::float4(0, 0, 0, 1);
    }

    // DLSS-G / Reflex options MUST be configured on the PRESENTING thread (DLSS-G PG
    // §6.0): when set on another thread the eOn update may never be applied to
    // presents, which manifests as numFramesActuallyPresented stuck at 0. AdoptSwapChain
    // runs on the swapchain-creation thread, so we defer enabling to the first present.
    void EnsureFeaturesOnPresentThread()
    {
        if (g_featuresEnabledOnPresent) return;
        g_featuresEnabledOnPresent = true;
        Logf(0, "First present on thread %lu — enabling Reflex on the present thread.",
             (unsigned long)GetCurrentThreadId());
        EnableReflex();   // harmless; DLSS-G requires Reflex active

        // DLSS-G enable gated separately so we can test proxy-queue stability with FG
        // OFF (it currently crashes inside FG's first evaluate on the DUMMY inputs).
        char env[8] = {};
        DWORD n = GetEnvironmentVariableA("NR_SL_ENABLE_FG", env, sizeof(env));
        if (n > 0 && env[0] == '1')
            EnableDLSSG();
        else
            Logf(0, "DLSS-G enable SKIPPED (set NR_SL_ENABLE_FG=1 to turn FG on). "
                    "Running proxy-queue stability test with FG OFF.");
    }

    // Present-side markers. The token + sleep + sim-start + tagging now happen at
    // frame BEGIN (StreamlineProbe::BeginFrame); here we only close out the frame's
    // render-submit + present markers, reusing that token. If BeginFrame hasn't run
    // yet (first present before the C# driver fires) there is no token — skip.
    void EmitPresentMarkersPre()
    {
        EnsureFeaturesOnPresentThread();

        sl::FrameToken* token = g_curToken;
        if (!token) return;

        // DLSS-G requires the back-buffer index be queried each frame
        // (else eFailGetCurrentBackBufferIndexNotCalled).
        if (g_proxySC3) g_proxySC3->GetCurrentBackBufferIndex();

        slPCLSetMarker(sl::PCLMarker::eSimulationEnd,     *token);
        slPCLSetMarker(sl::PCLMarker::eRenderSubmitStart, *token);
        slPCLSetMarker(sl::PCLMarker::eRenderSubmitEnd,   *token);
        slPCLSetMarker(sl::PCLMarker::ePresentStart,      *token);
    }

    void PostPresentMarker()
    {
        if (!g_curToken) return;
        slPCLSetMarker(sl::PCLMarker::ePresentEnd, *g_curToken);

        // Query DLSS-G state HERE, on the present thread (DLSS-G PG §6.0 / the
        // "must be synchronized with the present thread" warning), so the
        // numFramesActuallyPresented count is read correctly.
        const uint64_t p = ++g_presentCount;
        if (p <= 3 || (p & 0x7F) == 0) LogDlssgState();
    }

    HRESULT STDMETHODCALLTYPE Hooked_SLPresent1(
        IDXGISwapChain1* This, UINT sync, UINT flags, const DXGI_PRESENT_PARAMETERS* pp)
    {
        EmitPresentMarkersPre();
        HRESULT hr = g_slOrigPresent1(This, sync, flags, pp);
        PostPresentMarker();
        return hr;
    }
    HRESULT STDMETHODCALLTYPE Hooked_SLPresent(IDXGISwapChain* This, UINT sync, UINT flags)
    {
        EmitPresentMarkersPre();
        HRESULT hr = g_slOrigPresent(This, sync, flags);
        PostPresentMarker();
        return hr;
    }

    void* PatchSlot(void* obj, UINT idx, void* hook)
    {
        void** vt = *reinterpret_cast<void***>(obj);
        DWORD old = 0;
        if (!VirtualProtect(vt + idx, sizeof(void*), PAGE_READWRITE, &old)) return nullptr;
        void* orig = vt[idx];
        vt[idx] = hook;
        VirtualProtect(vt + idx, sizeof(void*), old, &old);
        return orig;
    }

    void InstallPresentHookOnProxy(IDXGISwapChain1* proxy)
    {
        bool expected = false;
        if (!g_presentHooked.compare_exchange_strong(expected, true)) return;
        g_slOrigPresent  = reinterpret_cast<PFN_Present >(PatchSlot(proxy, kPresentVTIdx,  &Hooked_SLPresent));
        g_slOrigPresent1 = reinterpret_cast<PFN_Present1>(PatchSlot(proxy, kPresent1VTIdx, &Hooked_SLPresent1));
        const bool ok = g_slOrigPresent1 != nullptr;
        Logf(ok ? 0 : 2, "Per-frame tagging hook on SL proxy Present1: %s",
             ok ? "installed" : "FAILED");
        if (!ok) g_presentHooked.store(false);
    }

    // --- ID3D12Device::CreateCommandQueue hook --------------------------------
    // Make Unity's command queues SL proxies so DLSS-G can attach its async present
    // (mandatory manual hook eID3D12Device_CreateCommandQueue). Without an SL-proxied
    // PRESENT queue, DLSS-G counts 0 presents and never generates.
    //
    // ID3D12Device vtable: IUnknown(0-2) + ID3D12Object(3-6) + GetNodeCount(7) +
    // CreateCommandQueue(8).
    constexpr UINT kCreateCommandQueueVTIdx = 8;
    using PFN_CreateCommandQueue = HRESULT(STDMETHODCALLTYPE*)(
        ID3D12Device*, const D3D12_COMMAND_QUEUE_DESC*, REFIID, void**);
    PFN_CreateCommandQueue g_origCreateCommandQueue = nullptr;
    std::atomic<bool>      g_deviceHooked{false};
    ComPtr<ID3D12Device>   g_proxyDevice;   // SL proxy of Unity's device (cached)
    bool                   g_proxyDeviceTried = false;
    ComPtr<IDXGIFactory2>  g_proxyFactory;  // SL proxy factory (cached) for swapchain creation
    bool                   g_proxyFactoryTried = false;

    HRESULT STDMETHODCALLTYPE Hooked_CreateCommandQueue(
        ID3D12Device* This, const D3D12_COMMAND_QUEUE_DESC* desc, REFIID riid, void** ppQueue)
    {
        // Re-entrancy guard. SL's proxy device creates the underlying NATIVE queue by
        // calling This->CreateCommandQueue — but This is Unity's native device whose
        // vtable slot we patched, so that call re-enters THIS hook. Without the guard it
        // recurses into the proxy forever → stack overflow (the crash we saw right after
        // "Upgraded ID3D12Device v0 to v10"). On re-entry, go straight to the original.
        static thread_local bool t_inProxyCreate = false;
        if (t_inProxyCreate)
            return g_origCreateCommandQueue(This, desc, riid, ppQueue);

        // First call: register the native device with SL and obtain its proxy device.
        if (!g_proxyDeviceTried && g_inited.load())
        {
            g_proxyDeviceTried = true;
            if (!g_deviceSet)
            {
                g_device = This;
                sl::Result rd = slSetD3DDevice(This);
                g_deviceSet = (rd == sl::Result::eOk);
                Logf(g_deviceSet ? 0 : 2, "CreateCommandQueue hook: slSetD3DDevice -> %s "
                     "(thread %lu)", R(rd), (unsigned long)GetCurrentThreadId());
            }
            ID3D12Device* dev = This;
            sl::Result ru = slUpgradeInterface(reinterpret_cast<void**>(&dev));
            if (ru == sl::Result::eOk && dev && dev != This)
                g_proxyDevice.Attach(dev);   // take ownership of the proxy's ref
            Logf(ru == sl::Result::eOk ? 0 : 2,
                 "CreateCommandQueue hook: slUpgradeInterface(device) -> %s%s",
                 R(ru), g_proxyDevice ? " (proxy device cached)" : "");
        }

        // Create the queue THROUGH the proxy device so SL wraps it. No recursion: the
        // proxy device has SL's own vtable, not our patched native one.
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
                     "CreateCommandQueue routed via SL proxy device (Type=%d) -> hr=0x%08lx; "
                     "queue handed to Unity is now an SL proxy.",
                     desc ? (int)desc->Type : -1, (unsigned long)hr);
            }
            return hr;
        }

        // Fallback: SL not ready — native queue (DLSS-G won't attach to this one).
        return g_origCreateCommandQueue(This, desc, riid, ppQueue);
    }
}

namespace StreamlineProbe
{
    bool InitSL(LogFn log)
    {
        g_log = log;
        bool expected = false;
        if (!g_inited.compare_exchange_strong(expected, true)) return true;

        Logf(0, "slInit Streamline %u.%u.%u (manual hooking, preload)...",
             SL_VERSION_MAJOR, SL_VERSION_MINOR, SL_VERSION_PATCH);

        static const sl::Feature kFeatures[] = {
            sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL,
        };
        sl::Preferences pref{};
        pref.logLevel           = sl::LogLevel::eVerbose;
        pref.logMessageCallback = &SLLog;
        pref.featuresToLoad     = kFeatures;
        pref.numFeaturesToLoad  = (uint32_t)_countof(kFeatures);
        // Manual hooking (we attach to Unity's device/swapchain) + frame-based
        // resource tagging (required by slSetTagForFrame).
        pref.flags             |= sl::PreferenceFlags::eUseManualHooking;
        pref.flags             |= sl::PreferenceFlags::eUseFrameBasedResourceTagging;
        pref.engine             = sl::EngineType::eUnity;
        pref.engineVersion      = "6000.3";
        pref.projectId          = "a0f57b54-1daf-4934-90ae-c4035c19df04";
        pref.renderAPI          = sl::RenderAPI::eD3D12;

        sl::Result r = slInit(pref, sl::kSDKVersion);
        Logf(r == sl::Result::eOk ? 0 : 2, "slInit -> %s", R(r));
        if (r != sl::Result::eOk) { g_inited.store(false); return false; }
        return true;
    }

    bool IsInited() { return g_inited.load(std::memory_order_acquire); }

    void InstallDeviceQueueHook()
    {
        // OPT-IN ONLY. Proxying Unity's command queue crashes Unity the moment it uses
        // the SL-proxied queue (confirmed: editor dies right after "Upgraded ID3D12Device
        // v0 to v10"). Leaving this on would crash the editor on every launch since the
        // plugin loads on startup. Enable explicitly for player-only experiments:
        //     set NR_SL_PROXY_QUEUE=1  (before launching the player)
        char env[8] = {};
        DWORD n = GetEnvironmentVariableA("NR_SL_PROXY_QUEUE", env, sizeof(env));
        if (n == 0 || env[0] != '1')
        {
            Logf(0, "CreateCommandQueue proxy DISABLED (set NR_SL_PROXY_QUEUE=1 to enable; "
                    "it currently crashes Unity once the proxy queue is used).");
            return;
        }

        bool expected = false;
        if (!g_deviceHooked.compare_exchange_strong(expected, true)) return;

        // Create a throwaway device only to patch the (process-wide) ID3D12Device
        // vtable slot — Unity's later device shares the same vtable, exactly like the
        // DXGI factory hook. Released immediately; the patch persists in d3d12.dll.
        ComPtr<ID3D12Device> tmp;
        HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&tmp));
        if (FAILED(hr) || !tmp)
        {
            Logf(2, "InstallDeviceQueueHook: D3D12CreateDevice failed hr=0x%08lx",
                 (unsigned long)hr);
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

    bool IsQueueProxyActive() { return g_proxyDevice != nullptr; }

    // Lazily build an SL proxy DXGI factory (used to create the swapchain so SL
    // establishes the device/queue/swapchain proxy links — the piece the plain
    // vtable-hook + slUpgradeInterface(swapchain) path was missing, which left the
    // proxy queue unlinked and crashed SL's present).
    bool EnsureProxyFactory()
    {
        if (g_proxyFactory) return true;
        if (g_proxyFactoryTried) return false;
        g_proxyFactoryTried = true;

        ComPtr<IDXGIFactory2> nativeFactory;
        HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&nativeFactory));
        if (FAILED(hr) || !nativeFactory)
        { Logf(2, "EnsureProxyFactory: CreateDXGIFactory2 failed 0x%08lx", (unsigned long)hr); return false; }

        IDXGIFactory2* f = nativeFactory.Get();
        sl::Result r = slUpgradeInterface(reinterpret_cast<void**>(&f));
        if (r != sl::Result::eOk || !f || f == nativeFactory.Get())
        { Logf(2, "EnsureProxyFactory: slUpgradeInterface(factory) -> %s", R(r)); return false; }
        g_proxyFactory.Attach(f);   // own the proxy ref; nativeFactory releases native
        Logf(0, "EnsureProxyFactory: SL proxy factory ready.");
        return true;
    }

    // Create the swapchain via the SL proxy factory (re-entrancy is broken by the
    // factory vtable hook's guard). Returns an SL proxy swapchain in *ppSwapChain.
    HRESULT CreateSwapChainViaProxyFactory(IUnknown* queue, HWND hWnd,
        const DXGI_SWAP_CHAIN_DESC1* desc, const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fs,
        IDXGIOutput* out, IDXGISwapChain1** ppSwapChain)
    {
        if (!EnsureProxyFactory()) return E_FAIL;
        HRESULT hr = g_proxyFactory->CreateSwapChainForHwnd(queue, hWnd, desc, fs, out, ppSwapChain);
        Logf(SUCCEEDED(hr) ? 0 : 2, "CreateSwapChainViaProxyFactory -> hr=0x%08lx (SL proxy swapchain)",
             (unsigned long)hr);
        return hr;
    }

    // alreadyProxy=true when *ppSwapChain came back from CreateSwapChainViaProxyFactory
    // (already an SL proxy) — skip the in-place slUpgradeInterface(swapchain).
    void AdoptSwapChain(IDXGISwapChain1** ppSwapChain, IUnknown* presentQueue, bool alreadyProxy)
    {
        if (!IsInited() || !ppSwapChain || !*ppSwapChain) return;

        EnsureDevice(presentQueue);
        if (!g_deviceSet) { Logf(1, "AdoptSwapChain: device not set; leaving native swapchain."); return; }

        DXGI_SWAP_CHAIN_DESC1 desc{};
        (*ppSwapChain)->GetDesc1(&desc);
        g_w = desc.Width; g_h = desc.Height;

        if (!alreadyProxy)
        {
            void* iface = *ppSwapChain;
            sl::Result r = slUpgradeInterface(&iface);
            if (r != sl::Result::eOk || !iface)
            { Logf(2, "slUpgradeInterface(swapchain) -> %s; no FG.", R(r)); return; }
            *ppSwapChain = reinterpret_cast<IDXGISwapChain1*>(iface);
            Logf(0, "slUpgradeInterface(swapchain) -> eOk; Unity now presents through SL FG proxy.");
        }
        else
        {
            Logf(0, "Swapchain created via SL proxy factory (device/queue/swapchain links set).");
        }
        g_proxySwapchain = *ppSwapChain;   // borrowed; for per-frame GetCurrentBackBufferIndex
        g_proxySwapchain->QueryInterface(IID_PPV_ARGS(&g_proxySC3));

        CreateDummies(desc.Format);
        InstallPresentHookOnProxy(*ppSwapChain);   // close out present markers per frame
        // NOTE: Reflex + DLSS-G are enabled lazily on the PRESENT thread (DLSS-G PG §6.0).
        Logf(0, "STEP 2b: per-frame Reflex loop driven from Unity frame-begin; Reflex+DLSS-G "
                "enabled on first present (present thread). Watch present cadence ~2x.");
    }

    void BeginFrame()
    {
        if (!IsInited() || !g_dummiesReady) return;   // swapchain not adopted yet

        static bool s_loggedTid = false;
        if (!s_loggedTid)
        {
            s_loggedTid = true;
            Logf(0, "BeginFrame runs on thread %lu (compare to the present thread above).",
                 (unsigned long)GetCurrentThreadId());
        }

        sl::FrameToken* token = nullptr;
        uint32_t idx = g_frameIndex++;
        sl::Result rt = slGetNewFrameToken(token, &idx);
        if (rt != sl::Result::eOk || !token) { Logf(1, "slGetNewFrameToken -> %s", R(rt)); return; }
        g_curToken = token;   // shared with EmitPresentMarkersPre / PostPresentMarker

        // The whole point of step 2b: drive Reflex at the START of the frame. DLSS-G's
        // pacer is built on the Reflex frame loop; without slReflexSleep + an early
        // eSimulationStart the pacer has no frame interval to interpolate across.
        slReflexSleep(*token);
        slPCLSetMarker(sl::PCLMarker::eSimulationStart, *token);

        sl::ViewportHandle viewport{ 0 };
        sl::Extent ext{ 0, 0, g_w, g_h };

        sl::Resource rDepth  (sl::ResourceType::eTex2d, g_depth.Get(),   D3D12_RESOURCE_STATE_COMMON);
        sl::Resource rMvec   (sl::ResourceType::eTex2d, g_mvec.Get(),    D3D12_RESOURCE_STATE_COMMON);
        sl::Resource rHudless(sl::ResourceType::eTex2d, g_hudless.Get(), D3D12_RESOURCE_STATE_COMMON);
        sl::Resource rUI     (sl::ResourceType::eTex2d, g_ui.Get(),      D3D12_RESOURCE_STATE_COMMON);

        sl::ResourceTag tags[] = {
            sl::ResourceTag(&rDepth,   sl::kBufferTypeDepth,            sl::ResourceLifecycle::eValidUntilPresent, &ext),
            sl::ResourceTag(&rMvec,    sl::kBufferTypeMotionVectors,    sl::ResourceLifecycle::eValidUntilPresent, &ext),
            sl::ResourceTag(&rHudless, sl::kBufferTypeHUDLessColor,     sl::ResourceLifecycle::eValidUntilPresent, &ext),
            sl::ResourceTag(&rUI,      sl::kBufferTypeUIColorAndAlpha,  sl::ResourceLifecycle::eValidUntilPresent, &ext),
        };
        sl::Result rTag = slSetTagForFrame(*token, viewport, tags, (uint32_t)_countof(tags), nullptr);

        sl::Constants c{};
        identity(c.cameraViewToClip);
        identity(c.clipToCameraView);
        identity(c.clipToLensClip);
        identity(c.clipToPrevClip);
        identity(c.prevClipToClip);
        c.jitterOffset        = sl::float2(0, 0);
        c.mvecScale           = sl::float2(1, 1);
        c.cameraPinholeOffset = sl::float2(0, 0);
        c.cameraPos   = sl::float3(0, 0, 0);
        c.cameraUp    = sl::float3(0, 1, 0);
        c.cameraRight = sl::float3(1, 0, 0);
        c.cameraFwd   = sl::float3(0, 0, 1);
        c.cameraNear = 0.1f; c.cameraFar = 1000.0f;
        c.cameraFOV = 1.0f;
        c.cameraAspectRatio = g_h ? (float)g_w / (float)g_h : 1.0f;
        c.depthInverted        = sl::Boolean::eFalse;
        c.cameraMotionIncluded = sl::Boolean::eTrue;
        c.motionVectors3D      = sl::Boolean::eFalse;
        // Signal a hard reset for the first couple of frames (no valid history yet).
        c.reset                = (g_taggedFrames < 2) ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        sl::Result rC = slSetConstants(c, *token, viewport);

        const uint64_t f = ++g_taggedFrames;
        if (f <= 4 || (f & 0xFF) == 0 || rTag != sl::Result::eOk || rC != sl::Result::eOk)
            Logf((rTag != sl::Result::eOk || rC != sl::Result::eOk) ? 2 : 0,
                 "BEGIN frame #%llu: slSetTagForFrame -> %s, slSetConstants -> %s",
                 (unsigned long long)f, R(rTag), R(rC));
        // DLSS-G state is now queried on the present thread (see PostPresentMarker).
    }

    IUnknown* NativeIfProxy(IUnknown* maybeProxy)
    {
        if (!maybeProxy) return maybeProxy;
        void* native = nullptr;
        sl::Result r = slGetNativeInterface(maybeProxy, &native);
        if (r == sl::Result::eOk && native)
        {
            static bool s_logged = false;
            if (!s_logged) { s_logged = true; Logf(0, "NativeIfProxy: extracted native queue from SL proxy for DXGI."); }
            return reinterpret_cast<IUnknown*>(native);
        }
        return maybeProxy;   // not a proxy (or no proxying) — use as-is
    }

    void Shutdown()
    {
        if (!IsInited()) return;
        g_inited.store(false);
        g_proxySwapchain = nullptr; g_curToken = nullptr; g_proxySC3.Reset();
        g_depth.Reset(); g_mvec.Reset(); g_hudless.Reset(); g_ui.Reset(); g_device.Reset();
        // After this the CreateCommandQueue hook falls back to native (g_proxyDevice
        // null + g_inited false), so the persistent vtable patch stays safe.
        g_proxyDevice.Reset();
        g_proxyFactory.Reset();
        sl::Result r = slShutdown();
        Logf(r == sl::Result::eOk ? 0 : 1, "slShutdown -> %s", R(r));
    }
}
