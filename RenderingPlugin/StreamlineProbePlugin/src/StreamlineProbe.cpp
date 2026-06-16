// StreamlineProbe.cpp — see header.
//
// Step 2a (dummy tagging): after SL owns the present path (step 1), feed DLSS-G
// the 4 required inputs as DUMMY textures (Depth/MotionVectors/HUDLessColor/
// UIColorAndAlpha) every frame so SL actually starts generating frames. Output is
// garbage (inputs are blank), but it verifies the tag/constants API and that the
// present cadence doubles — before wiring the real path-tracer resources.
//
// Per-frame tagging happens in a vtable hook on SL's proxy swapchain Present1:
// right before SL presents we slGetNewFrameToken + slSetTagForFrame + slSetConstants.
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
    }

    void identity(sl::float4x4& m)
    {
        m.row[0] = sl::float4(1, 0, 0, 0);
        m.row[1] = sl::float4(0, 1, 0, 0);
        m.row[2] = sl::float4(0, 0, 1, 0);
        m.row[3] = sl::float4(0, 0, 0, 1);
    }

    // Per-frame: hand DLSS-G its (dummy) inputs + constants for this present.
    void TagThisFrame()
    {
        if (!g_dummiesReady) return;

        sl::FrameToken* token = nullptr;
        uint32_t idx = g_frameIndex++;
        sl::Result rt = slGetNewFrameToken(token, &idx);
        if (rt != sl::Result::eOk || !token) return;
        g_curToken = token;   // shared with the post-present ePresentEnd marker

        // NOTE: slReflexSleep belongs at FRAME START, not present. Calling it here
        // (the only hook a standalone plugin has) stalls the app to a few fps and
        // still yields 0 generated frames — DLSS-G's pacer needs frame-loop timing.
        // Real pacing is wired in the NativeRenderPlugin frame loop (step 2).

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
        c.reset                = sl::Boolean::eFalse;
        sl::Result rC = slSetConstants(c, *token, viewport);

        // DLSS-G requires the app to drive the back-buffer index + Reflex/PCL
        // present markers each frame (else eFailGetCurrentBackBufferIndexNotCalled
        // / eFailReflexNotDetectedAtRuntime). Fire a consistent marker set on this
        // frame's token; ePresentEnd is set after SL presents.
        if (g_proxySC3) g_proxySC3->GetCurrentBackBufferIndex();
        slPCLSetMarker(sl::PCLMarker::eSimulationStart,   *token);
        slPCLSetMarker(sl::PCLMarker::eSimulationEnd,     *token);
        slPCLSetMarker(sl::PCLMarker::eRenderSubmitStart, *token);
        slPCLSetMarker(sl::PCLMarker::eRenderSubmitEnd,   *token);
        slPCLSetMarker(sl::PCLMarker::ePresentStart,      *token);

        const uint64_t f = ++g_taggedFrames;
        if (f <= 4 || (f & 0xFF) == 0 || rTag != sl::Result::eOk || rC != sl::Result::eOk)
            Logf((rTag != sl::Result::eOk || rC != sl::Result::eOk) ? 2 : 0,
                 "TAG frame #%llu: slSetTagForFrame -> %s, slSetConstants -> %s",
                 (unsigned long long)f, R(rTag), R(rC));
        if (f <= 2 || (f & 0x7F) == 0) LogDlssgState();   // is FG generating? why not?
    }

    void PostPresentMarker()
    {
        if (g_curToken) slPCLSetMarker(sl::PCLMarker::ePresentEnd, *g_curToken);
    }

    HRESULT STDMETHODCALLTYPE Hooked_SLPresent1(
        IDXGISwapChain1* This, UINT sync, UINT flags, const DXGI_PRESENT_PARAMETERS* pp)
    {
        TagThisFrame();
        HRESULT hr = g_slOrigPresent1(This, sync, flags, pp);
        PostPresentMarker();
        return hr;
    }
    HRESULT STDMETHODCALLTYPE Hooked_SLPresent(IDXGISwapChain* This, UINT sync, UINT flags)
    {
        TagThisFrame();
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

    void AdoptSwapChain(IDXGISwapChain1** ppSwapChain, IUnknown* presentQueue)
    {
        if (!IsInited() || !ppSwapChain || !*ppSwapChain) return;

        EnsureDevice(presentQueue);
        if (!g_deviceSet) { Logf(1, "AdoptSwapChain: device not set; leaving native swapchain."); return; }

        DXGI_SWAP_CHAIN_DESC1 desc{};
        (*ppSwapChain)->GetDesc1(&desc);
        g_w = desc.Width; g_h = desc.Height;

        void* iface = *ppSwapChain;
        sl::Result r = slUpgradeInterface(&iface);
        if (r != sl::Result::eOk || !iface)
        { Logf(2, "slUpgradeInterface(swapchain) -> %s; no FG.", R(r)); return; }
        *ppSwapChain = reinterpret_cast<IDXGISwapChain1*>(iface);
        g_proxySwapchain = *ppSwapChain;   // borrowed; for per-frame GetCurrentBackBufferIndex
        g_proxySwapchain->QueryInterface(IID_PPV_ARGS(&g_proxySC3));
        Logf(0, "slUpgradeInterface(swapchain) -> eOk; Unity now presents through SL FG proxy.");

        EnableReflex();                            // DLSS-G requires Reflex active
        EnableDLSSG();
        CreateDummies(desc.Format);
        InstallPresentHookOnProxy(*ppSwapChain);   // tag every frame before SL presents
        Logf(0, "STEP 2a: dummy DLSS-G inputs tagged per frame; SL should now GENERATE "
                "(garbage output, watch present cadence ~2x + stability).");
    }

    void Shutdown()
    {
        if (!IsInited()) return;
        g_inited.store(false);
        g_proxySwapchain = nullptr; g_curToken = nullptr; g_proxySC3.Reset();
        g_depth.Reset(); g_mvec.Reset(); g_hudless.Reset(); g_ui.Reset(); g_device.Reset();
        sl::Result r = slShutdown();
        Logf(r == sl::Result::eOk ? 0 : 1, "slShutdown -> %s", R(r));
    }
}
