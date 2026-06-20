// SLDlssg.cpp — DLSS-G (Frame Generation) via Streamline. See SLDlssg.h.
//
// Ported from the proven StreamlineProbe machinery. Differences:
//   * slInit/slSetD3DDevice/slShutdown + logging are shared via SLCore (SLCore::Init at
//     plugin load already loaded DLSS_G/Reflex/PCL alongside DLSS_RR).
//   * Inputs are REAL path-tracer depth + motion vectors + camera constants (no dummies,
//     no HUDLessColor — SL interpolates the presented backbuffer for now).
//
// Frame timeline (token shared by index — see SLCore.h):
//   * MAIN thread, top of frame: SLCore::BeginFrame mints the frame token; SLReflex does
//     slReflexSleep + eSimulationStart. End of game logic: SLReflex eSimulationEnd.
//   * RENDER thread frame-begin event (data == the FrameToken*): SLCore::SetRenderFrame
//     pins the render/present side to that exact token.
//   * RENDER thread: ConsumeFrameInputs() tags depth/mvec + sets constants on the render
//     token (SLCore::CurrentFrameToken).
//   * PRESENT thread (SL proxy swapchain hook): eRenderSubmitStart/End, ePresentStart/End
//     on the render token.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_5.h>
#include <atomic>
#include <cstring>
#include <wrl/client.h>

#include "sl.h"
#include "sl_consts.h"
#include "sl_dlss_g.h"
#include "sl_pcl.h"
// Reflex (slReflexSetOptions/slReflexSleep) now lives in SLReflex.cpp; the frame token in
// SLCore. This file only owns the FG present path + the present-side PCL markers.

#include "SLCore.h" // shared slInit/slSetD3DDevice/slShutdown + logging/result helpers
#include "SLDlssg.h"

using Microsoft::WRL::ComPtr;

// Terse forwarders to the shared SLCore helpers (tag every line "SLDlssg").
#define Logf(level, ...) SLCore::Logf("SLDlssg", level, __VA_ARGS__)
#define R(r)             SLCore::ResultStr(r)

namespace
{
    std::atomic<int>       g_fgDesired{ -1 };
    std::atomic<bool>      g_fgApplied{ false };

    ComPtr<ID3D12Device>   g_device;
    UINT                   g_w = 0, g_h = 0;
    bool                   g_adopted = false;

    SLDlssg::FrameInputs   g_inputs{};
    bool                   g_haveRealInputs = false;
    uint32_t               g_appliedFrameIdx = 0xFFFFFFFFu;
    uint64_t               g_taggedFrames = 0;
    uint64_t               g_presentCount = 0;
    bool                   g_featuresEnabledOnPresent = false;
    IDXGISwapChain1*       g_proxySwapchain = nullptr;
    ComPtr<IDXGISwapChain3> g_proxySC3;

    constexpr UINT kPresentVTIdx  = 8;
    constexpr UINT kPresent1VTIdx = 22;
    using PFN_Present  = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, UINT);
    using PFN_Present1 = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain1*, UINT, UINT,
                                                     const DXGI_PRESENT_PARAMETERS*);
    PFN_Present       g_slOrigPresent  = nullptr;
    PFN_Present1      g_slOrigPresent1 = nullptr;
    std::atomic<bool> g_presentHooked{ false };

    void EnsureDevice(IUnknown* presentQueue)
    {
        if (SLCore::IsDeviceSet() || !presentQueue) return;
        ComPtr<ID3D12CommandQueue> queue;
        if (FAILED(presentQueue->QueryInterface(IID_PPV_ARGS(&queue))) || !queue)
        { Logf(2, "AdoptSwapChain: present 'device' arg is not an ID3D12CommandQueue."); return; }
        queue->GetDevice(IID_PPV_ARGS(&g_device));
        if (!g_device) { Logf(2, "AdoptSwapChain: queue->GetDevice failed."); return; }

        // Single shared device-set (guarded; the queue hook usually got here first).
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

    void ApplyDlssgMode(bool on)
    {
        if (g_fgApplied.load() == on) return;
        sl::ViewportHandle viewport{ 0 };
        sl::DLSSGOptions opt{};
        opt.mode = on ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
        opt.numFramesToGenerate = 1;
        opt.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
        sl::Result r = slDLSSGSetOptions(viewport, opt);
        Logf(r == sl::Result::eOk ? 0 : 2, "slDLSSGSetOptions(mode=%s, gen=1, retain) -> %s",
             on ? "eOn" : "eOff", R(r));
        if (r == sl::Result::eOk) g_fgApplied.store(on);
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

    void identity(sl::float4x4& m)
    {
        m.row[0] = sl::float4(1, 0, 0, 0);
        m.row[1] = sl::float4(0, 1, 0, 0);
        m.row[2] = sl::float4(0, 0, 1, 0);
        m.row[3] = sl::float4(0, 0, 0, 1);
    }

    void EnsureFeaturesOnPresentThread()
    {
        if (!g_featuresEnabledOnPresent)
        {
            g_featuresEnabledOnPresent = true;
            Logf(0, "First present on thread %lu (Reflex is driven from frame begin via SLReflex).",
                 (unsigned long)GetCurrentThreadId());
            if (g_fgDesired.load() < 0)
            {
                char env[8] = {};
                DWORD n = GetEnvironmentVariableA("NR_SL_ENABLE_FG", env, sizeof(env));
                const bool on = (n > 0 && env[0] == '1');
                g_fgDesired.store(on ? 1 : 0);
                Logf(0, on ? "DLSS-G initial state ON (NR_SL_ENABLE_FG=1)."
                           : "DLSS-G initial state OFF (call SL_SetFrameGeneration(1) to enable).");
            }
        }
        const int desired = g_fgDesired.load();
        if (desired >= 0)
            ApplyDlssgMode(desired != 0);
    }

    void EmitPresentMarkersPre()
    {
        EnsureFeaturesOnPresentThread();
        // eSimulationStart/eSimulationEnd are emitted on the MAIN thread (SLReflex) for the
        // frame currently being rendered; here we close out the render/present side of the
        // PCL timeline on the present thread, using the render-latched token for that frame.
        sl::FrameToken* token = SLCore::CurrentFrameToken();
        if (!token) return;
        if (g_proxySC3) g_proxySC3->GetCurrentBackBufferIndex();
        slPCLSetMarker(sl::PCLMarker::eRenderSubmitStart, *token);
        slPCLSetMarker(sl::PCLMarker::eRenderSubmitEnd,   *token);
        slPCLSetMarker(sl::PCLMarker::ePresentStart,      *token);
    }

    void PostPresentMarker()
    {
        sl::FrameToken* token = SLCore::CurrentFrameToken();
        if (!token) return;
        slPCLSetMarker(sl::PCLMarker::ePresentEnd, *token);
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

    // ID3D12Device::CreateCommandQueue hook (proxy queue for DLSS-G async present).
    constexpr UINT kCreateCommandQueueVTIdx = 8;
    using PFN_CreateCommandQueue = HRESULT(STDMETHODCALLTYPE*)(
        ID3D12Device*, const D3D12_COMMAND_QUEUE_DESC*, REFIID, void**);
    PFN_CreateCommandQueue g_origCreateCommandQueue = nullptr;
    std::atomic<bool>      g_deviceHooked{ false };
    ComPtr<ID3D12Device>   g_proxyDevice;
    bool                   g_proxyDeviceTried = false;
    ComPtr<IDXGIFactory2>  g_proxyFactory;
    bool                   g_proxyFactoryTried = false;

    HRESULT STDMETHODCALLTYPE Hooked_CreateCommandQueue(
        ID3D12Device* This, const D3D12_COMMAND_QUEUE_DESC* desc, REFIID riid, void** ppQueue)
    {
        static thread_local bool t_inProxyCreate = false;
        if (t_inProxyCreate)
            return g_origCreateCommandQueue(This, desc, riid, ppQueue);

        // First queue creation after Unity's device exists: register it with SL (slInit was
        // already done by SLDlssrr::InitSL at plugin load) and obtain the proxy device.
        if (!g_proxyDeviceTried)
        {
            g_proxyDeviceTried = true;
            // Single shared device-set (guarded). This is the earliest reliable point
            // ("immediately after creating the device"), so it usually wins the race with
            // the graphics-init event.
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
        g_proxyFactory.Attach(f);
        Logf(0, "EnsureProxyFactory: SL proxy factory ready.");
        return true;
    }
}

namespace SLDlssg
{
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

    bool IsQueueProxyActive() { return g_proxyDevice != nullptr; }

    HRESULT CreateSwapChainViaProxyFactory(IUnknown* queue, HWND hWnd,
        const DXGI_SWAP_CHAIN_DESC1* desc, const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fs,
        IDXGIOutput* out, IDXGISwapChain1** ppSwapChain)
    {
        if (!EnsureProxyFactory()) return E_FAIL;
        HRESULT hr = g_proxyFactory->CreateSwapChainForHwnd(queue, hWnd, desc, fs, out, ppSwapChain);
        Logf(SUCCEEDED(hr) ? 0 : 2, "CreateSwapChainViaProxyFactory -> hr=0x%08lx", (unsigned long)hr);
        return hr;
    }

    void AdoptSwapChain(IDXGISwapChain1** ppSwapChain, IUnknown* presentQueue, bool alreadyProxy)
    {
        if (!ppSwapChain || !*ppSwapChain) return;

        EnsureDevice(presentQueue);
        if (!SLCore::IsDeviceSet()) { Logf(1, "AdoptSwapChain: device not set; leaving native swapchain."); return; }

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
        g_proxySwapchain = *ppSwapChain;
        g_proxySwapchain->QueryInterface(IID_PPV_ARGS(&g_proxySC3));

        InstallPresentHookOnProxy(*ppSwapChain);
        g_adopted = true;
        Logf(0, "Swapchain adopted (%ux%u); DLSS-G ready. Toggle with SL_SetFrameGeneration.", g_w, g_h);
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

    static void ApplyRealInputs(const FrameInputs& in, sl::FrameToken& token)
    {
        sl::ViewportHandle viewport{ 0 };
        sl::Extent depthMvecExtent{ 0, 0, in.mvecDepthW, in.mvecDepthH };

        sl::Resource rDepth(sl::ResourceType::eTex2d, in.depth,         (uint32_t)in.depthState);
        sl::Resource rMvec (sl::ResourceType::eTex2d, in.motionVectors, (uint32_t)in.mvecState);

        // DLSS-G tags are eValidUntilPresent (SL reads them at present; no per-frame copy).
        // DLSS-G REQUIRES kBufferTypeDepth (see ProgrammingGuideDLSS_G §5.1/§5.2) — it does not
        // consume kBufferTypeLinearDepth (that is a DLSS-SR/RR input). Tagging linear depth here
        // leaves DLSS-G with no depth input and frame generation stops. The depth *values* fed
        // (pool.ViewZ) and the depthInverted flag are the tuning knobs for the is_dynamic test.
        sl::ResourceTag tags[] = {
            sl::ResourceTag(&rDepth, sl::kBufferTypeDepth,         sl::ResourceLifecycle::eValidUntilPresent, &depthMvecExtent),
            sl::ResourceTag(&rMvec,  sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent, &depthMvecExtent),
        };
        sl::Result rTag = slSetTagForFrame(token, viewport, tags, (uint32_t)_countof(tags), nullptr);

        sl::Constants c{};
        std::memcpy(&c.cameraViewToClip.row[0].x, in.cameraViewToClip, sizeof(float) * 16);
        std::memcpy(&c.clipToCameraView.row[0].x, in.clipToCameraView, sizeof(float) * 16);
        std::memcpy(&c.clipToPrevClip.row[0].x,   in.clipToPrevClip,   sizeof(float) * 16);
        std::memcpy(&c.prevClipToClip.row[0].x,   in.prevClipToClip,   sizeof(float) * 16);
        identity(c.clipToLensClip);
        c.jitterOffset        = sl::float2(-in.jitterX, -in.jitterY); // NGX/SL sign convention
        c.mvecScale           = sl::float2(in.mvecScaleX, in.mvecScaleY);
        c.cameraPinholeOffset = sl::float2(0, 0);
        c.cameraPos   = sl::float3(in.cameraPos[0],   in.cameraPos[1],   in.cameraPos[2]);
        c.cameraUp    = sl::float3(in.cameraUp[0],    in.cameraUp[1],    in.cameraUp[2]);
        c.cameraRight = sl::float3(in.cameraRight[0], in.cameraRight[1], in.cameraRight[2]);
        c.cameraFwd   = sl::float3(in.cameraFwd[0],   in.cameraFwd[1],   in.cameraFwd[2]);
        c.cameraNear        = in.cameraNear;
        c.cameraFar         = in.cameraFar;
        c.cameraFOV         = in.cameraFOV;
        c.cameraAspectRatio = in.cameraAspect;
        c.depthInverted        = in.depthInverted        ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        c.cameraMotionIncluded = in.cameraMotionIncluded ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        c.motionVectors3D      = in.motionVectors3D      ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        c.reset                = (in.reset || g_taggedFrames < 2) ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        sl::Result rC = slSetConstants(c, token, viewport);

        const uint64_t f = ++g_taggedFrames;
        if (f <= 4 || (f & 0xFF) == 0 || rTag != sl::Result::eOk || rC != sl::Result::eOk)
            Logf((rTag != sl::Result::eOk || rC != sl::Result::eOk) ? 2 : 0,
                 "FG frame #%llu: slSetTagForFrame -> %s, slSetConstants -> %s",
                 (unsigned long long)f, R(rTag), R(rC));
    }

    void ConsumeFrameInputs(const FrameInputs& inputs)
    {
        g_inputs = inputs;
        if (!g_haveRealInputs)
        {
            g_haveRealInputs = true;
            Logf(0, "First real DLSS-G inputs received (mvec/depth %ux%u, frame %ux%u).",
                 inputs.mvecDepthW, inputs.mvecDepthH, inputs.colorW, inputs.colorH);
        }
        sl::FrameToken* token = SLCore::CurrentFrameToken();
        if (!token)
        {
            Logf(1, "ConsumeFrameInputs: no render token latched yet (frame-begin event not run).");
            return;
        }
        const uint32_t idx = (uint32_t)(*token);
        if (g_appliedFrameIdx == idx) return;
        ApplyRealInputs(g_inputs, *token);
        g_appliedFrameIdx = idx;
    }

    void SetFrameGeneration(bool enable)
    {
        g_fgDesired.store(enable ? 1 : 0, std::memory_order_release);
        Logf(0, "SetFrameGeneration(%s) requested; applies on next present.", enable ? "ON" : "OFF");
    }

    bool IsFrameGenerationOn() { return g_fgApplied.load(std::memory_order_acquire); }

    void Shutdown()
    {
        g_fgDesired.store(-1); g_fgApplied.store(false); g_featuresEnabledOnPresent = false;
        g_haveRealInputs = false; g_inputs = {}; g_adopted = false;
        g_appliedFrameIdx = 0xFFFFFFFFu;
        g_proxySwapchain = nullptr; g_proxySC3.Reset();
        g_device.Reset();
        g_proxyDevice.Reset(); g_proxyFactory.Reset();
        // slShutdown is owned by the RR module (shared slInit).
    }
}
