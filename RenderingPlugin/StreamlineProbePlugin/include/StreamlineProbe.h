// StreamlineProbe.h
// Streamline (SL 2.11.1) integration for DLSS-G Frame Generation, manual-hooking
// mode, attached to Unity's D3D12 device/swapchain in-process.
//
// Step 1 (present-path takeover):
//   * InitSL()  — slInit at plugin load (preload, BEFORE Unity's swapchain
//                 exists) with eUseManualHooking + {DLSS_G, Reflex, PCL}.
//   * AdoptSwapChain() — called from the CreateSwapChainForHwnd hook right after
//                 Unity creates its native swapchain: slSetD3DDevice + capability
//                 log (once) + slUpgradeInterface(swapchain) to hand Unity SL's
//                 FG proxy swapchain, then slDLSSGSetOptions(mode=eOn). From then
//                 on Unity presents through SL → SL owns buffer mgmt + presentCommon.
//
// NOTE: we upgrade the SWAPCHAIN, not the factory: our factory vtable hook would
// make SL's proxy factory recurse back into our hook. The SDK explicitly supports
// upgrading the presentation interface (swapchain) immediately after creation.
//
// No resource tags yet (Depth/MVec/HUDLessColor/UIColorAndAlpha) — without them
// DLSS-G won't generate, but this proves SL can OWN Unity's present without the
// device-removal the raw extra-present route hit. Tags are step 2 (NativeRenderPlugin).
#pragma once

// NOTE: this header is included AFTER <windows.h> + <dxgi1_5.h> in both .cpp units,
// so HWND / DXGI_SWAP_CHAIN_DESC1 / etc. are already defined where used below.
struct ID3D12Device;
struct IUnknown;
struct IDXGISwapChain1;
struct IDXGIOutput;
struct DXGI_SWAP_CHAIN_DESC1;
struct DXGI_SWAP_CHAIN_FULLSCREEN_DESC;

namespace StreamlineProbe
{
    // level: 0=info, 1=warn, 2=error.
    using LogFn = void (*)(int level, const char* msg);

    // slInit in manual-hooking mode. Call at plugin load (no device needed).
    // Returns true on eOk. Safe to call once.
    bool InitSL(LogFn log);

    // Installs a vtable hook on ID3D12Device::CreateCommandQueue (via a throwaway
    // device's shared vtable) so that the command queues Unity later creates are
    // SL-proxied. DLSS-G's mandatory manual hook (eID3D12Device_CreateCommandQueue):
    // without an SL-proxied PRESENT queue, DLSS-G can't attach its async present and
    // never generates. Call at plugin load, BEFORE Unity creates its device/queues.
    void InstallDeviceQueueHook();

    // True once slInit succeeded.
    bool IsInited();

    // Adopt Unity's freshly-created swapchain into Streamline: set the device,
    // upgrade the swapchain to SL's FG proxy (replacing *ppSwapChain in place),
    // and enable DLSS-G. presentQueue is the swapchain's command queue (the
    // CreateSwapChainForHwnd "device" arg). No-op if SL isn't initialized.
    void AdoptSwapChain(IDXGISwapChain1** ppSwapChain, IUnknown* presentQueue,
                        bool alreadyProxy = false);

    // True when Unity's command queue is being SL-proxied (queue proxy installed).
    bool IsQueueProxyActive();

    // Create the swapchain through SL's proxy factory so SL establishes the
    // device/queue/swapchain proxy links (required for DLSS-G to attach + present
    // without the proxy-queue crash). The CALLER must guard re-entrancy: SL's proxy
    // factory calls the native factory's CreateSwapChainForHwnd, which re-enters the
    // caller's vtable hook. Returns an SL proxy swapchain. Pass the (proxy) queue.
    HRESULT CreateSwapChainViaProxyFactory(IUnknown* queue, HWND hWnd,
        const DXGI_SWAP_CHAIN_DESC1* desc, const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fs,
        IDXGIOutput* out, IDXGISwapChain1** ppSwapChain);

    // Per-frame, called at the START of Unity's frame (from a render-thread plugin
    // event issued on RenderPipelineManager.beginContextRendering). Mints this
    // frame's token, drives slReflexSleep + eSimulationStart, and tags the (dummy)
    // DLSS-G inputs + constants. The present hook reuses this token and only emits
    // the present-side markers. This is what gives the SL pacer a real per-frame
    // timeline (vs. the old present-only path that collapsed every marker onto
    // present). No-op until the swapchain has been adopted.
    void BeginFrame();

    // If 'maybeProxy' is an SL proxy interface, returns its underlying NATIVE
    // interface; otherwise returns it unchanged. Used to hand DXGI the native command
    // queue when Unity's queue has been SL-proxied (DXGI must not be given a proxy
    // queue — its present path crashes; SL still tracks the proxy by creation).
    IUnknown* NativeIfProxy(IUnknown* maybeProxy);

    // Request DLSS-G frame generation ON (enable=true) or OFF at runtime. Thread-safe:
    // it only records the desired state; the actual slDLSSGSetOptions call is applied on
    // the PRESENT thread at the next present (slDLSSGSetOptions is not thread-safe and
    // must run on the presenting thread — DLSS-G PG §6.0). OFF uses
    // eRetainResourcesWhenOff so re-enabling is stutter-free without recreating the
    // swapchain (PG §6.4). No-op (request still recorded) until the swapchain is adopted.
    void SetFrameGeneration(bool enable);

    // True if DLSS-G frame generation is currently applied (as last set on the present
    // thread). Reflects the actual applied mode, not a pending request.
    bool IsFrameGenerationOn();

    // slShutdown. Idempotent. Call before the device is destroyed.
    void Shutdown();
}
