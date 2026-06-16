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

struct ID3D12Device;
struct IUnknown;
struct IDXGISwapChain1;

namespace StreamlineProbe
{
    // level: 0=info, 1=warn, 2=error.
    using LogFn = void (*)(int level, const char* msg);

    // slInit in manual-hooking mode. Call at plugin load (no device needed).
    // Returns true on eOk. Safe to call once.
    bool InitSL(LogFn log);

    // True once slInit succeeded.
    bool IsInited();

    // Adopt Unity's freshly-created swapchain into Streamline: set the device,
    // upgrade the swapchain to SL's FG proxy (replacing *ppSwapChain in place),
    // and enable DLSS-G. presentQueue is the swapchain's command queue (the
    // CreateSwapChainForHwnd "device" arg). No-op if SL isn't initialized.
    void AdoptSwapChain(IDXGISwapChain1** ppSwapChain, IUnknown* presentQueue);

    // slShutdown. Idempotent. Call before the device is destroyed.
    void Shutdown();
}
