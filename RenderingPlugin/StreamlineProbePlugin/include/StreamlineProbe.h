// StreamlineProbe.h
// Log-only Streamline (SL 2.11.1) feasibility probe — the SL analog of the raw
// DlssgProbe. It answers ONE question before any swapchain work: can Streamline
// initialize in manual-hooking mode against Unity's already-created D3D12 device
// and report DLSS-G (Frame Generation) as supported, in-process?
//
//   * slInit({DLSS_G, Reflex, PCL}, eUseManualHooking) with our log callback,
//   * slSetD3DDevice(Unity device),
//   * slIsFeatureSupported(DLSS_G/Reflex) + slGetFeatureRequirements → log.
//
// It never creates a swapchain, tags no resources, and alters no presentation,
// so it cannot disturb Unity rendering. Owning the swapchain (SL's FG proxy via
// the CreateSwapChainForHwnd hook) is a later step gated on this probe passing.
//
// Lives in its own StreamlineProbePlugin DLL so the Streamline runtime stays
// fully isolated from the raw-NGX DlssgProbe + proxy in SwapChainHookPlugin
// (enable only ONE of the two via "Load on startup" at a time).
#pragma once

struct ID3D12Device;

namespace StreamlineProbe
{
    // level: 0=info, 1=warn, 2=error.
    using LogFn = void (*)(int level, const char* msg);

    // One-time: slInit + slSetD3DDevice + capability query (log-only). Call only
    // with a non-null device (null is ignored WITHOUT consuming the one-shot
    // latch, so the poller may retry until Unity's device appears).
    void Init(ID3D12Device* device, LogFn log);

    // True once Init() actually executed against a valid device.
    bool InitAttempted();

    // slShutdown. Idempotent.
    void Shutdown();
}
