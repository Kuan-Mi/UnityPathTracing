// SLCore.h
// Shared Streamline lifecycle for the SLDenoiser plugin: one slInit, one guarded
// slSetD3DDevice, one slShutdown, plus shared logging/result helpers. Every SL feature
// hosted in this plugin (DLSS-RR, DLSS-G, and later DLSS-SR / NIS) uses this — they must
// NOT call slInit/slSetD3DDevice/slShutdown themselves (Streamline is one instance per
// process, and a second slSetD3DDevice is rejected with "plugins already initialized").
//
// To add a feature later: add its sl::kFeature* to the feature list in SLCore.cpp::Init.
#pragma once

#include "sl_result.h" // sl::Result (lightweight; for ResultStr)

struct ID3D12Device;
namespace sl { struct FrameToken; }

namespace SLCore
{
    // level: 0=info, 1=warn, 2=error.
    using LogFn = void (*)(int level, const char* msg);

    // Set the Unity log bridge. Call before Init.
    void SetLog(LogFn log);

    // Formatted log with a per-feature tag, e.g. Logf("SLDlssrr", 0, "x=%d", x).
    void Logf(const char* tag, int level, const char* fmt, ...);

    // sl::Result -> string (shared switch).
    const char* ResultStr(sl::Result r);

    // slInit with this plugin's feature set (manual hooking + frame-based tagging).
    // Idempotent. Call once at plugin load, before the device/swapchain exist.
    bool Init(LogFn log);
    bool IsInited();

    // Single guarded slSetD3DDevice (+ DLSS-RR capability log). The FIRST successful call
    // wins; later racing calls (FG queue hook vs graphics-init event) are no-ops. Calling
    // slSetD3DDevice twice makes SL reject the second and leaves features dead.
    void SetDevice(ID3D12Device* device);
    bool IsDeviceSet();

    // slShutdown. Idempotent. Call before the device is destroyed.
    void Shutdown();

    // --- Shared per-frame token ---------------------------------------------------------
    // Streamline correlates everything belonging to one frame (constants, resource tags,
    // the Reflex sleep, the PCL sim->render->present markers) by FrameToken. The latency
    // math only works if a frame's calls all carry the SAME token. A token is therefore
    // minted EXACTLY ONCE per frame and that one pointer is shared across threads — matching
    // the donut/RTXPT pattern (mint in SimStart, reuse through present); re-minting per
    // index is NOT relied upon.
    //
    //   * Main / simulation thread (top of frame, before input): BeginFrame() mints + caches
    //     the token and returns it. SLReflex sleeps + marks eSimulationStart/eSimulationEnd.
    //   * The SAME pointer is forwarded to the render thread via IssuePluginEventAndData;
    //     SetRenderFrame() caches it as the "render" token, which DLSS-RR evaluate, DLSS-G
    //     tagging and the present-side PCL markers read via CurrentFrameToken(). Carrying it
    //     through the ordered render event keeps the render/present side pinned to the frame
    //     it is actually rendering even when the main thread has already advanced.

    // Main thread, top of frame. Mint + cache the frame token. Returns it, or nullptr if SL
    // is not initialized / the mint failed.
    sl::FrameToken* BeginFrame();
    // The token cached by the last BeginFrame (main-thread sim markers + sleep). May be null.
    sl::FrameToken* SimFrameToken();

    // Render thread. Cache the token forwarded from the main thread (frame-begin event data).
    // nullptr is ignored (leaves the previous render token in place).
    void            SetRenderFrame(sl::FrameToken* token);
    // The render-side token (frame currently being rendered/presented). Null before the first
    // SetRenderFrame.
    sl::FrameToken* CurrentFrameToken();
}
