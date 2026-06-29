// SLCore.h
// Shared Streamline lifecycle for the SLDenoiser plugin: one slInit, one guarded
// slSetD3DDevice, one slShutdown, plus shared logging/result helpers. Every SL feature
// hosted in this plugin (DLSS-RR, DLSS-G, and later DLSS-SR / NIS) uses this — they must
// NOT call slInit/slSetD3DDevice/slShutdown themselves (Streamline is one instance per
// process, and a second slSetD3DDevice is rejected with "plugins already initialized").
//
// To add a feature later: add its sl::kFeature* to the feature list in SLCore.cpp::Init.
#pragma once

#include <cstdint>

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

    // Per-adapter feature capability, cached on the first successful SetDevice. DLSS-G (Frame
    // Generation) is supported only on Ada (40-series) and newer; Reflex Low Latency on Maxwell
    // (900-series) and newer. On a 30-series card IsFGSupported() is false but IsReflexSupported()
    // is true — the SL proxy present path still runs (Reflex/PCL + presentCommon); only the
    // DLSS-G *mode* is gated off. Both are false until the device is set.
    bool IsFGSupported();
    bool IsReflexSupported();

    // slShutdown. Idempotent. Call before the device is destroyed.
    void Shutdown();

    // --- Shared per-frame token ---------------------------------------------------------
    // Streamline correlates everything belonging to one frame (constants, resource tags,
    // the Reflex sleep, the PCL sim->render->present markers) by FrameToken. The latency
    // math only works if a frame's calls all carry the SAME token. A token is therefore
    // minted EXACTLY ONCE per frame and that one pointer is shared across threads — matching
    // the donut/RTXPT pattern (mint in SimStart, reuse through present).
    //
    // The plugin is STATELESS about the token: C# holds the minted pointer (the return of
    // GetNewFrameToken, surfaced as SL_GetNewFrameToken) and passes it back into every call that
    // has a data channel — the Reflex sim markers, DLSS-RR evaluate (SLDlssrrFrameData::frameToken)
    // and DLSS-G's render-thread tagging (FrameInputs::frameToken). The DXGI Present hook is the
    // one path with no such channel; see the back-buffer-keyed association below.

    // Main thread, top of frame. Mint and return a fresh frame token (no caching), or nullptr
    // if SL is not initialized / the mint failed.
    sl::FrameToken* GetNewFrameToken();

    // --- Present-marker association by FIFO (PCL latency correctness) -------
    // Unity presents on a separate task thread that lags rendering, and the DXGI Present hook has
    // no data channel to learn the presenting frame's token. Tagging ePresentStart/End with the
    // wrong frame corrupts FrameView's PC-latency math (presentEnd(T) - simStart(T)).
    //
    // The render thread pushes each frame's token in submit order (RegisterPresentToken); the present
    // hook pops the oldest (ResolvePresentToken). In a flip-model swapchain present order == submit
    // order, so the queue head is exactly the frame being presented — order-exact. (The earlier
    // back-buffer-index map could mis-route a frame when the render thread raced ahead of present and
    // sampled a stale GetCurrentBackBufferIndex; validated against this FIFO it dropped ~1.8% of frames
    // and mis-tagged on occasion. Map code is retained, commented, in SLCore.cpp.) Resolve skips if the
    // token was recycled out from under it; a bounded depth cap discards the oldest entry if a frame is
    // ever registered but never presented (true occlusion drop), so the queue can't desync forever.
    // The backBufferIndex argument is now unused (kept so callers/hook sites need not change).
    void            RegisterPresentToken(sl::FrameToken* token, uint32_t backBufferIndex);
    sl::FrameToken* ResolvePresentToken(uint32_t backBufferIndex);
}
