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
    // the donut/RTXPT pattern (mint in SimStart, reuse through present).
    //
    // The plugin is otherwise STATELESS about the token: C# holds the minted pointer (the
    // return of BeginFrame, surfaced as SL_FrameBegin) and passes it back into every call
    // that has a data channel — the Reflex sim markers (SL_MarkSimulationEnd arg) and DLSS-RR
    // evaluate (SLDlssrrFrameData::frameToken). The ONE exception is the DLSS-G present path:
    //   * SetRenderFrame() latches the C#-forwarded token (frame-begin render event) so the
    //     DXGI Present hook — which Streamline/DXGI call with NO parameter we control — and
    //     DLSS-G's render-thread tagging (same frame's valid-until-present tags) can read it
    //     via CurrentFrameToken(). This single latch is irreducible: the present thread has
    //     no other way to learn the frame's token.

    // Main thread, top of frame. Mint and return a fresh frame token (no caching), or nullptr
    // if SL is not initialized / the mint failed.
    sl::FrameToken* BeginFrame();

    // Render thread. Latch the token forwarded from the main thread (frame-begin event data)
    // for the DLSS-G render-tag + present PCL markers. nullptr is ignored (keeps the previous).
    void            SetRenderFrame(sl::FrameToken* token);
    // The latched render/present token (frame currently being rendered/presented). Null before
    // the first SetRenderFrame. Used ONLY by the DLSS-G present path (see above).
    sl::FrameToken* CurrentFrameToken();
}
