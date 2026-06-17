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
    // math only works if a frame's calls all carry the SAME token, so the token is owned
    // here, not per feature. slGetNewFrameToken(idx) returns the same object for the same
    // index — every feature must read CurrentFrameToken rather than minting its own.
    //
    // BeginFrame: advance the frame index and mint its token. Call EXACTLY ONCE per Unity
    // frame, at the earliest render-thread tick (RenderPipelineManager.beginContextRendering).
    // Returns the new token (nullptr if SL is not initialized / mint failed).
    sl::FrameToken* BeginFrame();
    // The token for the frame currently in flight, or nullptr before the first BeginFrame.
    // Does NOT mint — readers (DLSS-RR evaluate, DLSS-G present markers) just consume it.
    sl::FrameToken* CurrentFrameToken();
    // Index of the current token (UINT32_MAX before the first BeginFrame).
    unsigned CurrentFrameIndex();
}
