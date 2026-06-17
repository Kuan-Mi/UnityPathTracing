// SLReflex.h
// Reflex Low Latency + PCL simulation marker via Streamline, hosted in the SLDenoiser
// plugin alongside DLSS-RR and DLSS-G.
//
// Reflex is INDEPENDENT of DLSS-G (see ProgrammingGuideReflex.md §NOTE: "the sub-features
// are distinct without any cross-dependencies"). Unlike DLSS-G it needs NO proxy swapchain,
// queue, or factory — only slReflexSetOptions + slReflexSleep + slPCLSetMarker against the
// shared frame token. That makes it editor-safe and usable with frame generation OFF.
//
// slInit/slSetD3DDevice/slShutdown + the shared per-frame token live in SLCore. The Reflex
// sleep + eSimulationStart marker are issued at frame begin (SLCore::BeginFrame); the
// remaining sim->render->present PCL markers are emitted by SLDlssg's present hook when
// frame generation owns the present path.
#pragma once

namespace sl { struct FrameToken; }

namespace SLReflex
{
    // Reflex mode: 0 = Off, 1 = Low Latency (On), 2 = Low Latency + Boost (On + Boost).
    // fpsCapUs is an optional FPS cap in microseconds (0 = uncapped); it works even with
    // mode Off. Recorded here and applied (slReflexSetOptions) on the next frame begin; safe
    // to call any time from any thread (e.g. a UI toggle).
    void SetMode(int mode, unsigned fpsCapUs = 0);
    // The currently applied mode (-1 until first applied).
    int  GetMode();
    // sl::ReflexState::lowLatencyAvailable — for gating Reflex UI only (do everything else
    // the same regardless; see the guide). False on non-NVIDIA / older hardware.
    bool IsLowLatencyAvailable();

    // Frame begin: apply pending options (once / on change), then slReflexSleep +
    // eSimulationStart for this frame's token. Idempotent per token (won't double-sleep).
    // Drive from SLCore::BeginFrame on the render-thread frame-begin tick.
    void OnFrameBegin(const sl::FrameToken& token);

    void Shutdown();
}
