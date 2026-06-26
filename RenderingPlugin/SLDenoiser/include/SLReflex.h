// SLReflex.h
// Reflex Low Latency + PCL markers via Streamline, hosted in the SLDenoiser
// plugin alongside DLSS-RR and DLSS-G.
//
// Reflex is INDEPENDENT of DLSS-G (see ProgrammingGuideReflex.md §NOTE: "the sub-features
// are distinct without any cross-dependencies"). Unlike DLSS-G it needs NO proxy swapchain,
// queue, or factory — only slReflexSetOptions + slReflexSleep + PCL markers against the
// shared frame token. That makes it editor-safe and usable with frame generation OFF.
//
// slInit/slSetD3DDevice/slShutdown + the shared per-frame token live in SLCore. The Reflex
// sleep + eSimulationStart marker are issued as separate main-thread frame-begin calls.
// All PCL marker calls live in SLReflex.cpp; render events and Present hooks call these
// semantic marker helpers rather than calling slPCLSetMarker directly.
#pragma once

#include <cstdint>

namespace sl { struct FrameToken; }

namespace SLReflex
{
    struct Stats
    {
        int lowLatencyAvailable = 0;
        int latencyReportAvailable = 0;
        int flashIndicatorDriverControlled = 0;
        uint32_t statsWindowMessage = 0;

        uint64_t frameID = 0;
        uint64_t totalGameToRenderLatencyUs = 0;
        uint64_t simDeltaUs = 0;
        uint64_t renderDeltaUs = 0;
        uint64_t presentDeltaUs = 0;
        uint64_t driverDeltaUs = 0;
        uint64_t osRenderQueueDeltaUs = 0;
        uint64_t gpuRenderDeltaUs = 0;
        uint32_t gpuActiveRenderTimeUs = 0;
        uint32_t gpuFrameTimeUs = 0;
    };

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
    bool GetStats(Stats& outStats);

    // MAIN thread, top of frame BEFORE input sampling: apply pending options (once / on
    // change), then slReflexSleep for this frame's token.
    // slReflexSleep paces the simulation thread, so it MUST run here (not on the render
    // thread) to actually reduce latency. Idempotent per token (won't double-sleep).
    void Sleep(const sl::FrameToken& token);

    // MAIN thread, immediately after the top-of-frame sleep: eSimulationStart.
    // Split from Sleep so the exported API names describe the exact operation.
    void MarkSimulationStart(const sl::FrameToken& token);

    // MAIN thread, end of game logic (simulation done, rendering about to begin):
    // eSimulationEnd. Idempotent per token.
    void MarkSimulationEnd(const sl::FrameToken& token);

    // Render thread. CPU render command submission window for this frame.
    void MarkRenderSubmitStart(const sl::FrameToken& token);
    void MarkRenderSubmitEnd(const sl::FrameToken& token);

    // Present thread. Native/SL proxy swapchain Present window for this frame.
    void MarkPresentStart(const sl::FrameToken& token);
    void MarkPresentEnd(const sl::FrameToken& token);

    // Subclass the game window's WndProc so the PCL stats ping (a registered window message)
    // can be observed. The WndProc only queues ping arrivals; C# consumes that queue at the
    // next main-thread frame begin and explicitly marks the frame token that will sample input.
    // This avoids guessing token ownership from the WndProc. hwnd is a Win32 HWND passed as
    // void* to keep <windows.h> out of this header. Player-only, idempotent; restored on Shutdown.
    void InstallPclPing(void* hwnd);

    // Main thread. Return and clear the number of PCL stats pings observed by the WndProc since
    // the previous consume. C# uses this to decide whether the current frame owns the ping.
    unsigned ConsumePclPingCount();

    // Main thread. Emit ePCLatencyPing on the frame that C# determined will consume the ping.
    void MarkPclLatencyPing(const sl::FrameToken& token, unsigned count = 1);

    void Shutdown();
}
