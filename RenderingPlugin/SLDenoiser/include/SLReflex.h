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
        uint64_t inputSampleTime = 0;
        uint64_t simStartTime = 0;
        uint64_t simEndTime = 0;
        uint64_t renderSubmitStartTime = 0;
        uint64_t renderSubmitEndTime = 0;
        uint64_t presentStartTime = 0;
        uint64_t presentEndTime = 0;
        uint64_t driverStartTime = 0;
        uint64_t driverEndTime = 0;
        uint64_t osRenderQueueStartTime = 0;
        uint64_t osRenderQueueEndTime = 0;
        uint64_t gpuRenderStartTime = 0;
        uint64_t gpuRenderEndTime = 0;
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

    // Legacy hook entry kept for the present-path caller. PCL ping reception now uses a C#
    // message thread via SetPclPingThreadId instead of subclassing Unity's WndProc.
    void InstallPclPing(void* hwnd);

    // Thread id that Streamline's PCL stats ping thread should PostThreadMessageW to.
    void SetPclPingThreadId(uint32_t threadId);

    // Main thread. Emit ePCLatencyPing on the frame that C# determined will consume the ping.
    void MarkPclLatencyPing(const sl::FrameToken& token, unsigned count = 1);

    // Main thread, on the frame whose input sampled a trigger (e.g. fire/click). Emits
    // eTriggerFlash for the Reflex Latency Analyzer's flash indicator (LDAT click-to-photon).
    // Idempotent per token. Optional — only meaningful with the hardware analyzer / flash overlay.
    void MarkTriggerFlash(const sl::FrameToken& token);

    void Shutdown();
}
