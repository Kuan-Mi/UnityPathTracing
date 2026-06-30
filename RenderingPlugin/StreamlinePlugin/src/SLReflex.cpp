// SLReflex.cpp - see SLReflex.h. Reflex Low Latency + all PCL markers via Streamline.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <atomic>
#include <cstring>

#include "sl.h"
#include "sl_reflex.h"
#include "sl_pcl.h"

#include "SLCore.h"
#include "SLReflex.h"

#define Logf(level, ...) SLCore::Logf("SLReflex", level, __VA_ARGS__)
#define R(r)             SLCore::ResultStr(r)

namespace
{
    // Desired (requested by C#) vs applied (last pushed to slReflexSetOptions). Default ON
    // (Low Latency) — the guide's QA checklist requires Reflex's default state to be On.
    std::atomic<int>      g_modeDesired{ 1 };
    std::atomic<unsigned> g_fpsCapDesired{ 0 };
    std::atomic<int>      g_modeApplied{ -1 };
    std::atomic<unsigned> g_fpsCapApplied{ 0xFFFFFFFFu };
    std::atomic<uint32_t> g_pclPingThreadDesired{ 0 };
    std::atomic<uint32_t> g_pclPingThreadApplied{ 0xFFFFFFFFu };

    // Dedupe: the index of the frame we last slept on / marked sim-start / marked sim-end.
    std::atomic<uint32_t> g_lastSleptIndex{ 0xFFFFFFFFu };
    std::atomic<uint32_t> g_lastSimStartIndex{ 0xFFFFFFFFu };
    std::atomic<uint32_t> g_lastSimEndIndex{ 0xFFFFFFFFu };
    std::atomic<uint32_t> g_lastRenderSubmitStartIndex{ 0xFFFFFFFFu };
    std::atomic<uint32_t> g_lastRenderSubmitEndIndex{ 0xFFFFFFFFu };
    std::atomic<uint32_t> g_lastPresentStartIndex{ 0xFFFFFFFFu };
    std::atomic<uint32_t> g_lastPresentEndIndex{ 0xFFFFFFFFu };
    std::atomic<uint32_t> g_lastFlashIndex{ 0xFFFFFFFFu };

    // --- PCL latency ping (so FrameView / ReflexTest can MEASURE PC latency) ---
    // C# owns a message thread and passes its Win32 thread id here. Streamline PCL then posts its
    // registered ping message to that thread. The C# thread immediately queues a virtual
    // InputSystem event, letting Unity's native input update decide which frame sampled the ping.

    sl::ReflexMode MapMode(int m)
    {
        switch (m)
        {
            case 1:  return sl::ReflexMode::eLowLatency;
            case 2:  return sl::ReflexMode::eLowLatencyWithBoost;
            default: return sl::ReflexMode::eOff;
        }
    }

    // slReflexSetOptions only when mode or FPS cap changed. Must be called at least once
    // even when Off (the guide: "needs to be called at least once, even when ... Off").
    void ApplyOptionsIfNeeded()
    {
        const int      mode = g_modeDesired.load(std::memory_order_acquire);
        const unsigned cap  = g_fpsCapDesired.load(std::memory_order_acquire);
        const uint32_t pingThread = g_pclPingThreadDesired.load(std::memory_order_acquire);
        if (g_modeApplied.load(std::memory_order_acquire) == mode &&
            g_fpsCapApplied.load(std::memory_order_acquire) == cap &&
            g_pclPingThreadApplied.load(std::memory_order_acquire) == pingThread)
            return;

        sl::ReflexOptions opt{};
        opt.mode         = MapMode(mode);
        opt.frameLimitUs = cap;
        opt.idThread     = pingThread;
        sl::Result r = slReflexSetOptions(opt);
        Logf(r == sl::Result::eOk ? 0 : 2,
             "slReflexSetOptions(mode=%d, frameLimitUs=%u, pclPingThread=%u) -> %s",
             mode, cap, pingThread, R(r));
        if (r == sl::Result::eOk)
        {
            g_modeApplied.store(mode, std::memory_order_release);
            g_fpsCapApplied.store(cap, std::memory_order_release);
            g_pclPingThreadApplied.store(pingThread, std::memory_order_release);
        }
    }
}

namespace SLReflex
{
    void SetMode(int mode, unsigned fpsCapUs)
    {
        if (mode < 0 || mode > 2) mode = 1;
        g_modeDesired.store(mode, std::memory_order_release);
        g_fpsCapDesired.store(fpsCapUs, std::memory_order_release);
        Logf(0, "SetMode(%d, fpsCapUs=%u) requested; applies on next frame begin.", mode, fpsCapUs);
    }

    int GetMode() { return g_modeApplied.load(std::memory_order_acquire); }

    void InstallPclPing(void* hwndV)
    {
        (void)hwndV;
        Logf(0, "InstallPclPing: WndProc hook skipped; using PCL idThread message path.");
    }

    void SetPclPingThreadId(uint32_t threadId)
    {
        const uint32_t prev = g_pclPingThreadDesired.exchange(threadId, std::memory_order_acq_rel);
        if (prev != threadId)
        {
            g_pclPingThreadApplied.store(0xFFFFFFFFu, std::memory_order_release);
            Logf(0, "SetPclPingThreadId(%u) requested; applies on next Reflex options update.", threadId);
        }
    }

    bool IsLowLatencyAvailable()
    {
        if (!SLCore::IsInited() || !SLCore::IsDeviceSet()) return false;
        sl::ReflexState st{};
        sl::Result r = slReflexGetState(st);
        if (r != sl::Result::eOk) { Logf(1, "slReflexGetState -> %s", R(r)); return false; }
        return st.lowLatencyAvailable;
    }

    bool GetStats(Stats& outStats)
    {
        std::memset(&outStats, 0, sizeof(outStats));
        if (!SLCore::IsInited() || !SLCore::IsDeviceSet()) return false;

        sl::ReflexState st{};
        sl::Result r = slReflexGetState(st);
        if (r != sl::Result::eOk)
        {
            Logf(1, "slReflexGetState(stats) -> %s", R(r));
            return false;
        }

        outStats.lowLatencyAvailable = st.lowLatencyAvailable ? 1 : 0;
        outStats.latencyReportAvailable = st.latencyReportAvailable ? 1 : 0;
        outStats.flashIndicatorDriverControlled = st.flashIndicatorDriverControlled ? 1 : 0;
        outStats.statsWindowMessage = st.statsWindowMessage;

        const sl::ReflexReport* report = nullptr;
        for (int i = sl::kReflexFrameReportCount - 1; i >= 0; --i)
        {
            if (st.frameReport[i].frameID != 0 && st.frameReport[i].gpuRenderEndTime != 0)
            {
                report = &st.frameReport[i];
                break;
            }
        }

        if (!report) return true;

        outStats.frameID = report->frameID;
        outStats.inputSampleTime = report->inputSampleTime;
        outStats.simStartTime = report->simStartTime;
        outStats.simEndTime = report->simEndTime;
        outStats.renderSubmitStartTime = report->renderSubmitStartTime;
        outStats.renderSubmitEndTime = report->renderSubmitEndTime;
        outStats.presentStartTime = report->presentStartTime;
        outStats.presentEndTime = report->presentEndTime;
        outStats.driverStartTime = report->driverStartTime;
        outStats.driverEndTime = report->driverEndTime;
        outStats.osRenderQueueStartTime = report->osRenderQueueStartTime;
        outStats.osRenderQueueEndTime = report->osRenderQueueEndTime;
        outStats.gpuRenderStartTime = report->gpuRenderStartTime;
        outStats.gpuRenderEndTime = report->gpuRenderEndTime;
        outStats.gpuActiveRenderTimeUs = report->gpuActiveRenderTimeUs;
        outStats.gpuFrameTimeUs = report->gpuFrameTimeUs;
        return true;
    }

    void Sleep(const sl::FrameToken& token)
    {
        if (!SLCore::IsInited() || !SLCore::IsDeviceSet()) return;

        // slReflexSetOptions + slReflexSleep must ALWAYS be issued (even when Reflex is Off);
        // the driver gates the actual sleep on the mode. See the guide's NOTEs in §4.0.
        ApplyOptionsIfNeeded();

        const uint32_t idx  = (uint32_t)token;
        const uint32_t prev = g_lastSleptIndex.exchange(idx, std::memory_order_acq_rel);
        if (prev == idx)
        {
            Logf(1, "duplicate slReflexSleep ignored for frame token %u.", idx);
            return;
        }

        // slReflexSleep is the latency-critical call: it paces the CPU so it does not run
        // unbounded ahead of the GPU (shallower render queue = lower latency). Placed at
        // the earliest frame tick so the sleep front-loads the frame.
        slReflexSleep(token);
    }

    void MarkSimulationStart(const sl::FrameToken& token)
    {
        if (!SLCore::IsInited() || !SLCore::IsDeviceSet()) return;
        const uint32_t idx  = (uint32_t)token;
        const uint32_t prev = g_lastSimStartIndex.exchange(idx, std::memory_order_acq_rel);
        if (prev == idx)
        {
            Logf(1, "duplicate eSimulationStart ignored for frame token %u.", idx);
            return;
        }
        slPCLSetMarker(sl::PCLMarker::eSimulationStart, token);
    }

    void MarkSimulationEnd(const sl::FrameToken& token)
    {
        if (!SLCore::IsInited() || !SLCore::IsDeviceSet()) return;
        const uint32_t idx  = (uint32_t)token;
        const uint32_t prev = g_lastSimEndIndex.exchange(idx, std::memory_order_acq_rel);
        if (prev == idx)
        {
            Logf(1, "duplicate eSimulationEnd ignored for frame token %u.", idx);
            return;
        }
        slPCLSetMarker(sl::PCLMarker::eSimulationEnd, token);
    }

    void MarkRenderSubmitStart(const sl::FrameToken& token)
    {
        if (!SLCore::IsInited() || !SLCore::IsDeviceSet()) return;
        const uint32_t idx  = (uint32_t)token;
        const uint32_t prev = g_lastRenderSubmitStartIndex.exchange(idx, std::memory_order_acq_rel);
        if (prev == idx)
        {
            Logf(1, "duplicate eRenderSubmitStart ignored for frame token %u.", idx);
            return;
        }
        slPCLSetMarker(sl::PCLMarker::eRenderSubmitStart, token);
    }

    void MarkRenderSubmitEnd(const sl::FrameToken& token)
    {
        if (!SLCore::IsInited() || !SLCore::IsDeviceSet()) return;
        const uint32_t idx  = (uint32_t)token;
        const uint32_t prev = g_lastRenderSubmitEndIndex.exchange(idx, std::memory_order_acq_rel);
        if (prev == idx)
        {
            Logf(1, "duplicate eRenderSubmitEnd ignored for frame token %u.", idx);
            return;
        }
        slPCLSetMarker(sl::PCLMarker::eRenderSubmitEnd, token);
    }

    void MarkPresentStart(const sl::FrameToken& token)
    {
        if (!SLCore::IsInited() || !SLCore::IsDeviceSet()) return;
        const uint32_t idx  = (uint32_t)token;
        const uint32_t prev = g_lastPresentStartIndex.exchange(idx, std::memory_order_acq_rel);
        if (prev == idx)
        {
            Logf(1, "duplicate ePresentStart ignored for frame token %u.", idx);
            return;
        }
        slPCLSetMarker(sl::PCLMarker::ePresentStart, token);
    }

    void MarkPresentEnd(const sl::FrameToken& token)
    {
        if (!SLCore::IsInited() || !SLCore::IsDeviceSet()) return;
        const uint32_t idx  = (uint32_t)token;
        const uint32_t prev = g_lastPresentEndIndex.exchange(idx, std::memory_order_acq_rel);
        if (prev == idx)
        {
            Logf(1, "duplicate ePresentEnd ignored for frame token %u.", idx);
            return;
        }
        slPCLSetMarker(sl::PCLMarker::ePresentEnd, token);
    }

    void MarkPclLatencyPing(const sl::FrameToken& token, unsigned count)
    {
        if (!SLCore::IsInited() || !SLCore::IsDeviceSet() || count == 0) return;
        for (unsigned i = 0; i < count; ++i)
            slPCLSetMarker(sl::PCLMarker::ePCLatencyPing, token);
    }

    void MarkTriggerFlash(const sl::FrameToken& token)
    {
        if (!SLCore::IsInited() || !SLCore::IsDeviceSet()) return;
        // eTriggerFlash drives the Reflex Latency Analyzer's flash indicator (LDAT click-to-photon
        // measurement). Emitted on the frame whose input sampled the trigger; dedup so multiple
        // clicks in one frame flash once.
        const uint32_t idx  = (uint32_t)token;
        const uint32_t prev = g_lastFlashIndex.exchange(idx, std::memory_order_acq_rel);
        if (prev == idx)
        {
            Logf(1, "duplicate eTriggerFlash ignored for frame token %u.", idx);
            return;
        }
        slPCLSetMarker(sl::PCLMarker::eTriggerFlash, token);
    }

    void Shutdown()
    {
        g_pclPingThreadDesired.store(0, std::memory_order_release);
        g_pclPingThreadApplied.store(0xFFFFFFFFu, std::memory_order_release);

        g_modeApplied.store(-1, std::memory_order_release);
        g_fpsCapApplied.store(0xFFFFFFFFu, std::memory_order_release);
        g_lastSleptIndex.store(0xFFFFFFFFu, std::memory_order_release);
        g_lastSimStartIndex.store(0xFFFFFFFFu, std::memory_order_release);
        g_lastSimEndIndex.store(0xFFFFFFFFu, std::memory_order_release);
        g_lastRenderSubmitStartIndex.store(0xFFFFFFFFu, std::memory_order_release);
        g_lastRenderSubmitEndIndex.store(0xFFFFFFFFu, std::memory_order_release);
        g_lastPresentStartIndex.store(0xFFFFFFFFu, std::memory_order_release);
        g_lastPresentEndIndex.store(0xFFFFFFFFu, std::memory_order_release);
        g_lastFlashIndex.store(0xFFFFFFFFu, std::memory_order_release);
        // slShutdown is owned by SLCore.
    }
}
