// SLReflex.cpp — see SLReflex.h. Reflex Low Latency + PCL eSimulationStart via Streamline.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <atomic>

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

    // Dedupe: the index of the frame we last slept on, so multiple begin ticks (or a stale
    // re-issue) in one frame don't call slReflexSleep twice.
    std::atomic<uint32_t> g_lastSleptIndex{ 0xFFFFFFFFu };

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
        if (g_modeApplied.load(std::memory_order_acquire) == mode &&
            g_fpsCapApplied.load(std::memory_order_acquire) == cap)
            return;

        sl::ReflexOptions opt{};
        opt.mode         = MapMode(mode);
        opt.frameLimitUs = cap;
        sl::Result r = slReflexSetOptions(opt);
        Logf(r == sl::Result::eOk ? 0 : 2,
             "slReflexSetOptions(mode=%d, frameLimitUs=%u) -> %s", mode, cap, R(r));
        if (r == sl::Result::eOk)
        {
            g_modeApplied.store(mode, std::memory_order_release);
            g_fpsCapApplied.store(cap, std::memory_order_release);
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

    bool IsLowLatencyAvailable()
    {
        if (!SLCore::IsInited() || !SLCore::IsDeviceSet()) return false;
        sl::ReflexState st{};
        sl::Result r = slReflexGetState(st);
        if (r != sl::Result::eOk) { Logf(1, "slReflexGetState -> %s", R(r)); return false; }
        return st.lowLatencyAvailable;
    }

    void OnFrameBegin(const sl::FrameToken& token)
    {
        if (!SLCore::IsInited() || !SLCore::IsDeviceSet()) return;

        // slReflexSetOptions + slReflexSleep + the PCL markers must ALWAYS be issued (even
        // when Reflex is Off) — PCL uses them to measure latency, and the driver gates the
        // actual sleep on the mode. See the guide's NOTEs in §4.0.
        ApplyOptionsIfNeeded();

        const uint32_t idx  = (uint32_t)token;
        const uint32_t prev = g_lastSleptIndex.exchange(idx, std::memory_order_acq_rel);
        if (prev == idx) return; // already slept this frame

        // slReflexSleep is the latency-critical call: it paces the CPU so it does not run
        // unbounded ahead of the GPU (shallower render queue = lower latency). Placed at
        // the earliest frame tick so the sleep front-loads the frame.
        sl::Result rs = slReflexSleep(token);
        slPCLSetMarker(sl::PCLMarker::eSimulationStart, token);

        static uint64_t s_frames = 0;
        const uint64_t f = ++s_frames;
        if (f <= 4 || (f & 0xFF) == 0 || rs != sl::Result::eOk)
            Logf(rs != sl::Result::eOk ? 1 : 0, "frame #%llu: slReflexSleep -> %s",
                 (unsigned long long)f, R(rs));
    }

    void Shutdown()
    {
        g_modeApplied.store(-1, std::memory_order_release);
        g_fpsCapApplied.store(0xFFFFFFFFu, std::memory_order_release);
        g_lastSleptIndex.store(0xFFFFFFFFu, std::memory_order_release);
        // slShutdown is owned by SLCore.
    }
}
