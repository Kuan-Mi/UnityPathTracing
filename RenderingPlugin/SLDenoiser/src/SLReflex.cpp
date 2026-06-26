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

    // Dedupe: the index of the frame we last slept on / marked sim-end, so multiple begin
    // ticks (or a re-issued SRP context) in one frame don't emit twice.
    std::atomic<uint32_t> g_lastSleptIndex{ 0xFFFFFFFFu };
    std::atomic<uint32_t> g_lastSimEndIndex{ 0xFFFFFFFFu };

    // --- PCL latency ping (so FrameView / ReflexTest can MEASURE PC latency) ---
    // FrameView shows "PCL: NA" unless the app answers the PCL stats ping: a registered window
    // message (PCLState::statsWindowMessage) the driver/tool posts to the game window. We
    // subclass the window's WndProc and queue that message until C# can choose the frame token
    // that will consume the simulated input. C# consumes that queue at the next main-thread
    // frame begin and emits ePCLatencyPing on that token. This is the 7th PCL marker (we already
    // emit Simulation/RenderSubmit/Present start+end). See ProgrammingGuidePCL.md §3.0.
    std::atomic<uint32_t>              g_pclPingMsg{ 0 };       // 0 until acquired from slPCLGetState
    HWND                               g_pingHwnd    = nullptr;
    WNDPROC                            g_origWndProc = nullptr;
    std::atomic<uint64_t>              g_wndMsgSeen{ 0 };       // any msg through our subclass (liveness)
    std::atomic<uint64_t>              g_pclPingQueued{ 0 };    // PCL ping messages waiting for C#
    std::atomic<uint64_t>              g_pclPingMarked{ 0 };    // PCL ping markers emitted

    LRESULT CALLBACK PclWndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
    {
        g_wndMsgSeen.fetch_add(1, std::memory_order_relaxed);
        const uint32_t ping = g_pclPingMsg.load(std::memory_order_acquire);
        if (ping != 0 && msg == ping)
        {
            const uint64_t n = g_pclPingQueued.fetch_add(1, std::memory_order_release) + 1;
            if (n <= 3)
                Logf(0, "PCL ping #%llu queued for main-thread token ownership.", (unsigned long long)n);
        }
        return CallWindowProcW(g_origWndProc, hwnd, msg, wp, lp);
    }

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

    void InstallPclPing(void* hwndV)
    {
        HWND hwnd = reinterpret_cast<HWND>(hwndV);
        if (!hwnd || g_origWndProc) return; // idempotent (subclass once)
        g_pingHwnd    = hwnd;
        g_origWndProc = reinterpret_cast<WNDPROC>(
            SetWindowLongPtrW(hwnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(&PclWndProc)));
        if (!g_origWndProc) { g_pingHwnd = nullptr; Logf(2, "InstallPclPing: SetWindowLongPtrW failed."); return; }
        Logf(0, "InstallPclPing: subclassed game window %p for PCL latency-ping.", (void*)hwnd);
    }

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

        // Acquire the PCL stats window-message id once it becomes available (it can be 0 until a
        // latency consumer such as FrameView/ReflexTest attaches), so the ping handler knows which
        // message to queue for the main-thread owner.
        if (g_pclPingMsg.load(std::memory_order_acquire) == 0)
        {
            sl::PCLState ps{};
            if (slPCLGetState(ps) == sl::Result::eOk && ps.statsWindowMessage != 0)
            {
                g_pclPingMsg.store(ps.statsWindowMessage, std::memory_order_release);
                Logf(0, "PCL stats ping acquired (statsWindowMessage=0x%x); PC latency now measurable.",
                     ps.statsWindowMessage);
            }
        }

        // Diagnostic (every ~256 frames): warn only while the PCL ping chain is BROKEN. SL posts
        // the ping (queued in PclWndProc, marked from C#) only while a latency consumer
        // actively captures AND the game window is foreground (PCLStatsPingThreadProc). Once any
        // ping arrives we stay silent — success is already logged by the first few pings.
        if (g_origWndProc && g_pclPingMarked.load(std::memory_order_relaxed) == 0)
        {
            static uint64_t s_diag = 0;
            if (((++s_diag) & 0xFF) == 0)
            {
                const uint64_t msgs = g_wndMsgSeen.load(std::memory_order_relaxed);
                Logf(1, "PCL diag: no latency pings yet (wndproc msgs=%llu, pingMsg=0x%x). %s",
                     (unsigned long long)msgs, g_pclPingMsg.load(std::memory_order_acquire),
                     msgs == 0 ? "Our WndProc gets NO messages -> subclass was bumped (Unity re-set the proc)."
                               : "WndProc alive but no ping -> start an ACTIVE latency capture (FrameView/PresentMon/ReflexTest) and keep the game window FOCUSED.");
            }
        }

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
        // DIAG: time the sleep in isolation (separately from slGetNewFrameToken, which SLCore
        // logs) so we can attribute the "SLReflexFrameBegin" 10ms marker to the right call.
        LARGE_INTEGER freq, t0, t1; QueryPerformanceFrequency(&freq); QueryPerformanceCounter(&t0);
        sl::Result rs = slReflexSleep(token);
        QueryPerformanceCounter(&t1);
        const double sleepMs = double(t1.QuadPart - t0.QuadPart) * 1000.0 / double(freq.QuadPart);
        slPCLSetMarker(sl::PCLMarker::eSimulationStart, token);

        static uint64_t s_frames = 0;
        const uint64_t f = ++s_frames;
        if (f <= 4 || (f & 0xFF) == 0 || rs != sl::Result::eOk || sleepMs > 1.0)
            Logf(rs != sl::Result::eOk ? 1 : 0, "frame #%llu: slReflexSleep -> %s (%.2f ms, mode=%d)",
                 (unsigned long long)f, R(rs), sleepMs, g_modeApplied.load(std::memory_order_acquire));
    }

    void MarkSimulationEnd(const sl::FrameToken& token)
    {
        if (!SLCore::IsInited() || !SLCore::IsDeviceSet()) return;
        const uint32_t idx  = (uint32_t)token;
        const uint32_t prev = g_lastSimEndIndex.exchange(idx, std::memory_order_acq_rel);
        if (prev == idx) return;
        slPCLSetMarker(sl::PCLMarker::eSimulationEnd, token);
    }

    unsigned ConsumePclPingCount()
    {
        const uint64_t n = g_pclPingQueued.exchange(0, std::memory_order_acq_rel);
        return n > 0xFFFFFFFFull ? 0xFFFFFFFFu : (unsigned)n;
    }

    void MarkPclLatencyPing(const sl::FrameToken& token, unsigned count)
    {
        if (!SLCore::IsInited() || !SLCore::IsDeviceSet() || count == 0) return;
        for (unsigned i = 0; i < count; ++i)
            slPCLSetMarker(sl::PCLMarker::ePCLatencyPing, token);

        const uint64_t n = g_pclPingMarked.fetch_add(count, std::memory_order_relaxed) + count;
        if (n <= 3)
            Logf(0, "PCL ping marker(s) emitted on C#-owned token=%p (count=%u, total=%llu).",
                 (const void*)&token, count, (unsigned long long)n);
    }

    void Shutdown()
    {
        // Restore the original WndProc BEFORE the DLL can unload — otherwise Unity would call
        // into freed code on the next window message and crash.
        if (g_origWndProc && g_pingHwnd)
            SetWindowLongPtrW(g_pingHwnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(g_origWndProc));
        g_origWndProc = nullptr;
        g_pingHwnd    = nullptr;
        g_pclPingMsg.store(0, std::memory_order_release);
        g_wndMsgSeen.store(0, std::memory_order_release);
        g_pclPingQueued.store(0, std::memory_order_release);
        g_pclPingMarked.store(0, std::memory_order_release);

        g_modeApplied.store(-1, std::memory_order_release);
        g_fpsCapApplied.store(0xFFFFFFFFu, std::memory_order_release);
        g_lastSleptIndex.store(0xFFFFFFFFu, std::memory_order_release);
        g_lastSimEndIndex.store(0xFFFFFFFFu, std::memory_order_release);
        // slShutdown is owned by SLCore.
    }
}
