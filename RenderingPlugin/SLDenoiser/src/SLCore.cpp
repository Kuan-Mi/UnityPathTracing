// SLCore.cpp — see SLCore.h. Shared Streamline lifecycle + logging for SLDenoiser.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>

#include "sl.h"
#include "sl_consts.h"

#include "SLCore.h"

namespace
{
    SLCore::LogFn     g_log = nullptr;
    std::atomic<bool> g_inited{ false };
    bool              g_deviceSet = false;
    // Per-adapter capability, cached on the first successful SetDevice (see SLCore.h).
    std::atomic<bool> g_fgSupported{ false };
    std::atomic<bool> g_reflexSupported{ false };

    // Present-marker association by FIFO (see SLCore.h). The DXGI Present hook has no data channel for
    // the frame token, so the render thread pushes each frame's token in submit order (RegisterPresent-
    // Token) and the present hook pops the oldest (ResolvePresentToken). In a flip-model swapchain
    // present order == submit order, so the queue head is exactly the frame being presented — this is
    // order-exact, unlike keying on the back-buffer index (which a render-ahead-of-present race can
    // mis-route, dropping or even mis-tagging a frame). kMaxQueueDepth caps drift: if a frame is ever
    // registered but never presented (a true occlusion drop), the oldest entry is discarded rather than
    // the queue growing forever. Register (render thread) / resolve (present thread) are mutex-guarded;
    // both are once-per-frame so contention is nil.
    //
    // Each entry carries the frame index captured at push: SL recycles FrameTokens in a ring, so if an
    // entry's token was recycled to a newer frame before it was popped, the index won't match and we
    // skip rather than mis-tag. (At observed depth ~0-1 the token is popped one present later, far
    // inside the recycle ring, so this is belt-and-suspenders.)
    struct PresentEntry { sl::FrameToken* token; uint32_t index; };
    std::mutex                   g_presentSlotMutex;   // guards g_presentQueue
    std::deque<PresentEntry>     g_presentQueue;
    uint32_t                     g_presentQueueDropCount = 0; // registered-but-never-presented overflow
    uint32_t                     g_presentAliasCount = 0;     // token recycled before its pop (rate-limited log)
    constexpr size_t             kMaxQueueDepth = 16;

    // Directory containing THIS module (SLDenoiser.dll). In a player build Unity copies
    // native plugins — and the SL runtime DLLs deployed beside us (sl.dlss_g.dll,
    // nvngx_dlssg.dll, sl.dlss_d.dll, …) — into <build>_Data\Plugins\x86_64\, NOT next to
    // the .exe. SL's default plugin search is the executable directory, so it fails to load
    // NGXCore (error 126) there. We pass this directory via Preferences::pathsToPlugins.
    const wchar_t* SelfModuleDir()
    {
        static std::wstring dir;
        static bool tried = false;
        if (tried) return dir.empty() ? nullptr : dir.c_str();
        tried = true;

        HMODULE self = nullptr;
        if (GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                reinterpret_cast<LPCWSTR>(&SelfModuleDir), &self) && self)
        {
            wchar_t path[MAX_PATH] = {};
            DWORD n = GetModuleFileNameW(self, path, MAX_PATH);
            if (n > 0 && n < MAX_PATH)
            {
                if (wchar_t* slash = wcsrchr(path, L'\\')) *slash = L'\0';
                dir = path;
            }
        }
        return dir.empty() ? nullptr : dir.c_str();
    }

    void SLLogCallback(sl::LogType type, const char* msg)
    {
        if (!g_log || !msg) return;
        const int lvl = (type == sl::LogType::eError) ? 2
                      : (type == sl::LogType::eWarn)  ? 1 : 0;
        char line[1024];
        _snprintf_s(line, sizeof(line), _TRUNCATE, "[NR/SL] %s", msg);
        size_t n = strnlen_s(line, sizeof(line));
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
        g_log(lvl, line);
    }
}

namespace SLCore
{
    void SetLog(LogFn log) { g_log = log; }

    void Logf(const char* tag, int level, const char* fmt, ...)
    {
        char buf[768];
        va_list ap; va_start(ap, fmt);
        _vsnprintf_s(buf, sizeof(buf), _TRUNCATE, fmt, ap);
        va_end(ap);
        const char* sev = (level == 2) ? " ERR" : (level == 1) ? " WRN" : "";
        char line[896];
        _snprintf_s(line, sizeof(line), _TRUNCATE, "[NR/%s%s] %s", tag, sev, buf);
        if (g_log) g_log(level, line);
        else { OutputDebugStringA(line); OutputDebugStringA("\n"); }
    }

    const char* ResultStr(sl::Result r)
    {
        switch (r)
        {
            case sl::Result::eOk:                           return "eOk";
            case sl::Result::eErrorDriverOutOfDate:         return "eErrorDriverOutOfDate";
            case sl::Result::eErrorOSDisabledHWS:           return "eErrorOSDisabledHWS";
            case sl::Result::eErrorNoSupportedAdapterFound: return "eErrorNoSupportedAdapterFound";
            case sl::Result::eErrorAdapterNotSupported:     return "eErrorAdapterNotSupported";
            case sl::Result::eErrorNoPlugins:               return "eErrorNoPlugins";
            case sl::Result::eErrorNotInitialized:          return "eErrorNotInitialized";
            case sl::Result::eErrorInitNotCalled:           return "eErrorInitNotCalled";
            case sl::Result::eErrorFeatureNotSupported:     return "eErrorFeatureNotSupported";
            case sl::Result::eErrorMissingProxy:            return "eErrorMissingProxy";
            case sl::Result::eErrorMissingInputParameter:   return "eErrorMissingInputParameter";
            case sl::Result::eErrorMissingConstants:        return "eErrorMissingConstants";
            case sl::Result::eErrorUnsupportedInterface:    return "eErrorUnsupportedInterface";
            case sl::Result::eErrorMissingOrInvalidAPI:     return "eErrorMissingOrInvalidAPI";
            default:                                        return "(other)";
        }
    }

    bool Init(LogFn log)
    {
        g_log = log;
        bool expected = false;
        if (!g_inited.compare_exchange_strong(expected, true)) return true;

        Logf("SLCore", 0, "slInit Streamline %u.%u.%u (DLSS-RR + DLSS-G, manual hooking)...",
             SL_VERSION_MAJOR, SL_VERSION_MINOR, SL_VERSION_PATCH);

        // Feature set for the whole plugin. ADD NEW FEATURES HERE (e.g. sl::kFeatureNIS,
        // sl::kFeatureDLSS for super-resolution) — this is the single extension point.
        // kFeatureImGUI = SL's in-engine debug overlay (Reflex/DLSS-G/common HUD, toggle
        // Ctrl+Shift+Home). It MUST be requested here or sl.imgui is "Ignoring plugin ...
        // not requested by the host"; it only loads with development SL binaries (no-op in
        // production) and renders through presentCommon, so it needs the SL proxy present path.
        static const sl::Feature kFeatures[] = {
            sl::kFeatureDLSS_RR, sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL,
            sl::kFeatureImGUI,
        };
        sl::Preferences pref{};
        pref.showConsole        = false;
        pref.logLevel           = sl::LogLevel::eDefault;
        pref.logMessageCallback = &SLLogCallback;
        pref.featuresToLoad     = kFeatures;
        pref.numFeaturesToLoad  = (uint32_t)_countof(kFeatures);
        // Point SL at the folder holding this DLL + the SL runtime/NGX DLLs (player build:
        // <build>_Data\Plugins\x86_64\). Without this SL searches only the .exe dir and fails
        // to load nvngx_dlssg.dll (NGXCore error 126 -> "no matching adapter found").
        static const wchar_t* s_pluginPath = SelfModuleDir();
        if (s_pluginPath)
        {
            pref.pathsToPlugins    = &s_pluginPath;
            pref.numPathsToPlugins = 1;
            Logf("SLCore", 0, "pathsToPlugins = %ls", s_pluginPath);
        }
        else
        {
            Logf("SLCore", 1, "could not resolve self-module dir; SL will search the .exe dir only.");
        }
        // Preferences::flags defaults to eDisableCLStateTracking | eAllowOTA | eLoadDownloadedPlugins.
        // eAllowOTA fires the NGX/SL OTA updater at init (the benign "updateNGXFeature ... failed
        // 0xbad00002" + "NGXCore: 126 / no matching adapter" log noise). We don't ship OTA'd plugins,
        // so ASSIGN flags explicitly (overwriting the struct default) WITHOUT the two OTA bits.
        pref.flags              = sl::PreferenceFlags::eDisableCLStateTracking
                                | sl::PreferenceFlags::eUseManualHooking
                                | sl::PreferenceFlags::eUseFrameBasedResourceTagging;
        pref.engine             = sl::EngineType::eUnity;
        pref.engineVersion      = "6000.3";
        pref.projectId          = "a0f57b54-1daf-4934-90ae-c4035c19df04";
        pref.renderAPI          = sl::RenderAPI::eD3D12;

        sl::Result r = slInit(pref, sl::kSDKVersion);
        Logf("SLCore", r == sl::Result::eOk ? 0 : 2, "slInit -> %s", ResultStr(r));
        if (r != sl::Result::eOk) { g_inited.store(false); return false; }
        return true;
    }

    bool IsInited() { return g_inited.load(std::memory_order_acquire); }

    void SetDevice(ID3D12Device* device)
    {
        if (!IsInited() || g_deviceSet || !device) return;
        sl::Result rd = slSetD3DDevice(device);
        g_deviceSet = (rd == sl::Result::eOk);
        Logf("SLCore", g_deviceSet ? 0 : 2, "slSetD3DDevice -> %s", ResultStr(rd));
        if (!g_deviceSet) return;

        LUID luid = device->GetAdapterLuid();
        sl::AdapterInfo ai{};
        ai.deviceLUID            = reinterpret_cast<uint8_t*>(&luid);
        ai.deviceLUIDSizeInBytes = sizeof(luid);

        // Cache per-adapter capability. DLSS-G needs Ada+ (40-series); Reflex needs Maxwell+
        // (900-series). On a 30-series card DLSS_G is unsupported but Reflex is: the SL proxy
        // present path still runs (Reflex/PCL + presentCommon) and only the DLSS-G mode is gated
        // off.
        auto supported = [&](sl::Feature f) { return slIsFeatureSupported(f, ai) == sl::Result::eOk; };
        const bool rrOk = supported(sl::kFeatureDLSS_RR);
        g_fgSupported.store(supported(sl::kFeatureDLSS_G),  std::memory_order_release);
        g_reflexSupported.store(supported(sl::kFeatureReflex), std::memory_order_release);
        Logf("SLCore", 0, "feature support: DLSS_RR=%d DLSS_G=%d Reflex=%d",
             (int)rrOk, (int)g_fgSupported.load(), (int)g_reflexSupported.load());
    }

    bool IsDeviceSet()      { return g_deviceSet; }
    bool IsFGSupported()     { return g_fgSupported.load(std::memory_order_acquire); }
    bool IsReflexSupported() { return g_reflexSupported.load(std::memory_order_acquire); }

    sl::FrameToken* GetNewFrameToken()
    {
        if (!IsInited()) return nullptr;
        // nullptr index = SL auto-increments its internal frame counter (matches donut's
        // SimStart). Mint and hand the pointer back to the caller; SLCore does not cache it.
        sl::FrameToken* token = nullptr;
        sl::Result r = slGetNewFrameToken(token, nullptr);

        if (r != sl::Result::eOk || !token)
        {
            Logf("SLCore", 1, "slGetNewFrameToken -> %s", ResultStr(r));
            return nullptr;
        }
        return token;
    }

    void RegisterPresentToken(sl::FrameToken* token, uint32_t /*backBufferIndex*/)
    {
        if (!token) return;
        // Render thread, after this frame's render commands are submitted: push its token in submit
        // order. The next Present pops the oldest, which (flip-model: present order == submit order) is
        // exactly this frame when its turn comes.
        const uint32_t idx = (uint32_t)(*token);
        std::lock_guard<std::mutex> lk(g_presentSlotMutex);

        // FIFO push. Cap the depth so a registered-but-never-presented frame (a true occlusion drop)
        // bounds the backlog instead of desyncing forever — the depth print reveals if this fires.
        g_presentQueue.push_back({ token, idx });
        if (g_presentQueue.size() > kMaxQueueDepth)
        {
            g_presentQueue.pop_front();
            if (++g_presentQueueDropCount == 1 || g_presentQueueDropCount % 64 == 0)
                Logf("SLCore", 1,
                     "present FIFO overflow (>%zu): oldest token dropped — a frame was registered "
                     "but never presented (dropped=%u).", kMaxQueueDepth, g_presentQueueDropCount);
        }
    }

    sl::FrameToken* ResolvePresentToken(uint32_t /*backBufferIndex*/)
    {
        std::lock_guard<std::mutex> lk(g_presentSlotMutex);

        // FIFO resolve: pop the oldest registered token — the frame this present flips.
        PresentEntry q{};
        const bool have = !g_presentQueue.empty();
        if (have) { q = g_presentQueue.front(); g_presentQueue.pop_front(); }

        // Per-present diagnostic: resolved frame index + remaining queue depth (idx 0xFFFFFFFF = the
        // queue was empty, e.g. pre-roll or a present with no prior register).
        Logf("SLCore", 0, "present fifo idx=%u depth=%zu",
             (have && q.token) ? (uint32_t)(*q.token) : 0xFFFFFFFFu, g_presentQueue.size());

        if (!have || !q.token) return nullptr;            // pre-roll, or a present with no registered frame
        // Lifetime guard: the FrameToken lives in SL's ring (memory stays valid) but may have been
        // recycled to a newer frame; if its current index no longer matches, skip to avoid mis-tag.
        if ((uint32_t)(*q.token) != q.index)
        {
            if (++g_presentAliasCount == 1 || g_presentAliasCount % 64 == 0)
                Logf("SLCore", 2,
                     "present token recycled (registered idx=%u, now=%u): markers skipped to avoid "
                     "mis-pairing (aliased=%u).", q.index, (uint32_t)(*q.token), g_presentAliasCount);
            return nullptr;
        }
        return q.token;
    }

    void Shutdown()
    {
        if (!IsInited()) return;
        g_inited.store(false);
        g_deviceSet = false;
        g_fgSupported.store(false);
        g_reflexSupported.store(false);
        {
            std::lock_guard<std::mutex> lk(g_presentSlotMutex);
            g_presentQueue.clear();
            g_presentAliasCount = 0;
            g_presentQueueDropCount = 0;
        }
        sl::Result r = slShutdown();
        Logf("SLCore", r == sl::Result::eOk ? 0 : 1, "slShutdown -> %s", ResultStr(r));
    }
}
