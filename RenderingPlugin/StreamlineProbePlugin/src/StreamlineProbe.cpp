// StreamlineProbe.cpp — see header.
//
// Capability-only Streamline init. Links sl.interposer.lib (the SL entry points
// resolve from sl.interposer.dll, which is shipped beside this plugin alongside
// sl.dlss_g/common/reflex/pcl + nvngx_dlssg).

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <atomic>
#include <cstdarg>
#include <cstdio>

#include "sl.h"
#include "sl_consts.h"

#include "StreamlineProbe.h"

namespace
{
    StreamlineProbe::LogFn g_log = nullptr;
    std::atomic<bool>      g_attempted{false};
    bool                   g_slInited = false;

    void Logf(int level, const char* fmt, ...)
    {
        char buf[768];
        va_list ap;
        va_start(ap, fmt);
        _vsnprintf_s(buf, sizeof(buf), _TRUNCATE, fmt, ap);
        va_end(ap);
        const char* tag = (level == 2) ? "[NR/StreamlineProbe ERR] "
                        : (level == 1) ? "[NR/StreamlineProbe WRN] "
                                       : "[NR/StreamlineProbe] ";
        char line[864];
        _snprintf_s(line, sizeof(line), _TRUNCATE, "%s%s", tag, buf);
        if (g_log) g_log(level, line);
        else { OutputDebugStringA(line); OutputDebugStringA("\n"); }
    }

    // Pipe Streamline's own logs into the Unity log (verbose in this dev build).
    void SLLog(sl::LogType type, const char* msg)
    {
        if (!g_log || !msg) return;
        const int lvl = (type == sl::LogType::eError) ? 2
                      : (type == sl::LogType::eWarn)  ? 1 : 0;
        char line[1024];
        _snprintf_s(line, sizeof(line), _TRUNCATE, "[NR/StreamlineProbe][SL] %s", msg);
        // SL messages usually carry a trailing newline; trim it for tidy logging.
        size_t n = strnlen_s(line, sizeof(line));
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
        g_log(lvl, line);
    }

    const char* ResultStr(sl::Result r)
    {
        switch (r)
        {
            case sl::Result::eOk:                          return "eOk";
            case sl::Result::eErrorDriverOutOfDate:        return "eErrorDriverOutOfDate";
            case sl::Result::eErrorOSOutOfDate:            return "eErrorOSOutOfDate";
            case sl::Result::eErrorOSDisabledHWS:          return "eErrorOSDisabledHWS (HW scheduling off)";
            case sl::Result::eErrorNoSupportedAdapterFound:return "eErrorNoSupportedAdapterFound";
            case sl::Result::eErrorAdapterNotSupported:    return "eErrorAdapterNotSupported";
            case sl::Result::eErrorNoPlugins:              return "eErrorNoPlugins";
            case sl::Result::eErrorNGXFailed:              return "eErrorNGXFailed";
            case sl::Result::eErrorNotInitialized:         return "eErrorNotInitialized";
            case sl::Result::eErrorInitNotCalled:          return "eErrorInitNotCalled";
            case sl::Result::eErrorFeatureMissing:         return "eErrorFeatureMissing";
            case sl::Result::eErrorFeatureNotSupported:    return "eErrorFeatureNotSupported";
            case sl::Result::eErrorFeatureFailedToLoad:    return "eErrorFeatureFailedToLoad";
            case sl::Result::eErrorMissingProxy:           return "eErrorMissingProxy";
            default:                                       return "(other)";
        }
    }
}

namespace StreamlineProbe
{
    void Init(ID3D12Device* device, LogFn log)
    {
        g_log = log;
        if (!device) return;                       // wait for a real device

        bool expected = false;
        if (!g_attempted.compare_exchange_strong(expected, true)) return;

        Logf(0, "Initializing Streamline %u.%u.%u (manual hooking, D3D12)...",
             SL_VERSION_MAJOR, SL_VERSION_MINOR, SL_VERSION_PATCH);

        static const sl::Feature kFeatures[] = {
            sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL,
        };

        sl::Preferences pref{};
        pref.showConsole        = false;
        pref.logLevel           = sl::LogLevel::eVerbose;   // dev diagnostics
        pref.logMessageCallback = &SLLog;
        pref.featuresToLoad     = kFeatures;
        pref.numFeaturesToLoad  = (uint32_t)_countof(kFeatures);
        // We attach to Unity's already-created device/swapchain, so SL must NOT
        // auto-hook DXGI/D3D12 creation.
        pref.flags             |= sl::PreferenceFlags::eUseManualHooking;
        pref.engine             = sl::EngineType::eUnity;
        pref.engineVersion      = "6000.3";
        pref.projectId          = "a0f57b54-1daf-4934-90ae-c4035c19df04";
        pref.renderAPI          = sl::RenderAPI::eD3D12;
        // pathsToPlugins left null: SL looks for sl.dlss_g/etc. beside
        // sl.interposer.dll, which we ship next to this plugin.

        sl::Result r = slInit(pref, sl::kSDKVersion);
        Logf(r == sl::Result::eOk ? 0 : 2, "slInit -> %s", ResultStr(r));
        if (r != sl::Result::eOk)
            return;
        g_slInited = true;

        sl::Result rd = slSetD3DDevice(device);
        Logf(rd == sl::Result::eOk ? 0 : 2, "slSetD3DDevice -> %s", ResultStr(rd));
        if (rd != sl::Result::eOk)
            return;

        // Adapter for capability query = the device's adapter LUID.
        LUID luid = device->GetAdapterLuid();
        sl::AdapterInfo ai{};
        ai.deviceLUID            = reinterpret_cast<uint8_t*>(&luid);
        ai.deviceLUIDSizeInBytes = sizeof(luid);

        sl::Result rg = slIsFeatureSupported(sl::kFeatureDLSS_G, ai);
        Logf(rg == sl::Result::eOk ? 0 : 1,
             "slIsFeatureSupported(DLSS_G / FrameGeneration) -> %s", ResultStr(rg));

        sl::Result rr = slIsFeatureSupported(sl::kFeatureReflex, ai);
        Logf(rr == sl::Result::eOk ? 0 : 1,
             "slIsFeatureSupported(Reflex) -> %s", ResultStr(rr));

        bool loaded = false;
        if (slIsFeatureLoaded(sl::kFeatureDLSS_G, loaded) == sl::Result::eOk)
            Logf(0, "DLSS_G plugin loaded = %d", (int)loaded);

        sl::FeatureRequirements req{};
        if (slGetFeatureRequirements(sl::kFeatureDLSS_G, req) == sl::Result::eOk)
            Logf(0, "DLSS-G requirements: maxViewports=%u numRequiredTags=%u "
                    "maxNumCPUThreads=%u flags=0x%llx",
                 req.maxNumViewports, req.numRequiredTags, req.maxNumCPUThreads,
                 (unsigned long long)req.flags);

        sl::FeatureVersion ver{};
        if (slGetFeatureVersion(sl::kFeatureDLSS_G, ver) == sl::Result::eOk)
            Logf(0, "DLSS-G version queried (SL plugin reports its version).");

        Logf(rg == sl::Result::eOk ? 0 : 1,
             "PROBE RESULT: Streamline %s on Unity's device; DLSS-G supported = %s.",
             g_slInited ? "initialized" : "NOT initialized",
             rg == sl::Result::eOk ? "YES" : "NO");
    }

    bool InitAttempted()
    {
        return g_attempted.load(std::memory_order_acquire);
    }

    void Shutdown()
    {
        if (!g_slInited) return;
        g_slInited = false;
        sl::Result r = slShutdown();
        Logf(r == sl::Result::eOk ? 0 : 1, "slShutdown -> %s", ResultStr(r));
    }
}
