// SLCore.cpp — see SLCore.h. Shared Streamline lifecycle + logging for SLDenoiser.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>

#include "sl.h"
#include "sl_consts.h"

#include "SLCore.h"

namespace
{
    SLCore::LogFn     g_log = nullptr;
    std::atomic<bool> g_inited{ false };
    bool              g_deviceSet = false;

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
        static const sl::Feature kFeatures[] = {
            sl::kFeatureDLSS_RR, sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL,
        };
        sl::Preferences pref{};
        pref.showConsole        = false;
        pref.logLevel           = sl::LogLevel::eDefault;
        pref.logMessageCallback = &SLLogCallback;
        pref.featuresToLoad     = kFeatures;
        pref.numFeaturesToLoad  = (uint32_t)_countof(kFeatures);
        pref.flags             |= sl::PreferenceFlags::eUseManualHooking;
        pref.flags             |= sl::PreferenceFlags::eUseFrameBasedResourceTagging;
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
        sl::Result rs = slIsFeatureSupported(sl::kFeatureDLSS_RR, ai);
        Logf("SLCore", rs == sl::Result::eOk ? 0 : 1, "slIsFeatureSupported(DLSS_RR) -> %s", ResultStr(rs));
    }

    bool IsDeviceSet() { return g_deviceSet; }

    void Shutdown()
    {
        if (!IsInited()) return;
        g_inited.store(false);
        g_deviceSet = false;
        sl::Result r = slShutdown();
        Logf("SLCore", r == sl::Result::eOk ? 0 : 1, "slShutdown -> %s", ResultStr(r));
    }
}
