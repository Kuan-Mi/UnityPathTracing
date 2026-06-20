// SLCore.cpp — see SLCore.h. Shared Streamline lifecycle + logging for SLDenoiser.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>
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

    // Render/present token latch (see SLCore.h). BeginFrame mints on the main thread and returns
    // the pointer to C# (no caching); C# forwards it to the render thread, where SetRenderFrame
    // latches it here for the DLSS-G render-tag + present PCL markers (the present hook has no
    // data channel of its own). Atomic: written on the render thread, read on the present thread.
    std::atomic<sl::FrameToken*> g_renderToken{ nullptr };

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
        static const sl::Feature kFeatures[] = {
            sl::kFeatureDLSS_RR, sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL,
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

        // Cache per-adapter capability. DLSS-G needs Ada+ (40-series); Reflex needs Maxwell+
        // (900-series). On a 30-series card DLSS_G is unsupported but Reflex is: the SL proxy
        // present path still runs (Reflex/PCL + presentCommon) and only the DLSS-G mode is gated
        // off (see SLDlssg::EmitPresentMarkersPre).
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

    sl::FrameToken* BeginFrame()
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

    void SetRenderFrame(sl::FrameToken* token)
    {
        if (token) g_renderToken.store(token, std::memory_order_release);
    }

    sl::FrameToken* CurrentFrameToken() { return g_renderToken.load(std::memory_order_acquire); }

    void Shutdown()
    {
        if (!IsInited()) return;
        g_inited.store(false);
        g_deviceSet = false;
        g_fgSupported.store(false);
        g_reflexSupported.store(false);
        g_renderToken.store(nullptr);
        sl::Result r = slShutdown();
        Logf("SLCore", r == sl::Result::eOk ? 0 : 1, "slShutdown -> %s", ResultStr(r));
    }
}
