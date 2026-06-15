// StreamlineContext.cpp — see header.
//
// Streamline bring-up Test 1: slInit on Unity's device + DLSS-G support query.
// Links against sl.interposer.lib; the sl.*.dll plugins must be staged next to
// the host executable (the player) so Streamline can load them.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <atomic>
#include <cstdarg>
#include <cstdio>

#include "sl.h"
#include "sl_helpers.h"

#include "StreamlineContext.h"

namespace
{
    std::atomic<bool>      g_Initialized{false};
    std::atomic<bool>      g_DlssgSupported{false};
    StreamlineContext::LogFn g_Logger = nullptr;

    void Logf(int level, const char* fmt, ...)
    {
        char buf[640];
        va_list ap;
        va_start(ap, fmt);
        int n = _vsnprintf_s(buf, sizeof(buf), _TRUNCATE, fmt, ap);
        va_end(ap);
        if (n < 0) buf[sizeof(buf) - 1] = '\0';

        const char* tag = (level == 2) ? "[NR/SL ERR] "
                        : (level == 1) ? "[NR/SL WRN] "
                                       : "[NR/SL] ";
        char prefixed[704];
        _snprintf_s(prefixed, sizeof(prefixed), _TRUNCATE, "%s%s", tag, buf);

        if (g_Logger) g_Logger(level, prefixed);
        else
        {
            char nl[720];
            _snprintf_s(nl, sizeof(nl), _TRUNCATE, "%s\n", prefixed);
            OutputDebugStringA(nl);
        }
    }

    // Bridge Streamline's own log messages into our logger.
    void SlLogCallback(sl::LogType type, const char* msg)
    {
        int lvl = (type == sl::LogType::eError) ? 2
                : (type == sl::LogType::eWarn)  ? 1 : 0;
        Logf(lvl, "[sl] %s", msg ? msg : "");
    }
}

namespace StreamlineContext
{
    void SetLogger(LogFn fn)
    {
        g_Logger = fn;
        Logf(0, "Logger attached");
    }

    bool Initialize(ID3D12Device* device)
    {
        if (g_Initialized.load(std::memory_order_acquire)) return true;
        if (!device) { Logf(2, "Initialize: device is null"); return false; }

        // DLSS-G depends on Reflex + PCL, so load all three.
        static const sl::Feature kFeatures[] = {
            sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL
        };

        sl::Preferences pref{};
        pref.logLevel           = sl::LogLevel::eDefault;
        pref.logMessageCallback = &SlLogCallback;
        pref.featuresToLoad     = kFeatures;
        pref.numFeaturesToLoad  = (uint32_t)(sizeof(kFeatures) / sizeof(kFeatures[0]));
        pref.engine             = sl::EngineType::eUnity;
        pref.engineVersion      = "6000";
        pref.renderAPI          = sl::RenderAPI::eD3D12;
        // Manual hooking: do NOT auto-interpose Unity's already-created dxgi /
        // device. This keeps slInit from perturbing the live swapchain — we only
        // want plugin load + the support query here. (The swapchain proxy comes
        // later, explicitly.) Keep CL state tracking off (default), no OTA.
        pref.flags = sl::PreferenceFlags::eDisableCLStateTracking
                   | sl::PreferenceFlags::eUseManualHooking;

        sl::Result r = slInit(pref, sl::kSDKVersion);
        if (r != sl::Result::eOk)
        {
            Logf(2, "slInit failed: %s", sl::getResultAsStr(r));
            return false;
        }

        // Hand Streamline the device Unity already created.
        r = slSetD3DDevice(device);
        if (r != sl::Result::eOk)
        {
            Logf(2, "slSetD3DDevice failed: %s", sl::getResultAsStr(r));
            slShutdown();
            return false;
        }

        g_Initialized.store(true, std::memory_order_release);
        Logf(0, "Streamline initialized on Unity's D3D12 device (device=%p)", (void*)device);

        // --- DLSS-G support query (the payload of Test 1) -------------------
        LUID luid = device->GetAdapterLuid();
        sl::AdapterInfo adapter{};
        adapter.deviceLUID            = reinterpret_cast<uint8_t*>(&luid);
        adapter.deviceLUIDSizeInBytes = sizeof(luid);

        sl::Result sup = slIsFeatureSupported(sl::kFeatureDLSS_G, adapter);
        g_DlssgSupported.store(sup == sl::Result::eOk, std::memory_order_release);

        if (sup == sl::Result::eOk)
        {
            Logf(0, "DLSS-G SUPPORTED on this GPU");
        }
        else
        {
            // Expected on the RTX 3060 (pre-Ada). Common reasons:
            //   eErrorAdapterNotSupported  -> hardware too old (no FG)
            //   eErrorDriverOutOfDate      -> driver below required version
            Logf(1, "DLSS-G unavailable: %s (expected on non-40-series cards)",
                 sl::getResultAsStr(sup));
        }

        // Detailed requirements (driver versions, vsync/HWS flags) for context.
        sl::FeatureRequirements req{};
        if (slGetFeatureRequirements(sl::kFeatureDLSS_G, req) == sl::Result::eOk)
        {
            Logf(0, "DLSS-G req: driverDetected=%u.%u.%u driverRequired=%u.%u.%u flags=0x%x",
                 req.driverVersionDetected.major, req.driverVersionDetected.minor,
                 req.driverVersionDetected.build,
                 req.driverVersionRequired.major, req.driverVersionRequired.minor,
                 req.driverVersionRequired.build, (unsigned)req.flags);
        }
        return true;
    }

    bool IsInitialized()    { return g_Initialized.load(std::memory_order_acquire); }
    bool IsDLSSGSupported() { return g_DlssgSupported.load(std::memory_order_acquire); }

    void Shutdown()
    {
        if (g_Initialized.exchange(false, std::memory_order_acq_rel))
        {
            sl::Result r = slShutdown();
            if (r != sl::Result::eOk) Logf(1, "slShutdown failed: %s", sl::getResultAsStr(r));
            else                      Logf(0, "Streamline shut down");
        }
        g_DlssgSupported.store(false, std::memory_order_release);
    }
}
