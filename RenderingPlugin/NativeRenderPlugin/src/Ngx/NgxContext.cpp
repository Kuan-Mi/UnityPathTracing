// NgxContext.cpp — see header.
//
// Test 1 of the DLSS-FG bring-up: initialize NGX on Unity's D3D12 device and
// report DLSS-FG support. Calls through to the NVIDIA NGX SDK (linked statically
// via nvsdk_ngx_d.lib).

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <atomic>
#include <cstdarg>
#include <cstdio>

#include "nvsdk_ngx.h"
#include "nvsdk_ngx_helpers.h"
#include "nvsdk_ngx_defs_dlssg.h"

#include "NgxContext.h"

namespace
{
    // Project identity reported to NGX. The project ID is an arbitrary GUID-style
    // string for telemetry; NVIDIA does not require a registered ID when using
    // the CUSTOM/UNITY engine type.
    const char*    kProjectId     = "unity-pathtracing-dlssfg";
    const char*    kEngineVersion = "1.0.0";
    // Where NGX may write logs/data. Must already exist; "." (cwd, i.e. next to
    // the host exe) always does.
    const wchar_t* kAppDataPath   = L".";

    std::atomic<bool>     g_Initialized{false};
    std::atomic<bool>     g_FgAvailable{false};
    NVSDK_NGX_Parameter*  g_CapParams = nullptr;

    NgxContext::LogFn     g_Logger = nullptr;

    void Logf(int level, const char* fmt, ...)
    {
        char buf[640];
        va_list ap;
        va_start(ap, fmt);
        int n = _vsnprintf_s(buf, sizeof(buf), _TRUNCATE, fmt, ap);
        va_end(ap);
        if (n < 0) buf[sizeof(buf) - 1] = '\0';

        const char* tag = (level == 2) ? "[NR/Ngx ERR] "
                        : (level == 1) ? "[NR/Ngx WRN] "
                                       : "[NR/Ngx] ";
        char prefixed[704];
        _snprintf_s(prefixed, sizeof(prefixed), _TRUNCATE, "%s%s", tag, buf);

        if (g_Logger)
        {
            g_Logger(level, prefixed);
        }
        else
        {
            char withNl[720];
            _snprintf_s(withNl, sizeof(withNl), _TRUNCATE, "%s\n", prefixed);
            OutputDebugStringA(withNl);
        }
    }

    // Read an unsigned int from the capability map, returning a default when the
    // parameter is absent (older snippet / feature not present).
    unsigned int GetUI(const char* name, unsigned int dflt)
    {
        unsigned int v = dflt;
        if (g_CapParams && NVSDK_NGX_SUCCEED(g_CapParams->Get(name, &v)))
            return v;
        return dflt;
    }

    // Log everything we can learn about DLSS-FG from the capability map. This is
    // the actual payload of Test 1.
    void ReportFrameGenerationSupport()
    {
        unsigned int available    = GetUI(NVSDK_NGX_Parameter_FrameGeneration_Available, 0);
        unsigned int needsDriver  = GetUI(NVSDK_NGX_Parameter_FrameGeneration_NeedsUpdatedDriver, 0);
        unsigned int minDrvMajor  = GetUI(NVSDK_NGX_Parameter_FrameGeneration_MinDriverVersionMajor, 0);
        unsigned int minDrvMinor  = GetUI(NVSDK_NGX_Parameter_FrameGeneration_MinDriverVersionMinor, 0);
        unsigned int initResult   = GetUI(NVSDK_NGX_Parameter_FrameGeneration_FeatureInitResult, 0);

        g_FgAvailable.store(available != 0, std::memory_order_release);

        if (available)
        {
            Logf(0, "DLSS-FG AVAILABLE on this GPU (FeatureInitResult=0x%08x)", initResult);
        }
        else if (needsDriver)
        {
            Logf(1, "DLSS-FG unavailable: driver too old. Need >= %u.%u "
                    "(FeatureInitResult=0x%08x)", minDrvMajor, minDrvMinor, initResult);
        }
        else
        {
            // Expected path on the RTX 3060 dev box (pre-Ada hardware).
            Logf(1, "DLSS-FG unavailable on this GPU (hardware not supported / pre-Ada). "
                    "FeatureInitResult=0x%08x. This is expected on non-40-series cards.",
                 initResult);
        }
    }
}

namespace NgxContext
{
    void SetLogger(LogFn fn)
    {
        g_Logger = fn;
        Logf(0, "Logger attached");
    }

    bool Initialize(ID3D12Device* device)
    {
        if (g_Initialized.load(std::memory_order_acquire)) return true;
        if (!device)
        {
            Logf(2, "Initialize: device is null");
            return false;
        }

        // Init NGX on Unity's existing device. NGX is GPU-agnostic here; this
        // call is expected to succeed on the RTX 3060 as well as the target 4060.
        NVSDK_NGX_Result r = NVSDK_NGX_D3D12_Init_with_ProjectID(
            kProjectId, NVSDK_NGX_ENGINE_TYPE_UNITY, kEngineVersion,
            kAppDataPath, device);
        if (NVSDK_NGX_FAILED(r))
        {
            Logf(2, "NVSDK_NGX_D3D12_Init_with_ProjectID failed: 0x%08x (%ls)",
                 r, GetNGXResultAsString(r));
            return false;
        }

        // Capability map is pre-populated with feature support info — use it
        // (not AllocateParameters) so the FrameGeneration.* query has data.
        r = NVSDK_NGX_D3D12_GetCapabilityParameters(&g_CapParams);
        if (NVSDK_NGX_FAILED(r) || !g_CapParams)
        {
            Logf(2, "GetCapabilityParameters failed: 0x%08x (%ls)",
                 r, GetNGXResultAsString(r));
            NVSDK_NGX_D3D12_Shutdown1(device);
            return false;
        }

        g_Initialized.store(true, std::memory_order_release);
        Logf(0, "NGX initialized on Unity's D3D12 device (device=%p)", (void*)device);

        ReportFrameGenerationSupport();
        return true;
    }

    bool IsInitialized()
    {
        return g_Initialized.load(std::memory_order_acquire);
    }

    bool IsFrameGenerationAvailable()
    {
        return g_FgAvailable.load(std::memory_order_acquire);
    }

    void Shutdown(ID3D12Device* /*device*/)
    {
        // Only release what we exclusively own: our capability parameter map.
        //
        // We deliberately do NOT call NVSDK_NGX_D3D12_Shutdown1 here. NGX is a
        // process-global subsystem already initialized and owned by NRI (the
        // Denoiser plugin's DLSS-RR/SR upscalers). Tearing it down from here
        // would either race NRI's own shutdown (the source of the earlier
        // 0xbad00007 NotInitialized warning) or, worse, pull NGX out from under
        // an active DLSS-RR instance. The process exit reclaims NGX regardless.
        if (g_CapParams)
        {
            NVSDK_NGX_D3D12_DestroyParameters(g_CapParams);
            g_CapParams = nullptr;
        }
        // Reset so a later device re-init re-runs the (idempotent) NGX init and
        // re-queries support against the new device.
        g_Initialized.store(false, std::memory_order_release);
        g_FgAvailable.store(false, std::memory_order_release);
        Logf(0, "NgxContext released (NGX lifecycle left to NRI)");
    }
}
