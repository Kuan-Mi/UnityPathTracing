// SLDlssrr.cpp — see SLDlssrr.h.
//
// DLSS Ray Reconstruction via Streamline. Per frame on Unity's command list:
//   slGetNewFrameToken -> slDLSSDSetOptions (on change) -> slSetConstants ->
//   slSetTagForFrame(all RR guides) -> slEvaluateFeature(kFeatureDLSS_RR).
// SL manages the resource state transitions for tagged resources; the host is
// responsible for restoring command-list state afterwards — we rely on Unity
// rebinding its own pipeline for subsequent passes (same as the NRI path).

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "sl.h"
#include "sl_consts.h"
#include "sl_dlss.h"
#include "sl_dlss_d.h"
#include "sl_reflex.h"
#include "sl_pcl.h"

#include "SLDlssrr.h"
#include "SLDlssrrFrameData.h"

namespace
{
    SLDlssrr::LogFn   g_log = nullptr;
    std::atomic<bool> g_inited{ false };
    bool              g_deviceSet = false;

    // Per-instance (== per-viewport) tracked options so we only re-issue slDLSSDSetOptions
    // when something changes (matches DLRRInstance recreate-on-change behaviour).
    struct InstanceState
    {
        uint32_t outputWidth  = 0;
        uint32_t outputHeight = 0;
        uint8_t  mode         = 0xFF;
        uint8_t  preset       = 0xFF;
        bool     optionsSet   = false;
    };
    std::unordered_map<int, InstanceState> g_instances;
    std::mutex                             g_instanceMutex;
    int                                    g_nextInstanceId = 1;
    uint32_t                               g_frameIndex     = 0;

    // QueryOptimalRenderSize cache, keyed by (outputWidth, outputHeight, mode).
    struct OptKey
    {
        uint32_t w, h; uint8_t m;
        bool operator==(const OptKey& o) const { return w == o.w && h == o.h && m == o.m; }
    };
    struct OptKeyHash
    {
        size_t operator()(const OptKey& k) const
        { return std::hash<uint64_t>()(((uint64_t)k.w << 24) | ((uint64_t)k.h << 8) | k.m); }
    };
    std::unordered_map<OptKey, sl::DLSSDOptimalSettings, OptKeyHash> g_optCache;
    std::mutex                                                       g_optMutex;

    void Logf(int level, const char* fmt, ...)
    {
        char buf[768];
        va_list ap; va_start(ap, fmt);
        _vsnprintf_s(buf, sizeof(buf), _TRUNCATE, fmt, ap);
        va_end(ap);
        const char* tag = (level == 2) ? "[NR/SLDlssrr ERR] "
                        : (level == 1) ? "[NR/SLDlssrr WRN] "
                                       : "[NR/SLDlssrr] ";
        char line[864];
        _snprintf_s(line, sizeof(line), _TRUNCATE, "%s%s", tag, buf);
        if (g_log) g_log(level, line);
        else { OutputDebugStringA(line); OutputDebugStringA("\n"); }
    }

    void SLLog(sl::LogType type, const char* msg)
    {
        if (!g_log || !msg) return;
        const int lvl = (type == sl::LogType::eError) ? 2
                      : (type == sl::LogType::eWarn)  ? 1 : 0;
        char line[1024];
        _snprintf_s(line, sizeof(line), _TRUNCATE, "[NR/SLDlssrr][SL] %s", msg);
        size_t n = strnlen_s(line, sizeof(line));
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
        g_log(lvl, line);
    }

    const char* R(sl::Result r)
    {
        switch (r)
        {
            case sl::Result::eOk:                           return "eOk";
            case sl::Result::eErrorDriverOutOfDate:         return "eErrorDriverOutOfDate";
            case sl::Result::eErrorNoSupportedAdapterFound: return "eErrorNoSupportedAdapterFound";
            case sl::Result::eErrorAdapterNotSupported:     return "eErrorAdapterNotSupported";
            case sl::Result::eErrorNoPlugins:               return "eErrorNoPlugins";
            case sl::Result::eErrorNotInitialized:          return "eErrorNotInitialized";
            case sl::Result::eErrorInitNotCalled:           return "eErrorInitNotCalled";
            case sl::Result::eErrorFeatureNotSupported:     return "eErrorFeatureNotSupported";
            case sl::Result::eErrorMissingInputParameter:   return "eErrorMissingInputParameter";
            case sl::Result::eErrorMissingConstants:        return "eErrorMissingConstants";
            case sl::Result::eErrorMissingOrInvalidAPI:     return "eErrorMissingOrInvalidAPI";
            default:                                        return "(other)";
        }
    }

    sl::DLSSMode MapMode(uint8_t m)
    {
        // C# UpscalerMode: 0 NATIVE,1 ULTRA_QUALITY,2 QUALITY,3 BALANCED,4 PERFORMANCE,5 ULTRA_PERFORMANCE
        switch (m)
        {
            case 0:  return sl::DLSSMode::eDLAA;            // native render == output
            case 1:  return sl::DLSSMode::eUltraQuality;
            case 2:  return sl::DLSSMode::eMaxQuality;
            case 3:  return sl::DLSSMode::eBalanced;
            case 4:  return sl::DLSSMode::eMaxPerformance;
            case 5:  return sl::DLSSMode::eUltraPerformance;
            default: return sl::DLSSMode::eDLAA;
        }
    }

    sl::DLSSDPreset MapPreset(uint8_t p)
    {
        switch (p)
        {
            case 4:  return sl::DLSSDPreset::ePresetD;
            case 5:  return sl::DLSSDPreset::ePresetE;
            default: return sl::DLSSDPreset::eDefault;
        }
    }

    void CopyMatrix(sl::float4x4& dst, const float src[16])
    {
        // Raw Unity Matrix4x4 bytes (column-major) -> row-major sl::float4x4 performs the
        // column->row-vector transpose Streamline expects (same as StreamlineProbe).
        std::memcpy(&dst.row[0].x, src, sizeof(float) * 16);
    }
}

namespace SLDlssrr
{
    bool InitSL(LogFn log)
    {
        g_log = log;
        bool expected = false;
        if (!g_inited.compare_exchange_strong(expected, true)) return true;

        Logf(0, "slInit Streamline %u.%u.%u for DLSS-RR + DLSS-G (manual hooking)...",
             SL_VERSION_MAJOR, SL_VERSION_MINOR, SL_VERSION_PATCH);

        // DLSS_RR (evaluate-time) + DLSS_G/Reflex/PCL (present-path frame generation).
        // One slInit serves both; the FG present/queue hooks are installed only in the
        // player (see SLDenoiserPlugin.cpp editor auto-detect).
        static const sl::Feature kFeatures[] = {
            sl::kFeatureDLSS_RR, sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL,
        };
        sl::Preferences pref{};
        // eDefault (warnings/errors only) + no console: eVerbose logs every NGX SRV-cache hit
        // and VRAM op per frame through the Unity log callback, a real per-frame CPU cost.
        // Bump to eVerbose + showConsole only when debugging SL itself.
        pref.showConsole        = false;
        pref.logLevel           = sl::LogLevel::eDefault;
        pref.logMessageCallback = &SLLog;
        pref.featuresToLoad     = kFeatures;
        pref.numFeaturesToLoad  = (uint32_t)_countof(kFeatures);
        // Manual hooking: we attach to Unity's device + evaluate on Unity's command list,
        // SL does not own the swapchain. Frame-based tagging is required by slSetTagForFrame.
        pref.flags             |= sl::PreferenceFlags::eUseManualHooking;
        pref.flags             |= sl::PreferenceFlags::eUseFrameBasedResourceTagging;
        pref.engine             = sl::EngineType::eUnity;
        pref.engineVersion      = "6000.3";
        pref.projectId          = "a0f57b54-1daf-4934-90ae-c4035c19df04";
        pref.renderAPI          = sl::RenderAPI::eD3D12;

        sl::Result r = slInit(pref, sl::kSDKVersion);
        Logf(r == sl::Result::eOk ? 0 : 2, "slInit -> %s", R(r));
        if (r != sl::Result::eOk) { g_inited.store(false); return false; }
        return true;
    }

    bool IsInited() { return g_inited.load(std::memory_order_acquire); }

    bool IsDeviceSet() { return g_deviceSet; }

    void SetDevice(ID3D12Device* device)
    {
        if (!IsInited() || g_deviceSet || !device) return;
        sl::Result rd = slSetD3DDevice(device);
        Logf(rd == sl::Result::eOk ? 0 : 2, "slSetD3DDevice -> %s", R(rd));
        g_deviceSet = (rd == sl::Result::eOk);
        if (!g_deviceSet) return;

        LUID luid = device->GetAdapterLuid();
        sl::AdapterInfo ai{};
        ai.deviceLUID            = reinterpret_cast<uint8_t*>(&luid);
        ai.deviceLUIDSizeInBytes = sizeof(luid);
        sl::Result rs = slIsFeatureSupported(sl::kFeatureDLSS_RR, ai);
        Logf(rs == sl::Result::eOk ? 0 : 1, "slIsFeatureSupported(DLSS_RR) -> %s", R(rs));
    }

    int CreateInstance()
    {
        std::scoped_lock lock(g_instanceMutex);
        int id = g_nextInstanceId++;
        g_instances[id] = InstanceState{};
        Logf(0, "Created SL DLSS-RR instance %d (viewport %d).", id, id);
        return id;
    }

    void DestroyInstance(int id)
    {
        std::scoped_lock lock(g_instanceMutex);
        auto it = g_instances.find(id);
        if (it == g_instances.end()) return;
        if (IsInited())
        {
            sl::ViewportHandle viewport{ (uint32_t)id };
            sl::Result r = slFreeResources(sl::kFeatureDLSS_RR, viewport);
            Logf(r == sl::Result::eOk ? 0 : 1, "slFreeResources(DLSS_RR, vp %d) -> %s", id, R(r));
        }
        g_instances.erase(it);
        Logf(0, "Destroyed SL DLSS-RR instance %d.", id);
    }

    void Dispatch(SLDlssrrFrameData* data, ID3D12GraphicsCommandList* cmdList)
    {
        if (!IsInited() || !g_deviceSet || !data || !cmdList) return;
        if (data->outputWidth == 0 || data->outputHeight == 0) return;

        sl::ViewportHandle viewport{ (uint32_t)data->instanceId };

        // --- frame token ---
        sl::FrameToken* token = nullptr;
        uint32_t idx = g_frameIndex++;
        sl::Result rt = slGetNewFrameToken(token, &idx);
        if (rt != sl::Result::eOk || !token) { Logf(1, "slGetNewFrameToken -> %s", R(rt)); return; }

        // --- options (only when changed) ---
        {
            std::scoped_lock lock(g_instanceMutex);
            auto it = g_instances.find(data->instanceId);
            if (it == g_instances.end()) { Logf(1, "Dispatch: unknown instance %d", data->instanceId); return; }
            InstanceState& st = it->second;

            if (!st.optionsSet || st.outputWidth != data->outputWidth ||
                st.outputHeight != data->outputHeight || st.mode != data->upscalerMode ||
                st.preset != data->preset)
            {
                sl::DLSSDOptions opt{};
                opt.mode                = MapMode(data->upscalerMode);
                opt.outputWidth         = data->outputWidth;
                opt.outputHeight        = data->outputHeight;
                opt.colorBuffersHDR     = sl::Boolean::eTrue;     // DLSS-RR is HDR only
                opt.normalRoughnessMode = sl::DLSSDNormalRoughnessMode::ePacked; // roughness in normal.w
                opt.alphaUpscalingEnabled = sl::Boolean::eFalse;
                // Used by the specular-hit-distance path; harmless when specular MV are supplied.
                CopyMatrix(opt.worldToCameraView, data->worldToViewMatrix);
                CopyMatrix(opt.cameraViewToWorld, data->viewToWorldMatrix);
                const sl::DLSSDPreset preset = MapPreset(data->preset);
                opt.dlaaPreset             = preset;
                opt.qualityPreset          = preset;
                opt.balancedPreset         = preset;
                opt.performancePreset      = preset;
                opt.ultraPerformancePreset = preset;
                opt.ultraQualityPreset     = preset;

                sl::Result ro = slDLSSDSetOptions(viewport, opt);
                Logf(ro == sl::Result::eOk ? 0 : 2,
                     "slDLSSDSetOptions(vp %d, mode %d, %ux%u, preset %u) -> %s",
                     data->instanceId, (int)opt.mode, opt.outputWidth, opt.outputHeight,
                     (unsigned)data->preset, R(ro));
                if (ro == sl::Result::eOk)
                {
                    st.outputWidth  = data->outputWidth;
                    st.outputHeight = data->outputHeight;
                    st.mode         = data->upscalerMode;
                    st.preset       = data->preset;
                    st.optionsSet   = true;
                }
            }
        }

        // --- common constants ---
        sl::Constants c{};
        CopyMatrix(c.cameraViewToClip, data->cameraViewToClip);
        CopyMatrix(c.clipToCameraView, data->clipToCameraView);
        CopyMatrix(c.clipToPrevClip,   data->clipToPrevClip);
        CopyMatrix(c.prevClipToClip,   data->prevClipToClip);
        c.clipToLensClip.row[0] = sl::float4(1, 0, 0, 0);
        c.clipToLensClip.row[1] = sl::float4(0, 1, 0, 0);
        c.clipToLensClip.row[2] = sl::float4(0, 0, 1, 0);
        c.clipToLensClip.row[3] = sl::float4(0, 0, 0, 1);
        // Negate to match the NRI DLSS-RR path (DLRRInstance: cameraJitter = {-x,-y}) and the
        // NGX/SL sign convention: jitterOffset is the opposite sign of the applied projection
        // offset. Passing it un-negated makes DLSS reproject the wrong way → the image "shakes".
        c.jitterOffset        = sl::float2(-data->cameraJitter[0], -data->cameraJitter[1]);
        c.mvecScale           = sl::float2(data->mvecScale[0], data->mvecScale[1]);
        c.cameraPinholeOffset = sl::float2(0, 0);
        c.cameraPos   = sl::float3(data->cameraPos[0],   data->cameraPos[1],   data->cameraPos[2]);
        c.cameraUp    = sl::float3(data->cameraUp[0],    data->cameraUp[1],    data->cameraUp[2]);
        c.cameraRight = sl::float3(data->cameraRight[0], data->cameraRight[1], data->cameraRight[2]);
        c.cameraFwd   = sl::float3(data->cameraFwd[0],   data->cameraFwd[1],   data->cameraFwd[2]);
        c.cameraNear        = data->cameraNear;
        c.cameraFar         = data->cameraFar;
        c.cameraFOV         = data->cameraFOV;
        c.cameraAspectRatio = data->cameraAspect;
        c.depthInverted        = data->depthInverted        ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        c.cameraMotionIncluded = data->cameraMotionIncluded ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        c.motionVectors3D      = data->motionVectors3D      ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        c.reset                = data->reset                ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        sl::Result rc = slSetConstants(c, *token, viewport);

        // --- tag resources ---
        sl::Extent renderExtent{ 0, 0, data->renderWidth,  data->renderHeight };
        sl::Extent outputExtent{ 0, 0, data->outputWidth,  data->outputHeight };

        sl::Resource rInput (sl::ResourceType::eTex2d, data->inputTex,           data->inputState);
        sl::Resource rOutput(sl::ResourceType::eTex2d, data->outputTex,          data->outputState);
        sl::Resource rMv    (sl::ResourceType::eTex2d, data->mvTex,              data->mvState);
        sl::Resource rDepth (sl::ResourceType::eTex2d, data->depthTex,           data->depthState);
        sl::Resource rDiff  (sl::ResourceType::eTex2d, data->diffuseAlbedoTex,   data->diffuseAlbedoState);
        sl::Resource rSpec  (sl::ResourceType::eTex2d, data->specularAlbedoTex,  data->specularAlbedoState);
        sl::Resource rNorm  (sl::ResourceType::eTex2d, data->normalRoughnessTex, data->normalRoughnessState);
        sl::Resource rSpecMv(sl::ResourceType::eTex2d, data->specularMvOrHitTex, data->specularMvOrHitState);

        const sl::BufferType specType = data->useSpecularMotionVector
                                      ? sl::kBufferTypeSpecularMotionVectors
                                      : sl::kBufferTypeSpecularHitDistance;

        // eValidUntilPresent: SL references these persistent path-tracer RTs DIRECTLY (no
        // copy). eOnlyValidNow marks a resource "volatile" → SL allocates+copies an internal
        // texture EVERY FRAME (per-tag) which churns VRAM and tanks the frame rate. These
        // pool textures are dedicated and not reused within the frame, so direct reference is
        // correct (matches donut/RTXPT DLSS tagging).
        const sl::ResourceLifecycle kLifecycle = sl::ResourceLifecycle::eValidUntilPresent;
        sl::ResourceTag tags[] = {
            sl::ResourceTag(&rInput,  sl::kBufferTypeScalingInputColor,  kLifecycle, &renderExtent),
            sl::ResourceTag(&rOutput, sl::kBufferTypeScalingOutputColor, kLifecycle, &outputExtent),
            sl::ResourceTag(&rDepth,  sl::kBufferTypeLinearDepth,        kLifecycle, &renderExtent),
            sl::ResourceTag(&rMv,     sl::kBufferTypeMotionVectors,      kLifecycle, &renderExtent),
            sl::ResourceTag(&rDiff,   sl::kBufferTypeAlbedo,             kLifecycle, &renderExtent),
            sl::ResourceTag(&rSpec,   sl::kBufferTypeSpecularAlbedo,     kLifecycle, &renderExtent),
            sl::ResourceTag(&rNorm,   sl::kBufferTypeNormalRoughness,    kLifecycle, &renderExtent),
            sl::ResourceTag(&rSpecMv, specType,                          kLifecycle, &renderExtent),
        };
        sl::Result rtag = slSetTagForFrame(*token, viewport, tags, (uint32_t)_countof(tags),
                                           reinterpret_cast<sl::CommandBuffer*>(cmdList));

        // --- evaluate ---
        const sl::BaseStructure* inputs[] = { &viewport };
        sl::Result re = slEvaluateFeature(sl::kFeatureDLSS_RR, *token, inputs, (uint32_t)_countof(inputs),
                                          reinterpret_cast<sl::CommandBuffer*>(cmdList));

        static uint64_t s_frames = 0;
        const uint64_t f = ++s_frames;
        if (f <= 4 || (f & 0xFF) == 0 ||
            rc != sl::Result::eOk || rtag != sl::Result::eOk || re != sl::Result::eOk)
        {
            Logf((rc != sl::Result::eOk || rtag != sl::Result::eOk || re != sl::Result::eOk) ? 2 : 0,
                 "frame #%llu vp %d: setConstants=%s setTag=%s evaluate=%s",
                 (unsigned long long)f, data->instanceId, R(rc), R(rtag), R(re));
        }
        // Host is responsible for restoring command-list state after evaluate; Unity rebinds
        // its own pipeline (descriptor heaps via the shared D3D12 heap hook, root sig, PSO)
        // for the next pass, so we intentionally do not restore here.
    }

    bool QueryOptimalRenderSize(unsigned outW, unsigned outH, unsigned char mode,
                                unsigned* outRenderW, unsigned* outRenderH)
    {
        if (!outRenderW || !outRenderH) return false;

        // NATIVE/DLAA: render resolution equals output resolution.
        if (mode == 0) { *outRenderW = outW; *outRenderH = outH; return true; }

        OptKey key{ outW, outH, mode };
        {
            std::scoped_lock lock(g_optMutex);
            auto it = g_optCache.find(key);
            if (it != g_optCache.end())
            {
                *outRenderW = it->second.optimalRenderWidth;
                *outRenderH = it->second.optimalRenderHeight;
                return true;
            }
        }

        if (!IsInited() || !g_deviceSet) return false;

        sl::DLSSDOptions opt{};
        opt.mode         = MapMode(mode);
        opt.outputWidth  = outW;
        opt.outputHeight = outH;
        opt.colorBuffersHDR = sl::Boolean::eTrue;

        sl::DLSSDOptimalSettings settings{};
        sl::Result r = slDLSSDGetOptimalSettings(opt, settings);
        if (r != sl::Result::eOk)
        {
            Logf(1, "slDLSSDGetOptimalSettings(%ux%u, mode %u) -> %s", outW, outH, (unsigned)mode, R(r));
            return false;
        }

        {
            std::scoped_lock lock(g_optMutex);
            g_optCache[key] = settings;
        }
        *outRenderW = settings.optimalRenderWidth;
        *outRenderH = settings.optimalRenderHeight;
        return true;
    }

    void Shutdown()
    {
        if (!IsInited()) return;
        g_inited.store(false);
        g_deviceSet = false;
        {
            std::scoped_lock lock(g_instanceMutex);
            g_instances.clear();
        }
        sl::Result r = slShutdown();
        Logf(r == sl::Result::eOk ? 0 : 1, "slShutdown -> %s", R(r));
    }
}
