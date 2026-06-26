// SLDlssg.cpp — DLSS-G (Frame Generation) via Streamline. See SLDlssg.h.
//
// Ported from the proven StreamlineProbe machinery. Differences:
//   * slInit/slSetD3DDevice/slShutdown + logging are shared via SLCore (SLCore::Init at
//     plugin load already loaded DLSS_G/Reflex/PCL alongside DLSS_RR).
//   * Inputs are REAL path-tracer depth + motion vectors + camera constants (no dummies,
//     no HUDLessColor — SL interpolates the presented backbuffer for now).
//
// The SL present path (proxy device/queue/swapchain + the Present hook) is SL's COMMON
// interposer and is installed on every adapter by SLHooks — it must be, because under
// eUseManualHooking the SL common plugin's presentCommon() only runs when presentation goes
// through an SL-upgraded swapchain. This file owns DLSS-G mode/input work and the pre-present
// backbuffer query required by FG. PCL marker emission belongs to SLReflex.cpp.
//
// Frame timeline (token shared by index — see SLCore.h):
//   * MAIN thread, top of frame: SLCore::GetNewFrameToken mints the frame token; C# then calls
//     SLReflex sleep and eSimulationStart separately. End of game logic: SLReflex eSimulationEnd.
//   * RENDER thread frame-begin/end events (data == the FrameToken*): SLCore::SetRenderFrame
//     latches the token, then SLReflex emits eRenderSubmitStart/End.
//   * RENDER thread: ConsumeFrameInputs() (FG only) tags depth/mvec + sets constants on the
//     render token (SLCore::CurrentFrameToken).
//   * PRESENT thread (native or SL proxy swapchain hook): SLHooks marks ePresentStart/End
//     around Present; this file only prepares DLSS-G state before Present.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_5.h>
#include <atomic>
#include <cstring>
#include <wrl/client.h>

#include "sl.h"
#include "sl_consts.h"
#include "sl_dlss_g.h"
// Reflex (slReflexSetOptions/slReflexSleep) now lives in SLReflex.cpp; the frame token in
// SLCore. SLHooks owns the actual DXGI Present hook.

#include "SLCore.h" // shared slInit/slSetD3DDevice/slShutdown + logging/result helpers
#include "SLDlssg.h"

using Microsoft::WRL::ComPtr;

// Terse forwarders to the shared SLCore helpers (tag every line "SLDlssg").
#define Logf(level, ...) SLCore::Logf("SLDlssg", level, __VA_ARGS__)
#define R(r)             SLCore::ResultStr(r)

namespace
{
    std::atomic<int>       g_fgDesired{ -1 };
    std::atomic<bool>      g_fgApplied{ false };
    // Whether slDLSSGSetOptions has been pushed at least once. DLSS-G must be configured into a
    // defined mode (even eOff) before the SL proxy swapchain's presentCommon() drives it; without
    // this, starting with FG OFF skipped slDLSSGSetOptions entirely (g_fgApplied already == false)
    // and the unconfigured DLSS-G removed the device on the first present.
    std::atomic<bool>      g_fgModeInitialized{ false };

    UINT                   g_w = 0, g_h = 0;
    bool                   g_adopted = false;

    SLDlssg::FrameInputs   g_inputs{};
    bool                   g_haveRealInputs = false;
    uint32_t               g_appliedFrameIdx = 0xFFFFFFFFu;
    uint64_t               g_taggedFrames = 0;
    uint64_t               g_presentCount = 0;
    bool                   g_featuresEnabledOnPresent = false;
    bool                   g_isDlssgModeOn = false;
    void ApplyDlssgMode(bool on)
    {
        // Skip only once DLSS-G has actually been configured at least once AND is already in the
        // requested mode. The first call must always go through — even for eOff — so DLSS-G is
        // initialized into a known state before presentCommon() starts driving the proxy swapchain.
        if (g_fgModeInitialized.load(std::memory_order_acquire) && g_fgApplied.load() == on) return;
        sl::ViewportHandle viewport{ 0 };
        sl::DLSSGOptions opt{};
        opt.mode = on ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
        opt.numFramesToGenerate = 1;
        opt.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
        sl::Result r = slDLSSGSetOptions(viewport, opt);
        Logf(r == sl::Result::eOk ? 0 : 2, "slDLSSGSetOptions(mode=%s, gen=1, retain) -> %s",
             on ? "eOn" : "eOff", R(r));
        if (r == sl::Result::eOk)
        {
            g_fgApplied.store(on);
            g_fgModeInitialized.store(true, std::memory_order_release);
        }
        g_isDlssgModeOn = on;
    }

    void LogDlssgState()
    {
        sl::ViewportHandle viewport{ 0 };
        sl::DLSSGState st{};
        sl::Result r = slDLSSGGetState(viewport, st, nullptr);
        if (r != sl::Result::eOk) { Logf(1, "slDLSSGGetState -> %s", R(r)); return; }
        const uint32_t s = (uint32_t)st.status;
        char flags[256] = "";
        if (s == 0) strcpy_s(flags, "eOk");
        else {
            if (s & (1u<<0)) strcat_s(flags, "ResolutionTooLow|");
            if (s & (1u<<1)) strcat_s(flags, "ReflexNotDetected|");
            if (s & (1u<<2)) strcat_s(flags, "HDRFormatNotSupported|");
            if (s & (1u<<3)) strcat_s(flags, "CommonConstantsInvalid|");
            if (s & (1u<<4)) strcat_s(flags, "GetCurrentBackBufferIndexNotCalled|");
        }
        Logf(s == 0 ? 0 : 1,
             "DLSSGState: status=0x%x [%s] framesPresentedSinceLast=%u maxGen=%u minDim=%u",
             s, flags, st.numFramesActuallyPresented, st.numFramesToGenerateMax,
             st.minWidthOrHeight);
    }

    void identity(sl::float4x4& m)
    {
        m.row[0] = sl::float4(1, 0, 0, 0);
        m.row[1] = sl::float4(0, 1, 0, 0);
        m.row[2] = sl::float4(0, 0, 1, 0);
        m.row[3] = sl::float4(0, 0, 0, 1);
    }

    void EnsureFeaturesOnPresentThread()
    {
        if (!g_featuresEnabledOnPresent)
        {
            g_featuresEnabledOnPresent = true;
            Logf(0, "First present on thread %lu (Reflex is driven from frame begin via SLReflex).",
                 (unsigned long)GetCurrentThreadId());
            if (g_fgDesired.load() < 0)
            {
                char env[8] = {};
                DWORD n = GetEnvironmentVariableA("NR_SL_ENABLE_FG", env, sizeof(env));
                const bool on = (n > 0 && env[0] == '1');
                g_fgDesired.store(on ? 1 : 0);
                Logf(0, on ? "DLSS-G initial state ON (NR_SL_ENABLE_FG=1)."
                           : "DLSS-G initial state OFF (call SL_SetFrameGeneration(1) to enable).");
            }
        }
        const int desired = g_fgDesired.load();
        if (desired >= 0)
            ApplyDlssgMode(desired != 0);
    }

    void PrepareDlssgForPresent(IDXGISwapChain3* proxySwapchain)
    {
        // DLSS-G mode application is FG-only; skip it entirely on a Reflex-only adapter.
        if (SLCore::IsFGSupported())
            EnsureFeaturesOnPresentThread();

        if (SLCore::IsFGSupported() && proxySwapchain) proxySwapchain->GetCurrentBackBufferIndex();
    }

    void PostPresentDlssgState()
    {
        const uint64_t p = ++g_presentCount;
        if (SLCore::IsFGSupported() && (p <= 3 || (p & 0x7F) == 0) && g_isDlssgModeOn) LogDlssgState();
    }

}

namespace SLDlssg
{
    static void ApplyRealInputs(const FrameInputs& in, sl::FrameToken& token)
    {
        sl::ViewportHandle viewport{ 0 };
        sl::Extent depthMvecExtent{ 0, 0, in.mvecDepthW, in.mvecDepthH };

        sl::Resource rDepth(sl::ResourceType::eTex2d, in.depth,         (uint32_t)in.depthState);
        sl::Resource rMvec (sl::ResourceType::eTex2d, in.motionVectors, (uint32_t)in.mvecState);

        // DLSS-G tags are eValidUntilPresent (SL reads them at present; no per-frame copy).
        // DLSS-G REQUIRES kBufferTypeDepth (see ProgrammingGuideDLSS_G §5.1/§5.2) — it does not
        // consume kBufferTypeLinearDepth (that is a DLSS-SR/RR input). Tagging linear depth here
        // leaves DLSS-G with no depth input and frame generation stops. The depth *values* fed
        // (pool.ViewZ) and the depthInverted flag are the tuning knobs for the is_dynamic test.
        sl::ResourceTag tags[] = {
            sl::ResourceTag(&rDepth, sl::kBufferTypeDepth,         sl::ResourceLifecycle::eValidUntilPresent, &depthMvecExtent),
            sl::ResourceTag(&rMvec,  sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent, &depthMvecExtent),
        };
        sl::Result rTag = slSetTagForFrame(token, viewport, tags, (uint32_t)_countof(tags), nullptr);

        sl::Constants c{};
        std::memcpy(&c.cameraViewToClip.row[0].x, in.cameraViewToClip, sizeof(float) * 16);
        std::memcpy(&c.clipToCameraView.row[0].x, in.clipToCameraView, sizeof(float) * 16);
        std::memcpy(&c.clipToPrevClip.row[0].x,   in.clipToPrevClip,   sizeof(float) * 16);
        std::memcpy(&c.prevClipToClip.row[0].x,   in.prevClipToClip,   sizeof(float) * 16);
        identity(c.clipToLensClip);
        c.jitterOffset        = sl::float2(-in.jitterX, -in.jitterY); // NGX/SL sign convention
        c.mvecScale           = sl::float2(in.mvecScaleX, in.mvecScaleY);
        c.cameraPinholeOffset = sl::float2(0, 0);
        c.cameraPos   = sl::float3(in.cameraPos[0],   in.cameraPos[1],   in.cameraPos[2]);
        c.cameraUp    = sl::float3(in.cameraUp[0],    in.cameraUp[1],    in.cameraUp[2]);
        c.cameraRight = sl::float3(in.cameraRight[0], in.cameraRight[1], in.cameraRight[2]);
        c.cameraFwd   = sl::float3(in.cameraFwd[0],   in.cameraFwd[1],   in.cameraFwd[2]);
        c.cameraNear        = in.cameraNear;
        c.cameraFar         = in.cameraFar;
        c.cameraFOV         = in.cameraFOV;
        c.cameraAspectRatio = in.cameraAspect;
        c.depthInverted        = in.depthInverted        ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        c.cameraMotionIncluded = in.cameraMotionIncluded ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        c.motionVectors3D      = in.motionVectors3D      ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        c.reset                = (in.reset || g_taggedFrames < 2) ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        sl::Result rC = slSetConstants(c, token, viewport);

        const uint64_t f = ++g_taggedFrames;
        if (f <= 4 || (f & 0xFF) == 0 || rTag != sl::Result::eOk || rC != sl::Result::eOk)
            Logf((rTag != sl::Result::eOk || rC != sl::Result::eOk) ? 2 : 0,
                 "FG frame #%llu: slSetTagForFrame -> %s, slSetConstants -> %s",
                 (unsigned long long)f, R(rTag), R(rC));
    }

    void ConsumeFrameInputs(const FrameInputs& inputs)
    {
        g_inputs = inputs;
        if (!g_haveRealInputs)
        {
            g_haveRealInputs = true;
            Logf(0, "First real DLSS-G inputs received (mvec/depth %ux%u, frame %ux%u).",
                 inputs.mvecDepthW, inputs.mvecDepthH, inputs.colorW, inputs.colorH);
        }
        sl::FrameToken* token = SLCore::CurrentFrameToken();
        if (!token)
        {
            Logf(1, "ConsumeFrameInputs: no render token latched yet (frame-begin event not run).");
            return;
        }
        const uint32_t idx = (uint32_t)(*token);
        if (g_appliedFrameIdx == idx) return;
        ApplyRealInputs(g_inputs, *token);
        g_appliedFrameIdx = idx;
    }

    void OnPresentPre(IDXGISwapChain3* proxySwapchain)
    {
        PrepareDlssgForPresent(proxySwapchain);
    }

    void OnPresentPost()
    {
        PostPresentDlssgState();
    }

    void OnSwapChainAdopted(unsigned width, unsigned height)
    {
        g_w = width;
        g_h = height;
        g_adopted = true;
        Logf(0, "Swapchain adopted (%ux%u); DLSS-G ready. Toggle with SL_SetFrameGeneration.", g_w, g_h);
    }

    void SetFrameGeneration(bool enable)
    {
        if (enable && !SLCore::IsFGSupported())
        {
            Logf(1, "SetFrameGeneration(ON) ignored: Frame Generation is unavailable on this adapter.");
            return;
        }
        g_fgDesired.store(enable ? 1 : 0, std::memory_order_release);
        Logf(0, "SetFrameGeneration(%s) requested; applies on next present.", enable ? "ON" : "OFF");
    }

    bool IsFrameGenerationOn()
    {
        return SLCore::IsFGSupported() && g_fgApplied.load(std::memory_order_acquire);
    }

    void Shutdown()
    {
        g_fgDesired.store(-1); g_fgApplied.store(false); g_fgModeInitialized.store(false);
        g_featuresEnabledOnPresent = false;
        g_haveRealInputs = false; g_inputs = {}; g_adopted = false;
        g_appliedFrameIdx = 0xFFFFFFFFu;
        g_w = 0; g_h = 0;
        // slShutdown is owned by the RR module (shared slInit).
    }
}
