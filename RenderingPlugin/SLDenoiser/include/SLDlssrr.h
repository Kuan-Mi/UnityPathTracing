// SLDlssrr.h
// Streamline (SL 2.11.1) integration for DLSS Ray Reconstruction (DLSS-RR), used as a
// standalone A/B comparison against the NRI DLSS-RR path (Denoiser.dll / DLRRInstance).
//
// Unlike the DLSS-G frame-generation probe (StreamlineProbePlugin), DLSS-RR is an
// EVALUATE-time feature: slEvaluateFeature(kFeatureDLSS_RR, ...) runs on the frame's
// command list during rendering. It does NOT take over the present path, so there is no
// device-queue / swapchain proxying — this is editor-safe and loaded on demand.
//
// Lifecycle:
//   InitSL()  — slInit({kFeatureDLSS_RR}, manual hooking + frame-based tagging) at load.
//   SetDevice() — slSetD3DDevice on Unity's device at the graphics-init event.
//   Per frame (render thread): CreateInstance/Dispatch -> token + options + constants +
//   tag + slEvaluateFeature on Unity's command list.
//   Shutdown() — slShutdown before the device dies.
#pragma once

struct ID3D12Device;
struct ID3D12GraphicsCommandList;
struct SLDlssrrFrameData;

namespace SLDlssrr
{
    // level: 0=info, 1=warn, 2=error.
    using LogFn = void (*)(int level, const char* msg);

    // slInit in manual-hooking mode with the DLSS_RR feature. Call once at plugin load.
    bool InitSL(LogFn log);
    bool IsInited();

    // slSetD3DDevice + DLSS-RR capability log. Idempotent + guarded: the FIRST successful
    // call wins; later calls (e.g. the FG queue hook and the graphics-init event both racing
    // to set it) are no-ops. Calling slSetD3DDevice twice makes SL reject the second
    // ("plugins already initialized") and would leave the device flag false → RR never runs.
    void SetDevice(ID3D12Device* device);

    // True once slSetD3DDevice has succeeded (shared by the RR and FG paths).
    bool IsDeviceSet();

    // Instance lifecycle (one viewport per instance; viewport id == instance id).
    int  CreateInstance();
    void DestroyInstance(int id);

    // Render-thread per-frame evaluate. Issued from IssuePluginEventAndData; the plugin
    // entry supplies Unity's open D3D12 command list (from CommandRecordingState).
    void Dispatch(SLDlssrrFrameData* data, ID3D12GraphicsCommandList* cmdList);

    // slDLSSDGetOptimalSettings (cached). mode matches the C# UpscalerMode byte.
    bool QueryOptimalRenderSize(unsigned outW, unsigned outH, unsigned char mode,
                                unsigned* outRenderW, unsigned* outRenderH);

    void Shutdown();
}
