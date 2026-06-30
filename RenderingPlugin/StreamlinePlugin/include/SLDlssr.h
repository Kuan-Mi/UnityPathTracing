// SLDlssr.h
// Streamline (SL 2.11.1) integration for DLSS Super Resolution (DLSS-SR / kFeatureDLSS),
// the upscaling counterpart of the DLSS-RR path (SLDlssrr). Replaces the NRI DLSS-SR path
// (NRIPlugin.dll / DLSRInstance).
//
// Like DLSS-RR, DLSS-SR is an EVALUATE-time feature: slEvaluateFeature(kFeatureDLSS, ...)
// runs on the frame's command list during rendering. It does NOT take over the present
// path, so there is no device-queue / swapchain proxying — editor-safe, loaded on demand.
//
// The SL lifecycle (slInit / slSetD3DDevice / slShutdown + logging) is shared across all
// features and lives in SLCore (see SLCore.h). This file is SR-specific only.
//
// Per frame (render thread): CreateInstance/Dispatch -> token + options + constants +
// tag + slEvaluateFeature on Unity's command list.
#pragma once

struct ID3D12GraphicsCommandList;
struct SLDlssrFrameData;

namespace SLDlssr
{
    // Instance lifecycle (one viewport per instance; viewport id == instance id).
    int  CreateInstance();
    void DestroyInstance(int id);

    // Render-thread per-frame evaluate. Issued from IssuePluginEventAndData; the plugin
    // entry supplies Unity's open D3D12 command list (from CommandRecordingState).
    void Dispatch(SLDlssrFrameData* data, ID3D12GraphicsCommandList* cmdList);

    // slDLSSGetOptimalSettings (cached). mode matches the C# UpscalerMode byte.
    bool QueryOptimalRenderSize(unsigned outW, unsigned outH, unsigned char mode,
                                unsigned* outRenderW, unsigned* outRenderH);

    // Release all SR viewport resources/instances. Call before SLCore::Shutdown.
    void Shutdown();
}
