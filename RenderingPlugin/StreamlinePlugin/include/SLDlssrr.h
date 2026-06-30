// SLDlssrr.h
// Streamline (SL 2.11.1) integration for DLSS Ray Reconstruction (DLSS-RR), used as a
// standalone A/B comparison against the NRI DLSS-RR path (NRIPlugin.dll / DLRRInstance).
//
// Unlike the DLSS-G frame-generation path (SLDlssg), DLSS-RR is an EVALUATE-time feature:
// slEvaluateFeature(kFeatureDLSS_RR, ...) runs on the frame's command list during
// rendering. It does NOT take over the present path, so there is no device-queue /
// swapchain proxying — this is editor-safe and loaded on demand.
//
// The SL lifecycle (slInit / slSetD3DDevice / slShutdown + logging) is shared across all
// features and lives in SLCore (see SLCore.h). This file is RR-specific only.
//
// Per frame (render thread): CreateInstance/Dispatch -> token + options + constants +
// tag + slEvaluateFeature on Unity's command list.
#pragma once

struct ID3D12GraphicsCommandList;
struct SLDlssrrFrameData;

namespace SLDlssrr
{
    // Instance lifecycle (one viewport per instance; viewport id == instance id).
    int  CreateInstance();
    void DestroyInstance(int id);

    // Render-thread per-frame evaluate. Issued from IssuePluginEventAndData; the plugin
    // entry supplies Unity's open D3D12 command list (from CommandRecordingState).
    void Dispatch(SLDlssrrFrameData* data, ID3D12GraphicsCommandList* cmdList);

    // slDLSSDGetOptimalSettings (cached). mode matches the C# UpscalerMode byte.
    bool QueryOptimalRenderSize(unsigned outW, unsigned outH, unsigned char mode,
                                unsigned* outRenderW, unsigned* outRenderH);

    // Release all RR viewport resources/instances. Call before SLCore::Shutdown.
    void Shutdown();
}
